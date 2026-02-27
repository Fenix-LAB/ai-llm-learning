"""
Multi-Agent con supervisor en LangChain.

Un agente supervisor que orquesta dos subagentes especialistas:
uno de planificacion de fin de semana y otro de planificacion de comidas.
El supervisor decide cual invocar segun la consulta del usuario.

Diagrama:

 usuario: "mis hijos quieren pasta para la cena y algo divertido el sabado"
 |
 v
 +--------------------------------------------------+
 |  Supervisor                                      |
 |                                                  |
 |  Decide que especialista(s) invocar:             |
 |    plan_weekend -> subagente de fin de semana     |
 |    plan_meal    -> subagente de comidas           |
 |                                                  |
 |  Cada subagente tiene sus propias herramientas:   |
 |                                                  |
 |  Fin de semana:                                  |
 |    get_current_date, get_weather, get_activities  |
 |                                                  |
 |  Comidas:                                        |
 |    find_recipes, check_fridge                     |
 |                                                  |
 |  El supervisor sintetiza las respuestas           |
 +--------------------------------------------------+
 |
 v
 respuesta integrada

Basado en agent_supervisor.py del repositorio Azure Agent Framework Demos,
adaptado para usar LangChain con bind_tools().
"""

import json
import logging
import os
import random
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# -- Configuracion --

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(override=True)


def make_llm() -> ChatOpenAI:
    """Crea una instancia de ChatOpenAI con la configuracion del .env."""
    return ChatOpenAI(
        model=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.7")),
    )


def run(llm_with_tools, messages: list, tool_map: dict) -> str:
    """Ejecuta el bucle de tool calling hasta obtener una respuesta final."""
    while True:
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))


# -- Subagente 1: planificacion de fin de semana --


@tool
def get_current_date() -> str:
    """Obtiene la fecha actual del sistema en formato YYYY-MM-DD."""
    logger.info("Obteniendo la fecha actual")
    return datetime.now().strftime("%Y-%m-%d")


@tool
def get_weather(city: str, date: str) -> dict:
    """Devuelve datos meteorologicos simulados para una ciudad y fecha."""
    logger.info("Obteniendo el clima para %s en %s", city, date)
    if random.random() < 0.05:
        return {"temperatura": 22, "descripcion": "soleado"}
    return {"temperatura": 15, "descripcion": "lluvioso"}


@tool
def get_activities(city: str, date: str) -> list[dict]:
    """Devuelve una lista de actividades para una ciudad y fecha dadas."""
    logger.info("Obteniendo actividades para %s en %s", city, date)
    return [
        {"nombre": "Senderismo", "ubicacion": city},
        {"nombre": "Playa", "ubicacion": city},
        {"nombre": "Museo", "ubicacion": city},
    ]


weekend_tools = [get_current_date, get_weather, get_activities]
weekend_tool_map = {t.name: t for t in weekend_tools}
weekend_llm = make_llm().bind_tools(weekend_tools)


def run_weekend_agent(query: str) -> str:
    """Ejecuta el subagente de fin de semana con su propio contexto."""
    messages = [
        SystemMessage(
            content=(
                "Ayudas a la gente a planear su fin de semana y elegir las "
                "mejores actividades segun el clima. Si una actividad seria "
                "desagradable con el clima, no la sugieras. Incluye la fecha "
                "del fin de semana en tu respuesta."
            )
        ),
        HumanMessage(content=query),
    ]
    return run(weekend_llm, messages, weekend_tool_map)


# -- Subagente 2: planificacion de comidas --


@tool
def find_recipes(query: str) -> list[dict]:
    """Devuelve recetas basadas en una consulta."""
    logger.info("Buscando recetas para '%s'", query)
    if "pasta" in query.lower():
        return [
            {
                "titulo": "Pasta Primavera",
                "ingredientes": ["pasta", "verduras", "aceite de oliva"],
                "pasos": ["Cocinar la pasta.", "Saltear las verduras."],
            }
        ]
    if "tofu" in query.lower():
        return [
            {
                "titulo": "Salteado de Tofu",
                "ingredientes": ["tofu", "salsa de soja", "verduras"],
                "pasos": ["Cortar el tofu en cubos.", "Saltear con las verduras."],
            }
        ]
    return [
        {
            "titulo": "Sandwich de Queso a la Plancha",
            "ingredientes": ["pan", "queso", "mantequilla"],
            "pasos": ["Untar mantequilla.", "Colocar queso.", "Cocinar hasta dorar."],
        }
    ]


@tool
def check_fridge() -> list[str]:
    """Devuelve los ingredientes actualmente en el refrigerador."""
    logger.info("Revisando ingredientes del refrigerador")
    if random.random() < 0.5:
        return ["pasta", "salsa de tomate", "pimientos", "aceite de oliva"]
    return ["tofu", "salsa de soja", "brocoli", "zanahorias"]


meal_tools = [find_recipes, check_fridge]
meal_tool_map = {t.name: t for t in meal_tools}
meal_llm = make_llm().bind_tools(meal_tools)


def run_meal_agent(query: str) -> str:
    """Ejecuta el subagente de comidas con su propio contexto."""
    messages = [
        SystemMessage(
            content=(
                "Ayudas a la gente a planear comidas y elegir las mejores recetas. "
                "Incluye los ingredientes e instrucciones de cocina. "
                "Indica lo que la persona necesita comprar cuando falten "
                "ingredientes en su refrigerador."
            )
        ),
        HumanMessage(content=query),
    ]
    return run(meal_llm, messages, meal_tool_map)


# -- Herramientas del supervisor (wrappers de subagentes) --


@tool
def plan_weekend(query: str) -> str:
    """Planifica un fin de semana segun la consulta del usuario."""
    logger.info("Delegando al subagente de fin de semana")
    return run_weekend_agent(query)


@tool
def plan_meal(query: str) -> str:
    """Planifica una comida segun la consulta del usuario."""
    logger.info("Delegando al subagente de comidas")
    return run_meal_agent(query)


supervisor_tools = [plan_weekend, plan_meal]
supervisor_tool_map = {t.name: t for t in supervisor_tools}
supervisor_llm = make_llm().bind_tools(supervisor_tools)


# -- Punto de entrada --


def main() -> None:
    query = "Mis hijos quieren pasta para la cena y algo divertido el sabado en San Francisco."

    messages = [
        SystemMessage(
            content=(
                "Eres un supervisor que gestiona dos agentes especialistas: "
                "uno de planificacion de fin de semana y otro de planificacion "
                "de comidas. Divide la solicitud del usuario, decide que "
                "especialista invocar (o ambos), y sintetiza una respuesta "
                "final util. Responde en espanol."
            )
        ),
        HumanMessage(content=query),
    ]

    print("\n--- Multi-Agent con Supervisor (LangChain) ---\n")
    print(f"[Usuario]: {query}\n")
    respuesta = run(supervisor_llm, messages, supervisor_tool_map)
    print(respuesta)


if __name__ == "__main__":
    main()
