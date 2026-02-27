"""
Tool Calling (llamada a herramientas) con LangChain.

Un agente de planificacion de fin de semana que demuestra como el LLM
puede invocar funciones Python para obtener datos externos antes de
generar su respuesta.

Diagrama:

 usuario: "Que puedo hacer este fin de semana en San Francisco?"
 |
 v
 +--------------------------------------------------+
 |  Bucle de tool calling                           |
 |                                                  |
 |  1. LLM decide que herramientas necesita         |
 |  2. Se ejecutan las herramientas elegidas        |
 |  3. Los resultados se envian de vuelta al LLM    |
 |  4. LLM genera la respuesta final                |
 +--------------------------------------------------+
 |
 v
 respuesta con recomendaciones personalizadas

Basado en agent_tools.py del repositorio Azure Agent Framework Demos,
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


# -- Herramientas --


@tool
def get_current_date() -> str:
    """Obtiene la fecha actual del sistema en formato YYYY-MM-DD."""
    logger.info("Obteniendo la fecha actual")
    return datetime.now().strftime("%Y-%m-%d")


@tool
def get_weather(city: str) -> dict:
    """Devuelve datos meteorologicos simulados para una ciudad."""
    logger.info("Obteniendo el clima para %s", city)
    opciones = [
        {"temperatura": 22, "descripcion": "soleado"},
        {"temperatura": 18, "descripcion": "nublado"},
        {"temperatura": 15, "descripcion": "lluvioso"},
        {"temperatura": 25, "descripcion": "caluroso y despejado"},
    ]
    return {"ciudad": city, **random.choice(opciones)}


@tool
def get_activities(city: str, date: str) -> list[dict]:
    """Devuelve una lista de actividades para una ciudad y fecha dadas."""
    logger.info("Obteniendo actividades para %s en %s", city, date)
    return [
        {"nombre": "Senderismo", "ubicacion": city},
        {"nombre": "Playa", "ubicacion": city},
        {"nombre": "Museo", "ubicacion": city},
    ]


# -- Bucle de tool calling --

tools = [get_current_date, get_weather, get_activities]
tool_map = {t.name: t for t in tools}


def run(llm_with_tools, messages: list) -> str:
    """Ejecuta el bucle de tool calling hasta obtener una respuesta final.

    Envia los mensajes al LLM; si responde con tool_calls, ejecuta cada
    herramienta, agrega los resultados y vuelve a invocar al LLM. Repite
    hasta que el LLM responda sin tool_calls.
    """
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


# -- Punto de entrada --


def main() -> None:
    llm = ChatOpenAI(
        model=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.7")),
    )
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "Eres un asistente de planificacion de fin de semana. "
                "Ayudas a los usuarios a elegir actividades segun el clima. "
                "Si una actividad seria desagradable con el clima, no la sugieras. "
                "Incluye la fecha del fin de semana en tu respuesta. "
                "Responde en espanol."
            )
        ),
        HumanMessage(content="Que puedo hacer este fin de semana en San Francisco?"),
    ]

    print("\n--- Tool Calling con LangChain ---\n")
    respuesta = run(llm_with_tools, messages)
    print(respuesta)


if __name__ == "__main__":
    main()
