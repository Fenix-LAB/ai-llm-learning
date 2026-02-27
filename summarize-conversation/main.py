"""
Compactacion de contexto mediante resumen de conversacion con LangChain.

Cuando una conversacion crece, los mensajes acumulados pueden exceder la
ventana de contexto del modelo o volverse costosos. Este ejemplo monitorea
el uso acumulado de tokens y, cuando se cruza un umbral, le pide al LLM
resumir la conversacion hasta ese momento. El resumen reemplaza los
mensajes anteriores y libera espacio para futuros turnos.

Diagrama:

 chain.invoke("mensaje del usuario")
 |
 v
 +--------------------------------------------------+
 |       SummarizationCallback (nivel chat)         |
 |                                                  |
 |  1. Revisar uso acumulado de tokens              |
 |  2. Si pasa el umbral -> resumir mensajes previos|
 |     con el LLM y reemplazarlos por el resumen    |
 |  3. Continuar la ejecucion normal                |
 |  4. Registrar tokens nuevos de la respuesta      |
 +--------------------------------------------------+
 |
 v
 respuesta

En Azure Agent Framework esto se implementa como un AgentMiddleware
con metodos before/after. En LangChain, usamos un callback para el
conteo de tokens y manejamos el historial de mensajes manualmente
con ChatMessageHistory.

Este ejemplo esta basado en agent_summarization.py del repositorio
de Azure Samples, adaptado para usar LangChain.
"""

import logging
import os
import random
from typing import Any

import tiktoken
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# -- Logging --
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)

# -- Configuracion --
load_dotenv(override=True)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.7"))


# -- Herramientas simuladas (datos ficticios) --


def get_weather(city: str) -> str:
    """Devuelve datos del clima simulados para una ciudad."""
    conditions = ["soleado", "nublado", "lluvioso", "parcialmente nublado", "tormentoso"]
    temp = random.randint(5, 35)
    condition = random.choice(conditions)
    return (
        f"Clima en {city}: {condition}, {temp} C. "
        f"Humedad: {random.randint(30, 90)}%. "
        f"Viento: {random.randint(5, 30)} km/h."
    )


def get_activities(city: str) -> str:
    """Devuelve actividades populares simuladas para una ciudad."""
    all_activities = {
        "San Francisco": [
            "Paseo por el Golden Gate",
            "Visitar Fisherman's Wharf",
            "Explorar el barrio de Mission",
            "Tour en bicicleta por la costa",
        ],
        "Portland": [
            "Visitar Powell's Books",
            "Recorrer el jardin japones",
            "Caminar por Forest Park",
            "Explorar la escena gastronomica local",
        ],
        "Seattle": [
            "Visitar Pike Place Market",
            "Subir a la Space Needle",
            "Recorrer el museo de aviacion",
            "Paseo por el lago Union",
        ],
    }
    activities = all_activities.get(
        city,
        ["Explorar el centro historico", "Probar la comida local", "Visitar museos"],
    )
    return f"Actividades en {city}: {', '.join(activities)}."


# -- Conteo de tokens con tiktoken --

_encoding = tiktoken.encoding_for_model(LLM_MODEL)


def count_tokens(text: str) -> int:
    """Cuenta los tokens de un texto usando tiktoken."""
    return len(_encoding.encode(text))


def count_message_tokens(messages: list) -> int:
    """Cuenta el total de tokens en una lista de mensajes de LangChain."""
    total = 0
    for msg in messages:
        # Overhead por mensaje (rol + delimitadores)
        total += 4
        total += count_tokens(msg.content)
    # Overhead final
    total += 2
    return total


# -- Resumen de conversacion --

SUMMARIZE_PROMPT = (
    "Eres un asistente de resumen. Condensa la siguiente conversacion "
    "en un resumen conciso que preserve todos los hechos clave, decisiones y contexto "
    "necesarios para continuar la conversacion. Escribe el resumen en tercera persona. "
    "Se conciso, pero no pierdas detalles importantes como ciudades especificas, "
    "condiciones del clima o recomendaciones que se hayan mencionado."
)


def summarize_history(
    llm: ChatOpenAI,
    messages: list,
) -> str:
    """Llama al LLM para resumir los mensajes de la conversacion."""
    conversation_lines = []
    for msg in messages:
        role = "usuario" if isinstance(msg, HumanMessage) else "asistente"
        if isinstance(msg, SystemMessage):
            role = "sistema"
        conversation_lines.append(f"{role}: {msg.content}")

    conversation_text = "\n".join(conversation_lines)

    summary_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", SUMMARIZE_PROMPT),
                ("human", "Resume esta conversacion:\n\n{conversation}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    return summary_chain.invoke({"conversation": conversation_text})


# -- Clase para manejar la conversacion con resumen automatico --


class SummarizingChat:
    """Maneja una conversacion multi-turno con resumen automatico.

    Cuando el uso acumulado de tokens supera un umbral configurable,
    la conversacion anterior se resume y se reemplaza por un unico
    mensaje de resumen. Esto mantiene manejable la ventana de contexto
    en conversaciones largas.

    Equivale al SummarizationMiddleware de Azure Agent Framework,
    adaptado al patron de LangChain con ChatMessageHistory y callbacks.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        system_prompt: str,
        token_threshold: int = 1000,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.token_threshold = token_threshold
        self.context_tokens = 0
        self.history = InMemoryChatMessageHistory()
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                    MessagesPlaceholder("history"),
                    ("human", "{question}"),
                ]
            )
            | llm
            | StrOutputParser()
        )

    def _check_and_summarize(self) -> None:
        """Revisa el uso de tokens y resume si supera el umbral."""
        messages = self.history.messages

        if self.context_tokens > self.token_threshold and len(messages) > 2:
            logger.info(
                "[Resumen] Uso de tokens (%d) excede el umbral (%d). "
                "Resumiendo %d mensajes del historial...",
                self.context_tokens,
                self.token_threshold,
                len(messages),
            )

            summary_text = summarize_history(self.llm, messages)
            logger.info(
                "[Resumen] Resumen generado: %s",
                summary_text[:200] + "..." if len(summary_text) > 200 else summary_text,
            )

            # Reemplazar historial con un unico mensaje de resumen
            self.history.clear()
            self.history.add_message(
                AIMessage(
                    content=f"[Resumen de la conversacion anterior]\n{summary_text}"
                )
            )

            # Reiniciar contador de tokens
            self.context_tokens = count_message_tokens(self.history.messages)
            logger.info("[Resumen] Historial compactado a 1 mensaje de resumen")
        else:
            logger.info(
                "[Resumen] Uso de tokens: %d / %d umbral. No hace falta resumir.",
                self.context_tokens,
                self.token_threshold,
            )

    def chat(self, user_message: str) -> str:
        """Envia un mensaje al agente y devuelve la respuesta."""
        # Antes de ejecutar: revisar si hay que compactar
        self._check_and_summarize()

        # Ejecutar la cadena con el callback de OpenAI para conteo de tokens
        with get_openai_callback() as cb:
            response = self.chain.invoke(
                {
                    "history": self.history.messages,
                    "question": user_message,
                }
            )

        # Guardar mensajes en el historial
        self.history.add_user_message(user_message)
        self.history.add_ai_message(response)

        # Registrar tokens
        new_tokens = cb.total_tokens
        self.context_tokens += new_tokens
        logger.info(
            "[Resumen] Este turno uso %d tokens (prompt=%d, completion=%d). "
            "Contexto acumulado: %d",
            new_tokens,
            cb.prompt_tokens,
            cb.completion_tokens,
            self.context_tokens,
        )

        return response


# -- Punto de entrada --


def main() -> None:
    """Ejecuta una conversacion multi-turno que dispara el resumen."""
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    system_prompt = (
        "Eres un asistente util para planear el fin de semana. Ayuda a la gente "
        "a planear su fin de semana revisando el clima y sugiriendo actividades. "
        "Se amable y da recomendaciones detalladas."
    )

    # Usar un umbral bajo para que el resumen se dispare rapido en la demo
    chat = SummarizingChat(llm=llm, system_prompt=system_prompt, token_threshold=500)

    # Simulacion de conversacion multi-turno
    conversations = [
        # Turno 1
        (
            "Como estara el clima en San Francisco este fin de semana?",
            lambda: get_weather("San Francisco"),
        ),
        # Turno 2
        (
            "Y Portland? Como estara el clima y que actividades puedo hacer ahi?",
            lambda: f"{get_weather('Portland')} {get_activities('Portland')}",
        ),
        # Turno 3 -- para este punto deberiamos estar cerca del umbral
        (
            "Que tal Seattle? Dame el panorama completo: clima y cosas para hacer.",
            lambda: f"{get_weather('Seattle')} {get_activities('Seattle')}",
        ),
        # Turno 4 -- aqui deberia dispararse el resumen
        (
            "De todas las ciudades que mencionamos, cual tiene la mejor combinacion de clima y actividades?",
            None,
        ),
        # Turno 5 -- despues del resumen, el agente deberia conservar contexto
        (
            "Perfecto, vamos con esa ciudad. Que deberia empacar?",
            None,
        ),
    ]

    print("\n=== Compactacion de contexto con resumen de conversacion ===")
    print(f"    Umbral de tokens: {chat.token_threshold}")
    print(
        "    El middleware resumira la conversacion cuando el uso de tokens supere el umbral.\n"
    )

    for user_msg, tool_fn in conversations:
        # Si hay herramienta, inyectar su resultado como contexto adicional
        if tool_fn:
            tool_result = tool_fn()
            full_message = f"{user_msg}\n\n[Datos disponibles]: {tool_result}"
        else:
            full_message = user_msg

        print(f"[Usuario]: {user_msg}")
        response = chat.chat(full_message)
        print(f"[Agente]:  {response}\n")

    print(f"    Conteo final de tokens en contexto: {chat.context_tokens}")


if __name__ == "__main__":
    main()
