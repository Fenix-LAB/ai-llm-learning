# filepath: /Users/chris/Documents/repositories/ai-projects/ai-llm-learning/sqlite-history/main.py
"""
Historial de conversacion persistente con SQLite y LangChain.

Cuando una aplicacion de chat se reinicia, el historial en memoria se
pierde. Este ejemplo demuestra como persistir los mensajes en una base
de datos SQLite para que la conversacion sobreviva reinicios.

Diagrama:

 chat("mensaje del usuario")
 |
 v
 +--------------------------------------------------+
 |  SQLiteChatHistory (BaseChatMessageHistory)      |
 |                                                  |
 |  1. Cargar mensajes previos desde SQLite         |
 |     usando el session_id                         |
 |  2. Ejecutar la cadena LCEL con el historial     |
 |  3. Guardar mensajes nuevos en SQLite            |
 +--------------------------------------------------+
 |
 v
 respuesta (con memoria de turnos anteriores)

En Azure Agent Framework esto se implementa como un BaseHistoryProvider
con metodos get_messages/save_messages. En LangChain, subclaseamos
BaseChatMessageHistory y lo integramos con RunnableWithMessageHistory.

Este ejemplo esta basado en agent_history_sqlite.py del repositorio
de Azure Samples, adaptado para usar LangChain.
"""

import json
import logging
import os
import random
import sqlite3
import uuid

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
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


# -- Herramienta simulada (datos ficticios) --


def get_weather(city: str) -> str:
    """Devuelve datos del clima simulados para una ciudad."""
    conditions = ["soleado", "nublado", "lluvioso", "tormentoso"]
    temp = random.randint(10, 30)
    condition = conditions[random.randint(0, 3)]
    return f"El clima en {city} esta {condition} con una temperatura maxima de {temp} C."


# -- Historial persistente con SQLite --


class SQLiteChatHistory(BaseChatMessageHistory):
    """Historial de mensajes de chat respaldado por SQLite.

    Cada mensaje se guarda como una fila individual en la tabla `messages`
    con el `session_id` correspondiente. Esto permite:
    - Multiples sesiones en la misma base de datos
    - Recuperar el historial completo de una sesion por orden de insercion
    - Persistencia basada en archivos sin servicios externos

    Equivale al SQLiteHistoryProvider de Azure Agent Framework,
    adaptado a la interfaz BaseChatMessageHistory de LangChain.
    """

    def __init__(self, session_id: str, db_path: str) -> None:
        self.session_id = session_id
        self.db_path = db_path
        # check_same_thread=False es necesario porque RunnableWithMessageHistory
        # puede invocar la lectura del historial desde un thread distinto al que
        # creo la conexion. En produccion se usaria un pool de conexiones.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages (session_id)
            """
        )
        self._conn.commit()
        logger.info(
            "[SQLite] Conexion abierta a '%s' (session: %s)",
            self.db_path,
            self.session_id,
        )

    @property
    def messages(self) -> list[BaseMessage]:
        """Recupera todos los mensajes de esta sesion desde SQLite."""
        cursor = self._conn.execute(
            "SELECT message_json FROM messages WHERE session_id = ? ORDER BY id",
            (self.session_id,),
        )
        rows = cursor.fetchall()
        if not rows:
            return []
        items = [json.loads(row[0]) for row in rows]
        return messages_from_dict(items)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Guarda una lista de mensajes en la base de datos SQLite."""
        self._conn.executemany(
            "INSERT INTO messages (session_id, message_json) VALUES (?, ?)",
            [
                (self.session_id, json.dumps(message_to_dict(msg)))
                for msg in messages
            ],
        )
        self._conn.commit()
        logger.info(
            "[SQLite] %d mensaje(s) guardados en session '%s'",
            len(messages),
            self.session_id,
        )

    def clear(self) -> None:
        """Elimina todos los mensajes de esta sesion."""
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (self.session_id,),
        )
        self._conn.commit()
        logger.info("[SQLite] Historial limpiado para session '%s'", self.session_id)

    def close(self) -> None:
        """Cierra la conexion a SQLite."""
        self._conn.close()
        logger.info("[SQLite] Conexion cerrada")

    def get_session_stats(self) -> dict:
        """Devuelve estadisticas de la sesion actual."""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (self.session_id,),
        )
        msg_count = cursor.fetchone()[0]

        cursor = self._conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM messages",
        )
        total_sessions = cursor.fetchone()[0]

        cursor = self._conn.execute(
            "SELECT SUM(LENGTH(message_json)) FROM messages WHERE session_id = ?",
            (self.session_id,),
        )
        total_bytes = cursor.fetchone()[0] or 0

        return {
            "session_id": self.session_id,
            "message_count": msg_count,
            "total_sessions_in_db": total_sessions,
            "storage_bytes": total_bytes,
        }


# -- Funciones auxiliares --


def create_chain(llm: ChatOpenAI):
    """Crea la cadena LCEL con soporte para historial de mensajes."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system}"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def inspect_db(db_path: str) -> None:
    """Muestra el contenido de la base de datos SQLite para inspeccion."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT session_id, COUNT(*) as msg_count "
        "FROM messages GROUP BY session_id ORDER BY session_id"
    )
    sessions = cursor.fetchall()
    print("\n    --- Contenido de la base de datos SQLite ---")
    print(f"    Archivo: {db_path}")
    print(f"    Sesiones encontradas: {len(sessions)}")
    for sid, count in sessions:
        print(f"      Session {sid[:8]}...: {count} mensajes")

    # Mostrar los mensajes de la ultima sesion
    if sessions:
        last_sid = sessions[-1][0]
        cursor = conn.execute(
            "SELECT id, message_json FROM messages "
            "WHERE session_id = ? ORDER BY id",
            (last_sid,),
        )
        print(f"\n    Mensajes de la sesion {last_sid[:8]}...:")
        for row_id, msg_json in cursor.fetchall():
            data = json.loads(msg_json)
            msg_type = data.get("type", "desconocido")
            content = data.get("data", {}).get("content", "")
            preview = content[:80] + "..." if len(content) > 80 else content
            print(f"      [{row_id}] {msg_type}: {preview}")

    conn.close()


# -- Punto de entrada --


def main() -> None:
    """Demuestra una sesion con SQLite que persiste el historial en un archivo local."""
    db_path = "sqlite-history/chat_history.sqlite3"
    session_id = str(uuid.uuid4())

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    system_prompt = (
        "Eres un agente de clima util. Cuando el usuario pregunte por el "
        "clima de una ciudad, usa los datos proporcionados para dar una "
        "respuesta amable y concisa. Recuerda las ciudades que se han "
        "mencionado en la conversacion."
    )

    chain = create_chain(llm)

    # -- Fase 1: Iniciar una conversacion --
    print("\n=== Sesion persistente en SQLite ===")
    print("--- Fase 1: Iniciando conversacion ---\n")

    history1 = SQLiteChatHistory(session_id=session_id, db_path=db_path)

    # Funcion para obtener el historial por session_id
    # RunnableWithMessageHistory la llama internamente
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda sid: SQLiteChatHistory(session_id=sid, db_path=db_path),
        input_messages_key="question",
        history_messages_key="history",
    )

    config1 = {"configurable": {"session_id": session_id}}

    # Turno 1: Preguntar por Tokio
    weather_tokyo = get_weather("Tokio")
    question1 = f"Como esta el clima en Tokio?\n\n[Datos disponibles]: {weather_tokyo}"
    print(f"[Usuario]: Como esta el clima en Tokio?")
    response1 = chain_with_history.invoke(
        {"system": system_prompt, "question": question1},
        config=config1,
    )
    print(f"[Agente]:  {response1}\n")

    # Turno 2: Preguntar por Paris
    weather_paris = get_weather("Paris")
    question2 = f"Y Paris?\n\n[Datos disponibles]: {weather_paris}"
    print(f"[Usuario]: Y Paris?")
    response2 = chain_with_history.invoke(
        {"system": system_prompt, "question": question2},
        config=config1,
    )
    print(f"[Agente]:  {response2}\n")

    # Mostrar estadisticas de la sesion
    stats = history1.get_session_stats()
    print(f"    Mensajes en la sesion: {stats['message_count']}")
    print(f"    Almacenamiento: {stats['storage_bytes']} bytes")

    # Cerrar la conexion (simular cierre de la aplicacion)
    history1.close()

    # -- Fase 2: Simular un reinicio de la aplicacion --
    print("\n--- Fase 2: Reanudando despues del 'reinicio' ---\n")
    print("    (Se creo una nueva instancia de SQLiteChatHistory")
    print(f"     reconectando al mismo session_id: {session_id[:8]}...)\n")

    # Crear una nueva instancia del historial y del agente
    # (como si la aplicacion se hubiera reiniciado)
    chain_with_history2 = RunnableWithMessageHistory(
        chain,
        lambda sid: SQLiteChatHistory(session_id=sid, db_path=db_path),
        input_messages_key="question",
        history_messages_key="history",
    )

    config2 = {"configurable": {"session_id": session_id}}

    # Turno 3: Preguntar algo que requiere memoria de turnos anteriores
    question3 = "Cual de las ciudades por las que pregunte tuvo mejor clima?"
    print(f"[Usuario]: {question3}")
    response3 = chain_with_history2.invoke(
        {"system": system_prompt, "question": question3},
        config=config2,
    )
    print(f"[Agente]:  {response3}\n")

    # Turno 4: Preguntar por una tercera ciudad para seguir acumulando historial
    weather_london = get_weather("Londres")
    question4 = f"Y como esta el clima en Londres?\n\n[Datos disponibles]: {weather_london}"
    print(f"[Usuario]: Y como esta el clima en Londres?")
    response4 = chain_with_history2.invoke(
        {"system": system_prompt, "question": question4},
        config=config2,
    )
    print(f"[Agente]:  {response4}\n")

    # Mostrar estadisticas finales
    history_final = SQLiteChatHistory(session_id=session_id, db_path=db_path)
    stats = history_final.get_session_stats()
    print(f"    Mensajes totales en la sesion: {stats['message_count']}")
    print(f"    Sesiones en la BD: {stats['total_sessions_in_db']}")
    print(f"    Almacenamiento: {stats['storage_bytes']} bytes")
    history_final.close()

    # Inspeccionar la base de datos
    inspect_db(db_path)

    print(
        f"\n    El archivo '{db_path}' persiste en disco."
        "\n    Puedes eliminarlo manualmente o ejecutar de nuevo para agregar mas sesiones.\n"
    )


if __name__ == "__main__":
    main()
