"""
Memoria de largo plazo con Redis y LangChain.

A diferencia del historial de chat (sqlite-history), que guarda todos los
mensajes crudos de la conversacion, la memoria de largo plazo extrae
**hechos destilados** de cada turno y los almacena de forma independiente.
Esto permite al agente recordar preferencias y datos clave del usuario
entre sesiones sin necesidad de reenviar toda la conversacion.

Diagrama:

 chat("Recuerda que mi ciudad favorita es Tokio")
 |
 v
 +--------------------------------------------------+
 |  MemoryEnhancedChat                              |
 |                                                  |
 |  ANTES de responder:                             |
 |    Buscar memorias relevantes en Redis            |
 |    (FT.SEARCH con texto completo)                |
 |    Inyectar memorias como contexto del sistema   |
 |                                                  |
 |  Ejecutar la cadena LCEL -> respuesta            |
 |                                                  |
 |  DESPUES de responder:                           |
 |    Extraer hechos nuevos del turno con el LLM    |
 |    Guardar cada hecho en Redis como un Hash      |
 |    (indexado por RediSearch para busqueda)        |
 +--------------------------------------------------+
 |
 v
 respuesta

En Azure Agent Framework esto se implementa con RedisContextProvider
(que guarda y busca contexto conversacional en Redis usando RediSearch).
En LangChain, implementamos el mismo patron manualmente: un LLM extrae
hechos, los guarda en Redis Hashes con un indice RediSearch, y antes de
cada turno busca memorias relevantes para inyectarlas como contexto.

Este ejemplo esta basado en agent_memory_redis.py del repositorio
de Azure Samples, adaptado para usar LangChain.
"""

import json
import logging
import os
import random
import time
import uuid

import redis
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
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
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


# -- Herramienta simulada (datos ficticios) --


def get_weather(city: str) -> str:
    """Devuelve datos del clima simulados para una ciudad."""
    conditions = ["soleado", "nublado", "lluvioso", "tormentoso"]
    temp = random.randint(10, 30)
    condition = conditions[random.randint(0, 3)]
    return f"El clima en {city} esta {condition} con una temperatura maxima de {temp} C."


# -- Almacen de memoria en Redis con RediSearch --


class RedisMemoryStore:
    """Almacena hechos (memorias) en Redis y los busca con RediSearch.

    Cada memoria se guarda como un Hash de Redis con los campos:
      - content: el texto del hecho (ej: "La ciudad favorita del usuario es Tokio")
      - user_id: identificador del usuario
      - timestamp: momento en que se creo la memoria

    Se crea un indice RediSearch sobre el campo 'content' para busqueda
    de texto completo (FT.SEARCH), lo que permite encontrar memorias
    relevantes para la pregunta actual del usuario.

    Equivale al RedisContextProvider de Azure Agent Framework.
    """

    INDEX_NAME = "idx:memories"
    PREFIX = "memory:"

    def __init__(self, redis_url: str, user_id: str) -> None:
        self.r = redis.from_url(redis_url, decode_responses=True)
        self.user_id = user_id
        # Escapar guiones en el user_id para queries de tags de RediSearch.
        # Los guiones son caracteres especiales en la sintaxis de tags.
        self._escaped_user_id = user_id.replace("-", "\\-")
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Crea el indice RediSearch si no existe."""
        try:
            self.r.ft(self.INDEX_NAME).info()
            logger.info("[Redis] Indice '%s' ya existe", self.INDEX_NAME)
        except redis.ResponseError:
            # El indice no existe, crearlo
            from redis.commands.search.field import TagField, TextField
            from redis.commands.search.index_definition import (
                IndexDefinition,
                IndexType,
            )

            schema = (
                TextField("content", weight=1.0),
                TagField("user_id"),
            )
            definition = IndexDefinition(
                prefix=[self.PREFIX],
                index_type=IndexType.HASH,
            )
            self.r.ft(self.INDEX_NAME).create_index(
                schema,
                definition=definition,
            )
            logger.info("[Redis] Indice '%s' creado", self.INDEX_NAME)

    def save_memory(self, content: str) -> str:
        """Guarda un hecho en Redis como un Hash.

        Antes de guardar, verifica si ya existe una memoria con contenido
        identico para este usuario (deduplicacion).
        """
        # Deduplicar: buscar si ya existe este hecho exacto
        existing = self.get_all_memories()
        normalized = content.strip().lower()
        for mem in existing:
            if mem.strip().lower() == normalized:
                logger.info("[Redis] Memoria duplicada, omitida: '%s'", content)
                return ""

        memory_id = f"{self.PREFIX}{uuid.uuid4().hex[:12]}"
        self.r.hset(
            memory_id,
            mapping={
                "content": content,
                "user_id": self.user_id,
                "timestamp": str(int(time.time())),
            },
        )
        logger.info("[Redis] Memoria guardada: '%s'", content)
        return memory_id

    # Stop words en espanol e ingles para filtrar de las queries de busqueda.
    # RediSearch usa AND por defecto, asi que enviar palabras comunes
    # como "es", "mi", "que" causa que la busqueda falle cuando esos
    # terminos no estan en el contenido indexado.
    _STOP_WORDS = frozenset(
        {
            # Espanol
            "a", "al", "algo", "como", "con", "cual", "de", "del", "el",
            "en", "es", "eso", "esta", "esto", "fue", "ha", "hay", "la",
            "las", "le", "lo", "los", "me", "mi", "muy", "no", "nos", "o",
            "para", "pero", "por", "que", "se", "si", "sin", "so", "son",
            "su", "te", "tu", "un", "una", "uno", "y", "ya",
            # Ingles
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "do",
            "for", "from", "has", "have", "he", "her", "his", "how", "i",
            "if", "in", "is", "it", "its", "my", "not", "of", "on", "or",
            "our", "she", "so", "that", "the", "their", "them", "they",
            "this", "to", "was", "we", "what", "when", "which", "who",
            "will", "with", "you", "your",
        }
    )

    def _extract_keywords(self, text: str) -> list[str]:
        """Extrae palabras clave de un texto, removiendo stop words y puntuacion."""
        import unicodedata

        # Normalizar acentos para mejorar coincidencias
        # (ej: "Paris" coincide con "París")
        normalized = unicodedata.normalize("NFKD", text)
        ascii_text = "".join(
            c for c in normalized if not unicodedata.combining(c)
        )

        # Extraer solo palabras alfanumericas
        words = []
        for word in ascii_text.split():
            clean = "".join(c for c in word if c.isalnum())
            if clean and clean.lower() not in self._STOP_WORDS and len(clean) > 1:
                words.append(clean)

        return words

    def search_memories(self, query: str, max_results: int = 5) -> list[str]:
        """Busca memorias relevantes usando RediSearch (texto completo).

        Extrae palabras clave de la consulta del usuario, las une con OR (|)
        y filtra por user_id. Si no hay palabras clave significativas,
        devuelve todas las memorias del usuario como fallback.

        Usa FT.SEARCH con BM25 sobre el campo 'content'.
        """
        try:
            from redis.commands.search.query import Query

            keywords = self._extract_keywords(query)

            if not keywords:
                # Sin palabras clave, devolver todas las memorias del usuario
                logger.info("[Redis] Sin palabras clave, devolviendo todas las memorias")
                return self.get_all_memories()[:max_results]

            # Unir con OR (|) para que cualquier termino coincida
            escaped_terms = [self._escape_query(kw) for kw in keywords]
            text_query = " | ".join(escaped_terms)

            full_query = f"@user_id:{{{self._escaped_user_id}}} ({text_query})"

            q = (
                Query(full_query)
                .return_fields("content")
                .paging(0, max_results)
            )
            results = self.r.ft(self.INDEX_NAME).search(q)

            memories = [doc.content for doc in results.docs]
            logger.info(
                "[Redis] Busqueda '%s' (keywords: %s) -> %d memoria(s)",
                query[:40],
                ", ".join(keywords[:5]),
                len(memories),
            )
            return memories
        except redis.ResponseError as e:
            logger.info("[Redis] Error en busqueda: %s", e)
            return []

    def get_all_memories(self) -> list[str]:
        """Devuelve todas las memorias del usuario."""
        try:
            from redis.commands.search.query import Query

            q = (
                Query(f"@user_id:{{{self._escaped_user_id}}}")
                .return_fields("content", "timestamp")
                .paging(0, 100)
            )
            results = self.r.ft(self.INDEX_NAME).search(q)
            return [doc.content for doc in results.docs]
        except redis.ResponseError:
            return []

    def clear_memories(self) -> int:
        """Elimina todas las memorias del usuario."""
        memories = self.get_all_memories()
        count = 0
        for key in self.r.scan_iter(f"{self.PREFIX}*"):
            if self.r.hget(key, "user_id") == self.user_id:
                self.r.delete(key)
                count += 1
        logger.info("[Redis] %d memoria(s) eliminada(s) para usuario '%s'", count, self.user_id)
        return count

    def close(self) -> None:
        """Cierra la conexion a Redis."""
        self.r.close()
        logger.info("[Redis] Conexion cerrada")

    @staticmethod
    def _escape_query(text: str) -> str:
        """Escapa caracteres especiales de la sintaxis de RediSearch."""
        special_chars = r"@.{}()[]!|&~*^$\-:+=><%#\"'/"
        escaped = []
        for char in text:
            if char in special_chars:
                escaped.append(f"\\{char}")
            else:
                escaped.append(char)
        return "".join(escaped)


# -- Extraccion de hechos con el LLM --

EXTRACT_FACTS_PROMPT = (
    "Analiza el siguiente turno de conversacion y extrae hechos clave "
    "sobre el usuario que valga la pena recordar para futuras interacciones. "
    "Ejemplos de hechos utiles: preferencias, ciudades favoritas, "
    "nombres, alergias, intereses, decisiones tomadas.\n\n"
    "Reglas:\n"
    "- Devuelve SOLO un JSON array de strings, cada uno un hecho conciso.\n"
    "- Si no hay hechos nuevos que extraer, devuelve un array vacio: []\n"
    "- No incluyas hechos triviales como saludos.\n"
    "- Escribe los hechos en tercera persona (ej: 'El usuario prefiere...').\n\n"
    "Ejemplo de salida:\n"
    '[\"La ciudad favorita del usuario es Tokio\", '
    '\"El usuario prefiere temperaturas en Celsius\"]'
)


def extract_facts(llm: ChatOpenAI, user_message: str, ai_response: str) -> list[str]:
    """Usa el LLM para extraer hechos del turno de conversacion."""
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", EXTRACT_FACTS_PROMPT),
                (
                    "human",
                    "Usuario dijo: {user_msg}\nAsistente respondio: {ai_msg}",
                ),
            ]
        )
        | llm
        | StrOutputParser()
    )

    raw = chain.invoke({"user_msg": user_message, "ai_msg": ai_response})

    # Parsear el JSON
    try:
        # Limpiar posibles bloques de codigo markdown
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        facts = json.loads(cleaned)
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, str) and f.strip()]
    except (json.JSONDecodeError, ValueError):
        logger.info("[Memoria] No se pudo parsear la respuesta del LLM: %s", raw[:100])

    return []


# -- Clase principal: chat con memoria de largo plazo --


class MemoryEnhancedChat:
    """Chat multi-turno con memoria de largo plazo en Redis.

    Antes de cada turno:
      1. Busca memorias relevantes en Redis (FT.SEARCH)
      2. Inyecta las memorias como contexto en el system prompt

    Despues de cada turno:
      3. Extrae hechos nuevos del turno usando el LLM
      4. Guarda cada hecho en Redis para futuras sesiones

    Esto permite al agente recordar preferencias y datos del usuario
    sin necesidad de reenviar toda la conversacion anterior.

    Equivale al RedisContextProvider de Azure Agent Framework.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        system_prompt: str,
        memory_store: RedisMemoryStore,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.memory_store = memory_store
        self.conversation_history: list = []

    def _build_system_with_memories(self, query: str) -> str:
        """Construye el system prompt inyectando memorias relevantes."""
        memories = self.memory_store.search_memories(query)

        if not memories:
            return self.system_prompt

        memory_block = "\n".join(f"- {m}" for m in memories)
        return (
            f"{self.system_prompt}\n\n"
            f"Informacion recordada sobre este usuario:\n{memory_block}\n\n"
            f"Usa esta informacion para personalizar tu respuesta."
        )

    def chat(self, user_message: str, tool_data: str | None = None) -> str:
        """Envia un mensaje y devuelve la respuesta, con memoria de largo plazo."""
        # 1. Buscar memorias relevantes e inyectarlas en el system prompt
        enriched_system = self._build_system_with_memories(user_message)

        # 2. Construir el mensaje del usuario (con datos de herramienta si hay)
        full_message = user_message
        if tool_data:
            full_message = f"{user_message}\n\n[Datos disponibles]: {tool_data}"

        # 3. Ejecutar la cadena
        chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", enriched_system),
                    MessagesPlaceholder("history"),
                    ("human", "{question}"),
                ]
            )
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(
            {
                "history": self.conversation_history,
                "question": full_message,
            }
        )

        # 4. Guardar en historial de la sesion actual (en memoria)
        self.conversation_history.append(HumanMessage(content=full_message))
        self.conversation_history.append(AIMessage(content=response))

        # 5. Extraer hechos nuevos y guardarlos en Redis
        facts = extract_facts(self.llm, user_message, response)
        for fact in facts:
            self.memory_store.save_memory(fact)

        if facts:
            logger.info(
                "[Memoria] %d hecho(s) extraido(s) y guardados en Redis",
                len(facts),
            )
        else:
            logger.info("[Memoria] No se extrajeron hechos nuevos de este turno")

        return response

    def new_session(self) -> None:
        """Inicia una nueva sesion (limpia el historial en memoria, NO las memorias en Redis)."""
        self.conversation_history = []
        logger.info("[Memoria] Nueva sesion iniciada (memorias en Redis preservadas)")


# -- Punto de entrada --


def main() -> None:
    """Demuestra un agente con memoria de largo plazo en Redis."""
    # Verificar conectividad con Redis
    r = redis.from_url(REDIS_URL)
    try:
        r.ping()
    except redis.ConnectionError as e:
        logger.error("No se puede conectar a Redis en %s: %s", REDIS_URL, e)
        logger.error(
            "Asegurate de que Redis este corriendo: "
            "cd long-term-memory-redis && docker compose up -d"
        )
        return
    finally:
        r.close()

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    system_prompt = (
        "Eres un asistente de clima util. Personaliza tus respuestas "
        "usando el contexto proporcionado. Antes de responder, siempre "
        "revisa la informacion recordada sobre el usuario."
    )

    # Cada usuario tiene un ID unico para aislar memorias
    user_id = str(uuid.uuid4())

    memory_store = RedisMemoryStore(redis_url=REDIS_URL, user_id=user_id)
    chat = MemoryEnhancedChat(llm=llm, system_prompt=system_prompt, memory_store=memory_store)

    print("\n=== Agente con memoria de largo plazo en Redis ===")
    print(f"    User ID: {user_id[:8]}...")
    print(f"    Redis: {REDIS_URL}")
    print("    Las memorias persisten entre sesiones y se buscan con RediSearch.\n")

    # -- Paso 1: Ensenarle preferencias al agente --
    print("--- Paso 1: Ensenando preferencias ---\n")

    msg1 = "Recuerda que mi ciudad favorita es Tokio y prefiero Celsius."
    print(f"[Usuario]: {msg1}")
    resp1 = chat.chat(msg1)
    print(f"[Agente]:  {resp1}\n")

    # -- Paso 2: Nueva sesion - el agente deberia recordar las preferencias --
    print("--- Paso 2: Nueva sesion -- recordando preferencias ---\n")
    chat.new_session()

    msg2 = "Cual es mi ciudad favorita?"
    print(f"[Usuario]: {msg2}")
    resp2 = chat.chat(msg2)
    print(f"[Agente]:  {resp2}\n")

    # -- Paso 3: Usar herramienta con memoria --
    print("--- Paso 3: Uso de herramientas con memoria ---\n")

    msg3 = "Como esta el clima en Paris?"
    weather_data = get_weather("Paris")
    print(f"[Usuario]: {msg3}")
    resp3 = chat.chat(msg3, tool_data=weather_data)
    print(f"[Agente]:  {resp3}\n")

    msg4 = "Por que ciudad acabo de preguntar y como estuvo el clima?"
    print(f"[Usuario]: {msg4}")
    resp4 = chat.chat(msg4)
    print(f"[Agente]:  {resp4}\n")

    # -- Paso 4: Mostrar las memorias almacenadas en Redis --
    print("--- Memorias almacenadas en Redis ---\n")
    all_memories = memory_store.get_all_memories()
    if all_memories:
        for i, mem in enumerate(all_memories, 1):
            print(f"    {i}. {mem}")
    else:
        print("    (sin memorias)")

    # -- Inspeccion directa de Redis --
    print("\n--- Inspeccion de Redis ---\n")
    keys = [
        k for k in memory_store.r.scan_iter(f"{RedisMemoryStore.PREFIX}*")
    ]
    print(f"    Claves con prefijo '{RedisMemoryStore.PREFIX}': {len(keys)}")
    for key in keys[:5]:
        data = memory_store.r.hgetall(key)
        content = data.get("content", "")
        preview = content[:70] + "..." if len(content) > 70 else content
        print(f"      {key}: {preview}")

    memory_store.close()

    print(
        f"\n    Las memorias persisten en Redis (contenedor 'ltm_redis')."
        "\n    Para limpiar: docker exec ltm_redis redis-cli FLUSHALL"
        "\n    Para detener: cd long-term-memory-redis && docker compose down\n"
    )


if __name__ == "__main__":
    main()
