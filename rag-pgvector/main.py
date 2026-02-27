"""
Recuperacion de conocimiento (RAG) con PostgreSQL, pgvector y LangChain.

Busqueda hibrida: vectorial (semantica) + texto completo (palabras clave).

Diagrama:

 Input --> Retriever (PostgreSQL hibrido) --> Contexto --> LLM --> Respuesta
             |                                              ^
             | busca con la pregunta del usuario            |
             v                                              |
         +-------------------+                              |
         | Base de           |------------------------------+
         | conocimiento      |   productos relevantes
         | (PostgreSQL +     |
         |  pgvector)        |
         +-------------------+

El retriever ejecuta una busqueda hibrida que combina:
  1. Busqueda semantica: similitud coseno entre embeddings (pgvector).
  2. Busqueda por palabras clave: texto completo con tsvector de PostgreSQL.

Los resultados se fusionan con Reciprocal Rank Fusion (RRF) para obtener
mejor recuperacion que cualquiera de los dos metodos por separado.

Este ejemplo esta basado en agent_knowledge_postgres.py del repositorio
de Azure Samples, adaptado para usar LangChain en lugar de Microsoft Agent Framework.
"""

import logging
import os
from typing import Any

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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

POSTGRES_USER = os.getenv("POSTGRES_USER", "chris")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "chrisa7")
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_pgvector")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Dimension reducida de embeddings para eficiencia
EMBEDDING_DIMENSIONS = 256
EMBED_MODEL = "text-embedding-3-small"

# Cliente de OpenAI para generar embeddings
embed_client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text: str) -> list[float]:
    """Obtiene un vector de embedding para el texto dado."""
    response = embed_client.embeddings.create(
        input=text, model=EMBED_MODEL, dimensions=EMBEDDING_DIMENSIONS
    )
    return response.data[0].embedding


# -- Catalogo de productos (datos de ejemplo en espanol) --

PRODUCTS = [
    {
        "name": "Botas de Senderismo TrailBlaze",
        "category": "Calzado",
        "price": 149.99,
        "description": (
            "Botas de senderismo impermeables con suelas Vibram, soporte de tobillo "
            "y forro transpirable Gore-Tex. Ideales para senderos rocosos y condiciones humedas."
        ),
    },
    {
        "name": "Mochila SummitPack 40L",
        "category": "Mochilas",
        "price": 89.95,
        "description": (
            "Mochila ligera de 40 litros con compartimento para hidratacion, cubierta de lluvia "
            "y cinturon de cadera ergonomico. Perfecta para excursiones de un dia o con pernocta."
        ),
    },
    {
        "name": "Chaqueta de Plumon ArcticShield",
        "category": "Ropa",
        "price": 199.00,
        "description": (
            "Chaqueta de plumon de ganso 800-fill con clasificacion de -28 C. "
            "Incluye carcasa resistente al agua, diseno comprimible y capucha ajustable."
        ),
    },
    {
        "name": "Remo para Kayak RiverRun",
        "category": "Deportes Acuaticos",
        "price": 74.50,
        "description": (
            "Remo de fibra de vidrio para kayak con ferula ajustable y anillos antigoteo. "
            "Ligero (795 g), apto para kayak recreativo y de travesia."
        ),
    },
    {
        "name": "Bastones de Trekking TerraFirm",
        "category": "Accesorios",
        "price": 59.99,
        "description": (
            "Bastones de trekking plegables de fibra de carbono con empunaduras de corcho "
            "y puntas de tungsteno. Ajustables de 60 a 137 cm, con amortiguacion anti-vibracion."
        ),
    },
    {
        "name": "Binoculares ClearView 10x42",
        "category": "Optica",
        "price": 129.00,
        "description": (
            "Binoculares de prisma de techo con aumento 10x y lentes objetivos de 42 mm. "
            "Cargados con nitrogeno y resistentes al agua. Ideales para observacion de aves y fauna."
        ),
    },
    {
        "name": "Linterna Frontal LED NightGlow",
        "category": "Iluminacion",
        "price": 34.99,
        "description": (
            "Linterna frontal recargable de 350 lumenes con modo de luz roja y haz ajustable. "
            "Clasificacion IPX6 de resistencia al agua, hasta 40 horas en modo bajo."
        ),
    },
    {
        "name": "Saco de Dormir CozyNest",
        "category": "Camping",
        "price": 109.00,
        "description": (
            "Saco de dormir tipo momia para tres estaciones, con clasificacion de -6 C. "
            "Aislamiento sintetico, saco de compresion incluido. Pesa 1.1 kg."
        ),
    },
]


# -- Base de conocimiento (PostgreSQL + pgvector) --


def create_knowledge_db(conn: psycopg.Connection) -> None:
    """Crea el catalogo de productos en PostgreSQL con pgvector e indices de texto completo."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    conn.execute("DROP TABLE IF EXISTS products")
    conn.execute(
        f"""
        CREATE TABLE products (
            id          SERIAL PRIMARY KEY,
            name        TEXT NOT NULL,
            category    TEXT NOT NULL,
            price       REAL NOT NULL,
            description TEXT NOT NULL,
            embedding   vector({EMBEDDING_DIMENSIONS})
        )
        """
    )
    # Indice GIN para busqueda de texto completo sobre nombre + descripcion (idioma espanol)
    conn.execute(
        "CREATE INDEX ON products USING GIN (to_tsvector('spanish', name || ' ' || description))"
    )

    logger.info(
        "[Base de conocimiento] Generando embeddings para %d productos...",
        len(PRODUCTS),
    )
    for product in PRODUCTS:
        text_for_embedding = (
            f"{product['name']} - {product['category']}: {product['description']}"
        )
        embedding = get_embedding(text_for_embedding)
        conn.execute(
            "INSERT INTO products (name, category, price, description, embedding) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                product["name"],
                product["category"],
                product["price"],
                product["description"],
                embedding,
            ),
        )

    conn.commit()
    logger.info("[Base de conocimiento] Catalogo cargado con embeddings.")


# -- SQL de busqueda hibrida con Reciprocal Rank Fusion (RRF) --
# Combina resultados de similitud vectorial y busqueda de texto completo

HYBRID_SEARCH_SQL = f"""
WITH semantic_search AS (
    SELECT id, RANK() OVER (ORDER BY embedding <=> %(embedding)s::vector({EMBEDDING_DIMENSIONS})) AS rank
    FROM products
    ORDER BY embedding <=> %(embedding)s::vector({EMBEDDING_DIMENSIONS})
    LIMIT 20
),
keyword_search AS (
    SELECT id, RANK() OVER (ORDER BY ts_rank_cd(to_tsvector('spanish', name || ' ' || description), query) DESC)
    FROM products, plainto_tsquery('spanish', %(query)s) query
    WHERE to_tsvector('spanish', name || ' ' || description) @@ query
    ORDER BY ts_rank_cd(to_tsvector('spanish', name || ' ' || description), query) DESC
    LIMIT 20
)
SELECT
    COALESCE(semantic_search.id, keyword_search.id) AS id,
    COALESCE(1.0 / (%(k)s + semantic_search.rank), 0.0) +
    COALESCE(1.0 / (%(k)s + keyword_search.rank), 0.0) AS score
FROM semantic_search
FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
ORDER BY score DESC
LIMIT %(limit)s
"""


# -- Retriever personalizado para busqueda hibrida en PostgreSQL --


class PostgresHybridRetriever(BaseRetriever):
    """Retriever de LangChain que ejecuta busqueda hibrida en PostgreSQL.

    Combina busqueda semantica (pgvector, similitud coseno) con busqueda
    por palabras clave (tsvector) usando Reciprocal Rank Fusion (RRF).
    Esto proporciona mejor recuperacion que cualquiera de los dos metodos
    por separado.
    """

    db_conn: Any  # psycopg.Connection
    max_results: int = 3

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        """Ejecuta busqueda hibrida y devuelve documentos relevantes."""
        logger.info("[Retriever] Generando embedding para la consulta...")
        query_embedding = get_embedding(query)

        logger.info(
            "[Retriever] Ejecutando busqueda hibrida (semantica + palabras clave)..."
        )
        cursor = self.db_conn.execute(
            HYBRID_SEARCH_SQL,
            {
                "embedding": query_embedding,
                "query": query,
                "k": 60,
                "limit": self.max_results,
            },
        )
        result_ids = [row[0] for row in cursor.fetchall()]

        if not result_ids:
            logger.info(
                "[Retriever] No se encontraron productos para: %s", query
            )
            return []

        # Obtener detalles completos de los productos encontrados
        documents = []
        for product_id in result_ids:
            row = self.db_conn.execute(
                "SELECT name, category, price, description FROM products WHERE id = %s",
                (product_id,),
            ).fetchone()
            if row:
                documents.append(
                    Document(
                        page_content=(
                            f"- **{row[0]}** ({row[1]}, ${row[2]:.2f}): {row[3]}"
                        ),
                        metadata={
                            "name": row[0],
                            "category": row[1],
                            "price": row[2],
                        },
                    )
                )

        logger.info(
            "[Retriever] %d producto(s) encontrado(s) para: %s",
            len(documents),
            query,
        )
        return documents


# -- Formatear documentos para el contexto del LLM --


def format_docs(docs: list[Document]) -> str:
    """Formatea los documentos del retriever como texto para el prompt."""
    if not docs:
        return "No se encontraron productos relevantes en el catalogo."
    header = "Informacion relevante de productos de nuestro catalogo:\n"
    return header + "\n".join(doc.page_content for doc in docs)


# -- Configuracion del LLM y la cadena RAG --

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente de compras de equipo para actividades al aire libre de la tienda 'TrailBuddy'. "
            "Responde las preguntas del cliente usando SOLO la informacion de productos proporcionada en el contexto. "
            "Si no se encuentran productos relevantes en el contexto, di que no tienes informacion sobre ese articulo. "
            "Incluye precios al recomendar productos.",
        ),
        (
            "system",
            "Contexto:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def setup_db() -> psycopg.Connection:
    """Conecta a PostgreSQL y carga la base de conocimiento."""
    logger.info(
        "[Base de conocimiento] Conectando a PostgreSQL en %s:%s...",
        POSTGRES_HOST,
        POSTGRES_PORT,
    )
    conn = psycopg.connect(POSTGRES_URL)
    create_knowledge_db(conn)
    return conn


# -- Punto de entrada --


def main() -> None:
    """Demuestra el patron de recuperacion de conocimiento (RAG) con busqueda hibrida."""
    db_conn = setup_db()

    # Crear el retriever
    retriever = PostgresHybridRetriever(db_conn=db_conn, max_results=3)

    # Construir la cadena RAG
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    queries = [
        # Consulta 1: Deberia encontrar botas de senderismo y bastones de trekking
        "Estoy planeando una excursion. Que botas y bastones me recomiendan?",
        # Consulta 2: Busqueda semantica -- "observar fauna" deberia encontrar binoculares
        "Quiero algo para observar fauna silvestre",
        # Consulta 3: Deberia encontrar la chaqueta de plumon
        "Necesito algo abrigado para acampar en invierno, tal vez una chaqueta?",
        # Consulta 4: Sin coincidencia -- demuestra manejo de "sin conocimiento"
        "Tienen tablas de surf?",
    ]

    print(
        "\n=== Demo de Recuperacion de Conocimiento (RAG) con PostgreSQL y Busqueda Hibrida ==="
    )
    print(
        "    (pgvector [semantica] + tsvector [palabras clave] con RRF)\n"
    )

    for query in queries:
        print(f"[Usuario]: {query}")
        response = rag_chain.invoke(query)
        print(f"[Agente]:  {response}\n")

    db_conn.close()


if __name__ == "__main__":
    main()
