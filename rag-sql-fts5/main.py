"""
Recuperacion de conocimiento (RAG) con SQLite FTS5 y LangChain.

Diagrama:

 Input --> Retriever (SQLite FTS5) --> Contexto --> LLM --> Respuesta

El retriever busca en una base de datos SQLite FTS5 usando el mensaje
del usuario para encontrar productos relevantes. Los resultados se
inyectan como contexto al LLM antes de generar la respuesta.

Este ejemplo crea un pequeno catalogo de productos y usa un
retriever personalizado de LangChain para inyectar filas relevantes
en el contexto del LLM.
"""

import logging
import re
import sqlite3
import os
from typing import Any

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


# -- Catalogo de productos (datos de ejemplo) --

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
            "Bastones de trekking plegables de fibra de carbono con empunaduras de corcho y puntas de tungsteno. "
            "Ajustables de 60 a 137 cm, con amortiguacion anti-vibracion."
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


# -- Base de conocimiento (SQLite + FTS5) --


def create_knowledge_db(db_path: str) -> sqlite3.Connection:
    """Crea (o recrea) el catalogo de productos en SQLite con un indice FTS5."""
    conn = sqlite3.connect(db_path, check_same_thread=False)

    # Eliminar tablas existentes para empezar de nuevo
    conn.execute("DROP TABLE IF EXISTS products_fts")
    conn.execute("DROP TABLE IF EXISTS products")

    conn.execute(
        """
        CREATE TABLE products (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name  TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            description TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO products (name, category, price, description) VALUES (?, ?, ?, ?)",
        [(p["name"], p["category"], p["price"], p["description"]) for p in PRODUCTS],
    )

    # Construir indice de busqueda de texto completo sobre nombre, categoria y descripcion
    conn.execute(
        """
        CREATE VIRTUAL TABLE products_fts USING fts5(
            name, category, description,
            content='products',
            content_rowid='id'
        )
        """
    )
    conn.execute(
        "INSERT INTO products_fts (rowid, name, category, description) "
        "SELECT id, name, category, description FROM products"
    )
    conn.commit()
    logger.info("[Base de conocimiento] Creada con %d productos", len(PRODUCTS))
    return conn


# -- Retriever personalizado para SQLite FTS5 --


class SQLiteFTS5Retriever(BaseRetriever):
    """Retriever de LangChain que busca productos en SQLite FTS5.

    Sigue el patron de "recuperacion de conocimiento" donde se busca
    de manera determinista en la base de conocimiento antes de que el
    LLM genere su respuesta.
    """

    db_conn: Any  # sqlite3.Connection
    max_results: int = 3

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        """Ejecuta una consulta FTS5 y devuelve documentos relevantes."""
        # Extraer palabras, filtrar cortas (len <= 2 elimina "a", "de", "el", etc.)
        words = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ]+", query)
        tokens = [w.lower() for w in words if len(w) > 2]
        if not tokens:
            logger.info("[Retriever] No se encontraron tokens validos en: %s", query)
            return []

        fts_query = " OR ".join(tokens)
        logger.info("[Retriever] Consulta FTS5: %s", fts_query)

        try:
            cursor = self.db_conn.execute(
                """
                SELECT p.name, p.category, p.price, p.description
                FROM products_fts fts
                JOIN products p ON fts.rowid = p.id
                WHERE products_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, self.max_results),
            )
            results = [
                Document(
                    page_content=(
                        f"- **{row[0]}** ({row[1]}, ${row[2]:.2f}): {row[3]}"
                    ),
                    metadata={"name": row[0], "category": row[1], "price": row[2]},
                )
                for row in cursor.fetchall()
            ]
            logger.info(
                "[Retriever] %d producto(s) encontrado(s) para: %s",
                len(results),
                query,
            )
            return results
        except Exception as e:
            logger.warning("Consulta FTS fallo para: %s - %s", fts_query, e)
            return []


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

# Crear la base de conocimiento en archivo local
DB_PATH = "knowledge.sqlite3"
db_conn = create_knowledge_db(DB_PATH)

# Crear el retriever
retriever = SQLiteFTS5Retriever(db_conn=db_conn, max_results=3)

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


# -- Punto de entrada --


def main() -> None:
    """Demuestra el patron de recuperacion de conocimiento (RAG) con varias consultas."""
    queries = [
        # Consulta 1: Deberia encontrar botas de senderismo y bastones de trekking
        "Estoy planeando una excursion. Que botas y bastones me recomiendan?",
        # Consulta 2: Sin coincidencia -- demuestra manejo de "sin conocimiento"
        "Tienen tablas de surf?",
        # Consulta 3: Deberia encontrar binoculares
        "Quiero algo para observar fauna silvestre",
    ]

    print("\n=== Demo de Recuperacion de Conocimiento (RAG) con SQLite FTS5 ===\n")

    for query in queries:
        print(f"[Usuario]: {query}")
        response = rag_chain.invoke(query)
        print(f"[Agente]:  {response}\n")

    db_conn.close()


if __name__ == "__main__":
    main()
