# Knowledge Retrieval (RAG) con SQLite FTS5 y LangChain

## Descripcion

Ejemplo de Retrieval-Augmented Generation (RAG) usando SQLite FTS5 como motor de busqueda de texto completo y LangChain como framework de orquestacion con un LLM de OpenAI.

El patron implementado es simple: antes de que el LLM responda, un **retriever personalizado** busca en una base de datos SQLite con indice FTS5 los productos mas relevantes segun la pregunta del usuario. Los resultados se inyectan como contexto en el prompt del LLM.

Este ejemplo esta basado en [agent_knowledge_sqlite.py](https://github.com/Azure-Samples/python-agentframework-demos/blob/main/examples/agent_knowledge_sqlite.py) del repositorio de Azure Samples, adaptado para usar LangChain en lugar de Microsoft Agent Framework.

## Diagrama

```plaintext
 Input --> Retriever (SQLite FTS5) --> Contexto --> LLM --> Respuesta
             |                                       ^
             |  busca con la pregunta del usuario    |
             v                                       |
         +----------------+                          |
         | Base de        |--------------------------+
         | conocimiento   |   productos relevantes
         | (SQLite FTS5)  |
         +----------------+
```

## Componentes principales

| Componente | Descripcion |
|---|---|
| `SQLiteFTS5Retriever` | Retriever personalizado de LangChain (`BaseRetriever`) que ejecuta consultas FTS5 sobre SQLite y devuelve `Document`s con la informacion de productos. |
| `create_knowledge_db()` | Crea la base de datos SQLite en memoria con una tabla de productos y un indice FTS5 sobre nombre, categoria y descripcion. |
| `rag_chain` | Cadena de LangChain (LCEL) que conecta: retriever -> formateo -> prompt -> LLM -> parser de salida. |
| `format_docs()` | Funcion que convierte los documentos del retriever en texto legible para el prompt. |

## Como funciona FTS5

SQLite FTS5 (Full-Text Search 5) es una extension de SQLite que permite busqueda de texto completo. En este ejemplo:

1. Se crea una tabla virtual `products_fts` con indice FTS5 sobre las columnas `name`, `category` y `description`.
2. Cuando el usuario hace una pregunta, el retriever extrae las palabras clave (tokens) de la pregunta.
3. Se construye una consulta FTS5 con los tokens unidos por `OR` (ej: `"hiking OR boots OR recommend"`).
4. SQLite FTS5 devuelve los productos que coinciden, ordenados por relevancia (`rank`).

## Requisitos

- Python >= 3.13
- Una API key de OpenAI

## Instalacion

```bash
# Crear el entorno virtual (si no existe)
uv sync
```

## Configuracion

Crear un archivo `.env` en la raiz del proyecto con las siguientes variables:

```env
OPENAI_API_KEY="tu-api-key-aqui"
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_LLM_TEMPERATURE=0.7
```

| Variable | Descripcion | Valor por defecto |
|---|---|---|
| `OPENAI_API_KEY` | API key de OpenAI (requerida) | - |
| `DEFAULT_LLM_MODEL` | Modelo de OpenAI a utilizar | `gpt-4o-mini` |
| `DEFAULT_LLM_TEMPERATURE` | Temperatura del LLM (creatividad) | `0.7` |

## Ejecucion

```bash
uv run rag-sql-fts5/main.py
```

## Ejemplo de salida

```
=== Demo de Recuperacion de Conocimiento (RAG) con SQLite FTS5 ===

[Usuario]: Estoy planeando una excursion. Que botas y bastones me recomiendan?
[Agente]:  Te recomiendo las **Botas de Senderismo TrailBlaze** ($149.99)
           y los **Bastones de Trekking TerraFirm** ($59.99)...

[Usuario]: Tienen tablas de surf?
[Agente]:  No tengo informacion sobre ese articulo.

[Usuario]: Quiero algo para observar fauna silvestre
[Agente]:  Te recomiendo los **Binoculares ClearView 10x42** ($129.00)...
```

## Estructura del proyecto

```
rag-sqlite-fts5/
  knowledge.sqlite3     # Base de datos SQLite con el catalogo (se genera al ejecutar)
  main.py               # Codigo principal con el ejemplo RAG
  README.md              # Documentacion del proyecto
```

## Dependencias

| Paquete | Uso |
|---|---|
| `langchain` | Framework de orquestacion para LLMs |
| `langchain-openai` | Integracion de LangChain con la API de OpenAI |
| `python-dotenv` | Carga de variables de entorno desde `.env` |

## Diferencias con el ejemplo original (Azure Agent Framework)

| Concepto | Azure Agent Framework | LangChain |
|---|---|---|
| Proveedor de contexto | `BaseContextProvider` con metodo `before_run` | `BaseRetriever` con metodo `_get_relevant_documents` |
| Inyeccion de contexto | `context.extend_messages()` agrega mensajes al historial | El retriever pasa documentos al prompt via la cadena LCEL |
| Cadena de ejecucion | `agent.run(query)` maneja todo internamente | Cadena LCEL explicita: `retriever -> format -> prompt -> llm -> parser` |
| Base de datos | Igual: SQLite en memoria con FTS5 | Igual: SQLite en memoria con FTS5 |

## Referencias

- [SQLite FTS5 Extension](https://www.sqlite.org/fts5.html)
- [LangChain Custom Retrievers](https://python.langchain.com/docs/how_to/custom_retriever/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/concepts/lcel/)
- [Ejemplo original (Azure Agent Framework)](https://github.com/Azure-Samples/python-agentframework-demos/blob/main/examples/agent_knowledge_sqlite.py)

