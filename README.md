# AI/LLM Learning -- Ejemplos de LangChain

Repositorio de aprendizaje sobre IA y LLMs. Cada carpeta contiene un ejemplo
funcional que demuestra un patron o tecnica especifica, adaptado desde los
ejemplos de [Azure Agent Framework](https://github.com/Azure-Samples/python-agentframework-demos)
a **LangChain**.

## Ejemplos

| Ejemplo | Descripcion | Infraestructura |
|---|---|---|
| [`rag-sql-fts5`](rag-sql-fts5/) | RAG con SQLite FTS5: busqueda de texto completo sobre productos usando un indice FTS5 en memoria. | SQLite (en memoria) |
| [`rag-pgvector`](rag-pgvector/) | RAG hibrido con PostgreSQL pgvector: busqueda semantica (embeddings) + busqueda por palabras clave (tsvector), fusionadas con Reciprocal Rank Fusion. | PostgreSQL + pgvector (Docker, puerto 5433) |
| [`summarize-conversation`](summarize-conversation/) | Compactacion de contexto: monitoreo de tokens y resumen automatico de la conversacion cuando se cruza un umbral configurable. | Ninguna (en memoria) |
| [`sqlite-history`](sqlite-history/) | Historial de conversacion persistente con SQLite: los mensajes sobreviven reinicios de la aplicacion usando `BaseChatMessageHistory`. | SQLite (archivo local) |
| [`long-term-memory-redis`](long-term-memory-redis/) | Memoria de largo plazo con Redis: extrae hechos destilados de cada turno con el LLM y los almacena en Redis con indice RediSearch para busqueda de texto completo. | Redis Stack (Docker, puerto 6379) |

## Equivalencias Azure Agent Framework -> LangChain

| Concepto | Azure Agent Framework | LangChain |
|---|---|---|
| Framework principal | `AgentKernel` | `ChatOpenAI` + cadenas LCEL |
| Historial de chat | `BaseHistoryProvider` | `BaseChatMessageHistory` + `RunnableWithMessageHistory` |
| Compactacion de contexto | `AgentMiddleware` (before/after) | Logica manual con `get_openai_callback` + `tiktoken` |
| Memoria de largo plazo | `RedisContextProvider` | `RedisMemoryStore` (clase personalizada con `redis-py`) |
| Retriever (RAG) | `BaseKnowledgeProvider` | `BaseRetriever` de LangChain |
| Busqueda hibrida | No incluido en framework base | pgvector (semantica) + tsvector (keywords) + RRF |
| Prompt templates | Configuracion del agente | `ChatPromptTemplate` + `MessagesPlaceholder` |
| Output parsing | Respuesta directa del agente | `StrOutputParser`, `JsonOutputParser` |

## Requisitos

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes y entornos virtuales)
- Una API key de OpenAI
- Docker (para los ejemplos que usan PostgreSQL o Redis)

## Instalacion

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd ai-llm-learning

# Instalar dependencias con uv
uv sync
```

## Configuracion

Crear un archivo `.env` en la raiz del proyecto:

```env
# LLM Settings
OPENAI_API_KEY="tu-api-key-aqui"
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_LLM_TEMPERATURE=0.9

# PostgreSQL Settings (solo para rag-pgvector)
POSTGRES_USER=chris
POSTGRES_PASSWORD=tu-password
POSTGRES_DB=rag_pgvector
POSTGRES_HOST=localhost
POSTGRES_PORT=5433

# Redis Settings (solo para long-term-memory-redis)
REDIS_URL=redis://localhost:6379
```

## Dependencias

| Paquete | Version | Uso |
|---|---|---|
| `langchain` | >= 1.2.10 | Framework de orquestacion LLM |
| `langchain-openai` | >= 1.1.10 | Integracion con modelos de OpenAI |
| `langchain-community` | >= 0.4.1 | Integraciones comunitarias |
| `openai` | >= 2.24.0 | Cliente de OpenAI (usado por langchain-openai) |
| `pgvector` | >= 0.4.2 | Extension de PostgreSQL para vectores (rag-pgvector) |
| `psycopg[binary]` | >= 3.3.3 | Driver PostgreSQL para Python (rag-pgvector) |
| `redis` | >= 7.2.1 | Cliente Redis con soporte RediSearch (long-term-memory-redis) |
| `tiktoken` | >= 0.12.0 | Tokenizador de OpenAI para conteo de tokens (summarize-conversation) |
| `python-dotenv` | >= 1.2.1 | Carga de variables de entorno desde `.env` |

## Estructura del proyecto

```
ai-llm-learning/
  .env                          Variables de entorno (no versionado)
  pyproject.toml                Dependencias del proyecto (uv)
  README.md                    Este archivo
  rag-sql-fts5/                RAG con SQLite FTS5
  rag-pgvector/                RAG hibrido con pgvector
  summarize-conversation/      Compactacion de contexto
  sqlite-history/              Historial persistente con SQLite
  long-term-memory-redis/      Memoria de largo plazo con Redis
```