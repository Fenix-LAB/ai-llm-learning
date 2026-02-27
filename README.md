# AI/LLM Learning - Guia de Estudio

Repositorio de aprendizaje con ejemplos practicos de patrones comunes en
aplicaciones con LLMs, adaptados de
[Azure Agent Framework](https://github.com/Azure-Samples/python-agentframework-demos)
a **LangChain**.

---

## Indice

1. [Mapa general](#1-mapa-general)
2. [Conceptos fundamentales](#2-conceptos-fundamentales)
3. [Tool Calling](#3-tool-calling)
4. [Multi-Agent (Supervisor)](#4-multi-agent-supervisor)
5. [Historial de chat persistente](#5-historial-de-chat-persistente)
6. [Compactacion de contexto (resumen)](#6-compactacion-de-contexto-resumen)
7. [Memoria a largo plazo](#7-memoria-a-largo-plazo)
8. [RAG (Retrieval-Augmented Generation)](#8-rag-retrieval-augmented-generation)
9. [Diferencias que confunden](#9-diferencias-que-confunden)
10. [Equivalencias Azure Agent Framework - LangChain](#10-equivalencias-azure-agent-framework---langchain)
11. [Referencia rapida de ejecucion](#11-referencia-rapida-de-ejecucion)

---

## 1. Mapa general

Orden recomendado de estudio. Cada bloque construye sobre el anterior:

```
+------------------------------------------------------------------+
|                                                                  |
|  (1) Tool Calling         Lo mas basico: el LLM invoca funciones |
|       tool-calling/                                              |
|            |                                                     |
|            v                                                     |
|  (2) Multi-Agent          Multiples agentes coordinados          |
|       multi-agent/        (reutiliza tool calling)               |
|            |                                                     |
|            v                                                     |
|  (3) Historial            Persistir la conversacion              |
|       sqlite-history/     entre reinicios                        |
|            |                                                     |
|            v                                                     |
|  (4) Compactacion         Resumir conversacion larga             |
|       summarize-          para no exceder la ventana             |
|       conversation/       de contexto                            |
|            |                                                     |
|            v                                                     |
|  (5) Memoria largo plazo  Recordar hechos del usuario            |
|       long-term-          entre sesiones distintas               |
|       memory-redis/                                              |
|            |                                                     |
|            v                                                     |
|  (6) RAG                  Inyectar conocimiento externo          |
|       rag-sql-fts5/       desde una base de datos                |
|       rag-pgvector/                                              |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 2. Conceptos fundamentales

Antes de entrar a los ejemplos, estos son los bloques basicos
que aparecen en todo el repositorio.

### Mensajes y roles

Todo chat con un LLM es una lista de mensajes. Cada mensaje tiene un **rol**:

```
+------------------+---------------------------------------------------+
| Rol              | Para que sirve                                    |
+------------------+---------------------------------------------------+
| SystemMessage    | Instrucciones para el LLM (personalidad, reglas). |
|                  | Va al inicio, el usuario no lo ve.                |
+------------------+---------------------------------------------------+
| HumanMessage     | Lo que dice el usuario.                           |
+------------------+---------------------------------------------------+
| AIMessage        | Lo que responde el LLM.                           |
+------------------+---------------------------------------------------+
| ToolMessage      | Resultado de una herramienta ejecutada.            |
|                  | Solo aparece en tool calling.                     |
+------------------+---------------------------------------------------+
```

El LLM siempre recibe la lista completa y genera el siguiente AIMessage.

### Ventana de contexto

El LLM tiene un limite de tokens que puede recibir + generar en una sola
llamada. Todo lo que envias (system + historial + herramientas + pregunta)
debe caber en esa ventana.

```
+---------------------------------------------------------------+
|                   Ventana de contexto (ej. 128k tokens)       |
|                                                               |
|  [SystemMessage]                                              |
|  [HumanMessage 1]                                             |
|  [AIMessage 1]                                                |
|  [HumanMessage 2]                                             |
|  [AIMessage 2]           <-- crece con cada turno             |
|  ...                                                          |
|  [HumanMessage N]                                             |
|  [AIMessage N]  <-- respuesta generada                        |
|                                                               |
|  Si la lista crece mucho --> se excede la ventana --> error    |
+---------------------------------------------------------------+
```

Esto motiva los patrones de **compactacion** (seccion 6) y
**memoria a largo plazo** (seccion 7).

### Temperature

Controla la aleatoriedad de las respuestas:

```
temperature = 0.0  -->  Determinista, siempre la misma respuesta
temperature = 0.7  -->  Balance entre creatividad y coherencia
temperature = 1.0  -->  Mas creativo, menos predecible
```

Para tool calling y tareas precisas conviene 0.0-0.3.
Para conversacion general, 0.7-0.9.

---

## 3. Tool Calling

> Carpeta: [`tool-calling/`](tool-calling/)

### Que es

El LLM no puede ejecutar codigo, acceder a APIs ni saber la fecha actual.
**Tool calling** le permite al LLM **pedir** que ejecutemos una funcion
de Python por el, recibir el resultado y usarlo en su respuesta.

### Flujo paso a paso

```
 usuario: "Que puedo hacer este fin de semana en SF?"
    |
    v
 +----------------------------------------------+
 |  LLM analiza la pregunta y decide:           |
 |  "Necesito 3 herramientas"                   |
 |                                               |
 |  Responde con tool_calls (NO texto):          |
 |    - get_current_date()                       |
 |    - get_weather("San Francisco")             |
 |    - get_activities("San Francisco", "2026-02-28")
 +----------------------------------------------+
    |
    v
 +----------------------------------------------+
 |  Nuestro codigo ejecuta cada funcion          |
 |  y envia los resultados como ToolMessage      |
 +----------------------------------------------+
    |
    v
 +----------------------------------------------+
 |  LLM recibe los resultados y ahora SI         |
 |  genera una respuesta de texto final           |
 |  integrando todos los datos                   |
 +----------------------------------------------+
    |
    v
 "Este sabado 28 de febrero el clima en SF sera
  soleado (22 C). Te recomiendo senderismo..."
```

### Concepto clave: el bucle de tool calling

El LLM puede necesitar multiples rondas de herramientas. Por eso
se implementa como un **bucle**:

```
mientras el LLM responda con tool_calls:
    ejecutar cada herramienta
    enviar resultados al LLM
cuando responda con texto:
    retornar la respuesta
```

En Azure Agent Framework, `agent.run()` ejecuta este bucle internamente.
En LangChain no hay un `agent.run()` equivalente, asi que lo implementamos
con una funcion `run()` explicita. Es simple pero hay que entenderlo.

### Anatomia de un @tool

```python
@tool
def get_weather(city: str) -> dict:
    """Devuelve datos meteorologicos simulados para una ciudad."""
    return {"temperatura": 22, "descripcion": "soleado"}
```

Tres cosas importan:
- **El nombre** de la funcion: el LLM lo usa para referirse a ella
- **El docstring**: el LLM lo lee para decidir cuando usarla
- **Los type hints**: LangChain genera el JSON Schema automaticamente

**Tip**: si el LLM no invoca tu herramienta, revisa el docstring.
Un docstring vago = el LLM no sabe cuando usarla.

---

## 4. Multi-Agent (Supervisor)

> Carpeta: [`multi-agent/`](multi-agent/)

### Que es

Un **supervisor** que gestiona multiples **subagentes especialistas**.
Cada subagente sabe hacer una cosa bien (planear el fin de semana,
planear comidas). El supervisor decide a quien delegar.

### Flujo

```
 usuario: "Quiero pasta para cenar y algo divertido el sabado"
    |
    v
 +--------------------------------------------------+
 |  SUPERVISOR                                      |
 |  "Necesito ambos especialistas"                  |
 |                                                  |
 |  tool_calls:                                     |
 |    plan_meal("pasta para cenar")                 |
 |    plan_weekend("algo divertido el sabado")      |
 +----+------------------------+--------------------+
      |                        |
      v                        v
 +----------------+     +------------------+
 | SUBAGENTE      |     | SUBAGENTE        |
 | COMIDAS        |     | FIN DE SEMANA    |
 |                |     |                  |
 | find_recipes() |     | get_weather()    |
 | check_fridge() |     | get_activities() |
 |                |     | get_current_date()|
 +-------+--------+     +--------+---------+
         |                       |
         v                       v
      resultado               resultado
         |                       |
         +-----------+-----------+
                     |
                     v
 +--------------------------------------------------+
 |  SUPERVISOR sintetiza ambas respuestas           |
 |  en una respuesta final unificada                |
 +--------------------------------------------------+
    |
    v
 "Para cenar: Pasta Primavera con...
  Para el sabado: el clima sera lluvioso,
  te recomiendo visitar un museo..."
```

### Concepto clave: subagentes como herramientas

El patron es elegante: cada subagente se **envuelve** como un `@tool`
que el supervisor puede invocar:

```python
# El subagente tiene sus propias herramientas y su propio contexto
def run_meal_agent(query: str) -> str:
    messages = [SystemMessage(...), HumanMessage(content=query)]
    return run(meal_llm, messages, meal_tool_map)

# Se expone al supervisor como una herramienta mas
@tool
def plan_meal(query: str) -> str:
    """Planifica una comida segun la consulta del usuario."""
    return run_meal_agent(query)

# El supervisor solo ve plan_meal y plan_weekend
supervisor_llm = make_llm().bind_tools([plan_weekend, plan_meal])
```

**Ventaja**: el supervisor nunca ve los detalles internos de cada
subagente (sus herramientas, sus mensajes intermedios). Solo recibe
el resultado final. Esto mantiene su ventana de contexto limpia.

---

## 5. Historial de chat persistente

> Carpeta: [`sqlite-history/`](sqlite-history/)

### Que es

Por defecto, la lista de mensajes vive en memoria. Si el programa se
reinicia, se pierde todo. Este patron **persiste** los mensajes en
SQLite para que la conversacion sobreviva reinicios.

### Flujo

```
 chat("Hola, me llamo Carlos")
    |
    v
 +--------------------------------------------------+
 |  SQLiteChatHistory                               |
 |                                                  |
 |  1. SELECT mensajes WHERE session_id = "abc"     |
 |     -> [mensajes previos]                        |
 |                                                  |
 |  2. Agregar HumanMessage("Hola, me llamo Carlos")|
 |                                                  |
 |  3. Enviar todo al LLM -> AIMessage              |
 |                                                  |
 |  4. INSERT nuevos mensajes en SQLite             |
 +--------------------------------------------------+
    |
    v
 "Hola Carlos, en que te puedo ayudar?"

 --- programa se reinicia ---

 chat("Como me llamo?")
    |
    v
 +--------------------------------------------------+
 |  SQLiteChatHistory                               |
 |                                                  |
 |  1. SELECT -> recupera [Hola me llamo Carlos,    |
 |               Hola Carlos en que te puedo...]    |
 |                                                  |
 |  2. El LLM ve todo el historial anterior         |
 +--------------------------------------------------+
    |
    v
 "Te llamas Carlos"      <-- recuerda!
```

### Concepto clave: session_id

Un `session_id` identifica una conversacion. Multiples usuarios
o multiples conversaciones del mismo usuario usan IDs distintos:

```
session_id = "usuario-1-conv-1"  -->  [mensajes de esa conversacion]
session_id = "usuario-1-conv-2"  -->  [mensajes de otra conversacion]
session_id = "usuario-2-conv-1"  -->  [mensajes de otro usuario]
```

---

## 6. Compactacion de contexto (resumen)

> Carpeta: [`summarize-conversation/`](summarize-conversation/)

### Que es

Si la conversacion crece mucho, los mensajes acumulados pueden exceder
la ventana de contexto o volverse costosos. Este patron **resume**
automaticamente la conversacion cuando cruza un umbral de tokens.

### El problema

```
Turno 1:  [System, Human1, AI1]                        = 200 tokens
Turno 5:  [System, Human1, AI1, ..., Human5, AI5]      = 800 tokens
Turno 10: [System, Human1, AI1, ..., Human10, AI10]    = 2000 tokens
                                                          ^
                                                 crece sin limite!
```

### La solucion

```
 Turno 5: tokens_acumulados = 800 > umbral (500)
    |
    v
 +--------------------------------------------------+
 |  Compactacion                                    |
 |                                                  |
 |  1. Tomar todos los mensajes previos             |
 |  2. Pedirle al LLM: "Resume esta conversacion"  |
 |  3. Reemplazar los mensajes por:                 |
 |     SystemMessage("Resumen: el usuario pregunto  |
 |     sobre el clima en SF y Tokyo...")            |
 +--------------------------------------------------+
    |
    v
 Turno 6: [System, Resumen, Human6]                = 150 tokens
                                                      ^
                                          mucho mas compacto!
```

### Concepto clave: umbral de tokens

El umbral determina cuando se activa el resumen. Un umbral bajo
resume mas seguido (perdiendo detalle). Uno alto permite mas
contexto pero usa mas tokens por llamada.

```
umbral = 500   -->  resume rapido, bueno para demos
umbral = 4000  -->  mas contexto antes de resumir, mas natural
umbral = 16000 -->  casi nunca resume, cuesta mas por llamada
```

**Tip**: contar tokens no es trivial. Usamos `tiktoken` (la misma
libreria que usa OpenAI internamente) para contar con precision.

---

## 7. Memoria a largo plazo

> Carpeta: [`long-term-memory-redis/`](long-term-memory-redis/)

### Que es

El historial de chat (seccion 5) guarda **todos los mensajes crudos**.
La memoria a largo plazo extrae **hechos destilados** y los almacena
por separado. Esto permite recordar cosas del usuario entre sesiones
completamente distintas.

### Diferencia clave vs historial

```
+-----------------------------+-------------------------------------+
| Historial de chat           | Memoria a largo plazo               |
| (sqlite-history)            | (long-term-memory-redis)            |
+-----------------------------+-------------------------------------+
| Guarda: mensajes completos  | Guarda: hechos extraidos            |
| "Hola, me llamo Carlos y    | "nombre: Carlos"                   |
| vivo en Madrid, me gusta    | "ciudad: Madrid"                   |
| el sushi y tengo 2 gatos"   | "comida favorita: sushi"            |
|                              | "mascotas: 2 gatos"                |
+-----------------------------+-------------------------------------+
| Alcance: una sesion          | Alcance: todas las sesiones         |
+-----------------------------+-------------------------------------+
| Busqueda: secuencial         | Busqueda: por relevancia (texto)    |
+-----------------------------+-------------------------------------+
| Crece sin limite             | Solo hechos clave, compacto         |
+-----------------------------+-------------------------------------+
```

### Flujo

```
 chat("Recuerda que mi ciudad favorita es Tokio")
    |
    v
 +--------------------------------------------------+
 |  ANTES de responder:                             |
 |  Buscar en Redis: "ciudad favorita Tokio"        |
 |  -> No hay memorias previas                      |
 +--------------------------------------------------+
    |
    v
 LLM responde: "Anotado, Tokio es tu ciudad favorita"
    |
    v
 +--------------------------------------------------+
 |  DESPUES de responder:                           |
 |  Extraer hechos con el LLM:                      |
 |    "ciudad favorita del usuario: Tokio"           |
 |  Guardar en Redis como Hash indexado             |
 +--------------------------------------------------+

 --- dias despues, nueva sesion ---

 chat("Que clima hace en mi ciudad favorita?")
    |
    v
 +--------------------------------------------------+
 |  ANTES de responder:                             |
 |  Buscar en Redis: "ciudad favorita"              |
 |  -> Encontrado: "ciudad favorita: Tokio"         |
 |  Inyectar como contexto del sistema              |
 +--------------------------------------------------+
    |
    v
 LLM responde: "En Tokio el clima hoy es..."
                 ^
                 sabe que es Tokio por la memoria!
```

### Concepto clave: extraccion de hechos

Un segundo LLM (o el mismo con un prompt distinto) analiza cada turno
y extrae hechos en formato estructurado. No todo lo que dice el usuario
es un "hecho" que valga la pena recordar:

```
"Hola, que tal?"                    --> ningun hecho
"Me llamo Carlos"                   --> "nombre: Carlos"
"Me encanta la pasta con pesto"     --> "comida favorita: pasta con pesto"
"Que hora es?"                      --> ningun hecho
```

---

## 8. RAG (Retrieval-Augmented Generation)

> Carpetas: [`rag-sql-fts5/`](rag-sql-fts5/) y [`rag-pgvector/`](rag-pgvector/)

### Que es

El LLM solo sabe lo que vio durante su entrenamiento. **RAG** le da
acceso a conocimiento externo (una base de datos, documentos, etc.)
buscando informacion relevante y agregandola al prompt antes de
que el LLM responda.

### Flujo basico

```
 usuario: "Que laptop me recomiendas para programar?"
    |
    v
 +----------------------------------------------+
 |  RETRIEVER                                   |
 |  Busca en la base de datos con la pregunta   |
 |                                               |
 |  Resultados:                                  |
 |    - MacBook Pro M3, $2499, para desarrollo  |
 |    - ThinkPad X1, $1899, para programadores  |
 +----------------------------------------------+
    |
    v
 +----------------------------------------------+
 |  PROMPT                                       |
 |                                               |
 |  "Basandote en estos productos:               |
 |   [MacBook Pro M3..., ThinkPad X1...]         |
 |   Responde: Que laptop me recomiendas         |
 |   para programar?"                            |
 +----------------------------------------------+
    |
    v
 LLM genera respuesta usando el contexto inyectado
```

### Dos tipos de busqueda

Este repositorio implementa ambos tipos:

```
+----------------------------+------------------------------------------+
| Busqueda por palabras      | Busqueda semantica                       |
| clave (keyword)            | (vectorial)                              |
+----------------------------+------------------------------------------+
| rag-sql-fts5/              | rag-pgvector/                            |
+----------------------------+------------------------------------------+
| Busca coincidencias        | Busca por significado                    |
| exactas de palabras        | (embeddings)                             |
+----------------------------+------------------------------------------+
| "laptop programar"         | "computadora para escribir codigo"       |
| encuentra: "laptop"        | encuentra: "laptop para desarrollo"      |
|                            | (aunque no diga "laptop" ni "programar") |
+----------------------------+------------------------------------------+
| Rapida, simple             | Mas inteligente, requiere embeddings     |
+----------------------------+------------------------------------------+
| SQLite FTS5                | PostgreSQL + pgvector                    |
+----------------------------+------------------------------------------+
```

### Busqueda hibrida (rag-pgvector)

El ejemplo avanzado combina **ambos metodos** y fusiona los resultados
con Reciprocal Rank Fusion (RRF):

```
 pregunta del usuario
    |
    +---> Busqueda semantica (pgvector coseno)  --> ranking 1
    |
    +---> Busqueda keyword (tsvector)           --> ranking 2
    |
    v
 +----------------------------------------------+
 |  RRF (Reciprocal Rank Fusion)                |
 |                                               |
 |  Combina ambos rankings en uno solo.          |
 |  Un documento que aparece bien rankeado       |
 |  en AMBAS busquedas sube al top.              |
 +----------------------------------------------+
    |
    v
 Top-K documentos --> se inyectan al prompt del LLM
```

### Concepto clave: embeddings

Un embedding es un vector numerico que representa el **significado**
de un texto. Textos con significado similar tienen vectores cercanos:

```
"gato"    --> [0.21, 0.85, 0.12, ...]
"felino"  --> [0.23, 0.83, 0.14, ...]   <-- cercano a "gato"
"avion"   --> [0.91, 0.05, 0.77, ...]   <-- lejano de "gato"
```

Se generan con un modelo de embeddings (ej. `text-embedding-3-small`
de OpenAI) y se almacenan en la base de datos para busquedas rapidas.

---

## 9. Diferencias que confunden

### Tool calling vs RAG

```
+---------------------------+-------------------------------------------+
| Tool Calling              | RAG                                       |
+---------------------------+-------------------------------------------+
| El LLM DECIDE que llamar  | Nosotros SIEMPRE buscamos antes de llamar |
| (puede elegir 0 tools)    | al LLM (el retriever corre siempre)       |
+---------------------------+-------------------------------------------+
| El LLM ve el nombre y     | El LLM no sabe que hay un retriever,      |
| descripcion de cada tool   | solo ve el contexto inyectado             |
+---------------------------+-------------------------------------------+
| Para: acciones, APIs,     | Para: conocimiento, documentos,           |
| funciones dinamicas        | bases de datos estaticas                  |
+---------------------------+-------------------------------------------+
| Ejemplo: "que hora es?"   | Ejemplo: "que laptop me recomiendas?"     |
| -> get_current_date()     | -> busca en catalogo -> inyecta contexto  |
+---------------------------+-------------------------------------------+
```

**Tip**: se pueden combinar. Un `@tool` puede internamente hacer
una busqueda RAG. El LLM decide cuando necesita buscar informacion
y la herramienta ejecuta el retriever.

### Historial vs Memoria a largo plazo

```
+-------------------------------+---------------------------------------+
| Historial de chat             | Memoria a largo plazo                 |
| (sqlite-history)              | (long-term-memory-redis)              |
+-------------------------------+---------------------------------------+
| Guarda TODO lo que se dijo    | Guarda solo HECHOS destilados         |
+-------------------------------+---------------------------------------+
| Una sesion / conversacion     | Cruza sesiones (el usuario vuelve     |
|                               | semanas despues y lo recuerda)        |
+-------------------------------+---------------------------------------+
| Se envia completo al LLM     | Se buscan solo memorias relevantes    |
| en cada turno                 | a la pregunta actual                  |
+-------------------------------+---------------------------------------+
| Crece linealmente             | Crece lento (solo hechos clave)       |
+-------------------------------+---------------------------------------+
| Problema: puede exceder la   | No tiene ese problema (hechos cortos) |
| ventana de contexto           |                                       |
+-------------------------------+---------------------------------------+
```

### Compactacion vs Memoria a largo plazo

Ambos resuelven el problema de "demasiado contexto", pero de
formas diferentes:

```
+-------------------------------+---------------------------------------+
| Compactacion (resumen)        | Memoria a largo plazo                 |
+-------------------------------+---------------------------------------+
| Comprime la conversacion      | Extrae hechos y los guarda aparte    |
| ACTUAL en un resumen          |                                       |
+-------------------------------+---------------------------------------+
| El resumen reemplaza los      | Los hechos se inyectan como contexto  |
| mensajes anteriores           | adicional del sistema                 |
+-------------------------------+---------------------------------------+
| Pierde detalle (es resumen)   | Preciso (son hechos individuales)     |
+-------------------------------+---------------------------------------+
| Solo dentro de una sesion     | Persiste entre sesiones              |
+-------------------------------+---------------------------------------+
| Util cuando: conversacion     | Util cuando: el usuario vuelve       |
| larga en una sola sesion      | dias despues y quieres recordarlo    |
+-------------------------------+---------------------------------------+
```

### bind_tools vs un agente con AgentExecutor

```
+-------------------------------+---------------------------------------+
| bind_tools + bucle manual     | AgentExecutor (langchain legacy)      |
+-------------------------------+---------------------------------------+
| Tu controlas el bucle         | El executor controla el bucle         |
+-------------------------------+---------------------------------------+
| Mas explicito y educativo     | Mas "magico", menos control           |
+-------------------------------+---------------------------------------+
| Funciona en langchain 1.x     | Movido a langgraph (no disponible     |
|                               | en langchain 1.2.10)                  |
+-------------------------------+---------------------------------------+
| Es lo que usamos en este      |                                       |
| repositorio                   |                                       |
+-------------------------------+---------------------------------------+
```

---

## 10. Equivalencias Azure Agent Framework - LangChain

```
+------------------------------------------+--------------------------------------------+
| Azure Agent Framework                    | LangChain (este repositorio)               |
+------------------------------------------+--------------------------------------------+
| @tool (agent_framework)                  | @tool (langchain_core.tools)               |
+------------------------------------------+--------------------------------------------+
| Agent(tools=[...])                       | llm.bind_tools([...])                      |
+------------------------------------------+--------------------------------------------+
| agent.run("pregunta")                    | run(llm_with_tools, messages, tool_map)    |
|                                          | (bucle explicito)                          |
+------------------------------------------+--------------------------------------------+
| Agent supervisor con tools=[plan_a,      | Mismo patron: subagentes como @tool        |
| plan_b] que wrappean subagentes          | wrappers del supervisor                    |
+------------------------------------------+--------------------------------------------+
| InMemoryHistoryProvider                  | InMemoryChatMessageHistory                 |
+------------------------------------------+--------------------------------------------+
| SqliteHistoryProvider                    | SQLiteChatHistory personalizado            |
|                                          | (BaseChatMessageHistory)                   |
+------------------------------------------+--------------------------------------------+
| AgentMiddleware (before/after)           | Callbacks de LangChain                     |
| para compactacion de contexto            | + conteo de tokens con tiktoken            |
+------------------------------------------+--------------------------------------------+
| RedisContextProvider                     | Redis Hashes + RediSearch                  |
| (memoria a largo plazo)                  | (implementacion manual)                    |
+------------------------------------------+--------------------------------------------+
| Knowledge con PostgreSQL                 | BaseRetriever personalizado                |
| (pgvector + tsvector)                    | + pgvector + tsvector + RRF                |
+------------------------------------------+--------------------------------------------+
```

---

## 11. Referencia rapida de ejecucion

### Requisitos

```bash
# Instalar dependencias
uv sync

# Variables de entorno en .env
OPENAI_API_KEY=sk-...
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_LLM_TEMPERATURE=0.9
```

### Ejemplos

| # | Carpeta | Comando | Servicios externos |
|---|---------|---------|-------------------|
| 1 | [`tool-calling/`](tool-calling/) | `uv run python tool-calling/main.py` | Ninguno |
| 2 | [`multi-agent/`](multi-agent/) | `uv run python multi-agent/main.py` | Ninguno |
| 3 | [`sqlite-history/`](sqlite-history/) | `uv run python sqlite-history/main.py` | Ninguno |
| 4 | [`summarize-conversation/`](summarize-conversation/) | `uv run python summarize-conversation/main.py` | Ninguno |
| 5 | [`long-term-memory-redis/`](long-term-memory-redis/) | `uv run python long-term-memory-redis/main.py` | Redis (Docker) |
| 6 | [`rag-sql-fts5/`](rag-sql-fts5/) | `uv run python rag-sql-fts5/main.py` | Ninguno |
| 7 | [`rag-pgvector/`](rag-pgvector/) | `uv run python rag-pgvector/main.py` | PostgreSQL (Docker) |

Para los que requieren Docker:

```bash
# Redis (memoria a largo plazo)
cd long-term-memory-redis && docker compose up -d

# PostgreSQL + pgvector (RAG hibrido)
cd rag-pgvector && docker compose up -d
```

---

> Cada carpeta tiene su propio `README.md` con detalles adicionales
> y la referencia al ejemplo original de Azure Agent Framework.
