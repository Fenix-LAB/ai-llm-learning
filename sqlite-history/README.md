# Historial de Conversacion Persistente con SQLite y LangChain

## Descripcion

Ejemplo de historial de conversacion persistente usando SQLite y LangChain con un LLM de OpenAI.

Cuando una aplicacion de chat se reinicia, el historial almacenado en memoria se pierde. Este ejemplo demuestra como persistir los mensajes de la conversacion en una base de datos SQLite para que el agente **recuerde turnos anteriores** incluso despues de un reinicio de la aplicacion.

El patron clave es:
1. Guardar cada mensaje (usuario y agente) como una fila en SQLite asociado a un `session_id`
2. Al iniciar una nueva instancia, reconectar al mismo `session_id` para recuperar el historial
3. El LLM recibe los mensajes previos y puede responder preguntas que requieren contexto de turnos anteriores

En Azure Agent Framework esto se implementa como un `BaseHistoryProvider` con metodos `get_messages`/`save_messages`. En LangChain, subclaseamos `BaseChatMessageHistory` y lo integramos con `RunnableWithMessageHistory`.

## Diagrama

```plaintext
 chat("mensaje del usuario")
 |
 v
 +--------------------------------------------------+
 |  RunnableWithMessageHistory                      |
 |                                                  |
 |  1. Obtener session_id de la config              |
 |  2. Crear SQLiteChatHistory(session_id)          |
 |  3. Cargar mensajes previos desde SQLite         |
 |  4. Inyectar historial en el prompt              |
 |  5. Ejecutar la cadena LCEL                      |
 |  6. Guardar mensajes nuevos en SQLite            |
 +--------------------------------------------------+
 |
 v
 respuesta (con memoria de turnos anteriores)
```

### Flujo detallado de persistencia

```plaintext
 Fase 1: Conversacion normal
 ============================

 [Usuario]: "Como esta el clima en Tokio?"
         |
         v
 +---------------------------+     +---------------------+
 | RunnableWithMessageHistory|---->| SQLiteChatHistory   |
 |                           |     |                     |
 | 1. Lee historial (vacio)  |     | messages -> []      |
 | 2. Ejecuta chain          |     |                     |
 | 3. Guarda H1 + A1         |     | INSERT H1, A1       |
 +---------------------------+     +---------------------+
         |
         v
 "El clima en Tokio esta soleado..."

 [Usuario]: "Y Paris?"
         |
         v
 +---------------------------+     +---------------------+
 | RunnableWithMessageHistory|---->| SQLiteChatHistory   |
 |                           |     |                     |
 | 1. Lee historial [H1,A1]  |     | SELECT -> [H1, A1]  |
 | 2. Ejecuta chain          |     |                     |
 | 3. Guarda H2 + A2         |     | INSERT H2, A2       |
 +---------------------------+     +---------------------+


 === REINICIO DE LA APLICACION ===
    (conexion cerrada, objetos destruidos)


 Fase 2: Reconexion al mismo session_id
 ========================================

 [Usuario]: "Cual ciudad tuvo mejor clima?"
         |
         v
 +---------------------------+     +---------------------+
 | RunnableWithMessageHistory|---->| SQLiteChatHistory   |
 |   (nueva instancia)       |     |   (nueva instancia) |
 |                           |     |                     |
 | 1. Lee historial          |     | SELECT -> [H1, A1,  |
 |    [H1,A1,H2,A2]          |     |            H2, A2]  |
 | 2. Ejecuta chain          |     |                     |
 | 3. Guarda H3 + A3         |     | INSERT H3, A3       |
 +---------------------------+     +---------------------+
         |
         v
 "Tokio tuvo mejor clima..." (recuerda las ciudades anteriores)
```

## Por que persistir el historial

### El problema: historial en memoria

Con `InMemoryChatMessageHistory` (usado en el ejemplo `summarize-conversation`), los mensajes viven en la RAM del proceso. Esto significa:

```plaintext
 Proceso Python (PID 12345)
 +---------------------------+
 |  InMemoryChatMessageHistory|
 |  [H1, A1, H2, A2]         |
 +---------------------------+

         |
    CTRL+C / crash / deploy
         |
         v

 Proceso Python (PID 67890)     <-- nuevo proceso
 +---------------------------+
 |  InMemoryChatMessageHistory|
 |  []  <-- historial perdido |
 +---------------------------+
```

### La solucion: SQLite como almacen

Con `SQLiteChatHistory`, los mensajes se guardan en un archivo `.sqlite3` que sobrevive reinicios:

```plaintext
 Proceso 1                        Proceso 2 (post-reinicio)
 +------------------+             +------------------+
 | SQLiteChatHistory|             | SQLiteChatHistory|
 | session_id: abc  |             | session_id: abc  |
 +--------+---------+             +--------+---------+
          |                                |
          v                                v
 +--------------------------------------------+
 |          chat_history.sqlite3              |
 |                                            |
 |  id | session_id | message_json            |
 |  1  | abc        | {"type":"human",...}     |
 |  2  | abc        | {"type":"ai",...}        |
 |  3  | abc        | {"type":"human",...}     |
 |  4  | abc        | {"type":"ai",...}        |
 +--------------------------------------------+
          (persiste en disco)
```

### Comparacion de estrategias de almacenamiento

| Estrategia | Persistencia | Latencia | Escalabilidad | Caso de uso |
|---|---|---|---|---|
| `InMemoryChatMessageHistory` | No (se pierde al reiniciar) | Nula | Un solo proceso | Prototipos, demos, tests |
| **SQLite** (este ejemplo) | Si (archivo local) | Baja (~1ms) | Un solo proceso/servidor | Apps locales, desarrollo, bots simples |
| Redis | Si (servidor externo) | Baja (~1ms red local) | Multi-proceso, distribuido | Produccion, multiples instancias |
| PostgreSQL | Si (servidor externo) | Media (~5ms red) | Multi-proceso, distribuido | Produccion con busqueda avanzada |
| Cosmos DB / DynamoDB | Si (nube) | Media (~10-50ms) | Global, serverless | Produccion en la nube a escala |

## Componentes principales

| Componente | Descripcion |
|---|---|
| `SQLiteChatHistory` | Clase que extiende `BaseChatMessageHistory` de LangChain. Persiste mensajes en SQLite usando `session_id` como clave. Equivale al `SQLiteHistoryProvider` de Azure Agent Framework. |
| `BaseChatMessageHistory` | Clase abstracta de LangChain que define la interfaz para almacenar historial: `messages` (propiedad), `add_messages()`, `clear()`. |
| `RunnableWithMessageHistory` | Wrapper de LangChain que automatiza la carga y guardado del historial en cada invocacion de la cadena. Equivale a la integracion automatica de `context_providers` en Azure Agent Framework. |
| `get_weather()` | Funcion simulada que devuelve datos ficticios de clima para la demo. |
| `inspect_db()` | Funcion auxiliar que muestra el contenido de la base de datos SQLite para verificar la persistencia. |
| `create_chain()` | Crea la cadena LCEL: `prompt -> llm -> parser`. |

## Estructura de la tabla SQLite

```plaintext
 +---------------------------------------------------+
 |                   messages                         |
 +---------------------------------------------------+
 | id (INTEGER PK)  | AUTO INCREMENT                 |
 | session_id (TEXT) | Identificador de la sesion     |
 | message_json (TEXT)| Mensaje serializado como JSON |
 +---------------------------------------------------+

 Indices:
   - PRIMARY KEY en id (B-Tree, automatico)
   - idx_messages_session en session_id (B-Tree, manual)
```

### Ejemplo del contenido de la tabla

```
 id | session_id  | message_json
 ---+-------------+--------------------------------------------------
  1 | abc-123...  | {"type":"human","data":{"content":"Como esta..."}}
  2 | abc-123...  | {"type":"ai","data":{"content":"El clima en..."}}
  3 | abc-123...  | {"type":"human","data":{"content":"Y Paris?"}}
  4 | abc-123...  | {"type":"ai","data":{"content":"En Paris esta..."}}
  5 | xyz-789...  | {"type":"human","data":{"content":"Hola"}}
  6 | xyz-789...  | {"type":"ai","data":{"content":"Hola, en que..."}}
```

Cada sesion tiene su propio `session_id` (UUID). Esto permite almacenar multiples conversaciones independientes en la misma base de datos. El indice `idx_messages_session` acelera las consultas filtradas por `session_id`.

### Por que cada mensaje es una fila separada

El ejemplo original de Azure usa este mismo patron (un `INSERT` por mensaje). Las ventajas son:

1. **Eficiencia en escritura**: agregar un mensaje es un solo `INSERT`, no un `UPDATE` del documento completo
2. **Orden garantizado**: el campo `id` auto-incremental preserva el orden cronologico
3. **Compatibilidad con bases de datos con limite de documento**: Cosmos DB, por ejemplo, tiene un limite de 2MB por documento
4. **Consultas parciales**: se pueden recuperar los ultimos N mensajes sin cargar toda la conversacion

La alternativa seria guardar toda la conversacion en una sola fila como un JSON array, pero eso requiere leer y reescribir todo el array en cada turno.

## Serializacion de mensajes con message_to_dict

LangChain proporciona `message_to_dict` y `messages_from_dict` para serializar y deserializar mensajes. Cada mensaje se convierte en un diccionario con esta estructura:

```json
{
  "type": "human",
  "data": {
    "content": "Como esta el clima en Tokio?",
    "additional_kwargs": {},
    "response_metadata": {},
    "type": "human",
    "name": null,
    "id": "run-abc123..."
  }
}
```

El campo `type` puede ser `human`, `ai`, `system`, `tool`, etc. Esto permite reconstruir el tipo correcto de `BaseMessage` al deserializar.

## RunnableWithMessageHistory: como funciona

`RunnableWithMessageHistory` es un wrapper que automatiza tres pasos:

```plaintext
 chain_with_history.invoke(
     {"question": "..."},
     config={"configurable": {"session_id": "abc-123"}}
 )

 Internamente:
 +---------------------------------------------------------+
 | 1. Extraer session_id de config["configurable"]         |
 |                                                         |
 | 2. Llamar get_session_history(session_id)               |
 |    -> Crea SQLiteChatHistory("abc-123", db_path)        |
 |    -> Lee mensajes previos: [H1, A1, H2, A2]           |
 |                                                         |
 | 3. Inyectar historial en el prompt via                  |
 |    MessagesPlaceholder("history")                       |
 |                                                         |
 | 4. Ejecutar la cadena:                                  |
 |    prompt -> llm -> parser                              |
 |                                                         |
 | 5. Guardar mensajes nuevos:                             |
 |    history.add_messages([HumanMessage, AIMessage])      |
 +---------------------------------------------------------+
```

### Parametros clave

| Parametro | Descripcion |
|---|---|
| `input_messages_key` | Nombre del campo en el input que contiene la pregunta del usuario (`"question"`) |
| `history_messages_key` | Nombre del `MessagesPlaceholder` en el prompt donde se inyecta el historial (`"history"`) |
| `get_session_history` | Funcion que recibe un `session_id` y devuelve una instancia de `BaseChatMessageHistory` |

## Sobre check_same_thread en SQLite

SQLite por defecto no permite usar una conexion creada en un thread desde otro thread. `RunnableWithMessageHistory` puede ejecutar la lectura del historial desde un thread diferente al del codigo principal (usa `concurrent.futures` internamente).

Para resolver esto, se usa `check_same_thread=False` al crear la conexion:

```python
self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
```

Esto es seguro en este ejemplo porque no hay escrituras concurrentes. En produccion con multiples usuarios simultaneos, se recomienda usar un pool de conexiones (por ejemplo, con SQLAlchemy) o una base de datos como PostgreSQL.

## Requisitos

- Python >= 3.13
- Una API key de OpenAI

## Instalacion

```bash
# Instalar dependencias del proyecto
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
uv run sqlite-history/main.py
```

El archivo `sqlite-history/chat_history.sqlite3` se crea automaticamente. Puedes eliminarlo para empezar de cero o ejecutar varias veces para acumular sesiones.

## Ejemplo de salida

```
=== Sesion persistente en SQLite ===
--- Fase 1: Iniciando conversacion ---

[SQLite] Conexion abierta a 'sqlite-history/chat_history.sqlite3' (session: 6704e13e...)
[Usuario]: Como esta el clima en Tokio?
[SQLite] 2 mensaje(s) guardados en session '6704e13e...'
[Agente]:  El clima en Tokio esta nublado con una temperatura maxima de 29 C...

[Usuario]: Y Paris?
[SQLite] 2 mensaje(s) guardados en session '6704e13e...'
[Agente]:  El clima en Paris esta tormentoso con una temperatura maxima de 24 C...

    Mensajes en la sesion: 4
    Almacenamiento: 1234 bytes
[SQLite] Conexion cerrada

--- Fase 2: Reanudando despues del 'reinicio' ---

    (Se creo una nueva instancia de SQLiteChatHistory
     reconectando al mismo session_id: 6704e13e...)

[Usuario]: Cual de las ciudades por las que pregunte tuvo mejor clima?
[SQLite] 2 mensaje(s) guardados en session '6704e13e...'
[Agente]:  Entre Tokio y Paris, Tokio tuvo mejor clima...

[Usuario]: Y como esta el clima en Londres?
[SQLite] 2 mensaje(s) guardados en session '6704e13e...'
[Agente]:  El clima en Londres esta soleado con una temperatura maxima de 27 C...

    Mensajes totales en la sesion: 8
    Sesiones en la BD: 1
    Almacenamiento: 2577 bytes

    --- Contenido de la base de datos SQLite ---
    Archivo: sqlite-history/chat_history.sqlite3
    Sesiones encontradas: 1
      Session 6704e13e...: 8 mensajes

    Mensajes de la sesion 6704e13e...:
      [1] human: Como esta el clima en Tokio?...
      [2] ai: El clima en Tokio esta nublado...
      [3] human: Y Paris?...
      [4] ai: El clima en Paris esta tormentoso...
      [5] human: Cual de las ciudades por las que pregunte tuvo mejor clima?
      [6] ai: Entre Tokio y Paris, Tokio tuvo mejor clima...
      [7] human: Y como esta el clima en Londres?...
      [8] ai: El clima en Londres esta soleado...

    El archivo 'sqlite-history/chat_history.sqlite3' persiste en disco.
    Puedes eliminarlo manualmente o ejecutar de nuevo para agregar mas sesiones.
```

Nota: los valores de clima y las respuestas varian entre ejecuciones porque los datos son aleatorios y el LLM no es determinista.

## Explorar la base de datos manualmente

Puedes inspeccionar el archivo SQLite directamente con el comando `sqlite3`:

```bash
# Abrir la base de datos
sqlite3 sqlite-history/chat_history.sqlite3

# Ver la estructura de la tabla
.schema messages

# Ver todas las sesiones
SELECT session_id, COUNT(*) as mensajes FROM messages GROUP BY session_id;

# Ver los mensajes de una sesion (reemplazar el session_id)
SELECT id, json_extract(message_json, '$.type') as tipo,
       substr(json_extract(message_json, '$.data.content'), 1, 60) as contenido
FROM messages
WHERE session_id = 'TU-SESSION-ID'
ORDER BY id;

# Salir
.quit
```

## Estructura del proyecto

```
sqlite-history/
  main.py               # Codigo principal con el historial persistente
  README.md             # Documentacion del proyecto
  chat_history.sqlite3  # Base de datos (generada al ejecutar, no se sube a git)
```

## Dependencias

| Paquete | Uso |
|---|---|
| `langchain` | Framework de orquestacion para LLMs |
| `langchain-openai` | Integracion de LangChain con la API de OpenAI |
| `python-dotenv` | Carga de variables de entorno desde `.env` |

Este ejemplo no requiere dependencias adicionales. SQLite viene incluido en la libreria estandar de Python (`sqlite3`), y `message_to_dict`/`messages_from_dict` son parte de `langchain-core`.

## Diferencias con el ejemplo original (Azure Agent Framework)

| Concepto | Azure Agent Framework | LangChain |
|---|---|---|
| Clase base | `BaseHistoryProvider` con `get_messages` / `save_messages` | `BaseChatMessageHistory` con propiedad `messages` / `add_messages` / `clear` |
| Integracion automatica | `context_providers=[sqlite_provider]` en el `Agent` | `RunnableWithMessageHistory(chain, get_session_history)` wrapping la cadena |
| Sesion | `agent.create_session(session_id=sid)` | `config={"configurable": {"session_id": sid}}` en cada invocacion |
| Serializacion | `message.to_json()` / `Message.from_json()` | `message_to_dict()` / `messages_from_dict()` |
| Threading | No aplica (async nativo) | Requiere `check_same_thread=False` en SQLite |
| Herramientas | `@tool` decorador con ejecucion automatica por el agente | Funciones regulares, resultado inyectado manualmente en el mensaje |

### Patron de reconexion

El patron de "reinicio" es identico en ambos frameworks:

```plaintext
 Azure:                              LangChain:
 ─────                               ─────────
 provider1 = SQLiteHistoryProvider() history1 = SQLiteChatHistory(sid, db)
 session = agent.create_session(sid) config = {"session_id": sid}
 agent.run("pregunta", session)      chain.invoke(input, config)
   |                                   |
   v REINICIO                          v REINICIO
   |                                   |
 provider2 = SQLiteHistoryProvider() history2 = SQLiteChatHistory(sid, db)
 session2 = agent2.create_session(sid) config = {"session_id": sid}
 agent2.run("pregunta2", session2)   chain.invoke(input, config)
```

En ambos casos, el segundo provider/history lee los mensajes previos del mismo `session_id` desde SQLite y el agente puede responder con contexto de la conversacion anterior.

## Relacion con el ejemplo summarize-conversation

Este ejemplo (`sqlite-history`) y `summarize-conversation` se complementan:

| Aspecto | sqlite-history | summarize-conversation |
|---|---|---|
| **Problema que resuelve** | Persistencia entre reinicios | Crecimiento ilimitado del historial |
| **Almacenamiento** | SQLite (disco) | InMemory (RAM) |
| **Historial crece?** | Si, indefinidamente | No, se compacta al superar el umbral |
| **Sobrevive reinicios?** | Si | No |

En una aplicacion de produccion, se combinarian ambos patrones: persistir en SQLite (o Redis/PostgreSQL) **y** resumir periodicamente para evitar que el historial crezca sin control.

## Referencias

- [LangChain BaseChatMessageHistory](https://python.langchain.com/docs/concepts/chat_history/)
- [LangChain RunnableWithMessageHistory](https://python.langchain.com/docs/how_to/message_history/)
- [LangChain message_to_dict / messages_from_dict](https://python.langchain.com/api_reference/core/messages/)
- [SQLite en Python (libreria estandar)](https://docs.python.org/3/library/sqlite3.html)
- [Ejemplo original (Azure Agent Framework)](https://github.com/Azure-Samples/python-agentframework-demos/blob/main/examples/agent_history_sqlite.py)
