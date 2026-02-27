# Memoria de Largo Plazo con Redis y LangChain

## Descripcion

Ejemplo de memoria de largo plazo usando Redis con RediSearch y LangChain con un LLM de OpenAI.

A diferencia del historial de chat (ejemplo `sqlite-history`), que guarda **todos los mensajes crudos** de la conversacion, la memoria de largo plazo extrae **hechos destilados** de cada turno y los almacena de forma independiente en Redis. Esto permite al agente recordar preferencias y datos clave del usuario entre sesiones sin necesidad de reenviar toda la conversacion.

El patron clave es:
1. **Antes** de cada turno: buscar memorias relevantes en Redis (FT.SEARCH con texto completo) e inyectarlas como contexto en el system prompt
2. Ejecutar la cadena LCEL para generar la respuesta
3. **Despues** de cada turno: usar el LLM para extraer hechos nuevos del turno y guardarlos en Redis como Hashes indexados por RediSearch

En Azure Agent Framework esto se implementa con `RedisContextProvider` (que guarda y busca contexto conversacional en Redis usando RediSearch). En LangChain, implementamos el mismo patron manualmente con `RedisMemoryStore` + `MemoryEnhancedChat`.

Este ejemplo esta basado en [agent_memory_redis.py](https://github.com/Azure-Samples/python-agentframework-demos/blob/main/examples/agent_memory_redis.py) del repositorio de Azure Samples, adaptado para usar LangChain.

## Diagrama

```plaintext
 chat("Recuerda que mi ciudad favorita es Tokio")
 |
 v
 +--------------------------------------------------+
 |  MemoryEnhancedChat                              |
 |                                                  |
 |  ANTES de responder:                             |
 |    Buscar memorias relevantes en Redis            |
 |    (FT.SEARCH con texto completo + OR)           |
 |    Inyectar memorias como contexto del sistema   |
 |                                                  |
 |  Ejecutar la cadena LCEL -> respuesta            |
 |                                                  |
 |  DESPUES de responder:                           |
 |    Extraer hechos nuevos del turno con el LLM    |
 |    Deduplicar contra memorias existentes         |
 |    Guardar cada hecho en Redis como un Hash      |
 |    (indexado por RediSearch para busqueda)        |
 +--------------------------------------------------+
 |
 v
 respuesta
```

### Flujo detallado de un turno

```plaintext
 chat("Cual es mi ciudad favorita?")
         |
         v
 +-------------------------------+
 | _build_system_with_memories() |
 |                               |
 | 1. Extraer keywords:          |
 |    "ciudad", "favorita"       |
 |    (sin stop words)           |
 |                               |
 | 2. FT.SEARCH con OR:          |
 |    @user_id:{uuid}            |
 |    (ciudad | favorita)        |
 |                               |
 | 3. Resultado:                 |
 |    "La ciudad favorita del    |
 |     usuario es Tokio"         |
 |                               |
 | 4. Inyectar en system prompt: |
 |    "Informacion recordada:    |
 |     - La ciudad favorita..."  |
 +-------------------------------+
         |
         v
 +-------------------------------+
 | Cadena LCEL                   |
 |                               |
 | system: prompt + memorias     |
 | history: mensajes de sesion   |
 | human: pregunta del usuario   |
 |                               |
 | ChatOpenAI -> StrOutputParser |
 +-------------------------------+
         |
         v
 "Tu ciudad favorita es Tokio"
         |
         v
 +-------------------------------+
 | extract_facts()               |
 |                               |
 | LLM analiza el turno y        |
 | extrae hechos como JSON array |
 |                               |
 | -> ["La ciudad favorita del   |
 |      usuario es Tokio"]       |
 +-------------------------------+
         |
         v
 +-------------------------------+
 | save_memory()                 |
 |                               |
 | Deduplicar: ya existe?        |
 |   SI -> omitir                |
 |   NO -> HSET memory:abc123   |
 |          content: "..."       |
 |          user_id: uuid        |
 |          timestamp: epoch     |
 +-------------------------------+
```

## Memoria de largo plazo vs. historial de chat

Esta es la diferencia fundamental entre este ejemplo y `sqlite-history`:

```plaintext
 Historial de chat (sqlite-history)
 ===================================

 Guarda TODOS los mensajes crudos:

 +----+---------+------------------------------------------+
 | #  | Rol     | Contenido                                |
 +----+---------+------------------------------------------+
 | 1  | human   | Hola, me llamo Carlos                    |
 | 2  | ai      | Hola Carlos, en que puedo ayudarte?      |
 | 3  | human   | Mi ciudad favorita es Tokio              |
 | 4  | ai      | Tokio es una ciudad increible!            |
 | 5  | human   | Como esta el clima ahi?                  |
 | 6  | ai      | El clima en Tokio esta soleado, 25C      |
 | 7  | human   | Y en Paris?                              |
 | 8  | ai      | En Paris esta nublado, 18C               |
 +----+---------+------------------------------------------+
 8 mensajes enviados al LLM en cada turno (crece sin limite)


 Memoria de largo plazo (este ejemplo)
 =======================================

 Extrae y guarda SOLO hechos destilados:

 +----+----------------------------------------------+
 | #  | Hecho                                        |
 +----+----------------------------------------------+
 | 1  | El usuario se llama Carlos                   |
 | 2  | La ciudad favorita del usuario es Tokio      |
 | 3  | El usuario prefiere temperaturas en Celsius  |
 +----+----------------------------------------------+
 3 hechos concisos, buscados por relevancia (no crece linealmente)
```

| Aspecto | Historial de chat | Memoria de largo plazo |
|---|---|---|
| Que guarda | Todos los mensajes crudos | Hechos destilados por el LLM |
| Crecimiento | Lineal (1 fila por mensaje) | Solo hechos nuevos y unicos |
| Busqueda | No hay (se envian todos) | Texto completo con RediSearch |
| Persistencia | SQLite (archivo local) | Redis (servicio externo) |
| Multi-sesion | Requiere mismo `session_id` | Automatico por `user_id` |
| Inyeccion | Todos los mensajes como historial | Solo memorias relevantes al turno |
| Ventana de contexto | Puede desbordar | Se mantiene acotada |

## Arquitectura en Redis

### Estructura de datos

Cada memoria se almacena como un **Hash de Redis**:

```plaintext
 memory:ae7c4deee8b6                (clave Redis)
 +----------------------------------------------+
 | content   | "La ciudad favorita del usuario   |
 |           |  es Tokio"                        |
 | user_id   | "6e88faae-1234-5678-..."          |
 | timestamp | "1772232487"                      |
 +----------------------------------------------+

 memory:d6baf3f05fea
 +----------------------------------------------+
 | content   | "El usuario prefiere temperaturas |
 |           |  en Celsius"                      |
 | user_id   | "6e88faae-1234-5678-..."          |
 | timestamp | "1772232489"                      |
 +----------------------------------------------+
```

### Indice RediSearch

El indice `idx:memories` permite busqueda de texto completo sobre los Hashes:

```plaintext
 FT.CREATE idx:memories
   ON HASH
   PREFIX 1 "memory:"
   SCHEMA
     content TEXT WEIGHT 1.0    <-- busqueda BM25 sobre el texto
     user_id TAG               <-- filtro exacto por usuario
```

### Busqueda con keywords y OR

La busqueda extrae palabras clave de la consulta del usuario, elimina stop words
(en espanol e ingles) y las une con OR para maximizar coincidencias:

```plaintext
 Entrada del usuario:
   "Cual es mi ciudad favorita?"

 1. Tokenizar y limpiar:
    ["Cual", "es", "mi", "ciudad", "favorita"]

 2. Filtrar stop words ("cual", "es", "mi"):
    ["ciudad", "favorita"]

 3. Construir query RediSearch:
    @user_id:{uuid} (ciudad | favorita)

 4. FT.SEARCH ejecuta BM25 sobre el campo 'content':
    -> "La ciudad favorita del usuario es Tokio" (match!)
```

Si no hay palabras clave significativas (solo stop words), se devuelven todas
las memorias del usuario como fallback.

## Extraccion de hechos con el LLM

Despues de cada turno, el LLM analiza el par (mensaje usuario, respuesta agente)
y extrae hechos relevantes en formato JSON:

```plaintext
 Prompt de extraccion:
 +--------------------------------------------------------+
 | "Analiza el turno y extrae hechos clave..."            |
 |                                                        |
 | Reglas:                                                |
 | - Devuelve SOLO un JSON array de strings               |
 | - Hechos concisos en tercera persona                   |
 | - Array vacio si no hay hechos nuevos                  |
 |                                                        |
 | Entrada:                                               |
 |   Usuario: "Mi ciudad favorita es Tokio"               |
 |   Agente:  "Tokio es una ciudad increible!"            |
 |                                                        |
 | Salida esperada:                                       |
 |   ["La ciudad favorita del usuario es Tokio"]          |
 +--------------------------------------------------------+
```

La deduplicacion se hace comparando el texto normalizado (lowercase) de cada
hecho nuevo contra las memorias existentes del usuario antes de guardar.

## Componentes principales

| Componente | Descripcion |
|---|---|
| `RedisMemoryStore` | Almacen de hechos en Redis con indice RediSearch. Metodos: `save_memory`, `search_memories`, `get_all_memories`, `clear_memories`. Equivale a `RedisContextProvider` de Azure Agent Framework. |
| `MemoryEnhancedChat` | Clase principal de chat que orquesta el ciclo de buscar memorias, ejecutar la cadena LCEL y extraer hechos nuevos. Mantiene un historial de sesion en memoria (separado de las memorias de largo plazo). |
| `extract_facts()` | Funcion que usa el LLM para analizar un turno de conversacion y devolver un array JSON de hechos destilados. |
| `get_weather()` | Herramienta simulada que devuelve datos ficticios del clima para demostrar la integracion con herramientas. |
| `_extract_keywords()` | Metodo que tokeniza la consulta del usuario, normaliza acentos, remueve stop words y devuelve las palabras clave para la busqueda. |
| `_escape_query()` | Metodo estatico que escapa caracteres especiales de la sintaxis de RediSearch en el texto de busqueda. |

## Equivalencias con Azure Agent Framework

| Azure Agent Framework | LangChain (este ejemplo) |
|---|---|
| `RedisContextProvider` | `RedisMemoryStore` (clase personalizada) |
| `RedisContextProvider.save()` | `RedisMemoryStore.save_memory()` |
| `RedisContextProvider.search()` | `RedisMemoryStore.search_memories()` |
| `AgentKernel.invoke()` con context | `MemoryEnhancedChat.chat()` |
| RediSearch `FT.CREATE` / `FT.SEARCH` | Mismo uso directo de `redis-py` con `ft()` |
| Extraccion de hechos integrada | `extract_facts()` con cadena LCEL dedicada |

## Requisitos

- Python >= 3.13
- Una API key de OpenAI
- Docker (para Redis Stack con RediSearch)

## Instalacion

```bash
# Crear el entorno virtual e instalar dependencias
uv sync

# Iniciar Redis Stack (incluye RediSearch)
cd long-term-memory-redis
docker compose up -d
```

## Configuracion

Crear un archivo `.env` en la raiz del proyecto con las siguientes variables:

```env
OPENAI_API_KEY="tu-api-key-aqui"
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_LLM_TEMPERATURE=0.9
REDIS_URL=redis://localhost:6379
```

## Ejecucion

```bash
uv run python long-term-memory-redis/main.py
```

### Salida esperada

```
=== Agente con memoria de largo plazo en Redis ===
    User ID: 6e88faae...
    Redis: redis://localhost:6379
    Las memorias persisten entre sesiones y se buscan con RediSearch.

--- Paso 1: Ensenando preferencias ---

[Usuario]: Recuerda que mi ciudad favorita es Tokio y prefiero Celsius.
[Redis] Memoria guardada: 'La ciudad favorita del usuario es Tokio'
[Redis] Memoria guardada: 'El usuario prefiere temperaturas en Celsius'
[Agente]:  Claro! Si necesitas saber el clima en Tokio...

--- Paso 2: Nueva sesion -- recordando preferencias ---

[Usuario]: Cual es mi ciudad favorita?
[Redis] Busqueda (keywords: ciudad, favorita) -> 1 memoria(s)
[Agente]:  Tu ciudad favorita es Tokio.

--- Paso 3: Uso de herramientas con memoria ---

[Usuario]: Como esta el clima en Paris?
[Agente]:  El clima en Paris esta soleado, 20 C.

--- Memorias almacenadas en Redis ---

    1. La ciudad favorita del usuario es Tokio
    2. El usuario prefiere temperaturas en Celsius
    3. El usuario esta interesado en el clima de Paris
```

## Docker: Redis Stack

Este ejemplo usa **Redis Stack** (`redis/redis-stack-server`), que incluye Redis
mas modulos adicionales como RediSearch (busqueda de texto completo e indices).

```yaml
# docker-compose.yml
services:
  redis:
    image: redis/redis-stack-server:latest
    container_name: ltm_redis
    ports:
      - "6379:6379"
    volumes:
      - ./redis-data:/data
    environment:
      - REDIS_ARGS=--save 60 1
```

### Comandos utiles

```bash
# Iniciar el contenedor
cd long-term-memory-redis && docker compose up -d

# Verificar que RediSearch esta cargado
docker exec ltm_redis redis-cli MODULE LIST

# Inspeccionar el indice
docker exec ltm_redis redis-cli FT.INFO idx:memories

# Buscar memorias manualmente
docker exec ltm_redis redis-cli FT.SEARCH idx:memories "Tokio" RETURN 1 content

# Ver todas las claves de memoria
docker exec ltm_redis redis-cli KEYS "memory:*"

# Limpiar todas las memorias
docker exec ltm_redis redis-cli FLUSHALL

# Detener el contenedor
cd long-term-memory-redis && docker compose down
```

## Dependencias

| Paquete | Uso |
|---|---|
| `langchain` | Framework de orquestacion LLM |
| `langchain-openai` | Integracion con modelos de OpenAI |
| `langchain-core` | Prompts, parsers, mensajes |
| `redis` (>= 7.2.1) | Cliente Redis con soporte para RediSearch (`ft()`) |
| `python-dotenv` | Carga de variables de entorno desde `.env` |

## Notas tecnicas

### Escapado de UUIDs en tags de RediSearch

Los UUIDs contienen guiones (`-`) que son caracteres especiales en la sintaxis
de tags de RediSearch. Para filtrar por `user_id` en un tag field, los guiones
deben escaparse con backslash:

```python
# Incorrecto (RediSearch interpreta los guiones como operadores):
"@user_id:{6e88faae-1234-5678-abcd-ef0123456789}"

# Correcto:
"@user_id:{6e88faae\\-1234\\-5678\\-abcd\\-ef0123456789}"
```

### Stop words y busqueda OR

RediSearch usa AND por defecto al buscar multiples terminos. Esto significa que
una query como "Cual es mi ciudad favorita" requiere que TODOS los terminos
("Cual", "es", "mi", "ciudad", "favorita") esten presentes en el contenido
indexado. Como la memoria almacenada es "La ciudad favorita del usuario es Tokio",
los terminos "Cual" y "mi" no estan presentes y la busqueda falla.

La solucion implementada:
1. Filtrar stop words comunes en espanol e ingles
2. Unir los terminos restantes con `|` (OR) en lugar de AND
3. Si no quedan palabras clave, devolver todas las memorias como fallback

### Diferencia entre `redis` y `redis-stack-server`

La imagen `redis:latest` solo incluye Redis base (key-value). Para usar
`FT.SEARCH`, `FT.CREATE` y otros comandos de RediSearch, se necesita
`redis/redis-stack-server` que incluye los modulos Search, JSON, y otros.
