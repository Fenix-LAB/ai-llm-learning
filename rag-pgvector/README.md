# Recuperacion de Conocimiento (RAG) con PostgreSQL pgvector y LangChain

## Descripcion

Ejemplo de Retrieval-Augmented Generation (RAG) usando PostgreSQL con la extension **pgvector** para busqueda hibrida y LangChain como framework de orquestacion con un LLM de OpenAI.

A diferencia del ejemplo con SQLite FTS5 (que solo usa busqueda por palabras clave), este ejemplo implementa **busqueda hibrida** combinando dos estrategias:

1. **Busqueda semantica** (pgvector): encuentra productos por similitud de significado usando embeddings vectoriales.
2. **Busqueda por palabras clave** (tsvector): encuentra productos por coincidencia exacta de palabras.

Los resultados de ambas busquedas se fusionan con **Reciprocal Rank Fusion (RRF)** para obtener mejor recuperacion que cualquiera de los dos metodos por separado.

## Diagrama

```plaintext
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
```

### Detalle de la busqueda hibrida

```plaintext
                    Pregunta del usuario
                            |
               +------------+------------+
               |                         |
               v                         v
    +--------------------+    +---------------------+
    | Busqueda semantica |    | Busqueda por        |
    | (pgvector)         |    | palabras clave      |
    |                    |    | (tsvector)           |
    | Genera embedding   |    |                     |
    | de la pregunta y   |    | Busca coincidencias |
    | compara similitud  |    | exactas de palabras |
    | coseno con los     |    | en nombre y         |
    | embeddings de los  |    | descripcion         |
    | productos          |    |                     |
    +--------------------+    +---------------------+
               |                         |
               | Top 20 resultados       | Top 20 resultados
               |                         |
               v                         v
         +----------------------------------+
         | Reciprocal Rank Fusion (RRF)     |
         |                                  |
         | score = 1/(k+rank_sem)           |
         |       + 1/(k+rank_kw)            |
         |                                  |
         | Combina rankings de ambas        |
         | busquedas en un score unificado  |
         +----------------------------------+
                        |
                        v
              Top N productos relevantes
```

## Busqueda hibrida vs. busqueda por palabras clave

| Caracteristica | SQLite FTS5 (rag-sql-fts5) | PostgreSQL pgvector (este ejemplo) |
|---|---|---|
| **Tipo de busqueda** | Solo palabras clave (texto completo) | Hibrida: semantica + palabras clave |
| **Encuentra sinonimos** | No. "observar fauna" no encuentra "binoculares" | Si. Los embeddings capturan relaciones semanticas |
| **Embeddings** | No usa | Si. Modelo `text-embedding-3-small` de OpenAI |
| **Motor de busqueda** | SQLite FTS5 (tabla virtual) | pgvector (similitud coseno) + tsvector (texto completo) |
| **Fusion de resultados** | No aplica (un solo metodo) | Reciprocal Rank Fusion (RRF) |
| **Dependencias** | Solo SQLite (incluido en Python) | PostgreSQL + extension pgvector + Docker |
| **Precision en espanol** | Limitada (tokenizacion basica) | Mejor (tsvector con diccionario 'spanish') |
| **Caso de uso** | Prototipos rapidos, datos pequenos | Produccion, busqueda semantica, datos grandes |

### Ejemplo comparativo

| Consulta | SQLite FTS5 | PostgreSQL pgvector |
|---|---|---|
| "botas senderismo" | Encuentra botas (coincidencia directa) | Encuentra botas (ambos metodos) |
| "observar fauna silvestre" | Puede encontrar binoculares si "fauna" esta en la descripcion | Encuentra binoculares por similitud semantica con "observacion de aves y fauna" |
| "algo abrigado para el frio" | Probablemente no encuentra nada | Encuentra la chaqueta por similitud semantica |

## Componentes principales

| Componente | Descripcion |
|---|---|
| `PostgresHybridRetriever` | Retriever personalizado de LangChain (`BaseRetriever`) que ejecuta busqueda hibrida (vectorial + texto completo) sobre PostgreSQL y devuelve `Document`s. |
| `create_knowledge_db()` | Crea la tabla de productos en PostgreSQL con columna de embeddings (`vector`) e indice GIN para texto completo. |
| `get_embedding()` | Genera un vector de embedding usando el modelo `text-embedding-3-small` de OpenAI con dimension reducida (256). |
| `HYBRID_SEARCH_SQL` | Consulta SQL con CTEs que ejecuta busqueda semantica y por palabras clave, fusionando resultados con RRF. |
| `rag_chain` | Cadena de LangChain (LCEL) que conecta: retriever -> formateo -> prompt -> LLM -> parser de salida. |
| `format_docs()` | Funcion que convierte los documentos del retriever en texto legible para el prompt. |

## Requisitos

- Python >= 3.13
- Docker (para PostgreSQL con pgvector)
- Una API key de OpenAI

## Instalacion

```bash
# Instalar dependencias del proyecto
uv sync

# Levantar PostgreSQL con pgvector usando Docker
cd rag-pgvector
docker compose up -d
```

## Configuracion

Crear un archivo `.env` en la raiz del proyecto con las siguientes variables:

```env
# LLM Settings
OPENAI_API_KEY="tu-api-key-aqui"
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_LLM_TEMPERATURE=0.7

# PostgreSQL Settings
POSTGRES_USER=chris
POSTGRES_PASSWORD=chrisa7
POSTGRES_DB=rag_pgvector
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
```

| Variable | Descripcion | Valor por defecto |
|---|---|---|
| `OPENAI_API_KEY` | API key de OpenAI (requerida) | - |
| `DEFAULT_LLM_MODEL` | Modelo de OpenAI a utilizar | `gpt-4o-mini` |
| `DEFAULT_LLM_TEMPERATURE` | Temperatura del LLM (creatividad) | `0.7` |
| `POSTGRES_USER` | Usuario de PostgreSQL | `chris` |
| `POSTGRES_PASSWORD` | Contrasena de PostgreSQL | `chrisa7` |
| `POSTGRES_DB` | Nombre de la base de datos | `rag_pgvector` |
| `POSTGRES_HOST` | Host de PostgreSQL | `localhost` |
| `POSTGRES_PORT` | Puerto de PostgreSQL | `5433` |

## Ejecucion

```bash
# Asegurarse de que PostgreSQL esta corriendo
cd rag-pgvector
docker compose up -d

# Ejecutar el ejemplo
cd ..
uv run rag-pgvector/main.py
```

## Ejemplo de salida

```
=== Demo de Recuperacion de Conocimiento (RAG) con PostgreSQL y Busqueda Hibrida ===
    (pgvector [semantica] + tsvector [palabras clave] con RRF)

[Usuario]: Estoy planeando una excursion. Que botas y bastones me recomiendan?
[Agente]:  Te recomiendo las **Botas de Senderismo TrailBlaze** ($149.99)
           y los **Bastones de Trekking TerraFirm** ($59.99)...

[Usuario]: Quiero algo para observar fauna silvestre
[Agente]:  Te recomiendo los **Binoculares ClearView 10x42** ($129.00)...

[Usuario]: Necesito algo abrigado para acampar en invierno, tal vez una chaqueta?
[Agente]:  Te recomiendo la **Chaqueta de Plumon ArcticShield** ($199.00)...

[Usuario]: Tienen tablas de surf?
[Agente]:  No tengo informacion sobre ese articulo.
```

## Estructura del proyecto

```
rag-pgvector/
  docker-compose.yml    # Configuracion de Docker para PostgreSQL + pgvector
  main.py               # Codigo principal con el ejemplo RAG
  README.md             # Documentacion del proyecto
```

## Dependencias

| Paquete | Uso |
|---|---|
| `langchain` | Framework de orquestacion para LLMs |
| `langchain-openai` | Integracion de LangChain con la API de OpenAI |
| `openai` | Cliente oficial de OpenAI (usado para generar embeddings) |
| `psycopg[binary]` | Driver de PostgreSQL para Python |
| `pgvector` | Soporte de vectores para psycopg (registra el tipo vector) |
| `python-dotenv` | Carga de variables de entorno desde `.env` |

## Docker - PostgreSQL con pgvector

Este proyecto usa la imagen `pgvector/pgvector:pg17` que incluye PostgreSQL 17 con la extension pgvector preinstalada.

```bash
# Levantar el contenedor
cd rag-pgvector
docker compose up -d

# Verificar que esta corriendo
docker compose ps

# Ver los logs
docker compose logs -f

# Detener el contenedor
docker compose down

# Detener y eliminar datos
docker compose down -v
rm -rf pgdata
```

Los datos se persisten en el directorio `pgdata/` gracias al volumen de Docker.

## Explorar los datos en PostgreSQL

Una vez ejecutado el ejemplo, puedes conectarte al contenedor y explorar la tabla de productos con sus embeddings directamente desde la terminal.

### Conectarse a la base de datos

```bash
# Abrir una sesion interactiva de psql dentro del contenedor
docker exec -it rag_pgvector_db psql -U chris -d rag_pgvector
```

### Ver la estructura de la tabla

```bash
# Describir la tabla products (columnas, tipos, etc.)
docker exec -it rag_pgvector_db psql -U chris -d rag_pgvector -c "\d products"
```

### Ver los productos (sin embeddings)

```bash
# Listar todos los productos con nombre, categoria, precio y descripcion
docker exec -it rag_pgvector_db psql -U chris -d rag_pgvector -c \
  "SELECT id, name, category, price FROM products ORDER BY id;"
```

### Ver los embeddings almacenados

```bash
# Ver los primeros 5 valores del vector de embedding de cada producto
docker exec -it rag_pgvector_db psql -U chris -d rag_pgvector -c \
  "SELECT id, name, left(embedding::text, 60) AS embedding_preview FROM products;"
```

```bash
# Ver la dimension del embedding de un producto
docker exec -it rag_pgvector_db psql -U chris -d rag_pgvector -c \
  "SELECT id, name, vector_dims(embedding) AS dimensiones FROM products;"
```

```bash
# Ver el embedding completo de un producto especifico (ej: id=1)
docker exec -it rag_pgvector_db psql -U chris -d rag_pgvector -c \
  "SELECT name, embedding FROM products WHERE id = 1;"
```

### Ejecutar una busqueda de similitud manual

```bash
# Ver los 3 productos mas similares al producto con id=1 (Botas TrailBlaze)
docker exec -it rag_pgvector_db psql -U chris -d rag_pgvector -c \
  "SELECT a.name AS producto, b.name AS similar_a, \
   (a.embedding <=> b.embedding)::numeric(10,4) AS distancia \
   FROM products a, products b \
   WHERE a.id != b.id AND b.id = 1 \
   ORDER BY distancia LIMIT 3;"
```

### Ver los indices de la tabla

```bash
# Listar los indices creados (incluye el GIN para texto completo)
docker exec -it rag_pgvector_db psql -U chris -d rag_pgvector -c \
  "\di+ products*"
```

### Probar la busqueda de texto completo

```bash
# Buscar productos que coincidan con "senderismo botas" usando tsvector
docker exec -it rag_pgvector_db psql -U chris -d rag_pgvector -c \
  "SELECT name, ts_rank_cd(to_tsvector('spanish', name || ' ' || description), query) AS rank \
   FROM products, plainto_tsquery('spanish', 'senderismo botas') query \
   WHERE to_tsvector('spanish', name || ' ' || description) @@ query \
   ORDER BY rank DESC;"
```

## Diferencias con el ejemplo original (Azure Agent Framework)

| Concepto | Azure Agent Framework | LangChain |
|---|---|---|
| Proveedor de contexto | `BaseContextProvider` con metodo `before_run` | `BaseRetriever` con metodo `_get_relevant_documents` |
| Inyeccion de contexto | `context.extend_messages()` agrega mensajes al historial | El retriever pasa documentos al prompt via la cadena LCEL |
| Cadena de ejecucion | `agent.run(query)` maneja todo internamente | Cadena LCEL explicita: `retriever -> format -> prompt -> llm -> parser` |
| Embeddings | Cliente `OpenAI` directo | Cliente `OpenAI` directo (mismo enfoque) |
| Base de datos | PostgreSQL con pgvector | PostgreSQL con pgvector (mismo enfoque) |
| Busqueda hibrida | SQL con CTEs y RRF | SQL con CTEs y RRF (mismo enfoque) |

## Como funciona la busqueda hibrida

### 1. Busqueda semantica (pgvector)

Cuando el usuario hace una pregunta, se genera un **embedding** (vector de 256 dimensiones) de la consulta usando el modelo `text-embedding-3-small` de OpenAI. Este vector se compara con los embeddings almacenados de cada producto usando **similitud coseno** (operador `<=>`).

### 2. Busqueda por palabras clave (tsvector)

Simultaneamente, PostgreSQL ejecuta una busqueda de **texto completo** usando el diccionario de idioma espanol (`'spanish'`). La funcion `plainto_tsquery` convierte la consulta del usuario en tokens normalizados, y `to_tsvector` hace lo mismo con los textos de los productos.

### 3. Fusion con RRF

Los resultados de ambas busquedas se combinan con **Reciprocal Rank Fusion (RRF)**:

```
score = 1/(k + rank_semantica) + 1/(k + rank_palabras_clave)
```

Donde `k` es una constante (60 por defecto) que controla cuanto peso se da a los resultados de alto ranking. Los productos que aparecen en ambas busquedas obtienen un score mas alto, mientras que los que solo aparecen en una tambien son considerados.

## Que es un embedding y como funciona la vectorizacion

### El concepto

Un **embedding** es una representacion numerica (un vector de numeros decimales) de un texto. La idea es que textos con significado similar produzcan vectores que apunten en direcciones parecidas en un espacio de alta dimension.

Por ejemplo, las palabras "botas de senderismo" y "calzado para montaña" generarian vectores cercanos entre si, aunque no comparten las mismas palabras.

### Como genera OpenAI los embeddings

OpenAI usa un modelo de red neuronal (Transformer) entrenado con millones de textos. El modelo `text-embedding-3-small` que usamos en este ejemplo:

1. **Recibe** un texto (ej: `"Botas de Senderismo TrailBlaze - Calzado: Botas impermeables..."`)
2. **Tokeniza** el texto: lo divide en sub-palabras (tokens) que el modelo entiende
3. **Procesa** los tokens a traves de multiples capas de atencion (Transformer)
4. **Produce** un vector de punto flotante con la dimension solicitada

```plaintext
Texto de entrada
      |
      v
+-----------------+
| Tokenizacion    |  "Botas" -> [tok_1]  "de" -> [tok_2]  "Senderismo" -> [tok_3] ...
+-----------------+
      |
      v
+-----------------+
| Transformer     |  Multiples capas de atencion que capturan
| (red neuronal)  |  relaciones semanticas entre tokens
+-----------------+
      |
      v
+-----------------+
| Proyeccion      |  Reduce/ajusta a la dimension solicitada (256)
+-----------------+
      |
      v
[0.0231, -0.0142, 0.0538, ..., -0.0089]   <-- vector de 256 dimensiones
```

### Ejemplo simplificado de vectorizacion

Imagina un espacio de solo 3 dimensiones donde cada eje representa un concepto:

```plaintext
           Eje Y (deportes acuaticos)
           ^
           |
     Kayak *
           |        * Mochila
           |       /
           +------/---------> Eje X (senderismo)
          /      /
         /  Botas *    * Bastones
        /
       v
    Eje Z (abrigo/frio)
       * Chaqueta    * Saco de dormir
```

En la realidad, en lugar de 3 ejes, tenemos **256 dimensiones** (una por cada posicion en el vector). Cada dimension captura algun aspecto del significado del texto. No podemos visualizar 256 dimensiones, pero la matematica funciona igual: textos similares producen vectores cercanos.

### Similitud coseno

Para comparar dos vectores, se calcula el **coseno del angulo** entre ellos:

```plaintext
                    A . B           (producto punto)
similitud = ─────────────────── = ──────────────────────
              ||A|| * ||B||       (magnitud A * magnitud B)

Resultado:
  1.0  = identicos (angulo 0)
  0.0  = sin relacion (angulo 90)
 -1.0  = opuestos (angulo 180)
```

PostgreSQL con pgvector usa el operador `<=>` para calcular la **distancia coseno** (que es `1 - similitud`), asi que valores menores significan mayor similitud.

## Estructura de la tabla y sus indices

Al ejecutar `\d products` en PostgreSQL, se ve esta estructura:

```
                                 Table "public.products"
   Column    |    Type     | Collation | Nullable |               Default
-------------+-------------+-----------+----------+--------------------------------------
 id          | integer     |           | not null | nextval('products_id_seq'::regclass)
 name        | text        |           | not null |
 category    | text        |           | not null |
 price       | real        |           | not null |
 description | text        |           | not null |
 embedding   | vector(256) |           |          |
Indexes:
    "products_pkey" PRIMARY KEY, btree (id)
    "products_to_tsvector_idx" gin (to_tsvector('spanish'::regconfig, (name || ' '::text) || description))
```

### Explicacion de cada columna

| Columna | Tipo | Para que sirve |
|---|---|---|
| `id` | `integer` | Identificador unico auto-incremental. PostgreSQL usa una secuencia (`products_id_seq`) para generar el siguiente valor automaticamente. |
| `name` | `text` | Nombre del producto. Se usa en la busqueda de texto completo. |
| `category` | `text` | Categoria del producto (Calzado, Mochilas, etc.). |
| `price` | `real` | Precio del producto (numero decimal de precision simple). |
| `description` | `text` | Descripcion detallada. Se usa en la busqueda de texto completo. |
| `embedding` | `vector(256)` | Vector de 256 dimensiones generado por OpenAI. Este es el tipo especial que agrega la extension pgvector. Almacena el significado semantico del producto. |

### Explicacion de los indices

Los indices son estructuras de datos que PostgreSQL mantiene **aparte de la tabla** para acelerar las busquedas. Sin indices, cada consulta tendria que recorrer todas las filas (scan secuencial).

#### 1. `products_pkey` -- Indice B-Tree (clave primaria)

```plaintext
Tipo: btree (arbol balanceado)
Columna: id

          [4]
         /   \
       [2]   [6]
      / \    / \
    [1] [3] [5] [7,8]

- Busqueda por id: O(log n) -- muy rapido
- Se crea automaticamente con PRIMARY KEY
- Permite buscar un producto por su id sin recorrer toda la tabla
```

#### 2. `products_to_tsvector_idx` -- Indice GIN (texto completo)

```plaintext
Tipo: GIN (Generalized Inverted Index)
Expresion: to_tsvector('spanish', name || ' ' || description)

Funciona como un indice invertido (similar a como funciona un buscador):

Palabra (lexema)    |  Filas donde aparece
--------------------+----------------------
"bot"               |  [1]           (botas)
"senderism"         |  [1, 5]        (senderismo)
"mochil"            |  [2]           (mochila)
"chaqueta"          |  [3]
"kayak"             |  [4]
"trekking"          |  [5]
"binocular"         |  [6]
"linterna"          |  [7]
"sac"               |  [8]           (saco)
"impermeabl"        |  [1]           (impermeables)
...

Nota: las palabras se guardan como "lexemas" (raices).
El diccionario 'spanish' sabe que:
  "impermeables" -> "impermeabl"
  "senderismo"   -> "senderism"
  "botas"        -> "bot"

Cuando buscas "senderismo botas", PostgreSQL:
1. Convierte la consulta en lexemas: "senderism" y "bot"
2. Busca en el indice GIN que filas contienen esos lexemas
3. Devuelve las filas sin recorrer toda la tabla
```

#### Por que no hay indice para los embeddings?

En este ejemplo, con solo 8 productos, PostgreSQL hace un scan secuencial (recorre todos los vectores) porque es mas eficiente que mantener un indice. Para tablas mas grandes (miles o millones de filas), se agregaria un indice HNSW o IVFFlat:

```sql
-- Indice HNSW (recomendado para datasets medianos/grandes)
CREATE INDEX ON products USING hnsw (embedding vector_cosine_ops);

-- Indice IVFFlat (mas rapido de crear, menos preciso)
CREATE INDEX ON products USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

## Referencias

- [pgvector - Extension de vectores para PostgreSQL](https://github.com/pgvector/pgvector)
- [Reciprocal Rank Fusion (RRF)](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)
- [LangChain Custom Retrievers](https://python.langchain.com/docs/how_to/custom_retriever/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/concepts/lcel/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Ejemplo original (Azure Agent Framework)](https://github.com/Azure-Samples/python-agentframework-demos/blob/main/examples/agent_knowledge_postgres.py)
