# Compactacion de Contexto mediante Resumen de Conversacion con LangChain

## Descripcion

Ejemplo de compactacion de contexto usando resumen automatico de conversacion con LangChain y un LLM de OpenAI.

Cuando una conversacion multi-turno crece, los mensajes acumulados pueden exceder la **ventana de contexto** del modelo (limite de tokens que puede procesar en una sola llamada) o generar costos innecesarios al enviar historial repetitivo. Este ejemplo monitorea el uso acumulado de tokens y, cuando se cruza un umbral configurable, le pide al LLM resumir la conversacion hasta ese momento. El resumen reemplaza los mensajes anteriores y libera espacio para futuros turnos.

En Azure Agent Framework esto se implementa como un `AgentMiddleware` con metodos `before`/`after`. En LangChain, usamos `get_openai_callback` para el conteo de tokens y manejamos el historial manualmente con `InMemoryChatMessageHistory`.

## Diagrama

```plaintext
 chat("mensaje del usuario")
 |
 v
 +--------------------------------------------------+
 |       SummarizingChat (nivel chat)               |
 |                                                  |
 |  1. Revisar uso acumulado de tokens              |
 |  2. Si pasa el umbral -> resumir mensajes        |
 |     previos con el LLM y reemplazarlos           |
 |     por el resumen                               |
 |  3. Ejecutar la cadena LCEL con el historial     |
 |  4. Registrar tokens nuevos de la respuesta      |
 +--------------------------------------------------+
 |
 v
 respuesta
```

### Flujo detallado de un turno de conversacion

```plaintext
 chat("Como esta el clima en Portland?")
         |
         v
 +------------------------------+
 | _check_and_summarize()       |
 |                              |
 | context_tokens > umbral?     |
 |                              |
 |   NO --> continuar           |
 |   SI --> resumir historial   |
 +------------------------------+
         |
    +----+----+
    |         |
    v         v
  [NO]      [SI]
    |         |
    |         v
    |   +----------------------------+
    |   | summarize_history()        |
    |   |                            |
    |   | 1. Formatear mensajes      |
    |   |    como texto plano        |
    |   | 2. Enviar al LLM con      |
    |   |    prompt de resumen       |
    |   | 3. Reemplazar historial    |
    |   |    por 1 mensaje resumen   |
    |   | 4. Reiniciar contador      |
    |   +----------------------------+
    |         |
    v         v
 +------------------------------+
 | chain.invoke()               |
 |                              |
 | ChatPromptTemplate:          |
 |   [system] prompt del agente |
 |   [history] mensajes previos |
 |   [human] pregunta actual    |
 |                              |
 | get_openai_callback():       |
 |   registrar tokens usados    |
 +------------------------------+
         |
         v
 +------------------------------+
 | Guardar en historial         |
 |                              |
 | history.add_user_message()   |
 | history.add_ai_message()    |
 | context_tokens += nuevos     |
 +------------------------------+
         |
         v
       respuesta
```

### Linea de tiempo de una conversacion de ejemplo

```plaintext
 Turno 1       Turno 2       Turno 3       Turno 4       Turno 5
 -------       -------       -------       -------       -------

 tokens:       tokens:       tokens:       tokens:       tokens:
 ~150          ~380          ~620          ~450          ~680

 [H1]          [H1]          [RESUMEN]     [RESUMEN]     [RESUMEN']
               [A1]             |          [H4]             |
               [H2]             v          [A4]             v
               [A2]          [H3]                        [H5]
                             [A3]

 Sin resumir   Sin resumir   Se resumio    Se resumio    Se resumio
                             turno 1-2     turno 3       turno 4
                             antes de      antes de      antes de
                             procesar      procesar      procesar

 H = HumanMessage    A = AIMessage    RESUMEN = historial compactado
```

## Por que es necesaria la compactacion de contexto

### El problema de la ventana de contexto

Cada modelo tiene un limite de tokens que puede procesar en una sola llamada:

| Modelo | Ventana de contexto | Costo aproximado (input) |
|---|---|---|
| `gpt-4o-mini` | 128,000 tokens | $0.15 / 1M tokens |
| `gpt-4o` | 128,000 tokens | $2.50 / 1M tokens |
| `gpt-3.5-turbo` | 16,385 tokens | $0.50 / 1M tokens |

Aunque 128K tokens parece mucho, en una aplicacion de chat de produccion:

- Cada turno consume entre 100-500 tokens (pregunta + respuesta)
- Un `system_prompt` largo puede usar 200-500 tokens
- En 50-100 turnos se puede llegar al limite
- El costo se acumula: enviar todo el historial en cada turno es redundante

### La solucion: resumir y compactar

En lugar de enviar todos los mensajes anteriores, se resumen en un unico mensaje que conserva los hechos clave. Esto reduce tanto el uso de tokens como el costo.

```plaintext
 ANTES (sin compactacion)          DESPUES (con compactacion)
 ─────────────────────────         ──────────────────────────────
 [system] Prompt                   [system] Prompt
 [human]  Pregunta 1               [ai] [Resumen: El usuario pregunto
 [ai]     Respuesta 1                    sobre clima en SF (soleado 28C),
 [human]  Pregunta 2                     Portland (lluvioso 15C) y
 [ai]     Respuesta 2                    Seattle (nublado 18C). Se
 [human]  Pregunta 3                     recomendaron actividades...]
 [ai]     Respuesta 3              [human] Pregunta actual
 [human]  Pregunta 4
 [ai]     Respuesta 4                 ~200 tokens (compacto)
 [human]  Pregunta actual

    ~800 tokens (creciente)
```

## Componentes principales

| Componente | Descripcion |
|---|---|
| `SummarizingChat` | Clase principal que maneja la conversacion multi-turno. Monitorea tokens acumulados y dispara el resumen cuando se supera el umbral. Equivale al `SummarizationMiddleware` de Azure Agent Framework. |
| `summarize_history()` | Funcion que formatea los mensajes del historial como texto plano y le pide al LLM que los condense en un resumen conciso. |
| `count_tokens()` | Cuenta tokens de un texto usando `tiktoken` (la misma libreria de tokenizacion que usa OpenAI internamente). |
| `count_message_tokens()` | Cuenta el total de tokens de una lista de mensajes de LangChain, incluyendo el overhead por rol y delimitadores. |
| `get_weather()` | Funcion simulada que devuelve datos ficticios de clima para la demo. |
| `get_activities()` | Funcion simulada que devuelve actividades turisticas ficticias para la demo. |
| `InMemoryChatMessageHistory` | Almacen de historial de LangChain que guarda los mensajes en memoria. Se limpia y reemplaza con el resumen cuando se dispara la compactacion. |
| `get_openai_callback` | Context manager de LangChain que intercepta las llamadas a OpenAI y registra tokens usados (prompt, completion, total). |

## Conteo de tokens: tiktoken vs. get_openai_callback

Este ejemplo usa **dos mecanismos** de conteo de tokens, cada uno con un proposito distinto:

### tiktoken (estimacion local, antes de la llamada)

`tiktoken` es la libreria de tokenizacion que usa OpenAI internamente. Nos permite estimar cuantos tokens ocupa un texto **sin hacer una llamada al API**. Lo usamos para:

- Contar los tokens del historial de mensajes despues de compactar
- Estimar si un mensaje individual es grande

```plaintext
 "Botas de senderismo impermeables"
         |
         v
 +------------------+
 | tiktoken.encode  |
 | (modelo: gpt-4o) |
 +------------------+
         |
         v
 [tok_1, tok_2, tok_3, tok_4]  -->  4 tokens
```

### get_openai_callback (conteo real, despues de la llamada)

`get_openai_callback` es un context manager de `langchain-community` que se coloca alrededor de la ejecucion de la cadena. Intercepta la respuesta del API de OpenAI y extrae los tokens **reales** reportados por el modelo:

```plaintext
 with get_openai_callback() as cb:
     response = chain.invoke(...)

 cb.prompt_tokens      # tokens enviados al modelo
 cb.completion_tokens  # tokens generados por el modelo
 cb.total_tokens       # prompt + completion
 cb.total_cost         # costo estimado en USD
```

### Por que usar ambos

| Mecanismo | Cuando se usa | Precision |
|---|---|---|
| `tiktoken` | Antes de la llamada, para estimar el historial compactado | Alta (misma tokenizacion que OpenAI) |
| `get_openai_callback` | Despues de la llamada, para acumular tokens reales | Exacta (dato del API) |

## Como funciona la compactacion paso a paso

### 1. Monitoreo de tokens

Despues de cada turno, se suman los tokens reales reportados por `get_openai_callback`:

```python
with get_openai_callback() as cb:
    response = self.chain.invoke(...)

self.context_tokens += cb.total_tokens
```

### 2. Verificacion del umbral

Antes de cada turno, se compara el acumulado con el umbral:

```python
if self.context_tokens > self.token_threshold and len(messages) > 2:
    # Disparar resumen
```

La condicion `len(messages) > 2` evita resumir cuando hay muy pocos mensajes (no tiene sentido resumir un solo intercambio).

### 3. Generacion del resumen

Se formatean los mensajes como texto plano y se envian al LLM con un prompt de resumen:

```plaintext
 Prompt de resumen:
 "Eres un asistente de resumen. Condensa la siguiente conversacion
  en un resumen conciso que preserve todos los hechos clave..."

 Conversacion:
 "usuario: Como esta el clima en SF?
  asistente: El clima en SF esta soleado, 28C...
  usuario: Y Portland?
  asistente: Portland esta lluvioso, 15C..."

         |
         v

 Resumen generado:
 "El usuario consulto el clima de San Francisco (soleado, 28C)
  y Portland (lluvioso, 15C). Se proporcionaron datos de
  humedad y viento para ambas ciudades."
```

### 4. Reemplazo del historial

El historial completo se reemplaza por un unico `AIMessage` con el resumen:

```python
self.history.clear()
self.history.add_message(
    AIMessage(content=f"[Resumen de la conversacion anterior]\n{summary_text}")
)
self.context_tokens = count_message_tokens(self.history.messages)
```

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
uv run summarize-conversation/main.py
```

## Ejemplo de salida

```
=== Compactacion de contexto con resumen de conversacion ===
    Umbral de tokens: 500
    El middleware resumira la conversacion cuando el uso de tokens supere el umbral.

[Resumen] Uso de tokens: 0 / 500 umbral. No hace falta resumir.
[Usuario]: Como estara el clima en San Francisco este fin de semana?
[Resumen] Este turno uso 188 tokens (prompt=91, completion=97). Contexto acumulado: 188
[Agente]:  El clima en San Francisco se presenta soleado con una temperatura de 28 C...

[Resumen] Uso de tokens: 188 / 500 umbral. No hace falta resumir.
[Usuario]: Y Portland? Como estara el clima y que actividades puedo hacer ahi?
[Resumen] Este turno uso 400 tokens (prompt=250, completion=150). Contexto acumulado: 588
[Agente]:  En Portland el clima estara lluvioso con 15 C...

[Resumen] Uso de tokens (588) excede el umbral (500). Resumiendo 4 mensajes del historial...
[Resumen] Resumen generado: El usuario consulto el clima de San Francisco (soleado, 28C)...
[Resumen] Historial compactado a 1 mensaje de resumen
[Usuario]: Que tal Seattle? Dame el panorama completo: clima y cosas para hacer.
[Resumen] Este turno uso 350 tokens (prompt=200, completion=150). Contexto acumulado: 430
[Agente]:  En Seattle el clima estara nublado con 18 C...

[Resumen] Uso de tokens: 430 / 500 umbral. No hace falta resumir.
[Usuario]: De todas las ciudades que mencionamos, cual tiene la mejor combinacion?
[Resumen] Este turno uso 280 tokens (prompt=180, completion=100). Contexto acumulado: 710
[Agente]:  Considerando las tres ciudades, San Francisco tiene la mejor combinacion...

[Resumen] Uso de tokens (710) excede el umbral (500). Resumiendo 5 mensajes del historial...
[Resumen] Resumen generado: El usuario comparo SF, Portland y Seattle...
[Resumen] Historial compactado a 1 mensaje de resumen
[Usuario]: Perfecto, vamos con esa ciudad. Que deberia empacar?
[Agente]:  Para tu viaje a San Francisco te recomiendo empacar...

    Conteo final de tokens en contexto: 520
```

Nota: los valores de tokens y las respuestas varian entre ejecuciones porque los datos de clima son aleatorios y el LLM no es determinista.

## Estructura del proyecto

```
summarize-conversation/
  main.py     # Codigo principal con la conversacion y resumen automatico
  README.md   # Documentacion del proyecto
```

## Dependencias

| Paquete | Uso |
|---|---|
| `langchain` | Framework de orquestacion para LLMs |
| `langchain-openai` | Integracion de LangChain con la API de OpenAI |
| `langchain-community` | Callbacks de la comunidad, incluido `get_openai_callback` |
| `tiktoken` | Tokenizacion local compatible con los modelos de OpenAI |
| `python-dotenv` | Carga de variables de entorno desde `.env` |

## Diferencias con el ejemplo original (Azure Agent Framework)

| Concepto | Azure Agent Framework | LangChain |
|---|---|---|
| Patron principal | `AgentMiddleware` con `before_run` / `after_run` | Clase `SummarizingChat` con logica manual antes/despues de `chain.invoke()` |
| Conteo de tokens | `context.token_count` disponible en el middleware | `get_openai_callback` (tokens reales) + `tiktoken` (estimacion local) |
| Historial de mensajes | `context.messages` gestionado por el framework | `InMemoryChatMessageHistory` gestionado manualmente |
| Resumen del historial | `context.replace_messages(summary)` | `history.clear()` + `history.add_message(resumen)` |
| Inyeccion en la cadena | Automatica (el middleware intercepta la ejecucion) | Manual (se pasa `history.messages` al prompt via `MessagesPlaceholder`) |
| Herramientas / datos | Funciones registradas como tools del agente | Funciones llamadas manualmente y resultado inyectado en el mensaje |

### Sobre middleware en LangChain

LangChain **no tiene** un parametro `middleware=` como Azure Agent Framework (`Agent(middleware=[...])`). Los equivalentes en LangChain son:

| Necesidad | Azure Agent Framework | LangChain |
|---|---|---|
| **Observar** (logging, conteo de tokens) | `AgentMiddleware.after_run()` | `callbacks=[handler]` con `BaseCallbackHandler` |
| **Modificar** (transformar input/output) | `AgentMiddleware.before_run()` | `RunnableLambda` en la cadena LCEL |
| **Interceptar** (resumir, filtrar) | `AgentMiddleware.before_run()` | Logica manual antes de `chain.invoke()` (como en este ejemplo) |

## Sobre la temperatura del LLM en el resumen

El resumen usa la misma instancia de `ChatOpenAI` con la temperatura configurada en `.env`. Para aplicaciones de produccion, seria recomendable usar una temperatura baja (0.0 - 0.3) para el resumen, ya que queremos precision factual, no creatividad:

```python
# Opcion recomendada para produccion
summary_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)
```

En este ejemplo mantenemos una sola instancia por simplicidad.

## Referencias

- [LangChain ChatMessageHistory](https://python.langchain.com/docs/concepts/chat_history/)
- [LangChain Callbacks](https://python.langchain.com/docs/concepts/callbacks/)
- [LangChain MessagesPlaceholder](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.MessagesPlaceholder.html)
- [tiktoken - Tokenizador de OpenAI](https://github.com/openai/tiktoken)
- [OpenAI Token Usage](https://platform.openai.com/docs/guides/rate-limits/usage-tiers)
- [Ejemplo original (Azure Agent Framework)](https://github.com/Azure-Samples/python-agentframework-demos/blob/main/examples/agent_summarization.py)
