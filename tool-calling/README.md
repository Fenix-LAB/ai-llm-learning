# Tool Calling (llamada a herramientas)

Ejemplo de como un LLM puede invocar funciones Python automaticamente
para obtener datos externos antes de generar su respuesta.

Se implementa un agente planificador de fin de semana con 3 herramientas:

| Herramienta       | Descripcion                                      |
|--------------------|--------------------------------------------------|
| `get_current_date` | Devuelve la fecha actual del sistema (YYYY-MM-DD)|
| `get_weather`      | Devuelve datos meteorologicos simulados          |
| `get_activities`   | Devuelve actividades disponibles para una ciudad |

## Flujo

```
usuario: "Que puedo hacer este fin de semana en San Francisco?"
  |
  v
LLM decide que herramientas necesita
  |
  v
get_current_date() -> "2026-02-28"
get_weather("San Francisco") -> {temperatura: 22, descripcion: "soleado"}
get_activities("San Francisco", "2026-02-28") -> [{nombre: "Senderismo", ...}]
  |
  v
LLM genera respuesta integrando los datos
```

El LLM decide de forma autonoma que herramientas invocar (puede ser 0, 1 o
varias) segun la pregunta del usuario.

## Equivalencia con Azure Agent Framework

| Azure Agent Framework                | LangChain                              |
|---------------------------------------|----------------------------------------|
| `@tool` (agent_framework)            | `@tool` (langchain_core.tools)         |
| `Agent(tools=[...])`                 | `llm.bind_tools([...])`               |
| `agent.run("pregunta")`              | bucle `run()` con `invoke` + `ToolMessage` |

En Azure Agent Framework, `agent.run()` maneja el bucle internamente.
En LangChain se necesita un bucle explicito que ejecute las herramientas
y devuelva los resultados al LLM hasta que responda sin tool calls.

## Requisitos

- `OPENAI_API_KEY` en `.env`
- Dependencias del proyecto (`langchain`, `langchain-openai`, etc.)

## Ejecucion

```bash
uv run python tool-calling/main.py
```

## Basado en

[`agent_tools.py`](https://github.com/Azure-Samples/python-agentframework-demos/blob/main/examples/spanish/agent_tools.py)
del repositorio Azure Agent Framework Demos.
