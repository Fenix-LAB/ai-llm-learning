# Multi-Agent con Supervisor

Ejemplo de un agente supervisor que orquesta dos subagentes especialistas.
El supervisor recibe la consulta del usuario, decide que especialista invocar
(o ambos), y sintetiza una respuesta final.

## Subagentes

| Subagente | Herramientas | Responsabilidad |
|-----------|-------------|-----------------|
| Fin de semana | `get_current_date`, `get_weather`, `get_activities` | Planificar actividades segun el clima |
| Comidas | `find_recipes`, `check_fridge` | Sugerir recetas con los ingredientes disponibles |

## Flujo

```
usuario: "Mis hijos quieren pasta para la cena y algo divertido el sabado"
  |
  v
+--------------------------------------------------+
|  Supervisor                                      |
|  Decide que especialista(s) invocar              |
+--------+-----------------+-----------------------+
         |                 |
         v                 v
  plan_meal(...)    plan_weekend(...)
         |                 |
         v                 v
  +-------------+  +----------------+
  | Subagente   |  | Subagente      |
  | de comidas  |  | de fin de      |
  |             |  | semana         |
  | find_recipes|  | get_weather    |
  | check_fridge|  | get_activities |
  +------+------+  | get_current_   |
         |         | date           |
         |         +-------+--------+
         v                 v
+--------------------------------------------------+
|  Supervisor sintetiza respuesta final            |
+--------------------------------------------------+
  |
  v
respuesta integrada
```

Cada subagente tiene su propio contexto de mensajes y herramientas.
El supervisor solo ve el resultado final de cada subagente, no los
detalles internos de sus llamadas a herramientas.

## Equivalencia con Azure Agent Framework

| Azure Agent Framework | LangChain |
|-----------------------|-----------|
| `Agent(tools=[...])` para subagente | `llm.bind_tools([...])` + `run()` |
| `@tool async def plan_weekend(query): await agent.run(query)` | `@tool def plan_weekend(query): run_weekend_agent(query)` |
| `Agent(tools=[plan_weekend, plan_meal])` supervisor | `llm.bind_tools([plan_weekend, plan_meal])` supervisor |
| `supervisor.run("consulta")` | `run(supervisor_llm, messages, tool_map)` |

El patron es el mismo: los subagentes se exponen como herramientas al supervisor.
En Azure Agent Framework, `agent.run()` maneja el bucle internamente. En LangChain
usamos la misma funcion `run()` con bucle explicito para cada nivel.

## Requisitos

- `OPENAI_API_KEY` en `.env`
- Dependencias del proyecto (`langchain`, `langchain-openai`, etc.)

## Ejecucion

```bash
uv run python multi-agent/main.py
```

## Basado en

[`agent_supervisor.py`](https://github.com/Azure-Samples/python-agentframework-demos/blob/main/examples/agent_supervisor.py)
del repositorio Azure Agent Framework Demos.
