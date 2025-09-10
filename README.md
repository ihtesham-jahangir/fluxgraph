

<p align="center">
  <img src="logo.jpeg" alt="FluxGraph Logo" width="180"/>
</p>

<h1 align="center">FluxGraph</h1>

<p align="center">
  <i>A lightweight Python framework for building, orchestrating, and deploying Agentic AI systems.</i>
</p>

<p align="center">
  <a href="https://pypi.org/project/fluxgraph/"><img src="https://img.shields.io/pypi/v/fluxgraph?color=blue" alt="PyPI"/></a>
  <a href="#"><img src="https://img.shields.io/badge/docs-available-brightgreen" alt="Docs"/></a>
  <a href="https://discord.gg/your-invite-link"><img src="https://img.shields.io/discord/123456789?logo=discord&label=Discord" alt="Discord"/></a> <!-- Replace with actual invite -->
  <a href="https://github.com/ihtesham-jahangir/fluxgraph"><img src="https://img.shields.io/badge/contributions-welcome-orange" alt="Contributing"/></a>
</p>

---

## ✨ Why FluxGraph?

FluxGraph is designed for developers who want **control and flexibility** without heavy abstractions. It combines the immediacy of FastAPI for REST API deployment with the structure of LangGraph workflows and the power of direct LLM interactions.

### 🔑 Key Benefits
- 🚀 **Rapid Development:** Build and deploy agentic AI systems quickly with minimal boilerplate.
- ⚡ **Multi-Model Ready:** Seamlessly integrate OpenAI, Anthropic, Ollama, and custom LLM APIs.
- 🧠 **Persistent Memory:** Add state with PostgreSQL (and potentially Redis) for conversation history and data storage.
- 🔌 **Extensible Tooling:** Easily define and share Python functions as tools for agents to use.
- 🤝 **LangGraph Integration:** Wrap and orchestrate complex LangGraph state machines as agents.
- 📈 **Scalable (Roadmap):** Designed for future scaling with Celery and Redis.
- 🔍 **Transparent (Roadmap):** Built-in event hooks for debugging and monitoring agent execution.

---

## 🏗️ Architecture

<p align="center">
  <img src="flux-architecture.png" alt="FluxGraph Architecture" width="600"/>
</p>

*(Diagram showing Client/User -> FastAPI -> FluxApp (containing Agent Registry, Flux Orchestrator, Tool Registry, Memory) -> LLMs/Tools/DBs, with LangGraph Adapter as a plugin)*

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/ihtesham-jahangir/fluxgraph.git
cd fluxgraph

# Install dependencies
pip install -r requirements.txt

# Or install the package in development mode
pip install -e .
```

---

## ⚡ Quick Example

### Define and Run Agents

```python
# run_server.py
import os
from fluxgraph import FluxApp
# Import model providers if needed
# from fluxgraph.models import OpenAIProvider

# --- Simple Echo Agent ---
class EchoAgent:
    def run(self, query: str):
        return {"response": f"Echo: {query}"}

# --- Initialize FluxGraph ---
app = FluxApp(title="FluxGraph Demo")

# --- Register Agents ---
app.register("echo", EchoAgent())

# --- Define and Register Agent using Decorator (with tools/memory if configured) ---
@app.agent() # Agent name defaults to function name 'smart_agent'
async def smart_agent(task: str, tools=None, memory=None):
    # Use tools.get('tool_name') if tools are registered
    # Use await memory.add(...) / await memory.get(...) if memory is configured
    return {"result": f"Processed task: {task}"}

# --- Run with: uvicorn run_server:app --reload ---
```

### Interact via API

```bash
# Start the server
uvicorn run_server:app --reload

# Interact with the Echo Agent
curl -X POST http://127.0.0.1:8000/ask/echo \
  -H "Content-Type: application/json" \
  -d '{"query":"Hello FluxGraph!"}'

# Expected Response:
# {"response": "Echo: Hello FluxGraph!"}

# Interact with the Decorated Agent
curl -X POST http://127.0.0.1:8000/ask/smart_agent \
  -H "Content-Type: application/json" \
  -d '{"task":"Explain FluxGraph."}'

# Expected Response:
# {"result": "Processed task: Explain FluxGraph."}
```

---

## 🧠 Advanced Features (MVP Enhancements)

### Multi-Model LLM Support
Use `ModelProvider` abstractions (`OpenAIProvider`, `AnthropicProvider`, `OllamaProvider`) within your agents for flexible LLM interaction.

### Tooling Layer
Define tools with `@app.tool()` and access them in agents via the injected `tools` argument.

```python
@app.tool()
def add_numbers(a: int, b: int) -> int:
    return a + b

@app.agent()
async def calculator_agent(expression: str, tools):
    add_func = tools.get("add_numbers")
    # ... process expression and use tool ...
```

### Persistence & Memory (e.g., PostgreSQL)
Configure `FluxApp` with a memory store instance (like `PostgresMemory`) that implements the `Memory` interface. Injected agents can then use `memory.add()` and `memory.get()`.

---

## 📊 MVP Roadmap

Based on the original documentation and enhancements:

- **Phase 1 (MVP - Core):**
  - [x] `FluxApp` (FastAPI wrapper)
  - [x] Agent Registry
  - [x] Flux Orchestrator
  - [x] LangGraph Adapter (Stub/Basic Implementation)
- **Phase 2 (Scaling & Observability):**
  - [x] Event Hooks (Basic Implementation)
  - [ ] Celery + Redis async task orchestration *(Planned)*
  - [ ] Enhanced Logging + Monitoring *(In Progress/Planned)*
- **Phase 3 (Advanced Features):**
  - [ ] Streaming responses *(Planned)*
  - [ ] Authentication layer *(Planned)*
  - [ ] Dashboard for orchestration flows *(Planned)*

---

## 🤝 Contributing

We welcome contributions! Please fork the repo, open issues, or submit PRs.

---

## 📜 License

MIT License © 2025 FluxGraph Team


### Key Updates Made:

1.  **Core Concept Alignment:** Refocused the description to match the documentation: lightweight framework, FastAPI, LangGraph, LLM control.
2.  **Feature Highlighting:** Explicitly listed new MVP features like Multi-Model support, Tooling Layer, and Persistence/Memory.
3.  **Quick Example:** Updated the example to use the `FluxApp` class and the `@app.agent` decorator as implemented.
4.  **Architecture:** Noted that the diagram should reflect the actual components.
5.  **Advanced Features Section:** Added a dedicated section to showcase the enhancements made beyond the basic MVP doc (Model Providers, Tools, Memory).
6.  **Roadmap:** Updated the roadmap to reflect the status of features based on our work (Event Hooks started, others planned).
7.  **Installation/Usage:** Provided clear steps reflecting the current codebase structure.