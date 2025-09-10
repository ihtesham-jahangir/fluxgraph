<p align="center">
  <img src="logo.jpeg" alt="FluxGraph Logo" width="180"/>
</p>

<h1 align="center">FluxGraph</h1>

<p align="center">
  <em>A lightweight, developer-first framework for building, orchestrating, and deploying powerful Agentic AI systems.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/fluxgraph/"><img src="https://img.shields.io/pypi/v/fluxgraph?color=blue" alt="PyPI"/></a>
  <a href="#"><img src="https://img.shields.io/badge/docs-available-brightgreen" alt="Docs"/></a>
  <a href="https://discord.gg/your-invite-link"><img src="https://img.shields.io/discord/123456789?logo=discord&label=Discord" alt="Discord"/></a> <!-- Replace with actual invite -->
  <a href="https://github.com/ihtesham-jahangir/fluxgraph"><img src="https://img.shields.io/badge/contributions-welcome-orange" alt="Contributing"/></a>
</p>

---

## 🌟 Why FluxGraph?

Tired of wrestling with overly complex agent frameworks? FluxGraph strips away the boilerplate and gives you **direct control** over LLMs, **structured orchestration** with LangGraph, and **instant deployment** via FastAPI. It's built for developers who want to build *powerful* agents *quickly* and *cleanly*.

### ✅ Key Benefits
- ⚡ **Rapid Prototyping:** From idea to API in minutes, not hours.
- 🔧 **Full Control:** Direct LLM API calls for maximum flexibility.
- 🧱 **Structured Flows:** Harness LangGraph for complex agent logic.
- 🚀 **Instant APIs:** Deploy with FastAPI/Uvicorn out-of-the-box.
- 🔌 **Extensible Tools:** Easily add Python functions for custom logic.
- 🌐 **Multi-Model Ready:** Works seamlessly with OpenAI, Anthropic, Ollama, and custom APIs.
- 💾 **Persistent Memory:** Add state (PostgreSQL, Redis planned) for contextual agents.
- 📈 **Scalable (Roadmap):** Built for future growth with Celery + Redis.

---

## 🏗️ Core Architecture

<p align="center">
  <img src="flux-architecture.png" alt="FluxGraph Architecture" width="600"/>
</p>

FluxGraph provides a clean, modular architecture:
*   **Client/User** ↔ **FastAPI REST API Layer**
*   **FluxApp** (the core) integrates:
    *   **Agent Registry**: Keeps track of your agents.
    *   **Flux Orchestrator**: Executes agent workflows.
    *   **Tooling Layer**: Extends agents with custom functions.
    *   **LangGraph Adapter**: Plug in sophisticated LangGraph state machines.
    *   **Persistence/Memory**: (Optional) Store data with PostgreSQL, Redis, etc.
*   Agents interact with **LLM Providers** (OpenAI, Anthropic, Ollama) and external **Tools/DBs**.

---

## 📦 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/ihtesham-jahangir/fluxgraph.git
cd fluxgraph

# Install dependencies
pip install -r requirements.txt

# Or install in development mode for easier code changes
pip install -e .
```

---

## 🚀 Quickstart & Examples

### 1. The Absolute Basics: An Echo Agent

Get up and running with the simplest possible agent.

**`echo_example.py`**
```python
from fluxgraph import FluxApp

class EchoAgent:
    def run(self, message: str):
        # The 'run' method is the heart of your agent.
        # Arguments map directly to JSON keys in the API request.
        return {"reply": f"Echo: {message}"}

# Initialize the FluxGraph application
app = FluxApp(title="My First Agents")

# Register your agent with a unique name
app.register("echo", EchoAgent())

# Run with: uvicorn echo_example:app --reload
```

**Run & Test:**
```bash
# 1. Start the server
uvicorn echo_example:app --reload

# 2. Test the agent in another terminal
curl -X POST http://127.0.0.1:8000/ask/echo \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, FluxGraph!"}'

# Expected Output:
# {"reply": "Echo: Hello, FluxGraph!"}
```

### 2. Simplify with Decorators: A Smart Greeting Agent

Use the `@app.agent` decorator for cleaner, function-based agents. FluxGraph automatically injects `tools` and `memory` if configured.

**`greeting_example.py`**
```python
from fluxgraph import FluxApp

app = FluxApp(title="Decorator Power!")

# Define and register in one go!
@app.agent() # Name defaults to 'greet_agent'
async def greet_agent(name: str, title: str = "there"):
    # This is an async function, great for I/O!
    greeting = f"Hello, {title} {name}! Nice to meet you."
    return {"greeting": greeting}

# Run with: uvicorn greeting_example:app --reload
```

**Run & Test:**
```bash
uvicorn greeting_example:app --reload
```
```bash
curl -X POST http://127.0.0.1:8000/ask/greet_agent \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "title": "Dr."}'

# Expected Output:
# {"greeting": "Hello, Dr. Alice! Nice to meet you."}
```

### 3. Supercharge with Tools: A Calculator Agent

Define reusable tools and let your agents use them.

**`tool_example.py`**
```python
from fluxgraph import FluxApp

app = FluxApp(title="Tools & Agents")

# --- Define a Tool ---
@app.tool() # Name defaults to 'add_numbers'
def add_numbers(x: int, y: int) -> int:
    """A simple addition tool."""
    return x + y

# --- Define an Agent that Uses the Tool ---
@app.agent() # Name defaults to 'calculator_agent'
async def calculator_agent(a: int, b: int, tools):
    # The 'tools' argument is automatically injected by FluxGraph
    add_tool = tools.get("add_numbers") # Retrieve the tool function
    result = add_tool(a, b) # Use the tool
    return {"a": a, "b": b, "operation": "add", "result": result}

# Run with: uvicorn tool_example:app --reload
```

**Run & Test:**
```bash
uvicorn tool_example:app --reload
```
```bash
curl -X POST http://127.0.0.1:8000/ask/calculator_agent \
  -H "Content-Type: application/json" \
  -d '{"a": 15, "b": 25}'

# Expected Output:
# {"a": 15, "b": 25, "operation": "add", "result": 30}
```

### 4. Connect to LLMs: A Poem Agent

Integrate with powerful LLMs using FluxGraph's provider abstractions.

**`llm_example.py`**
```python
# Requires: pip install openai
import os
from fluxgraph import FluxApp
from fluxgraph.models import OpenAIProvider # Or AnthropicProvider, OllamaProvider

app = FluxApp(title="LLM Magic")

# Configure your LLM provider (use environment variables for keys!)
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    openai_model = OpenAIProvider(api_key=openai_key, model="gpt-3.5-turbo")
else:
    print("Warning: OPENAI_API_KEY not found. LLM agent will not work.")
    openai_model = None

@app.agent(name="poem_writer") # Custom agent name
async def poem_agent(topic: str, tools, memory): # tools/memory injected
    if not openai_model:
        return {"error": "OpenAI key not configured."}

    prompt = f"Write a short, 4-line poem about {topic}."
    try:
        # Use the model provider's generate method
        response = await openai_model.generate(prompt, temperature=0.8)
        poem_text = response.get("text", "Inspiration failed...")
        return {"topic": topic, "poem": poem_text}
    except Exception as e:
        return {"error": f"Poem generation failed: {e}"}

# Run with:
# export OPENAI_API_KEY='your-key-here'
# uvicorn llm_example:app --reload
```

**Run & Test:**
```bash
export OPENAI_API_KEY='your-actual-openai-api-key'
uvicorn llm_example:app --reload
```
```bash
curl -X POST http://127.0.0.1:8000/ask/poem_writer \
  -H "Content-Type: application/json" \
  -d '{"topic": "coding on a rainy day"}'

# Expected Output (varies):
# {"topic": "coding on a rainy day", "poem": "Raindrops tap..."}
```

### 5. Remember Things: PostgreSQL Memory (Conceptual)

*(Assumes `PostgresMemory` is implemented and configured)*

Give your agents a memory to recall past interactions.

**`memory_example.py` (Concept Setup)**
```python
# ... (imports)
# DATABASE_URL = os.getenv("DATABASE_URL")
# memory_store = PostgresMemory(DATABASE_URL) if DATABASE_URL else None
# app = FluxApp(title="Agent with Memory", memory_store=memory_store)

@app.agent()
async def chat_agent(user_input: str, session_id: str, memory): # 'memory' injected
    if not memory:
        return {"response": "Memory not configured."}

    # 1. Save the user's message
    await memory.add(session_id, {"role": "user", "content": user_input})

    # 2. Recall the last message
    history = await memory.get(session_id, limit=1)
    context = history[0]['content'] if history else "Nothing before."

    # 3. Generate a response (simplified)
    response_text = f"You said '{user_input}'. Before that, you said '{context}'."

    # 4. Save the agent's response
    await memory.add(session_id, {"role": "assistant", "content": response_text})

    return {"response": response_text}

# Run with:
# export DATABASE_URL='postgresql+asyncpg://user:pass@host:port/dbname'
# uvicorn memory_example:app --reload
```

---

## 📊 MVP Development Roadmap

FluxGraph is evolving rapidly:

- **Phase 1 (MVP - Core - ✅ Complete):**
  - ✅ `FluxApp` (FastAPI foundation)
  - ✅ Agent Registry
  - ✅ Flux Orchestrator
  - ✅ LangGraph Adapter (Basic)
- **Phase 2 (Scaling & Observability - 🚧 In Progress):**
  - ✅ Event Hooks (Basic)
  - 🔄 Celery + Redis async tasks *(Planned)*
  - 📊 Enhanced Logging & Monitoring *(Planned)*
- **Phase 3 (Advanced Features - 🔮 Future):**
  - 🌊 Streaming responses *(Planned)*
  - 🔐 Authentication layer *(Planned)*
  - 🎛️ Dashboard for flows *(Planned)*

---

## 🤝 Contributing

We love contributions! Feel free to fork the repo, open issues, or submit pull requests to help make FluxGraph even better.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.