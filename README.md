# FluxGraph

> A lightweight, developer-first framework for building and deploying AI agent systems with minimal boilerplate.

[![PyPI version](https://img.shields.io/pypi/v/fluxgraph?color=blue)](https://pypi.org/project/fluxgraph/)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://fluxgraph.readthedocs.io)
[![Discord](https://img.shields.io/discord/1243184424318402592?logo=discord&label=Discord)](https://discord.gg/Z9bAqjYvPc)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Why FluxGraph?

Building AI agents shouldn't require wrestling with complex frameworks. FluxGraph gives you direct control over LLMs, structured orchestration, and instant API deployment—all with clean, readable code.

**Key Features:**
- 🚀 **Rapid Development** - From concept to API in minutes
- 🔧 **Full Control** - Direct LLM integration without abstraction layers
- 🏗️ **Structured Workflows** - Built-in LangGraph integration for complex flows
- ⚡ **Instant APIs** - FastAPI-powered endpoints out of the box
- 🛠️ **Extensible Tools** - Add custom Python functions as agent tools
- 🌐 **Multi-Model Support** - Works with OpenAI, Anthropic, Ollama, and custom APIs
- 💾 **Persistent Memory** - Optional state management with PostgreSQL/Redis
- 🔍 **Built-in RAG** - Automatic Retrieval-Augmented Generation setup

## Quick Start

### Installation

```bash
pip install fluxgraph
```

### Your First Agent

Create a simple echo agent in under 10 lines:

```python
from fluxgraph import FluxApp

app = FluxApp(title="My AI Agents")

@app.agent()
def echo_agent(message: str):
    return {"reply": f"You said: {message}"}

# Run with: flux run app.py
```

Test it immediately:
```bash
curl -X POST http://127.0.0.1:8000/ask/echo_agent \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello FluxGraph!"}'
```

## Core Examples

### LLM-Powered Agent

```python
import os
from fluxgraph import FluxApp
from fluxgraph.models import OpenAIProvider

app = FluxApp(title="Smart Assistant")

# Configure LLM provider
llm = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)

@app.agent()
async def assistant(query: str):
    response = await llm.generate(
        f"You are a helpful assistant. Answer: {query}",
        temperature=0.7
    )
    return {"answer": response.get("text", "No response")}
```

### Tool-Enabled Agent

```python
@app.tool()
def calculate(expression: str) -> float:
    """Safely evaluate mathematical expressions."""
    try:
        return eval(expression.replace(" ", ""))
    except:
        return None

@app.agent()
async def math_agent(problem: str, tools):
    calc_tool = tools.get("calculate")
    if "+" in problem or "-" in problem:
        result = calc_tool(problem)
        return {"calculation": problem, "result": result}
    return {"error": "Not a math problem"}
```

### Memory-Enabled Chat Agent

```python
@app.agent()
async def chat_agent(message: str, session_id: str, memory):
    if not memory:
        return {"error": "Memory not configured"}
    
    # Store user message
    await memory.add(session_id, {
        "role": "user", 
        "content": message
    })
    
    # Get conversation history
    history = await memory.get(session_id, limit=5)
    
    # Generate contextual response (simplified)
    response = f"I remember our conversation. You just said: {message}"
    
    # Store response
    await memory.add(session_id, {
        "role": "assistant", 
        "content": response
    })
    
    return {"response": response, "session_id": session_id}
```

## Advanced Features

### Retrieval-Augmented Generation (RAG)

FluxGraph automatically sets up RAG when dependencies are installed:

```bash
pip install langchain langchain-chroma chromadb sentence-transformers
```

```python
@app.agent()
async def knowledge_agent(question: str, rag):
    if not rag:
        return {"error": "RAG not configured"}
    
    # Query knowledge base
    results = await rag.query(question, top_k=3)
    
    if not results:
        return {"answer": "No relevant information found"}
    
    # Use retrieved context (integrate with LLM in production)
    context = "\n".join([r.get("content", "") for r in results])
    return {
        "question": question,
        "context_used": len(results),
        "answer": f"Based on the knowledge base: {context[:200]}..."
    }
```

Add documents via API:
```bash
curl -X POST http://127.0.0.1:8000/rag/ingest \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### LangGraph Integration

For complex agent workflows:

```python
from langgraph import StateGraph
from fluxgraph.adapters import LangGraphAdapter

# Define your LangGraph workflow
def create_workflow():
    workflow = StateGraph(state_schema)
    workflow.add_node("step1", step1_function)
    workflow.add_node("step2", step2_function)
    workflow.add_edge("step1", "step2")
    return workflow.compile()

# Register with FluxGraph
graph_workflow = create_workflow()
app.register("complex_agent", LangGraphAdapter(graph_workflow))
```

## Configuration

### Environment Variables

```bash
# LLM Providers
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Database (optional)
export DATABASE_URL="postgresql://user:pass@localhost/db"

# RAG Settings (optional)
export RAG_PERSIST_DIR="./knowledge_base"
```

### Production Setup

```python
from fluxgraph import FluxApp
from fluxgraph.core import PostgresMemory, DatabaseManager

# Initialize with production settings
db = DatabaseManager(os.getenv("DATABASE_URL"))
memory = PostgresMemory(db)

app = FluxApp(
    title="Production AI System",
    memory_store=memory,
    auto_init_rag=True,
    cors_enabled=True
)
```

## Deployment

### Local Development
```bash
flux run app.py
# or
uvicorn app:app --reload
```

### Production
```bash
gunicorn app:app -w 4 -k uvicorn.workers.UnicornWorker
```

### Docker
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install fluxgraph[all]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## API Reference

### Agent Endpoints
- `POST /ask/{agent_name}` - Execute an agent
- `GET /agents` - List all registered agents
- `GET /health` - Health check

### RAG Endpoints (if enabled)
- `POST /rag/ingest` - Upload documents
- `GET /rag/status` - RAG system status

### Response Format
```json
{
  "success": true,
  "result": {...},
  "agent": "agent_name",
  "execution_time": 0.123
}
```

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│   Client    │───▶│   FastAPI API    │───▶│   FluxApp   │
└─────────────┘    └──────────────────┘    └─────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             ▼                             │
                    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
                    │  │   Agents    │   │    Tools    │   │   Memory    │     │
                    │  │  Registry   │   │  Registry   │   │    Store    │     │
                    │  └─────────────┘   └─────────────┘   └─────────────┘     │
                    │                             │                             │
                    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
                    │  │ LangGraph   │   │ LLM Models  │   │ RAG System  │     │
                    │  │  Adapter    │   │  (OpenAI,   │   │ (ChromaDB)  │     │
                    │  └─────────────┘   │ Anthropic)  │   └─────────────┘     │
                    │                    └─────────────┘                       │
                    └─────────────────────────────────────────────────────────┘
```

## Supported Models

| Provider | Models | Status |
|----------|---------|--------|
| OpenAI | GPT-3.5, GPT-4, GPT-4-turbo | ✅ |
| Anthropic | Claude 3 (Haiku, Sonnet, Opus) | ✅ |
| Ollama | All local models | ✅ |
| Custom APIs | Any OpenAI-compatible | ✅ |

## Roadmap

### Current (v0.x)
- [x] Core agent framework
- [x] FastAPI integration
- [x] Basic LLM providers
- [x] Tool system
- [x] Memory persistence
- [x] RAG integration

### Upcoming (v1.x)
- [ ] Async task queue (Celery/Redis)
- [ ] Streaming responses
- [ ] Advanced monitoring
- [ ] Agent marketplace
- [ ] Visual workflow builder

### Future (v2.x)
- [ ] Multi-agent conversations
- [ ] Built-in authentication
- [ ] Distributed deployment
- [ ] Enterprise features

## Community & Support

- **Documentation**: [docs.fluxgraph.com](https://docs.fluxgraph.com)
- **Discord**: [Join our community](https://discord.gg/Z9bAqjYvPc)
- **Issues**: [GitHub Issues](https://github.com/ihtesham-jahangir/fluxgraph/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ihtesham-jahangir/fluxgraph/discussions)

## Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if needed
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ by developers, for developers.**

*FluxGraph - Where AI agents meet production reality.*