<div align="center">
  <img src="logo.png" alt="FluxGraph Logo" width="200" height="200"/>
</div>

<h1 align="center">FluxGraph</h1>

<p align="center">
  <strong>Enterprise-grade framework for building, orchestrating, and deploying AI agent systems</strong>
</p>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/fluxgraph?color=blue&style=flat-square)](https://pypi.org/project/fluxgraph/)
[![Python](https://img.shields.io/pypi/pyversions/fluxgraph?style=flat-square)](https://pypi.org/project/fluxgraph/)
[![Downloads](https://img.shields.io/pypi/dm/fluxgraph?style=flat-square)](https://pypi.org/project/fluxgraph/)
[![License](https://img.shields.io/github/license/ihtesham-jahangir/fluxgraph?style=flat-square)](https://github.com/ihtesham-jahangir/fluxgraph/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen?style=flat-square)](https://fluxgraph.readthedocs.io)
[![Discord](https://img.shields.io/discord/1243184424318402592?logo=discord&label=Discord&style=flat-square)](https://discord.gg/Z9bAqjYvPc)

</div>

---

## Overview

FluxGraph is a production-ready framework designed for developers who need to build sophisticated AI agent systems without the complexity of traditional orchestration platforms. It provides direct LLM integration, structured workflow management, and enterprise-grade deployment capabilities.

### Core Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| **Agent Orchestration** | Register and manage multiple AI agents with unified API | ✅ Production |
| **LLM Integration** | Native support for OpenAI, Anthropic, Ollama, and custom APIs | ✅ Production |
| **Tool System** | Extensible Python function integration for agents | ✅ Production |
| **Memory Persistence** | PostgreSQL and Redis-backed conversation memory | ✅ Production |
| **RAG Integration** | Automatic Retrieval-Augmented Generation with ChromaDB | ✅ Production |
| **LangGraph Support** | Complex workflow orchestration with state management | ✅ Production |
| **FastAPI Backend** | High-performance REST API with automatic documentation | ✅ Production |
| **Async Processing** | Non-blocking operations with background task support | 🚧 Beta |
| **Monitoring & Logging** | Structured logging with request tracing | 🚧 Beta |

---

## Installation

### Standard Installation
```bash
pip install fluxgraph
```

### Development Installation
```bash
git clone https://github.com/ihtesham-jahangir/fluxgraph.git
cd fluxgraph
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Optional Dependencies
```bash
# RAG capabilities
pip install fluxgraph[rag]

# Memory persistence
pip install fluxgraph[postgres]

# Complete installation
pip install fluxgraph[all]
```

---

## Quick Start

### Basic Agent Implementation

```python
from fluxgraph import FluxApp

# Initialize application
app = FluxApp(title="AI Agent API", version="1.0.0")

@app.agent()
def echo_agent(message: str) -> dict:
    """Simple echo agent for testing."""
    return {"response": f"Echo: {message}"}

# Start server: flux run app.py
```

### LLM-Powered Agent

```python
import os
from fluxgraph import FluxApp
from fluxgraph.models import OpenAIProvider

app = FluxApp(title="Smart Assistant API")

# Configure LLM provider
llm = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4",
    temperature=0.7
)

@app.agent()
async def assistant_agent(query: str) -> dict:
    """Intelligent assistant powered by GPT-4."""
    try:
        response = await llm.generate(
            prompt=f"You are a helpful assistant. Answer concisely: {query}",
            max_tokens=150
        )
        return {
            "query": query,
            "answer": response.get("text", "No response generated"),
            "model": response.get("model", "gpt-4")
        }
    except Exception as e:
        return {"error": f"Failed to generate response: {str(e)}"}
```

### Tool-Enhanced Agent

```python
@app.tool()
def web_search(query: str, max_results: int = 5) -> list:
    """Search the web for information."""
    # Implementation would use actual search API
    return [{"title": f"Result for {query}", "url": "https://example.com"}]

@app.tool()
def calculate(expression: str) -> float:
    """Evaluate mathematical expressions safely."""
    try:
        # Note: Use ast.literal_eval or similar for production safety
        return eval(expression.replace(" ", ""))
    except:
        raise ValueError(f"Invalid expression: {expression}")

@app.agent()
async def research_agent(task: str, tools) -> dict:
    """Agent with access to web search and calculation tools."""
    search_tool = tools.get("web_search")
    calc_tool = tools.get("calculate")
    
    if not search_tool or not calc_tool:
        return {"error": "Required tools not available"}
    
    # Use tools based on task type
    if "search" in task.lower():
        results = search_tool(task, max_results=3)
        return {"task": task, "search_results": results}
    elif any(op in task for op in ["+", "-", "*", "/"]):
        try:
            result = calc_tool(task)
            return {"task": task, "calculation": result}
        except ValueError as e:
            return {"error": str(e)}
    
    return {"message": "Task type not recognized"}
```

---

## Enterprise Features

### Memory-Persistent Agents

```python
from fluxgraph.core import PostgresMemory, DatabaseManager

# Configure database connection
db_manager = DatabaseManager(os.getenv("DATABASE_URL"))
memory_store = PostgresMemory(db_manager)

app = FluxApp(
    title="Conversational AI API",
    memory_store=memory_store
)

@app.agent()
async def conversation_agent(
    message: str, 
    session_id: str, 
    memory
) -> dict:
    """Persistent conversation agent with memory."""
    if not memory:
        return {"error": "Memory store not configured"}
    
    try:
        # Store user message
        await memory.add(session_id, {
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Retrieve conversation history
        history = await memory.get(session_id, limit=10)
        
        # Generate contextual response
        context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in reversed(history)
        ])
        
        # In production, use LLM with context
        response = f"Based on our conversation: {message}"
        
        # Store assistant response
        await memory.add(session_id, {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "message": message,
            "response": response,
            "session_id": session_id,
            "history_length": len(history)
        }
        
    except Exception as e:
        return {"error": f"Conversation processing failed: {str(e)}"}
```

### Retrieval-Augmented Generation

```python
# RAG is auto-initialized when dependencies are available
@app.agent()
async def knowledge_agent(question: str, rag) -> dict:
    """Agent with access to knowledge base via RAG."""
    if not rag:
        return {"error": "RAG system not configured"}
    
    try:
        # Query knowledge base
        retrieved_docs = await rag.query(
            question, 
            top_k=5,
            filters={"document_type": "policy"}  # Optional filtering
        )
        
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "No relevant information found in knowledge base",
                "sources": []
            }
        
        # Extract context from retrieved documents
        context = "\n\n".join([
            doc.get("content", "") for doc in retrieved_docs
        ])
        
        # Generate answer using LLM with retrieved context
        prompt = f"""
        Based on the following context, answer the question concisely and accurately.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        # Use configured LLM (example with OpenAI)
        llm_response = await llm.generate(prompt, max_tokens=200)
        
        return {
            "question": question,
            "answer": llm_response.get("text", "Could not generate answer"),
            "sources": [doc.get("metadata", {}) for doc in retrieved_docs],
            "context_length": len(context)
        }
        
    except Exception as e:
        return {"error": f"Knowledge retrieval failed: {str(e)}"}

# Document ingestion endpoint
@app.api.post("/knowledge/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Ingest documents into the knowledge base."""
    if not app.rag_connector:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Ingest into RAG system
        success = await app.rag_connector.ingest(
            temp_path,
            metadata={
                "filename": file.filename,
                "upload_time": datetime.utcnow().isoformat(),
                "content_type": file.content_type
            }
        )
        
        # Clean up temporary file
        os.remove(temp_path)
        
        if success:
            return {"message": f"Document '{file.filename}' ingested successfully"}
        else:
            raise HTTPException(status_code=500, detail="Ingestion failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")
```

---

## Configuration

### Environment Variables

```bash
# Required for LLM providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database configuration
DATABASE_URL=postgresql://user:password@localhost:5432/fluxgraph

# RAG configuration
RAG_PERSIST_DIR=./knowledge_base
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Application settings
FLUXGRAPH_DEBUG=false
FLUXGRAPH_LOG_LEVEL=INFO
FLUXGRAPH_CORS_ORIGINS=["http://localhost:3000"]
```

### Production Configuration

```python
from fluxgraph import FluxApp
from fluxgraph.core import PostgresMemory, DatabaseManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize production app
app = FluxApp(
    title="Production AI API",
    version="1.0.0",
    description="Enterprise AI agent system",
    memory_store=PostgresMemory(DatabaseManager(os.getenv("DATABASE_URL"))),
    cors_enabled=True,
    cors_origins=os.getenv("FLUXGRAPH_CORS_ORIGINS", "").split(","),
    auto_init_rag=True,
    debug=os.getenv("FLUXGRAPH_DEBUG", "false").lower() == "true"
)
```

---

## API Reference

### Agent Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask/{agent_name}` | Execute agent with parameters |
| `GET` | `/agents` | List all registered agents |
| `GET` | `/agents/{agent_name}` | Get agent details |
| `GET` | `/health` | System health check |

### RAG Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/knowledge/ingest` | Upload documents to knowledge base |
| `GET` | `/knowledge/status` | RAG system status |
| `DELETE` | `/knowledge/clear` | Clear knowledge base |

### Request/Response Format

**Request:**
```json
{
  "parameter1": "value1",
  "parameter2": "value2"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "response": "Agent output"
  },
  "metadata": {
    "agent": "agent_name",
    "execution_time": 0.145,
    "request_id": "uuid-string"
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "message": "Error description",
    "code": "ERROR_CODE",
    "details": {}
  },
  "metadata": {
    "request_id": "uuid-string",
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

---

## Deployment

### Local Development
```bash
# Using FluxGraph CLI
flux run app.py --host 0.0.0.0 --port 8000

# Using Uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment
```bash
# Using Gunicorn with Uvicorn workers
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# With environment configuration
gunicorn app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fluxgraph-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fluxgraph-api
  template:
    metadata:
      labels:
        app: fluxgraph-api
    spec:
      containers:
      - name: fluxgraph-api
        image: fluxgraph:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: fluxgraph-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: fluxgraph-secrets
              key: openai-key
```

---

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   Load Balancer │───▶│   FastAPI App   │───▶│   FluxGraph     │
│   (Nginx/ALB)   │    │   (Gunicorn)    │    │   Core Engine   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌────────────────────────────────┼────────────────────────────────┐
                       │                                │                                │
                       ▼                                ▼                                ▼
                ┌─────────────┐                 ┌─────────────┐                 ┌─────────────┐
                │             │                 │             │                 │             │
                │   Agent     │                 │    Tool     │                 │   Memory    │
                │  Registry   │                 │  Registry   │                 │    Store    │
                │             │                 │             │                 │ (Postgres)  │
                └─────────────┘                 └─────────────┘                 └─────────────┘
                       │                                │                                │
                       ▼                                ▼                                ▼
                ┌─────────────┐                 ┌─────────────┐                 ┌─────────────┐
                │             │                 │             │                 │             │
                │ LangGraph   │                 │ LLM Models  │                 │ RAG System  │
                │  Adapter    │                 │ (OpenAI,    │                 │ (ChromaDB)  │
                │             │                 │ Anthropic)  │                 │             │
                └─────────────┘                 └─────────────┘                 └─────────────┘
```

---

## Supported Integrations

### LLM Providers
| Provider | Models | Authentication |
|----------|--------|----------------|
| OpenAI | GPT-3.5, GPT-4, GPT-4-turbo | API Key |
| Anthropic | Claude 3 (Haiku, Sonnet, Opus) | API Key |
| Ollama | All local models | Local |
| Azure OpenAI | GPT models via Azure | API Key + Endpoint |
| Custom APIs | OpenAI-compatible endpoints | Configurable |

### Memory Backends
| Backend | Use Case | Configuration |
|---------|----------|---------------|
| PostgreSQL | Production, persistent storage | `DATABASE_URL` |
| Redis | Fast access, session storage | `REDIS_URL` |
| SQLite | Development, testing | Local file |
| In-Memory | Temporary, stateless | No configuration |

### RAG Systems
| Component | Implementation | Purpose |
|-----------|----------------|---------|
| Vector Store | ChromaDB | Document embedding storage |
| Embeddings | SentenceTransformers | Text vectorization |
| Document Loaders | Unstructured | Multi-format parsing |
| Chunking | LangChain | Text segmentation |

---

## Development Roadmap

### Version 1.0 (Current)
- [x] Core agent framework
- [x] Multi-LLM provider support
- [x] Tool system with decorators
- [x] Memory persistence layer
- [x] RAG integration
- [x] FastAPI REST endpoints
- [x] Production deployment support

### Version 1.1 (Q2 2024)
- [ ] Async task queues (Celery/Redis)
- [ ] Real-time streaming responses
- [ ] Advanced logging and monitoring
- [ ] Agent performance metrics
- [ ] Batch processing capabilities

### Version 1.2 (Q3 2024)
- [ ] Multi-agent conversation orchestration
- [ ] Built-in authentication and authorization
- [ ] Visual workflow designer
- [ ] Enterprise SSO integration
- [ ] Advanced error handling and recovery

### Version 2.0 (Q4 2024)
- [ ] Distributed agent execution
- [ ] Auto-scaling infrastructure
- [ ] Agent marketplace
- [ ] Advanced workflow templates
- [ ] Enterprise audit and compliance features

---

## Contributing

We welcome contributions from the community. Please read our contributing guidelines before submitting pull requests.

### Development Setup
```bash
# Clone repository
git clone https://github.com/ihtesham-jahangir/fluxgraph.git
cd fluxgraph

# Setup development environment
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black fluxgraph/
flake8 fluxgraph/
```

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Include comprehensive tests for new features
- Update documentation for API changes
- Submit detailed pull request descriptions

---

## Support and Community

- **📖 Documentation**: [https://fluxgraph.readthedocs.io](https://fluxgraph.readthedocs.io)
- **💬 Discord Community**: [Join Discussion](https://discord.gg/Z9bAqjYvPc)
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/ihtesham-jahangir/fluxgraph/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/ihtesham-jahangir/fluxgraph/discussions)
- **📧 Enterprise Support**: enterprise@fluxgraph.com

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p><strong>FluxGraph</strong> - Enterprise AI Agent Orchestration</p>
  <p>Built for production. Designed for developers.</p>
</div>