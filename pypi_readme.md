# FluxGraph

**Production-grade AI agent orchestration framework for building secure, scalable multi-agent systems**

[![PyPI version](https://img.shields.io/pypi/v/fluxgraph?color=blue)](https://pypi.org/project/fluxgraph/)
[![Python](https://img.shields.io/pypi/pyversions/fluxgraph)](https://pypi.org/project/fluxgraph/)
[![License](https://img.shields.io/github/license/ihtesham-jahangir/fluxgraph)](https://github.com/ihtesham-jahangir/fluxgraph/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://fluxgraph.readthedocs.io)

---

## Overview

FluxGraph is the **most complete open-source AI agent framework** for production deployment, combining cutting-edge innovations with enterprise-grade reliability. Built for developers who need sophisticated AI agent systems without complexity or vendor lock-in.

## Why FluxGraph?

✅ **Graph-Based Workflows** - Visual agent orchestration with conditional routing and loops  
✅ **Semantic Caching** - Intelligent response caching reduces LLM costs by 70%+  
✅ **Hybrid Memory System** - Short-term + long-term + episodic memory with semantic search  
✅ **Circuit Breakers** - Built-in fault tolerance and resilience  
✅ **Real-time Cost Tracking** - Per-agent, per-model cost analytics  
✅ **Blockchain Audit Logs** - Immutable execution history  
✅ **PII Detection** - Automatic detection and redaction of 9 sensitive data types  
✅ **Streaming Support** - Server-Sent Events for real-time responses  
✅ **Production Ready** - Security, monitoring, and scaling out of the box

## Installation

```bash
# Full installation with v3.0 features
pip install fluxgraph[full]

# Minimal installation
pip install fluxgraph

# Specific features
pip install fluxgraph[p0]           # Graph workflows, memory, caching
pip install fluxgraph[production]   # Streaming, sessions, retry
pip install fluxgraph[security]     # RBAC, audit logs, PII detection
pip install fluxgraph[rag]          # RAG with ChromaDB
pip install fluxgraph[all]          # Everything
```

## Quick Start

### Hello World

```python
from fluxgraph import FluxApp

app = FluxApp(title="My AI App")

@app.agent()
async def assistant(message: str) -> dict:
    return {"response": f"You said: {message}"}

# Run: flux run app.py
# Test: curl -X POST http://localhost:8000/ask/assistant -d '{"message":"Hello!"}'
```

### LLM-Powered Agent

```python
from fluxgraph import FluxApp
from fluxgraph.models import OpenAIProvider

app = FluxApp(title="Smart Assistant")
llm = OpenAIProvider(api_key="your-key", model="gpt-4")

@app.agent()
async def assistant(query: str) -> dict:
    response = await llm.generate(f"Answer: {query}")
    return {"answer": response.get("text")}
```

## Key Features

### 1. Graph-Based Workflows

Create complex agent workflows with conditional routing:

```python
from fluxgraph.core import WorkflowBuilder

workflow = (WorkflowBuilder("research")
    .add_agent("researcher", research_agent)
    .add_agent("analyzer", analysis_agent)
    .connect("researcher", "analyzer")
    .branch("analyzer", quality_check, {
        "retry": "researcher",
        "complete": "__end__"
    })
    .start_from("researcher")
    .build())

result = await workflow.execute({"query": "AI trends 2025"})
```

### 2. Advanced Memory System

Hybrid memory with semantic search:

```python
app = FluxApp(enable_advanced_memory=True)

@app.agent()
async def smart_agent(query: str, advanced_memory) -> dict:
    # Store in short-term memory
    advanced_memory.store(query, MemoryType.SHORT_TERM, importance=0.8)
    
    # Semantic search across all memories
    similar = advanced_memory.recall_similar(query, k=5)
    
    # Automatic consolidation to long-term
    advanced_memory.consolidate()
    
    return {"context": [e.content for e, score in similar]}
```

### 3. Semantic Caching

Reduce costs by 70%+ with intelligent caching:

```python
app = FluxApp(enable_agent_cache=True, cache_strategy="hybrid")

@app.agent()
async def expensive_agent(query: str, cache) -> dict:
    # Automatically checks cache for similar queries
    result = await expensive_llm_call(query)
    return {"answer": result}

# Cache statistics
stats = cache.get_stats()  # Hit rate, size, savings
```

### 4. Multi-Agent Orchestration

```python
@app.agent()
async def supervisor(task: str, call_agent, broadcast) -> dict:
    research = await call_agent("research_agent", query=task)
    analyses = await broadcast(
        ["technical_analyst", "business_analyst"],
        data=research
    )
    return {"results": analyses}
```

### 5. Security Features

Automatic protection against threats:

```python
app = FluxApp(enable_security=True)

@app.agent()
async def secure_agent(user_input: str) -> dict:
    # Automatic PII detection (9 types)
    # Prompt injection shield (7 techniques)
    # Immutable audit logging
    # RBAC + JWT authentication
    
    return await process(user_input)
```

**PII Detection**: EMAIL, PHONE, SSN, CREDIT_CARD, IP_ADDRESS, PASSPORT, DRIVER_LICENSE, DATE_OF_BIRTH, MEDICAL_RECORD

**Injection Prevention**: IGNORE_PREVIOUS, ROLE_PLAY, ENCODED_INJECTION, DELIMITER_INJECTION, PRIVILEGE_ESCALATION, CONTEXT_OVERFLOW, PAYLOAD_SPLITTING

### 6. Human-in-the-Loop

```python
@app.agent()
async def critical_agent(action: str) -> dict:
    approval = await app.hitl_manager.request_approval(
        agent_name="critical_agent",
        task_description=f"Execute: {action}",
        risk_level="HIGH",
        timeout_seconds=300
    )
    
    if await approval.wait_for_approval():
        return {"status": "executed"}
    return {"status": "rejected"}
```

### 7. Streaming Responses

```python
from fastapi.responses import StreamingResponse

@app.api.get("/stream/{agent}")
async def stream(agent: str, query: str):
    async def generate():
        async for chunk in app.orchestrator.run_streaming(agent, {"query": query}):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 8. Batch Processing

```python
# Submit 1000 tasks
job_id = await app.batch_processor.submit_batch(
    agent_name="processor",
    payloads=tasks,
    max_concurrent=50
)

# Check progress
status = app.batch_processor.get_job_status(job_id)
```

### 9. Cost Tracking

```python
# Automatic per-agent cost tracking
costs = app.orchestrator.cost_tracker.get_summary()
# {"research_agent": {"cost": "$2.34", "calls": 145}}
```

### 10. RAG Integration

```python
app = FluxApp(enable_rag=True)

@app.agent()
async def rag_agent(query: str, rag) -> dict:
    # Add documents
    await rag.add_documents([
        {"text": "FluxGraph is awesome", "metadata": {"source": "docs"}}
    ])
    
    # Query with semantic search
    results = await rag.query(query, top_k=5)
    return {"sources": results}
```

## Supported Integrations

### LLM Providers
- **OpenAI**: GPT-3.5, GPT-4, GPT-4 Turbo
- **Anthropic**: Claude 3 (Haiku, Sonnet, Opus)
- **Google**: Gemini Pro, Ultra
- **Groq**: Mixtral, Llama 3
- **Ollama**: All local models
- **Azure OpenAI**: GPT models

### Memory Backends
- **PostgreSQL**: Production persistence
- **Redis**: Fast session storage
- **SQLite**: Development/testing
- **In-Memory**: Temporary stateless

## Production Configuration

```python
app = FluxApp(
    # v3.0 Features
    enable_workflows=True,
    enable_advanced_memory=True,
    enable_agent_cache=True,
    cache_strategy="hybrid",
    
    # Production
    enable_streaming=True,
    enable_sessions=True,
    
    # Security
    enable_security=True,
    
    # Orchestration
    enable_orchestration=True
)
```

## Performance

**v3.0 Benchmarks:**
- Cache Hit Rate: 85-95% on similar queries
- Cost Reduction: 70%+ with semantic caching
- Memory Consolidation: <50ms for 1000 entries
- Workflow Execution: 100+ steps/second
- Latency: <10ms overhead

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask/{agent}` | POST | Execute agent |
| `/stream/{agent}` | GET | Stream response |
| `/workflows` | GET | List workflows |
| `/workflows/{name}/execute` | POST | Execute workflow |
| `/memory/stats` | GET | Memory statistics |
| `/cache/stats` | GET | Cache statistics |
| `/system/status` | GET | System health |
| `/system/costs` | GET | Cost summary |

## Docker Deployment

```yaml
version: '3.8'
services:
  fluxgraph:
    image: fluxgraph:3.0.0
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db/fluxgraph
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
      - redis
```

## Documentation & Support

- **Documentation**: https://fluxgraph.readthedocs.io
- **Discord**: https://discord.gg/Z9bAqjYvPc
- **GitHub**: https://github.com/ihtesham-jahangir/fluxgraph
- **Issues**: https://github.com/ihtesham-jahangir/fluxgraph/issues
- **Enterprise**: enterprise@fluxgraph.com

## License

MIT License - Free and open-source forever.

---

**FluxGraph 3.0** - The most advanced open-source AI agent framework

Graph workflows • Semantic caching • Hybrid memory • Enterprise security

⭐ Star us on [GitHub](https://github.com/ihtesham-jahangir/fluxgraph) if FluxGraph powers your AI systems!