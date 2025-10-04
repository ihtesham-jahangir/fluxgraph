<div align="center">
  <img src="https://raw.githubusercontent.com/ihtesham-jahangir/fluxgraph/main/logo.png" alt="FluxGraph Logo" width="200" height="200"/>
  
  # FluxGraph

  **Production-grade AI agent orchestration framework for building secure, scalable multi-agent systems**

  [![PyPI version](https://img.shields.io/pypi/v/fluxgraph?color=blue&style=flat-square)](https://pypi.org/project/fluxgraph/)
  [![Python](https://img.shields.io/pypi/pyversions/fluxgraph?style=flat-square)](https://pypi.org/project/fluxgraph/)
  [![Downloads](https://img.shields.io/pypi/dm/fluxgraph?style=flat-square)](https://pypi.org/project/fluxgraph/)
  [![License](https://img.shields.io/github/license/ihtesham-jahangir/fluxgraph?style=flat-square)](https://github.com/ihtesham-jahangir/fluxgraph/blob/main/LICENSE)
  [![Discord](https://img.shields.io/discord/1243184424318402592?logo=discord&label=Discord&style=flat-square)](https://discord.gg/Z9bAqjYvPc)
</div>

---

## Why FluxGraph?

FluxGraph is the most complete open-source AI agent framework for production environments, combining enterprise-grade features with zero vendor lock-in.

**Unique capabilities not found in other open-source frameworks:**
- Native circuit breakers for automatic failure recovery
- Real-time cost tracking per agent and model
- Blockchain-style immutable audit logs for compliance
- Built-in PII detection (9 types) and prompt injection shields (7 techniques)
- Agent-to-agent handoff protocol with context preservation
- Production-ready from installation: streaming, sessions, retry, validation

## Architecture

<div align="center">
  <img src="https://raw.githubusercontent.com/ihtesham-jahangir/fluxgraph/main/architecture.png" alt="FluxGraph Architecture" width="100%"/>
</div>

### System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client Applications                         │
│                   (Web, Mobile, CLI, API Clients)                   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Load Balancer        │
                    │  (Nginx/CloudFlare)     │
                    └────────────┬────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────┐
│                                │                                    │
│              ┌─────────────────▼─────────────────┐                 │
│              │    FluxGraph API Layer            │                 │
│              │  (FastAPI + Gunicorn Workers)     │                 │
│              └─────────────────┬─────────────────┘                 │
│                                │                                    │
│              ┌─────────────────▼─────────────────┐                 │
│              │   FluxGraph Core Orchestrator     │                 │
│              │  • Agent Registry & Routing       │                 │
│              │  • Circuit Breakers              │                 │
│              │  • Cost Tracking                 │                 │
│              │  • Security Layer                │                 │
│              └─────────────────┬─────────────────┘                 │
│                                │                                    │
│       ┌────────────┬───────────┼───────────┬────────────┐          │
│       │            │           │           │            │          │
│  ┌────▼────┐  ┌───▼────┐  ┌──▼───┐  ┌────▼─────┐ ┌───▼────┐     │
│  │  Agent  │  │  Tool  │  │Memory│  │   RAG    │ │  MCP   │     │
│  │Registry │  │Registry│  │Store │  │  System  │ │ Server │     │
│  └────┬────┘  └───┬────┘  └──┬───┘  └────┬─────┘ └───┬────┘     │
│       │           │          │           │           │          │
└───────┼───────────┼──────────┼───────────┼───────────┼──────────┘
        │           │          │           │           │
   ┌────▼─────┐┌───▼────┐┌────▼─────┐┌────▼─────┐┌───▼────┐
   │  Custom  ││External││PostgreSQL││ ChromaDB ││Protocol│
   │  Agents  ││  APIs  ││  Redis   ││Embeddings││ Tools  │
   └──────────┘└────────┘└──────────┘└──────────┘└────────┘
                    │
              ┌─────▼─────┐
              │   LLM     │
              │ Providers │
              │ (OpenAI,  │
              │Anthropic, │
              │  Groq)    │
              └───────────┘
```

### Data Flow

```
User Request → API Gateway → Security Layer → Agent Router
                                    ↓
            ┌───────────────────────┴───────────────────────┐
            │                                               │
    ┌───────▼───────┐                           ┌──────────▼────────┐
    │  Audit Logger │                           │  Cost Tracker     │
    │  (Blockchain) │                           │  (Real-time)      │
    └───────────────┘                           └───────────────────┘
            │                                               │
            └───────────────────┬───────────────────────────┘
                                ▼
                        ┌───────────────┐
                        │ Agent Engine  │
                        │ • PII Check   │
                        │ • Injection   │
                        │ • Circuit     │
                        └───────┬───────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ┌─────────┐ ┌─────────┐ ┌─────────┐
              │  Tools  │ │ Memory  │ │   RAG   │
              └─────────┘ └─────────┘ └─────────┘
                    │           │           │
                    └───────────┼───────────┘
                                ▼
                        ┌───────────────┐
                        │  LLM Provider │
                        └───────┬───────┘
                                │
                        Response Generated
                                │
                                ▼
                    Stream/Return to User
```

---

## Quick Start

```bash
pip install fluxgraph[all]
```

### Basic Agent

```python
from fluxgraph import FluxApp

app = FluxApp(
    enable_streaming=True,
    enable_sessions=True,
    enable_security=True
)

@app.agent()
async def assistant(query: str, tools, memory) -> dict:
    """Enterprise-ready agent with automatic security and monitoring."""
    response = await llm.generate(query)
    return {"answer": response}
```

### Multi-Agent System

```python
@app.agent()
async def supervisor(task: str, call_agent, broadcast) -> dict:
    """Orchestrates specialized agents with context preservation."""
    research = await call_agent("research_agent", query=task)
    results = await broadcast(["analysis_agent", "summary_agent"], data=research)
    return {"results": results}
```

## Core Features

### Production Readiness
- **Streaming Responses**: SSE and WebSocket support for real-time output
- **Session Management**: SQLite and PostgreSQL persistence with conversation history
- **Retry Logic**: Exponential backoff with configurable policies
- **Output Validation**: Pydantic-based schema validation
- **Circuit Breakers**: Automatic failure detection and recovery

### Enterprise Security
- **Audit Logging**: Immutable, blockchain-style logs for GDPR, HIPAA, SOC2 compliance
- **PII Detection**: Automatic detection and redaction of 9 PII types (email, SSN, credit cards, etc.)
- **Prompt Injection Shield**: Protection against 7 injection techniques
- **RBAC + JWT**: Role-based access control with JWT authentication

### Advanced Orchestration
- **Agent Handoffs**: Context-aware delegation between agents
- **Human-in-the-Loop**: Approval workflows for critical decisions
- **Batch Processing**: Priority queues for high-volume tasks
- **Task Adherence**: Goal tracking and compliance monitoring
- **Cost Tracking**: Real-time cost analysis per agent and model

### Ecosystem Integration
- **MCP Support**: Full Model Context Protocol compatibility
- **Multi-Modal**: Image and audio processing capabilities
- **Agent Versioning**: A/B testing and rollback support
- **RAG Integration**: Built-in retrieval-augmented generation

## Installation Options

```bash
# Full installation
pip install fluxgraph[all]

# Feature-specific
pip install fluxgraph[production]      # Streaming, sessions, retry
pip install fluxgraph[security]        # Audit, PII, RBAC
pip install fluxgraph[orchestration]   # Handoffs, HITL, batch
pip install fluxgraph[rag]            # RAG capabilities
pip install fluxgraph[postgres]       # PostgreSQL persistence
```

## Feature Comparison

| Feature | FluxGraph | LangGraph | CrewAI | AutoGen |
|---------|-----------|-----------|---------|---------|
| **Streaming Responses** | ✅ Native | ⚠️ Callbacks | ❌ | ❌ |
| **Immutable Audit Logs** | ✅ | ❌ | ❌ | ❌ |
| **PII Detection** | ✅ 9 types | ❌ | ❌ | ❌ |
| **Prompt Injection Shield** | ✅ 7 techniques | ❌ | ❌ | ❌ |
| **Circuit Breakers** | ✅ Unique | ❌ | ❌ | ❌ |
| **Real-time Cost Tracking** | ✅ Per-agent | ❌ | ❌ | ❌ |
| **Agent Handoffs** | ✅ Context-aware | ⚠️ Basic | ⚠️ Basic | ✅ |
| **HITL Workflows** | ✅ | ❌ | ❌ | ✅ |
| **MCP Support** | ✅ Full | ⚠️ Adapters | ❌ | ❌ |
| **Session Management** | ✅ SQL-backed | ⚠️ External | ⚠️ Basic | ⚠️ Manual |

## Production Examples

### Streaming Responses

```python
from fastapi.responses import StreamingResponse

@app.api.get("/stream/{agent_name}")
async def stream_agent(agent_name: str, query: str):
    async def generate():
        async for chunk in app.orchestrator.run_streaming(agent_name, {"query": query}):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Human-in-the-Loop

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
        return {"status": "executed", "result": execute_action(action)}
    return {"status": "rejected", "reason": approval.rejection_reason}
```

### Batch Processing

```python
@app.api.post("/batch/process")
async def batch_process(agent_name: str, tasks: list):
    job_id = await app.batch_processor.submit_batch(
        agent_name=agent_name,
        payloads=tasks,
        priority=0,
        max_concurrent=50
    )
    return {"job_id": job_id, "tasks": len(tasks)}
```

### Security Features

```python
@app.agent()
async def secure_agent(user_input: str) -> dict:
    # Automatic prompt injection detection
    is_safe, detections = app.prompt_shield.is_safe(user_input)
    if not is_safe:
        return {"error": "Security violation", "detections": detections}
    
    # Automatic PII redaction
    redacted_input, pii_found = app.pii_detector.redact(user_input)
    
    # Immutable audit logging
    app.audit_logger.log(
        AuditEventType.AGENT_EXECUTION,
        user_id="user_123",
        details={"pii_count": len(pii_found)}
    )
    
    return {"response": process(redacted_input), "pii_redacted": len(pii_found)}
```

## Production Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  fluxgraph:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/fluxgraph
      - FLUXGRAPH_ENABLE_SECURITY=true
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
      - redis
    command: gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker

  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fluxgraph
spec:
  replicas: 5
  template:
    spec:
      containers:
      - name: fluxgraph
        image: fluxgraph:latest
        env:
        - name: FLUXGRAPH_ENABLE_STREAMING
          value: "true"
        - name: FLUXGRAPH_ENABLE_SECURITY
          value: "true"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask/{agent_name}` | POST | Execute agent with cost tracking |
| `/stream/{agent_name}` | GET | Stream agent responses in real-time |
| `/sessions` | POST | Create conversation session |
| `/sessions/{id}` | GET | Retrieve session history |
| `/approvals` | GET | List pending HITL requests |
| `/approvals/{id}/approve` | POST | Approve HITL request |
| `/batch` | POST | Submit batch processing job |
| `/batch/{job_id}` | GET | Check batch job status |
| `/system/status` | GET | System health and circuit breakers |
| `/system/costs` | GET | Real-time cost breakdown |
| `/audit/verify` | GET | Verify audit log integrity |

## Supported LLM Providers

| Provider | Models | Streaming | Cost Tracking |
|----------|--------|-----------|---------------|
| OpenAI | GPT-3.5, GPT-4, GPT-4 Turbo | ✅ | ✅ |
| Anthropic | Claude 3 (Haiku, Sonnet, Opus) | ✅ | ✅ |
| Google | Gemini Pro, Gemini Ultra | ✅ | ✅ |
| Groq | Mixtral, Llama 3 | ✅ | ✅ |
| Ollama | All local models | ✅ | ❌ |
| Azure OpenAI | GPT models | ✅ | ✅ |

## Performance

- **Execution Speed**: 45 simple tasks/min, 32 multi-agent tasks/min
- **Cost Tracking**: <1% error rate across all providers
- **Memory**: Low overhead with FastAPI async architecture
- **Scalability**: Horizontal scaling with Kubernetes

## Development Roadmap

### ✅ v2.0 (Current)
All enterprise features complete: streaming, security, orchestration, MCP support

### 🚧 v2.1 (Q1 2026)
- Visual workflow designer
- Enterprise SSO (SAML, OAuth2)
- Analytics dashboard
- SOC2/HIPAA certifications

### 📋 v2.2 (Q2 2026)
- Distributed execution
- Auto-scaling infrastructure
- Multi-cloud deployment

## Community

- **Documentation**: [fluxgraph.readthedocs.io](https://fluxgraph.readthedocs.io)
- **Discord**: [Join Community](https://discord.gg/Z9bAqjYvPc)
- **GitHub**: [Issues](https://github.com/ihtesham-jahangir/fluxgraph/issues) | [Discussions](https://github.com/ihtesham-jahangir/fluxgraph/discussions)
- **Enterprise Support**: enterprise@fluxgraph.com

## Contributing

```bash
git clone https://github.com/ihtesham-jahangir/fluxgraph.git
cd fluxgraph
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

We welcome contributions in:
- Core features and LLM integrations
- Security enhancements
- Documentation and tutorials
- Testing and benchmarks

## License

MIT License - Free and open-source forever. No vendor lock-in.

---

**FluxGraph**: Enterprise AI agent framework • Zero vendor lock-in • Production-ready

*Star us on GitHub if FluxGraph powers your AI systems!* ⭐