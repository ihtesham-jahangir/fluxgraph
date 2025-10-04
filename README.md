<div align="center">
  <img src="logo.jpeg" alt="FluxGraph Logo" width="200" height="200"/>
  
  # FluxGraph

  **Production-grade AI agent orchestration framework for building secure, scalable multi-agent systems**

  [![PyPI version](https://img.shields.io/pypi/v/fluxgraph?color=blue&style=flat-square)](https://pypi.org/project/fluxgraph/)
  [![Python](https://img.shields.io/pypi/pyversions/fluxgraph?style=flat-square)](https://pypi.org/project/fluxgraph/)
  [![Downloads](https://img.shields.io/pypi/dm/fluxgraph?style=flat-square)](https://pypi.org/project/fluxgraph/)
  [![License](https://img.shields.io/github/license/ihtesham-jahangir/fluxgraph?style=flat-square)](https://github.com/ihtesham-jahangir/fluxgraph/blob/main/LICENSE)
  [![Documentation](https://img.shields.io/badge/docs-available-brightgreen?style=flat-square)](https://fluxgraph.readthedocs.io)
  [![Discord](https://img.shields.io/discord/1243184424318402592?logo=discord&label=Discord&style=flat-square)](https://discord.gg/Z9bAqjYvPc)
</div>

---

## Overview

FluxGraph is the **most complete open-source AI agent framework** for production deployment, combining enterprise-grade features with zero vendor lock-in. Built for developers who need sophisticated AI agent systems without the complexity of traditional platforms.

### Why FluxGraph?

**Unique capabilities not found in other open-source frameworks:**
- ✅ **Native circuit breakers** for automatic failure recovery (unique in open-source)
- ✅ **Real-time cost tracking** per agent and model (not even Microsoft has this free)
- ✅ **Blockchain-style audit logs** for compliance (GDPR, HIPAA, SOC2)
- ✅ **Built-in PII detection** (9 types) and prompt injection shields (7 techniques)
- ✅ **Agent handoff protocol** (A2A) with context preservation
- ✅ **MCP support** for ecosystem compatibility
- ✅ **Production-ready from installation**: streaming, sessions, retry, validation

### Feature Comparison

| Feature | FluxGraph | LangGraph | CrewAI | AutoGen |
|---------|-----------|-----------|---------|---------|
| **Streaming Responses** | ✅ Native SSE | ⚠️ Callbacks | ❌ | ❌ |
| **Immutable Audit Logs** | ✅ Blockchain-style | ❌ | ❌ | ❌ |
| **PII Detection** | ✅ 9 types | ❌ | ❌ | ❌ |
| **Prompt Injection Shield** | ✅ 7 techniques | ❌ | ❌ | ❌ |
| **Circuit Breakers** | ✅ **Unique** | ❌ | ❌ | ❌ |
| **Real-time Cost Tracking** | ✅ Per-agent | ❌ | ❌ | ❌ |
| **Agent Handoffs** | ✅ Context-aware | ⚠️ Basic | ⚠️ Basic | ✅ |
| **HITL Workflows** | ✅ Built-in | ❌ | ❌ | ✅ |
| **MCP Support** | ✅ Full protocol | ⚠️ Adapters | ❌ | ❌ |
| **Session Management** | ✅ SQL-backed | ⚠️ External | ⚠️ Basic | ⚠️ Manual |
| **RAG Integration** | ✅ Native | ✅ Via adapters | ⚠️ Basic | ⚠️ Basic |
| **Agent Versioning** | ✅ A/B testing | ❌ | ❌ | ❌ |
| **Batch Processing** | ✅ Priority queues | ❌ | ❌ | ❌ |
| **RBAC + JWT Auth** | ✅ Granular | ❌ | ❌ | ❌ |

---

## Architecture

<div align="center">
  <img src="architecture.png" alt="FluxGraph Architecture" width="100%"/>
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

## Installation

### Quick Start
```bash
pip install fluxgraph[all]
```

### Feature-Specific Installations
```bash
# Production features (streaming, sessions, retry)
pip install fluxgraph[production]

# Security features (audit, PII, RBAC)
pip install fluxgraph[security]

# Advanced orchestration (handoffs, HITL, batch)
pip install fluxgraph[orchestration]

# RAG capabilities
pip install fluxgraph[rag]

# Memory persistence
pip install fluxgraph[postgres]
```

### Development Installation
```bash
git clone https://github.com/ihtesham-jahangir/fluxgraph.git
cd fluxgraph
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

---

## Quick Start Examples

### 1. Basic Agent with Enterprise Features

```python
from fluxgraph import FluxApp

# Initialize with enterprise features
app = FluxApp(
    title="Enterprise AI API",
    version="2.0.0",
    enable_streaming=True,      # SSE streaming
    enable_sessions=True,        # Session management
    enable_security=True,        # Audit + PII + RBAC
    enable_orchestration=True,   # Handoffs + HITL
    enable_mcp=True             # MCP protocol
)

@app.agent()
async def assistant(query: str, tools, memory) -> dict:
    """Enterprise agent with automatic security and monitoring."""
    # Automatic PII detection and prompt injection filtering
    # Circuit breakers prevent cascading failures
    # Cost tracking per execution
    # Audit logging for compliance
    
    response = await llm.generate(query)
    return {"answer": response, "secure": True}
```

### 2. Multi-Agent System with Handoffs

```python
@app.agent()
async def supervisor(task: str, call_agent, broadcast) -> dict:
    """Orchestrates specialized agents with context preservation."""
    
    # Delegate to specialist
    research = await call_agent("research_agent", query=task)
    
    # Broadcast to multiple agents
    results = await broadcast(
        ["analysis_agent", "summary_agent"],
        data=research
    )
    
    return {"results": results}

@app.agent()
async def research_agent(query: str, rag) -> dict:
    """Specialized research agent with RAG."""
    docs = await rag.query(query, top_k=5)
    return {"findings": docs}
```

### 3. Streaming Responses

```python
from fastapi.responses import StreamingResponse

@app.api.get("/stream/{agent_name}")
async def stream_agent(agent_name: str, query: str):
    """Stream agent responses in real-time."""
    
    async def generate():
        async for chunk in app.orchestrator.run_streaming(
            agent_name, 
            {"query": query}
        ):
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 4. Human-in-the-Loop Workflows

```python
@app.agent()
async def critical_agent(action: str) -> dict:
    """Agent requiring human approval for high-risk actions."""
    
    approval = await app.hitl_manager.request_approval(
        agent_name="critical_agent",
        task_description=f"Execute: {action}",
        risk_level="HIGH",
        timeout_seconds=300
    )
    
    if await approval.wait_for_approval():
        result = execute_action(action)
        return {"status": "executed", "result": result}
    
    return {"status": "rejected", "reason": approval.rejection_reason}
```

### 5. Batch Processing

```python
@app.api.post("/batch/process")
async def batch_process(agent_name: str, tasks: list):
    """Process thousands of tasks asynchronously."""
    
    job_id = await app.batch_processor.submit_batch(
        agent_name=agent_name,
        payloads=tasks,
        priority=0,
        max_concurrent=50
    )
    
    return {"job_id": job_id, "tasks": len(tasks)}

@app.api.get("/batch/{job_id}")
async def get_batch_status(job_id: str):
    status = app.batch_processor.get_job_status(job_id)
    return status  # {completed: 850, failed: 2, pending: 148}
```

### 6. Security Features

```python
@app.agent()
async def secure_agent(user_input: str) -> dict:
    """Agent with automatic security checks."""
    
    # Automatic prompt injection detection (7 techniques)
    is_safe, detections = app.prompt_shield.is_safe(user_input)
    if not is_safe:
        return {"error": "Security violation", "detections": detections}
    
    # Automatic PII redaction (9 types)
    redacted_input, pii_found = app.pii_detector.redact(user_input)
    
    # Immutable audit logging
    app.audit_logger.log(
        AuditEventType.AGENT_EXECUTION,
        user_id="user_123",
        details={"pii_count": len(pii_found)}
    )
    
    return {"response": process(redacted_input), "pii_redacted": len(pii_found)}
```

### 7. Memory-Persistent Conversations

```python
from fluxgraph.core import PostgresMemory, DatabaseManager

db_manager = DatabaseManager(os.getenv("DATABASE_URL"))
memory_store = PostgresMemory(db_manager)

app = FluxApp(memory_store=memory_store)

@app.agent()
async def conversation_agent(message: str, session_id: str, memory) -> dict:
    """Persistent conversation with full history."""
    
    # Store user message
    await memory.add(session_id, {
        "role": "user",
        "content": message,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Retrieve history
    history = await memory.get(session_id, limit=10)
    
    # Generate contextual response
    context = "\n".join([f"{msg['role']}: {msg['content']}" 
                         for msg in reversed(history)])
    response = await llm.generate(context)
    
    # Store response
    await memory.add(session_id, {
        "role": "assistant",
        "content": response,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return {"response": response, "history_length": len(history)}
```

### 8. RAG Integration

```python
@app.agent()
async def knowledge_agent(question: str, rag) -> dict:
    """Agent with knowledge base access."""
    
    # Query knowledge base
    retrieved_docs = await rag.query(
        question, 
        top_k=5,
        filters={"document_type": "policy"}
    )
    
    if not retrieved_docs:
        return {"answer": "No relevant information found"}
    
    # Generate answer with context
    context = "\n\n".join([doc.get("content", "") for doc in retrieved_docs])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    answer = await llm.generate(prompt, max_tokens=200)
    
    return {
        "question": question,
        "answer": answer,
        "sources": [doc.get("metadata", {}) for doc in retrieved_docs]
    }

# Document ingestion
@app.api.post("/knowledge/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Ingest documents into knowledge base."""
    temp_path = f"/tmp/{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    success = await app.rag_connector.ingest(
        temp_path,
        metadata={
            "filename": file.filename,
            "upload_time": datetime.utcnow().isoformat()
        }
    )
    
    os.remove(temp_path)
    return {"message": f"Document '{file.filename}' ingested successfully"}
```

### 9. MCP Protocol Integration

```python
# Register MCP-compatible tools
app.mcp_server.register_fluxgraph_tool(
    tool_name="web_search",
    tool_func=web_search_function,
    description="Search the web",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5}
        }
    }
)

@app.agent()
async def mcp_agent(task: str, tools) -> dict:
    """Agent using MCP-compatible tools."""
    result = await app.mcp_server.execute_tool("web_search", {"query": task})
    return result
```

### 10. Agent Versioning & A/B Testing

```python
# Register versions
app.version_manager.register_version(
    agent_name="assistant",
    version="1.0.0",
    agent_code=agent_v1_code,
    description="Original version"
)

app.version_manager.register_version(
    agent_name="assistant",
    version="2.0.0",
    agent_code=agent_v2_code,
    description="Improved with GPT-4"
)

# Start A/B test
test_id = app.version_manager.start_ab_test(
    agent_name="assistant",
    version_a="1.0.0",
    version_b="2.0.0",
    split_ratio=0.5
)

# Route users
version = app.version_manager.select_ab_version(test_id, user_id)

# Rollback if needed
app.version_manager.rollback("assistant")
```

---

## Enterprise Security Features

### Immutable Audit Logs

```python
# Automatic audit logging
app.audit_logger.log(
    AuditEventType.AGENT_EXECUTION,
    user_id="user_123",
    details={"agent": "assistant", "action": "query"},
    severity="INFO"
)

# Verify integrity (tamper-proof)
integrity = app.audit_logger.verify_integrity()
# {is_valid: True, total_entries: 1543, verified_entries: 1543}

# Export compliance report (GDPR, HIPAA, SOC2)
app.audit_logger.export_compliance_report(
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31),
    output_path="./compliance_2025.json"
)
```

### PII Detection (9 Types)

```python
# Supported: EMAIL, PHONE, SSN, CREDIT_CARD, IP_ADDRESS,
# PASSPORT, DRIVER_LICENSE, DATE_OF_BIRTH, MEDICAL_RECORD

detections = app.pii_detector.detect(user_input)
redacted, detected = app.pii_detector.redact(user_input)
# "Contact me at [REDACTED:email]"
```

### Prompt Injection Shield (7 Techniques)

```python
# Detects: IGNORE_PREVIOUS, ROLE_PLAY, ENCODED_INJECTION,
# DELIMITER_INJECTION, PRIVILEGE_ESCALATION, CONTEXT_OVERFLOW,
# SYSTEM_PROMPT_EXTRACTION

is_safe, detections = app.prompt_shield.is_safe(user_input)
if not is_safe:
    raise SecurityException("Prompt injection detected")
```

### RBAC with JWT

```python
# Create users
user = app.rbac_manager.create_user(
    username="john",
    email="john@company.com",
    password="secure_password",
    roles=[Role.DEVELOPER]
)

# Generate token
token = app.rbac_manager.generate_jwt_token(user, expires_in_hours=24)

# Check permissions
if app.rbac_manager.check_permission(user, Permission.EXECUTE_AGENT):
    result = await agent.run(query)
```

---

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
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - FLUXGRAPH_ENABLE_SECURITY=true
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

---

## API Reference

| Endpoint | Method | Description | Enterprise Feature |
|----------|--------|-------------|-------------------|
| `/ask/{agent_name}` | POST | Execute agent | Cost tracking, audit |
| `/stream/{agent_name}` | GET | Stream responses | SSE streaming |
| `/sessions` | POST | Create session | Session management |
| `/sessions/{id}` | GET | Get history | Conversation persistence |
| `/approvals` | GET | Pending HITL | Human-in-the-loop |
| `/approvals/{id}/approve` | POST | Approve request | HITL workflows |
| `/batch` | POST | Submit batch | Batch processing |
| `/batch/{job_id}` | GET | Batch status | Job monitoring |
| `/system/status` | GET | System health | Circuit breakers |
| `/system/costs` | GET | Cost summary | Real-time tracking |
| `/audit/verify` | GET | Verify integrity | Blockchain logs |

---

## Supported Integrations

### LLM Providers

| Provider | Models | Streaming | Cost Tracking |
|----------|--------|-----------|---------------|
| OpenAI | GPT-3.5, GPT-4, GPT-4 Turbo | ✅ | ✅ |
| Anthropic | Claude 3 (Haiku, Sonnet, Opus) | ✅ | ✅ |
| Google | Gemini Pro, Gemini Ultra | ✅ | ✅ |
| Groq | Mixtral, Llama 3 | ✅ | ✅ |
| Ollama | All local models | ✅ | ❌ |
| Azure OpenAI | GPT models | ✅ | ✅ |

### Memory Backends

| Backend | Use Case | Configuration |
|---------|----------|---------------|
| PostgreSQL | Production persistence | `DATABASE_URL` |
| Redis | Fast session storage | `REDIS_URL` |
| SQLite | Development/testing | Local file |
| In-Memory | Temporary stateless | None |

### RAG Components

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| Vector Store | ChromaDB | Embedding storage |
| Embeddings | SentenceTransformers | Vectorization |
| Loaders | Unstructured | Document parsing |
| Chunking | LangChain | Text segmentation |

---

## Performance

- **Execution Speed**: 45 simple tasks/min, 32 multi-agent tasks/min
- **Cost Tracking**: <1% error rate across all providers
- **Memory Overhead**: Low with FastAPI async architecture
- **Scalability**: Horizontal scaling with Kubernetes

### Real-time Cost Breakdown

```json
{
  "research_agent": {"total_cost": 2.34, "calls": 145},
  "summary_agent": {"total_cost": 0.87, "calls": 89},
  "assistant_agent": {"total_cost": 5.12, "calls": 312}
}
```

---

## Development Roadmap

### ✅ v2.0 (Current - October 2025)
Complete enterprise feature set: streaming, security, orchestration, MCP support

### 🚧 v2.1 (Q1 2026)
- Visual workflow designer
- Enterprise SSO (SAML, OAuth2)
- Analytics dashboard
- SOC2/HIPAA certifications

### 📋 v2.2 (Q2 2026)
- Distributed execution
- Auto-scaling infrastructure
- Multi-cloud deployment

---

## Community & Support

- **📖 Documentation**: [fluxgraph.readthedocs.io](https://fluxgraph.readthedocs.io)
- **💬 Discord**: [Join Community](https://discord.gg/Z9bAqjYvPc)
- **🐛 Issues**: [GitHub Issues](https://github.com/ihtesham-jahangir/fluxgraph/issues)
- **💡 Discussions**: [GitHub Discussions](https://github.com/ihtesham-jahangir/fluxgraph/discussions)
- **📧 Enterprise**: enterprise@fluxgraph.com

### Enterprise Support

For companies requiring custom integrations, dedicated SLA, security audits, compliance assistance, or training:

**Contact**: enterprise@fluxgraph.com

---

## Contributing

```bash
git clone https://github.com/ihtesham-jahangir/fluxgraph.git
cd fluxgraph
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

**Contribution areas:**
- Core features and LLM integrations
- Security enhancements
- Documentation and tutorials
- Testing and benchmarks

---

## License

MIT License - Free and open-source forever. No vendor lock-in.

See [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p><strong>FluxGraph 2.0</strong></p>
  <p>The most complete open-source AI agent framework for production</p>
  <p>Enterprise features • Zero vendor lock-in • Built for scale</p>
  <br/>
  <p><em>Star us on GitHub if FluxGraph powers your AI systems!</em></p>
</div>