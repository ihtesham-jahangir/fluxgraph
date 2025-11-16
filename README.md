<div align="center">

<img src="https://github.com/ihtesham-jahangir/fluxgraph/blob/main/logo.jpeg" alt="FluxGraph" width="180" height="180"/>

# FluxGraph

### Production AI That Actually Ships 🚀

**The only open-source agent framework built for teams who can't afford downtime**

[![PyPI](https://img.shields.io/pypi/v/fluxgraph?color=6366f1&style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/fluxgraph/)
[![Python](https://img.shields.io/pypi/pyversions/fluxgraph?style=flat-square&logo=python&logoColor=white&color=3776ab)](https://pypi.org/project/fluxgraph/)
[![License](https://img.shields.io/github/license/ihtesham-jahangir/fluxgraph?style=flat-square&color=10b981)](https://github.com/ihtesham-jahangir/fluxgraph/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1243184424318402592?logo=discord&label=Community&style=flat-square&color=5865F2)](https://discord.gg/Z9bAqjYvPc)
[![Docs](https://img.shields.io/badge/docs-ready-10b981?style=flat-square&logo=read-the-docs&logoColor=white)](https://fluxgraph.readthedocs.io)

[![Stars](https://img.shields.io/github/stars/ihtesham-jahangir/fluxgraph?style=social)](https://github.com/ihtesham-jahangir/fluxgraph)
[![Downloads](https://static.pepy.tech/badge/fluxgraph/month)](https://pepy.tech/project/fluxgraph)

[**📖 Documentation**](https://fluxgraph.readthedocs.io) • [**💬 Discord**](https://discord.gg/Z9bAqjYvPc) • [**🎯 Examples**](https://github.com/ihtesham-jahangir/fluxgraph/tree/main/examples) • [**🚀 Quick Start**](#-60-second-start)

</div>

---

## 🎯 Why FluxGraph?

> **Stop duct-taping agents together. Start shipping production AI.**

Most frameworks give you toy examples. FluxGraph gives you **enterprise infrastructure** — the boring, critical stuff that takes months to build yourself.

```python
# Other frameworks: Hope your workflow doesn't crash
workflow.run(data)  # 💥 Server reboot = start over

# FluxGraph: Automatic crash recovery
result = await workflow.execute(
    workflow_id="user_123",  # Crashes? Resume from last step ✅
    initial_data=data
)
```

<div align="center">

### **The FluxGraph Difference**

</div>

<table align="center">
<thead>
<tr>
<th width="25%">What You Need</th>
<th width="25%">❌ LangGraph</th>
<th width="25%">❌ CrewAI</th>
<th width="25%">✅ FluxGraph</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Survive server crashes</strong></td>
<td>Manual checkpointing</td>
<td>Start over</td>
<td><strong>Built-in (Postgres)</strong></td>
</tr>
<tr>
<td><strong>Know what went wrong</strong></td>
<td>Pay for LangSmith</td>
<td>Print debugging</td>
<td><strong>Database event logs</strong></td>
</tr>
<tr>
<td><strong>Prove compliance</strong></td>
<td>Roll your own</td>
<td>N/A</td>
<td><strong>Hash-chained audit trail</strong></td>
</tr>
<tr>
<td><strong>RAG that scales</strong></td>
<td>DIY pipelines</td>
<td>N/A</td>
<td><strong>Data-source agnostic</strong></td>
</tr>
<tr>
<td><strong>Stop paying for repeats</strong></td>
<td>N/A</td>
<td>N/A</td>
<td><strong>Semantic caching</strong></td>
</tr>
<tr>
<td><strong>Accurate cost tracking</strong></td>
<td>N/A</td>
<td>N/A</td>
<td><strong>Per-agent metrics</strong></td>
</tr>
<tr>
<td><strong>Block data leaks</strong></td>
<td>N/A</td>
<td>N/A</td>
<td><strong>9-type PII detection</strong></td>
</tr>
</tbody>
</table>

---

## ⚡ 60-Second Start

```bash
# Full power mode
pip install fluxgraph[all]

# Minimal install
pip install fluxgraph
```

### Your First Resilient Workflow

```python
import os
from fluxgraph.core.app import FluxApp
from fluxgraph.core.workflow_graph import WorkflowBuilder

# 1️⃣ Initialize with Postgres = instant persistence
app = FluxApp(database_url=os.getenv("DATABASE_URL"))

# 2️⃣ Define workflow (it's just Python)
def needs_revision(state):
    return "retry" if state.get("confidence", 0) < 0.8 else "done"

workflow = (
    WorkflowBuilder("ai_writer", checkpointer=app.checkpointer)
    .add_agent("writer", writer_agent)
    .add_agent("editor", editor_agent)
    .connect("writer", "editor")
    .branch("editor", needs_revision, {
        "retry": "writer",
        "done": "__end__"
    })
    .start_from("writer")
    .build()
)

# 3️⃣ Execute (crash-proof with workflow_id)
result = await workflow.execute(
    workflow_id="blog_post_456",
    initial_data={"topic": "AI in 2025"}
)
# Server crashes? Run again with same workflow_id → resumes automatically ✨
```

**That's it.** No manual state management. No lost work. No debugging nightmares.

---

## 🔥 Core Superpowers

### 1️⃣ **Never Lose Progress** — Resumable Workflows

<table>
<tr>
<td width="50%">

**Without FluxGraph**
```python
# Server crashes...
# All progress lost 💸
# Start from scratch 😢
workflow.run(data)
```

</td>
<td width="50%">

**With FluxGraph**
```python
# Server crashes...
# Resume from last step ✅
await workflow.execute(
    workflow_id="unique_123",
    initial_data=data
)
```

</td>
</tr>
</table>

**How it works:** Every step auto-saves to Postgres. On crash, FluxGraph detects the incomplete workflow and picks up exactly where it left off.

---

### 2️⃣ **Stop Debugging in the Dark** — Deep Observability

```python
# Every step, every state change, every error → structured database log
app = FluxApp(
    database_url=os.getenv("DATABASE_URL"),
    enable_workflows=True  # Automatic step logging
)

# Query your workflow history like data
logs = await app.workflow_logger.get_execution_logs(workflow_id="user_123")
```

**What you get:**
- 📊 Step-by-step execution timeline
- 🔍 Full state snapshots at each stage
- ⚠️ Error traces with context
- 📈 Performance metrics per agent

**Coming v3.3:** Visual monitoring dashboard (database foundation done in v3.2)

---

### 3️⃣ **Compliance Without the Pain** — Verifiable Audit Logs

```python
from fluxgraph.security.audit_logger import VerifiableAuditLogger, AuditEventType

@app.agent()
async def financial_agent(
    user_id: str,
    transaction: dict,
    audit_logger: VerifiableAuditLogger  # Auto-injected
) -> dict:
    # Log BEFORE executing (tamper-proof)
    log_hash = await audit_logger.log(
        action=AuditEventType.DATA_MODIFICATION,
        actor=user_id,
        data={"action": "wire_transfer", "amount": transaction["amount"]}
    )
    
    # ... execute transaction ...
    
    return {"status": "complete", "audit_hash": log_hash}
```

**The secret sauce:** Hash-chained entries. Each log cryptographically references the previous one. Change anything = entire chain breaks = instant detection.

```python
# Verify integrity anytime
report = await app.audit_logger.verify_log_chain()
# ✅ Valid | ❌ Tampered (with exact break point)
```

**Perfect for:** HIPAA, SOC2, financial services, or anywhere "prove it" matters.

---

### 4️⃣ **RAG That Actually Works** — Universal Data Pipeline

```python
from fluxgraph.core.rag import UniversalRAG

@app.agent()
async def knowledge_agent(query: str, rag: UniversalRAG) -> dict:
    # Ingest from ANYWHERE (not just file uploads)
    await rag.ingest_from_urls([
        "https://docs.yourproduct.com/api",
        "https://blog.yourproduct.com/latest"
    ], metadata={"source": "docs", "version": "2.0"})
    
    # Query with semantic search
    context = await rag.query(query, top_k=5)
    return {"response": "...", "sources": context}
```

**Why it's different:**
- 🔌 **Data-source agnostic**: Files, URLs, APIs, databases
- 🚀 **True pipeline**: Ingest → Embed → Store → Query
- 🎯 **Smart retrieval**: Hybrid search (keyword + semantic)

---

## 🛡️ Enterprise Security (Built-In)

```python
app = FluxApp(
    database_url=os.getenv("DATABASE_URL"),
    enable_security=True,        # ✅ PII detection + redaction
    enable_audit_logging=True,   # ✅ Verifiable audit trail
    enable_pii_detection=True    # ✅ Scan for 9 PII types
)
```

### What You Get

| Feature | What It Does | Use Case |
|---------|-------------|----------|
| **PII Detection** | Auto-scan for EMAIL, PHONE, SSN, CREDIT_CARD, etc. | Healthcare, finance, HR |
| **Prompt Injection Shield** | Block 7 common attacks (ROLE_PLAY, IGNORE_PREVIOUS) | User-facing agents |
| **RBAC** | Role-based access control | Multi-tenant apps |
| **Verifiable Audits** | Cryptographic proof of log integrity | Compliance, legal |

---

## 🎛️ Human-in-the-Loop (HITL)

Critical decision? Pause and ask a human.

```python
@app.agent()
async def review_agent(action: str, hitl) -> dict:  # `hitl` auto-injected
    # Request approval (non-blocking)
    approval = await hitl.request_approval(
        task_description=f"Execute high-risk action: {action}",
        risk_level="HIGH",
        timeout_seconds=300  # 5 min deadline
    )
    
    if await approval.wait_for_approval():
        return {"status": "APPROVED", "executed": True}
    
    return {"status": "REJECTED", "executed": False}
```

**Perfect for:** Content moderation, financial transactions, legal reviews.

---

## 🚀 Production-Ready Out of the Box

### Docker Compose (Recommended)

```yaml
version: '3.8'
services:
  fluxgraph:
    image: fluxgraph:3.2.0
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/fluxgraph
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

```bash
docker-compose up -d
# 🎉 Production-grade AI in 30 seconds
```

### Kubernetes (Scale to Infinity)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fluxgraph
spec:
  replicas: 3  # Auto-scaling ready
  template:
    spec:
      containers:
      - name: fluxgraph
        image: fluxgraph:3.2.0
        env:
        - name: DATABASE_URL
          value: postgresql://user:pass@postgres-svc:5432/fluxgraph
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

---

## 🧩 Pick Your Power Level

```bash
# 🎯 Minimal (just agents)
pip install fluxgraph

# ⚡ Core features (workflows, memory, caching)
pip install fluxgraph[p0,chains]

# 🏭 Production stack
pip install fluxgraph[production,orchestration,security,postgres]

# 🌐 RAG-powered
pip install fluxgraph[rag,postgres]

# 🚀 Everything (recommended)
pip install fluxgraph[all]
```

### Available Extras

| Extra | What You Get |
|-------|-------------|
| `p0` | Graph workflows, advanced memory, semantic caching |
| `production` | Streaming, sessions, retry logic |
| `security` | RBAC, audit logs, PII detection, prompt shield |
| `orchestration` | Agent handoffs, HITL, batch processing |
| `rag` | ChromaDB, embeddings, document processing |
| `postgres` | Persistence, checkpointing, workflow logs |
| `chains` | LCEL-style chaining, distributed tracing |
| `all` | Everything (batteries included) |

---

## 🌐 API Endpoints (REST-Ready)

| Endpoint | Method | What It Does |
|----------|--------|--------------|
| `/ask/{agent}` | `POST` | Execute agent |
| `/stream/{agent}` | `GET` | Stream response (SSE) |
| `/workflows` | `GET` | List all workflows |
| `/workflows/{name}/execute` | `POST` | Run workflow (pass `workflow_id`) |
| `/rag/ingest_urls` | `POST` | Add knowledge from URLs |
| `/audit/verify` | `GET` | Check audit log integrity |
| `/health` | `GET` | System status |
| `/tracing/traces` | `GET` | Distributed traces |

**Example:**
```bash
curl -X POST http://localhost:8000/workflows/research/execute \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "user_request_789",
    "initial_data": {"query": "AI trends 2025"}
  }'
```

---

## 🤖 Supported LLM Providers

| Provider | Models | Streaming | Cost Tracking | Multimodal |
|----------|--------|-----------|---------------|------------|
| **OpenAI** | GPT-3.5, GPT-4, GPT-4 Turbo | ✅ | ✅ | ✅ |
| **Anthropic** | Claude 3 (Haiku, Sonnet, Opus) | ✅ | ✅ | ✅ |
| **Google** | Gemini Pro, Ultra | ✅ | ✅ | ✅ |
| **Groq** | Mixtral, Llama 3 | ✅ | ✅ | ❌ |
| **Ollama** | All local models | ✅ | ✅ | ❌ |
| **Azure OpenAI** | GPT models | ✅ | ✅ | ✅ |

**Switch providers without changing code:**
```python
app = FluxApp(llm_provider="anthropic")  # That's it
```

---

## 🗺️ Roadmap (We Ship Fast)

### ✅ v3.2 (Current — Q4 2025)
- ✅ Persistent conditional workflows
- ✅ Verifiable audit logs (hash-chained)
- ✅ Database-backed observability
- ✅ RAG-as-a-pipeline
- ✅ Intelligent hierarchical delegation
- ✅ Standardized multimodal API
- ✅ LCEL-style chains + distributed tracing

### 🚧 v3.3 (Q1 2026)
- 🎨 Visual workflow designer UI
- 📊 Observability & monitoring dashboard
- 🧠 Agent learning & auto-optimization

### 📋 v3.4 (Q2 2026)
- 🌍 Distributed agent execution (multi-host)
- 📈 Auto-scaling workflows
- 🔐 Enterprise SSO

---

## 🤝 Community & Support

<div align="center">

### Join 1000+ Developers Building with FluxGraph

[![Discord](https://img.shields.io/discord/1243184424318402592?logo=discord&label=Discord%20Community&style=for-the-badge&color=5865F2)](https://discord.gg/Z9bAqjYvPc)
[![GitHub Discussions](https://img.shields.io/github/discussions/ihtesham-jahangir/fluxgraph?style=for-the-badge&logo=github)](https://github.com/ihtesham-jahangir/fluxgraph/discussions)

</div>

**📖 Documentation:** [fluxgraph.readthedocs.io](https://fluxgraph.readthedocs.io)  
**💬 Discord:** [Join Community](https://discord.gg/Z9bAqjYvPc)  
**🐛 GitHub Issues:** [Report Bugs](https://github.com/ihtesham-jahangir/fluxgraph/issues)  
**💼 Enterprise:** enterprise@fluxgraph.com

---

## 🛠️ Contributing

We ❤️ contributions! Here's how to get started:

```bash
# Clone the repo
git clone https://github.com/ihtesham-jahangir/fluxgraph.git
cd fluxgraph

# Set up dev environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev,postgres]"

# Run tests
pytest tests/ -v

# Make changes, submit PR 🚀
```

**We need help with:**
- 📝 Documentation & tutorials
- 🧪 Test coverage
- 🔌 New integrations (vector DBs, LLM providers)
- 🐛 Bug reports (please include reproduction steps)

---

<div align="center">

## ⭐ Why Star FluxGraph?

**If you're tired of:**
- 🔥 Workflows that die on server restart
- 🕵️ Debugging without logs
- 🏗️ Building infrastructure instead of features
- 💸 Paying for proprietary monitoring tools

**Then FluxGraph is for you.**

<br>

### **Star us if you believe production AI shouldn't be this hard** ⭐

<br>

[![Star History](https://img.shields.io/github/stars/ihtesham-jahangir/fluxgraph?style=social)](https://github.com/ihtesham-jahangir/fluxgraph)

<br>

---

<sub>Built with 💙 by developers who've been burned by "toy" frameworks</sub>

**[⬆ Back to Top](#fluxgraph)**

</div>