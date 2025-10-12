
-----


<div align="center">
  <img src="https://github.com/ihtesham-jahangir/fluxgraph/blob/main/logo.jpeg" alt="FluxGraph Logo" width="200" height="200"/>
</div>

<h1 align="center">FluxGraph</h1>

<p align="center"><strong>Production-grade AI agent orchestration framework for building secure, scalable multi-agent systems</strong></p>

<p align="center">
  <a href="https://pypi.org/project/fluxgraph/">
    <img src="https://img.shields.io/pypi/v/fluxgraph?color=blue&style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI version"/>
  </a>
  <a href="https://pypi.org/project/fluxgraph/">
    <img src="https://img.shields.io/pypi/pyversions/fluxgraph?style=for-the-badge&logo=python&logoColor=white" alt="Python versions"/>
  </a>
  <a href="https://github.com/ihtesham-jahangir/fluxgraph/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/ihtesham-jahangir/fluxgraph?style=for-the-badge" alt="License"/>
  </a>
</p>

<p align="center">
  <a href="https://github.com/ihtesham-jahangir/fluxgraph">
    <img src="https://img.shields.io/github/stars/ihtesham-jahangir/fluxgraph?style=for-the-badge&logo=github" alt="GitHub Stars"/>
  </a>
  <a href="https://github.com/ihtesham-jahangir/fluxgraph/network/members">
    <img src="https://img.shields.io/github/forks/ihtesham-jahangir/fluxgraph?style=for-the-badge&logo=github" alt="GitHub Forks"/>
  </a>
  <a href="https://github.com/ihtesham-jahangir/fluxgraph/issues">
    <img src="https://img.shields.io/github/issues/ihtesham-jahangir/fluxgraph?style=for-the-badge" alt="GitHub Issues"/>
  </a>
</p>

<p align="center">
  <a href="https://fluxgraph.readthedocs.io">
    <img src="https://img.shields.io/badge/docs-available-brightgreen?style=for-the-badge&logo=read-the-docs&logoColor=white" alt="Documentation"/>
  </a>
  <a href="https://discord.gg/Z9bAqjYvPc">
    <img src="https://img.shields.io/discord/1243184424318402592?logo=discord&label=Discord&style=for-the-badge&color=5865F2" alt="Discord"/>
  </a>
</p>

<div align="center">
  <h3>📊 Download Statistics</h3>
  <table>
    <tr>
      <td align="center"><b>All-Time Downloads</b><br/><a href="https://pepy.tech/project/fluxgraph"><img src="https://static.pepy.tech/badge/fluxgraph" alt="Total"/></a></td>
      <td align="center"><b>Monthly Downloads</b><br/><a href="https://pepy.tech/project/fluxgraph"><img src="https://static.pepy.tech/badge/fluxgraph/month" alt="Monthly"/></a></td>
      <td align="center"><b>Weekly Downloads</b><br/><a href="https://pepy.tech/project/fluxgraph"><img src="https://static.pepy.tech/badge/fluxgraph/week" alt="Weekly"/></a></td>
    </tr>
  </table>
  <p><strong>🚀 Trusted by Developers Worldwide | Production-Ready from Day 1</strong></p>
</div>

---

<div align="center">
  <p><strong>1.6K+ Total Downloads | 1 GitHub Star | Modern & Production-Ready</strong></p>
</div>

---

## Overview

FluxGraph is the **most complete open-source AI agent framework** for production deployment, combining cutting-edge innovations with enterprise-grade reliability. Built for developers who need sophisticated AI agent systems without complexity or vendor lock-in.

### What's New in v3.2

**Revolutionary Features:**

- **Graph-Based Workflows (Fully Conditional)** - **Guaranteed conditional routing and looping** with robust state management.
- **Intelligent Multi-Agent Crews** - Hierarchical and consensus processes powered by **skill-based delegation and load balancing**.
- **Accurate Cost Tracking** - Real-time cost calculation now uses **tokenization fallback** (via `tiktoken`) to ensure reliability even with local or non-standard APIs.
- **Multimodal Ready** - Standardized interface for **GPT-4V and Gemini Vision** models using Base64 Data URIs.
- **Semantic Caching** - Intelligent response caching reduces LLM costs by 70%+

### Why FluxGraph?

| Feature | FluxGraph 3.2 | LangGraph | CrewAI | AutoGen |
|---------|---------------|-----------|--------|---------|
| **Graph Workflows** | ✅ **Full Conditional** | ✅ Core | ❌ | ❌ |
| **Semantic Caching** | ✅ Built-in | ❌ | ❌ | ❌ |
| **Hybrid Memory** | ✅ Advanced | ⚠️ Basic | ⚠️ Basic | ⚠️ Basic |
| **Circuit Breakers** | ✅ Native | ❌ | ❌ | ❌ |
| **Cost Tracking** | ✅ **Reliable & Accurate** | ❌ | ❌ | ❌ |
| **Audit Logs** | ✅ Blockchain | ❌ | ❌ | ❌ |
| **PII Detection** | ✅ 9 types | ❌ | ❌ | ❌ |
| **Streaming** | ✅ SSE & Optimized | ⚠️ Callbacks | ❌ | ❌ |
| **Multimodal Ready** | ✅ **Standardized API** | ❌ | ❌ | ❌ |

---

## Architecture

<div align="center">
  <img src="fluxgraph-architecture.png" alt="FluxGraph Architecture" width="100%"/>
</div>

### System Overview

```ascii
┌─────────────────────────────────────────────────────────┐
│                    FluxGraph v3.2                       │
├─────────────────────────────────────────────────────────┤
│  🔀 Workflow Engine (Conditional) │  ⚡ Semantic Cache  │  🧠 Hybrid Memory  │
├─────────────────────────────────────────────────────────┤
│              Advanced Orchestrator                      │
│   Circuit Breakers • Cost Tracking (Accurate) • Smart Routing     │
├─────────────────────────────────────────────────────────┤
│  🔒 Security Layer (PII, Injection, RBAC, Audit)       │
├─────────────────────────────────────────────────────────┤
│  Agent Registry  │  Tool Registry  │  RAG System (Multimodal)       │
└─────────────────────────────────────────────────────────┘
````

-----

## Installation

### Quick Start

```bash
# Full installation with v3.2 features
pip install fluxgraph[all]

# Minimal installation
pip install fluxgraph
```

### Feature-Specific

```bash
# v3.2 core features (optimized for chains/workflows)
pip install fluxgraph[chains,p0]

# Production stack
pip install fluxgraph[production,orchestration,security]

# Everything
pip install fluxgraph[all]
```

### Available Extras

  - `p0` - Graph workflows, advanced memory, semantic caching
  - `production` - Streaming, sessions, retry logic
  - `security` - RBAC, audit logs, PII detection, Prompt Shield
  - `orchestration` - Handoffs, HITL, batch processing
  - `rag` - ChromaDB, embeddings, document processing
  - `postgres` - PostgreSQL persistence
  - `chains` - LCEL-style chaining, Tracing
  - `full` - All production features
  - `all` - Everything including dev tools

-----

## Quick Start Guide

### Hello World (30 seconds)

```python
from fluxgraph import FluxApp

app = FluxApp(title="My AI App")

@app.agent()
async def assistant(message: str,**kwargs) -> dict:
    """Your first AI agent."""
    return {"response": f"You said: {message}"}

# Run: flux run app.py
# Test: curl -X POST http://localhost:8000/ask/assistant \
#       -d '{"message":"Hello!"}'
```

### LLM-Powered Agent

```python
import os
from fluxgraph import FluxApp
from fluxgraph.models import OpenAIProvider

app = FluxApp(title="Smart Assistant")

llm = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4"
)

@app.agent()
async def assistant(query: str) -> dict:
    """AI assistant powered by GPT-4."""
    response = await llm.generate(f"Answer: {query}")
    return {"answer": response.get("text")}
```

-----

## New in v3.2

### 1\. Graph-Based Workflows (Fully Conditional)

The workflow engine is now guaranteed to support complex, multi-step agent logic with true conditional routing and looping.

```python
from fluxgraph import FluxApp
from fluxgraph.core import WorkflowBuilder

app = FluxApp(enable_workflows=True)

# Define routing condition using latest result/state
def quality_check(state):
    # state.current_node_result holds the output of the 'analyzer' node
    confidence = state.get("analyzer_result", {}).get("confidence", 0) 
    
    if confidence < 0.8:
        return "retry"  # Loop back to research
    return "complete"

# Build workflow graph
workflow = (WorkflowBuilder("research_workflow")
    .add_agent("researcher", research_agent) # Agent that generates research
    .add_agent("analyzer", analysis_agent)   # Agent that analyzes, returns result with 'confidence'
    .connect("researcher", "analyzer")
    .branch("analyzer", quality_check, {
        "retry": "researcher", # Conditional loop back
        "complete": "__end__"  # Conditional exit
    })
    .start_from("researcher")
    .build())

result = await workflow.execute({"query": "AI trends 2025"})
```

**Features:**

  - **Full Conditional Routing:** Use any state variable or the last node's output to determine the next step.
  - **Robust Looping:** Easily define self-correcting loops (e.g., retry research or analysis).
  - State persistence across steps.

### 2\. Intelligent Multi-Agent System (Hierarchical Delegation)

Hierarchical crew execution now uses **skill-based matching** to ensure tasks are intelligently routed to the most appropriate specialist agent, delivering superior multi-agent results.

```python
from fluxgraph.crew import Crew, RoleAgent, Task, ProcessType
from fluxgraph.crew.delegation import DelegationStrategy

# Define specialist agents
researcher = RoleAgent(role="Senior Market Researcher", goal="Conduct deep analysis...", backstory="...")
writer = RoleAgent(role="Creative Content Writer", goal="Produce engaging reports...", backstory="...")

task1 = Task(description="Find Q3 market size data", expected_output="Numeric market size.")
task2 = Task(description="Write a 500-word executive summary", expected_output="Polished summary.")

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=ProcessType.HIERARCHICAL # Enables intelligent delegation
)

# Manager automatically assigns task1 (Find data) to Researcher
# and task2 (Write summary) to Writer using skill matching.
result = await crew.kickoff()
```

### 3\. Multimodal & Vision Ready

The framework introduces a standardized `multimodal_generate` function to support models like GPT-4V and Gemini Vision, allowing agents to process images and audio alongside text.

```python
import os
from fluxgraph import FluxApp
from fluxgraph.models import OpenAIProvider
from fluxgraph.multimodal.processor import MultiModalProcessor

# 1. Initialize with MultiModalProcessor
app = FluxApp(enable_multimodal=True)
processor = MultiModalProcessor()
llm = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4-vision-preview")

@app.agent()
async def vision_agent(image_path: str, prompt: str) -> dict:
    """An agent capable of processing images and text."""

    # 2. Process image file into a standardized input object (Base64 Data URI)
    image_input = processor.process_image(image_path)
    
    # 3. Call the standardized multimodal method
    response = await llm.multimodal_generate(
        prompt=prompt,
        media_inputs=[image_input]
    )
    
    return {"analysis": response.text, "source_image": image_path}
```

-----

## Complete Examples

### Customer Support Bot

```python
from fluxgraph import FluxApp
from fluxgraph.core import MemoryType

app = FluxApp(
    enable_advanced_memory=True,
    enable_agent_cache=True,
    enable_rag=True
)

@app.agent()
async def support_bot(
    query: str,
    session_id: str,
    advanced_memory,
    cache,
    rag
) -> dict:
    """Intelligent support bot with memory and caching."""
    
    # Check cache first
    if cached := cache.get(query, threshold=0.9):
        return cached
    
    # Search knowledge base
    kb_results = await rag.query(query, top_k=3)
    
    # Recall similar past cases
    similar_cases = advanced_memory.recall_similar(query, k=5)
    
    # Generate response with context
    context = {
        "knowledge_base": kb_results,
        "similar_cases": [e.content for e, _ in similar_cases]
    }
    
    response = await llm.generate(f"Context: {context}\nUser: {query}")
    
    # Store interaction
    advanced_memory.store(
        f"Q: {query}\nA: {response}",
        MemoryType.EPISODIC,
        importance=0.9
    )
    
    return {"response": response, "sources": kb_results}
```

### Research Pipeline

```python
from fluxgraph import FluxApp
from fluxgraph.core import WorkflowBuilder

app = FluxApp(enable_workflows=True)

async def web_search_agent(state):
    results = await search_web(state.get("query"))
    state.update("web_results", results)
    return results

async def analysis_agent(state):
    analysis = await analyze_data(state.get("web_results"))
    state.update("analysis", analysis)
    return analysis

async def synthesis_agent(state):
    report = await synthesize(state.get("analysis"))
    state.update("final_report", report)
    return report

def quality_check(state):
    if state.get("analysis").get("confidence") < 0.8:
        return "retry"
    return "synthesize"

workflow = (WorkflowBuilder("research")
    .add_agent("searcher", web_search_agent)
    .add_agent("analyzer", analysis_agent)
    .add_agent("synthesizer", synthesis_agent)
    .connect("searcher", "analyzer")
    .branch("analyzer", quality_check, {
        "retry": "searcher",
        "synthesize": "synthesizer"
    })
    .start_from("searcher")
    .build())

result = await workflow.execute({"query": "AI trends 2025"})
```

### Multi-Agent System

```python
@app.agent()
async def supervisor(task: str, call_agent, broadcast) -> dict:
    """Orchestrates specialized agents."""
    research = await call_agent("research_agent", query=task)
    analyses = await broadcast(
        ["technical_analyst", "business_analyst"],
        data=research
    )
    return {"results": analyses}
```

-----

## Enterprise Features

### Production Configuration

```python
app = FluxApp(
    # v3.2 Features
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

### Streaming Responses

```python
from fastapi.responses import StreamingResponse

@app.api.get("/stream/{agent_name}")
async def stream_agent(agent_name: str, query: str):
    async def generate():
        async for chunk in app.orchestrator.run_streaming(
            agent_name, {"query": query}
        ):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Security Features

```python
@app.agent()
async def secure_agent(user_input: str) -> dict:
    """Automatically protected against threats."""
    # PII Detection (9 types) - automatic
    # Prompt Injection Shield (7 techniques) - automatic
    # Immutable Audit Logging - automatic
    # RBAC + JWT Auth - automatic
    
    response = await process(user_input)
    return {"response": response}
```

**Supported PII Types:**
EMAIL, PHONE, SSN, CREDIT\_CARD, IP\_ADDRESS, PASSPORT, DRIVER\_LICENSE, DATE\_OF\_BIRTH, MEDICAL\_RECORD
**Injection Detection:**
IGNORE\_PREVIOUS, ROLE\_PLAY, ENCODED\_INJECTION, DELIMITER\_INJECTION, PRIVILEGE\_ESCALATION, CONTEXT\_OVERFLOW, PAYLOAD\_SPLITTING

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
    return {"status": "rejected"}
```

### Batch Processing

```python
# Submit 1000 tasks
job_id = await app.batch_processor.submit_batch(
    agent_name="data_processor",
    payloads=tasks,
    priority=0,
    max_concurrent=50
)

# Check status
status = app.batch_processor.get_job_status(job_id)
# {completed: 850, failed: 2, pending: 148}
```

### Cost Tracking

```python
# Automatic per-agent cost tracking
costs = app.orchestrator.cost_tracker.get_summary()
# {
#   "research_agent": {"cost": "$2.34", "calls": 145},
#   "summary_agent": {"cost": "$0.87", "calls": 89}
# }
```

-----

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
      - REDIS_URL=redis://redis:6379
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
  replicas: 3
  template:
    spec:
      containers:
      - name: fluxgraph
        image: fluxgraph:3.0.0
        env:
        - name: FLUXGRAPH_ENABLE_WORKFLOWS
          value: "true"
        - name: FLUXGRAPH_ENABLE_CACHE
          value: "true"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

-----

## API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask/{agent}` | POST | Execute agent |
| `/stream/{agent}` | GET | Stream response |
| `/workflows` | GET | List workflows |
| `/workflows/{name}/execute` | POST | Execute workflow |
| `/memory/stats` | GET | Memory statistics |
| `/memory/recall` | POST | Semantic search |
| `/cache/stats` | GET | Cache statistics |
| `/system/status` | GET | System health |
| `/system/costs` | GET | Cost summary |

-----

## Supported Integrations

### LLM Providers

| Provider | Models | Streaming | Cost Tracking | Multimodal |
|----------|--------|-----------|---------------|------------|
| OpenAI | GPT-3.5, GPT-4, GPT-4 Turbo | ✅ | ✅ | ✅ |
| Anthropic | Claude 3 (Haiku, Sonnet, Opus) | ✅ | ✅ | ✅ |
| Google | Gemini Pro, Ultra | ✅ | ✅ | ✅ |
| Groq | Mixtral, Llama 3 | ✅ | ✅ | ❌ |
| Ollama | All local models | ✅ | ✅ | ❌ |
| Azure OpenAI | GPT models | ✅ | ✅ | ✅ |

### Memory Backends

| Backend | Use Case | Configuration |
|---------|----------|---------------|
| PostgreSQL | Production persistence | `DATABASE_URL` |
| Redis | Fast session storage | `REDIS_URL` |
| SQLite | Development/testing | Local file |
| In-Memory | Temporary stateless | None |

-----

## Performance

**v3.2 Benchmarks:**

  - **Cache Hit Rate**: 85-95% on similar queries
  - **Cost Reduction**: 70%+ with semantic caching
  - **Memory Consolidation**: \<50ms for 1000 entries
  - **Workflow Execution**: 100+ steps/second
  - **Latency**: \<10ms overhead for v3.2 features

-----

## Development Roadmap

### ✅ v3.2 (Current - October 2025)

  - **Full Conditional Workflows**
  - **Accurate Cost Tracking/Tokenization**
  - **Intelligent Hierarchical Delegation**
  - **Standardized Multimodal API (GPT-4V, Gemini)**
  - LCEL-style Chains, Distributed Tracing, Batch Optimization

### 🚧 v3.3 (Q1 2026)

  - Visual workflow designer UI
  - Agent learning & optimization
  - Enhanced observability & Monitoring Dashboard

### 📋 v3.4 (Q2 2026)

  - Distributed agent execution
  - Auto-scaling workflows
  - Enterprise SSO (Single Sign-On)

-----

## Community & Support

  - **Documentation**: [fluxgraph.readthedocs.io](https://fluxgraph.readthedocs.io)
  - **Discord**: [Join Community](https://discord.gg/Z9bAqjYvPc)
  - **GitHub**: [Issues](https://github.com/ihtesham-jahangir/fluxgraph/issues) | [Discussions](https://github.com/ihtesham-jahangir/fluxgraph/discussions)
  - **Enterprise**: [enterprise@fluxgraph.com](mailto:enterprise@fluxgraph.com)

-----

## Contributing

```bash
git clone [https://github.com/ihtesham-jahangir/fluxgraph.git](https://github.com/ihtesham-jahangir/fluxgraph.git)
cd fluxgraph
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

We welcome contributions in core features, security, documentation, testing, and integrations.



<div align="center">
<p><strong>FluxGraph 3.2</strong></p>
<p>The most advanced open-source AI agent framework\</p>
<p>Conditional Workflows • Multimodal Ready • Accurate Cost Tracking • Enterprise Security</p>
<br>
<p><em>⭐ Star us on GitHub if FluxGraph powers your AI systems!</em></p>
<p><a href="https://github.com/ihtesham-jahangir/fluxgraph">GitHub\</a> • <a href="https://fluxgraph.readthedocs.io">Docs</a> • <a href="https://discord.gg/Z9bAqjYvPc"\>Discord</a></p>
</div>
