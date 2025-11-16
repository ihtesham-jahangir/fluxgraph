<div align="center">
<img src="https://github.com/ihtesham-jahangir/fluxgraph/blob/main/logo.jpeg" alt="FluxGraph Logo" width="200" height="200"/>
</div>

<h1 align="center">FluxGraph</h1>

<p align="center"><strong>Production-Grade AI Agent Orchestration Framework</strong></p>
<p align="center">Build scalable, resilient, and observable multi-agent systems with enterprise-grade security.</p>

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
<a href="https://fluxgraph.readthedocs.io">
<img src="https://img.shields.io/badge/docs-available-brightgreen?style=for-the-badge&logo=read-the-docs&logoColor=white" alt="Documentation"/>
</a>
<a href="https://discord.gg/Z9bAqjYvPc">
<img src="https://img.shields.io/discord/1243184424318402592?logo=discord&label=Discord&style=for-the-badge&color=5865F2" alt="Discord"/>
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

<div align="center">
<h3>📊 Download Statistics</h3>
<table>
<tr>
<td align="center"><b>All-Time Downloads</b>



<a href="https://pepy.tech/project/fluxgraph"><img src="https://static.pepy.tech/badge/fluxgraph" alt="Total"/></a></td>
<td align="center"><b>Monthly Downloads</b>



<a href="https://pepy.tech/project/fluxgraph"><img src="https://static.pepy.tech/badge/fluxgraph/month" alt="Monthly"/></a></td>
<td align="center"><b>Weekly Downloads</b>



<a href="https://pepy.tech/project/fluxgraph"><img src="https://static.pepy.tech/badge/fluxgraph/week" alt="Weekly"/></a></td>
</tr>
</table>
</div>

Overview

FluxGraph is the most complete open-source AI agent framework for production deployment. Built for developers who need sophisticated, multi-agent systems that are resilient, observable, and secure without complexity or vendor lock-in.

While other frameworks provide the building blocks, FluxGraph provides the production-ready infrastructure out-of-the-box.

New Production-Grade Features

FluxGraph is built on a foundation of resilience and observability, powered by your PostgreSQL database.

🔀 Persistent, Resumable Workflows (P0): Never lose progress. If your server crashes, workflows automatically resume from the last completed step.

🔍 Deep Observability (P1): Stop debugging in the dark. Every step, state change, and error in your workflows is written to a structured database log, providing the foundation for our monitoring UI.

🛡️ Verifiable Audit Logs (P2): Go beyond simple logging. We provide a high-performance, hash-chained audit trail in Postgres. This proves log integrity for compliance (HIPAA, SOC2) without the overhead of a blockchain.

🔌 RAG-as-a-Pipeline (P2): Our RAG system is a true data pipeline. Ingest data from any source—local files, web URLs, or APIs—directly into your vector store.

Why FluxGraph?

Feature

FluxGraph 3.2

LangGraph

CrewAI

AutoGen

Graph Workflows

✅ Full Conditional

✅ Core

❌

❌

Resumable Workflows

✅ Built-in (Postgres)

⚠️ Manual Setup

❌

❌

Workflow Observability

✅ DB Event Log

⚠️ LangSmith

❌

❌

Verifiable Audit Log

✅ Hash-Chained (SQL)

❌

❌

❌

RAG Pipeline

✅ Data-Source Agnostic

⚠️ Manual Setup

❌

⚠️ Basic

Semantic Caching

✅ Built-in

❌

❌

❌

Cost Tracking

✅ Reliable & Accurate

❌

❌

❌

PII Detection

✅ 9 Types

❌

❌

❌

Streaming

✅ SSE & Optimized

⚠️ Callbacks

❌

❌

Multimodal Ready

✅ Standardized API

❌

❌

❌

Architecture

<div align="center">
<img src="fluxgraph-architecture.png" alt="FluxGraph Architecture" width="100%"/>
</div>

System Overview

┌──────────────────────────────────────────────────────────────────┐
│                         FluxGraph v3.2                           │
│     (Powered by PostgreSQL for State, Logging, & Audits)         │
├──────────────────────────────────────────────────────────────────┤
│ 🔀 Workflow Engine (Resumable) │ ⚡ Semantic Cache │ 🧠 Hybrid Memory │
├──────────────────────────────────────────────────────────────────┤
│                     Advanced Orchestrator                        │
│   Circuit Breakers • Cost Tracking (Accurate) • Smart Routing    │
├──────────────────────────────────────────────────────────────────┤
│ 🔒 Security Layer (PII, Injection, RBAC, Verifiable Audit Log)   │
├──────────────────────────────────────────────────────────────────┤
│ Agent Registry │ Tool Registry │ 🔌 RAG Pipeline (Data-Source Agnostic) │
└──────────────────────────────────────────────────────────────────┘


Installation

Quick Start

# Full installation with all v3.2 features
pip install fluxgraph[all]

# Minimal installation
pip install fluxgraph


Feature-Specific

# v3.2 core features (optimized for chains/workflows)
pip install fluxgraph[chains,p0]

# Production stack (Postgres, Orchestration, Security)
pip install fluxgraph[production,orchestration,security,postgres]

# Everything
pip install fluxgraph[all]


Available Extras

p0 - Graph workflows, advanced memory, semantic caching

production - Streaming, sessions, retry logic

security - RBAC, Verifiable Audit Log, PII detection, Prompt Shield

orchestration - Handoffs, HITL, batch processing

rag - ChromaDB, embeddings, document processing

postgres - PostgreSQL persistence, Workflow Checkpointing & Logging

chains - LCEL-style chaining, Tracing

full - All production features

all - Everything including dev tools

Core Concept Examples

1. Resumable & Observable Workflows (P0/P1)

Define your workflow once, and let FluxGraph handle persistence. Pass your app's checkpointer and workflow_logger to the WorkflowBuilder.

import os
from fluxgraph.core.app import FluxApp
from fluxgraph.core.workflow_graph import WorkflowBuilder

# 1. Init app with a database URL to enable persistence
app = FluxApp(database_url=os.getenv("DATABASE_URL"))

# Define routing condition
def quality_check(state):
    confidence = state.get("analyzer_result", {}).get("confidence", 0) 
    return "retry" if confidence < 0.8 else "complete"

# 2. Pass the app's checkpointer and logger to the builder
workflow = (WorkflowBuilder(
        "research_workflow",
        checkpointer=app.checkpointer, 
        logger=app.workflow_logger
    )
    .add_agent("researcher", research_agent)
    .add_agent("analyzer", analysis_agent)
    .connect("researcher", "analyzer")
    .branch("analyzer", quality_check, {
        "retry": "researcher",
        "complete": "__end__"
    })
    .start_from("researcher")
    .build())

# 3. Execute with a unique ID
# If this script crashes, run it again with the same workflow_id
# and FluxGraph will resume from the last completed step.
result = await workflow.execute(
    workflow_id="user_request_123", 
    initial_data={"query": "AI trends 2025"}
)


2. RAG-as-a-Pipeline (P2)

Ingest data from any source—not just files—and query it seamlessly.

from fluxgraph.core.rag import UniversalRAG

@app.agent()
async def knowledge_agent(query: str, rag: UniversalRAG) -> dict:
    
    # 1. Ingest data from any source (e.g., web URLs)
    await rag.ingest_from_urls(
        urls=["[https://blog.fluxgraph.ai/new-features-v3-2](https://blog.fluxgraph.ai/new-features-v3-2)"],
        metadata={"source": "blog", "topic": "v3.2"}
    )
    
    # 2. Query the vector store
    # The RAG connector is automatically injected into the agent
    context = await rag.query(query, top_k=3)
    
    return {"response": "Query complete", "context": context}

# You can also expose this as an API endpoint:
# POST /rag/ingest_urls
# Body: {"urls": ["..."]}


3. Verifiable Auditing (P2)

Create a tamper-evident, hash-chained log for critical events, essential for compliance.

from fluxgraph.security.audit_logger import VerifiableAuditLogger, AuditEventType

@app.agent()
async def finance_agent(
    user_id: str, 
    trade_details: dict,
    audit_logger: VerifiableAuditLogger # Injected by the app
) -> dict:
    
    # Log the critical, verifiable event before execution
    log_hash = await audit_logger.log(
        action=AuditEventType.DATA_MODIFICATION,
        actor=user_id,
        data={"action": "execute_trade", "details": trade_details}
    )
    
    # ... execute the trade logic ...
    
    return {"status": "trade_executed", "audit_hash": log_hash}

# You can verify the entire log integrity at any time
# integrity_report = await app.audit_logger.verify_log_chain()


Enterprise Features

Production Configuration

Enable all production features by providing a database_url.

import os
from fluxgraph.core.app import FluxApp

app = FluxApp(
    # Enable persistence for workflows, logs, and audits
    database_url=os.getenv("DATABASE_URL"),
    
    # Enable v3.0 features
    enable_workflows=True,
    enable_advanced_memory=True,
    enable_agent_cache=True,
    
    # Enable security
    enable_security=True,
    enable_audit_logging=True,
    enable_pii_detection=True,
    
    # Enable orchestration
    enable_orchestration=True
)


Verifiable Security

FluxGraph's security is built for enterprise compliance.

Verifiable Audit Logging (Hash-Chained):
We provide a high-performance, hash-chained log in PostgreSQL. Each entry hashes the previous, creating a tamper-evident chain. Any modification breaks the chain, which can be instantly detected by running app.audit_logger.verify_log_chain(). This provides the integrity of a blockchain with the performance of a SQL database.

PII Detection & Redaction: Automatically scan and redact 9 types of PII (EMAIL, PHONE, SSN, etc.) from inputs.

Prompt Injection Shield: Protect against 7 common injection techniques (ROLE_PLAY, IGNORE_PREVIOUS, etc.).

Human-in-the-Loop

Pause a workflow and request human approval before executing critical tasks.

@app.agent()
async def critical_agent(action: str, hitl) -> dict: # `hitl` is injected
    
    # Request approval and wait
    approval = await hitl.request_approval(
        task_description=f"Execute: {action}",
        risk_level="HIGH",
        timeout_seconds=300
    )
    
    if await approval.wait_for_approval():
        return {"status": "EXECUTED"}
        
    return {"status": "REJECTED"}


Production Deployment

Docker Compose

A DATABASE_URL is now essential for production features.

version: '3.8'
services:
  fluxgraph:
    build: .
    ports:
      - "8000:8000"
    environment:
      # CRITICAL: Set database for persistence, logging, and audits
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
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=fluxgraph

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data


Kubernetes

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
        image: fluxgraph:3.2.0
        env:
        - name: DATABASE_URL
          value: "postgresql://user:pass@postgres-svc:5432/fluxgraph"
        - name: FLUXGRAPH_ENABLE_WORKFLOWS
          value: "true"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"


API Reference

Endpoint

Method

Description

/ask/{agent}

POST

Execute agent

/stream/{agent}

GET

Stream agent response

/workflows

GET

List all workflows

/workflows/{name}/execute

POST

Execute a workflow (pass workflow_id)

/rag/ingest_urls

POST

Ingest knowledge from a list of URLs

/health

GET

System health check

/tracing/traces

GET

Get all distributed traces

/audit/verify

GET

Run integrity check on the verifiable audit log

Supported Integrations

LLM Providers

Provider

Models

Streaming

Cost Tracking

Multimodal

OpenAI

GPT-3.5, GPT-4, GPT-4 Turbo

✅

✅

✅

Anthropic

Claude 3 (Haiku, Sonnet, Opus)

✅

✅

✅

Google

Gemini Pro, Ultra

✅

✅

✅

Groq

Mixtral, Llama 3

✅

✅

❌

Ollama

All local models

✅

✅

❌

Azure OpenAI

GPT models

✅

✅

✅

Backends

Backend

Use Case

Configuration

PostgreSQL

Production Persistence, Workflow State, Audit Logs

DATABASE_URL

Redis

Fast session storage, Caching

REDIS_URL

SQLite

Development/testing

Local file

In-Memory

Temporary stateless

None

Development Roadmap

✅ v3.2 (Current - Q4 2025)

Persistent Conditional Workflows (P0)

Verifiable Audit Logs (P1 - Hash-Chained)

Database-Backed Observability Log (P2)

RAG-as-a-Pipeline (P2)

Intelligent Hierarchical Delegation

Standardized Multimodal API (GPT-4V, Gemini)

LCEL-style Chains, Distributed Tracing

🚧 v3.3 (Q1 2026)

Visual Workflow Designer UI

Enhanced Observability & Monitoring Dashboard

Note: Backend DB foundation is complete in v3.2

Agent Learning & Auto-Optimization

📋 v3.4 (Q2 2026)

Distributed Agent Execution (Multi-Host)

Auto-scaling Workflows

Enterprise SSO (Single Sign-On)

Community & Support

Documentation: fluxgraph.readthedocs.io

Discord: Join Community

GitHub: Issues | Discussions

Enterprise: enterprise@fluxgraph.com

Contributing

git clone [https://github.com/ihtesham-jahangir/fluxgraph.git](https://github.com/ihtesham-jahangir/fluxgraph.git)
cd fluxgraph
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,postgres]"
pytest tests/


We welcome contributions in core features, security, documentation, testing, and integrations.

<div align="center">
<p><strong>FluxGraph 3.2</strong></p>
<p>The Production-Grade AI Agent Framework</p>
<p>Resumable Workflows • Verifiable Audits • Deep Observability • RAG Pipelines</p>





<p><em>⭐ Star us on GitHub if FluxGraph powers your AI systems!</em></p>
<p><a href="https://github.com/ihtesham-jahangir/fluxgraph">GitHub</a> • <a href="https://fluxgraph.readthedocs.io">Docs</a> • <a href="https://discord.gg/Z9bAqjYvPc">Discord</a></p>
</div>