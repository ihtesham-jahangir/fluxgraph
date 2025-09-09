<p align="center">
  <img src="logo.png" alt="FluxGraph Logo" width="180"/>
</p>

<h1 align="center">FluxGraph</h1>

<p align="center">
  <i>Lightweight, modular framework for building agentic AI workflows with full control and flexibility.</i>
</p>

<p align="center">
  <a href="https://pypi.org/project/fluxgraph/"><img src="https://img.shields.io/pypi/v/fluxgraph?color=blue" alt="PyPI"/></a>
  <a href="#"><img src="https://img.shields.io/badge/docs-available-brightgreen" alt="Docs"/></a>
  <a href="#"><img src="https://img.shields.io/discord/123456789?logo=discord&label=Discord" alt="Discord"/></a>
  <a href="https://github.com/ihtesham-jahangir/fluxgraph"><img src="https://img.shields.io/badge/contributions-welcome-orange" alt="Contributing"/></a>
</p>

---

## ✨ Why FluxGraph?

FluxGraph integrates **FastAPI, Celery, Redis**, and supports **direct LLM API calls** with extensible tooling.  
It’s designed for developers who want **control + flexibility** without heavy abstractions.

### 🔑 Key Benefits
- 🚀 **2-3x faster development** – skip complex boilerplate  
- ⚡ **Flexible orchestration** – works with LangGraph, AutoGen, or custom FastAPI  
- 📈 **Scalable** – Redis + Celery for distributed workloads  
- 🔌 **Extensible tooling** – Python functions for DB queries, APIs, custom tasks  
- 🤖 **Multi-LLM ready** – OpenAI, Anthropic, Ollama, Local models  
- 🧠 **Persistence & Memory** – Redis, PostgreSQL, Vector DB  
- 🔄 **Feedback loop** – RLHF integration ready  

---

## 🏗️ Architecture

<p align="center">
  <img src="flux-architecture.png" alt="FluxGraph Architecture" width="600"/>
</p>

---

## 📦 Installation
```bash
git clone https://github.com/ihtesham-jahangir/fluxgraph.git
cd fluxgraph
pip install -r requirements.txt
```

---

## ⚡ Quick Example

### Before: Manual orchestration
```python
def process_query(query):
    # lots of boilerplate here...
    return f"Processed: {query}"
```

### After: With FluxGraph
```python
from fluxgraph import Agent, Orchestrator

class MyAgent(Agent):
    def run(self, query: str):
        return f"Processed: {query}"

orchestrator = Orchestrator()
orchestrator.register("my_agent", MyAgent())

response = orchestrator.run("my_agent", "Hello FluxGraph!")
print(response)
```

---

## 📊 Roadmap
- [x] MVP with FastAPI + Orchestrator  
- [x] Tooling Layer (Python functions)  
- [ ] RLHF feedback integration  
- [ ] Auto-scaling with Docker + Kubernetes  
- [ ] GUI dashboard for monitoring  

---

## 🤝 Contributing
We welcome contributions! Please fork the repo, open issues, or submit PRs.  

---

## 📜 License
MIT License © 2025 FluxGraph Team
