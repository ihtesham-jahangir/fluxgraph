Welcome to FluxGraph Documentation
====================================

**FluxGraph** is an enterprise-grade AI agent orchestration framework for building intelligent, multi-agent systems with advanced memory, caching, and workflow capabilities.

.. image:: https://img.shields.io/badge/version-3.0.0-blue
   :alt: Version 3.0.0

.. image:: https://img.shields.io/badge/license-MIT-green
   :alt: License MIT

Quick Start
-----------

Install FluxGraph:

.. code-block:: bash

   pip install fluxgraph

Create your first agent:

.. code-block:: python

   from fluxgraph import FluxApp

   app = FluxApp(title="My AI Agent", version="1.0.0")

   @app.agent(name="greeter")
   async def greeter(message: str):
       return {"response": f"Hello! You said: {message}"}

   if __name__ == "__main__":
       app.run()

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   architecture
   agents
   memory
   workflows
   llm_providers
   api_reference
   examples
   deployment


Features
--------

🚀 **Multi-Agent Orchestration**
   - Build complex agent workflows
   - Dynamic agent chaining
   - Parallel execution support

🧠 **Advanced Memory System**
   - Episodic and semantic memory
   - PostgreSQL-backed persistence
   - Context-aware recall

⚡ **High Performance**
   - Built-in caching layer
   - Async-first architecture
   - Real-time analytics dashboard

🔌 **LLM Provider Support**
   - OpenAI, Anthropic, Google Gemini
   - Groq, Ollama, HuggingFace
   - Custom provider integration

🛠️ **Production Ready**
   - CORS support
   - Security middleware
   - Performance monitoring
   - Error handling & logging


Architecture Overview
---------------------

FluxGraph follows a modular architecture:

.. code-block:: text

   ┌─────────────────────────────────────┐
   │         FastAPI Application         │
   ├─────────────────────────────────────┤
   │      Agent Orchestrator Layer       │
   ├──────────┬──────────┬───────────────┤
   │  Memory  │  Cache   │   Workflows   │
   ├──────────┴──────────┴───────────────┤
   │        LLM Provider Adapters        │
   ├─────────────────────────────────────┤
   │      Database & External APIs       │
   └─────────────────────────────────────┘


Use Cases
---------

✅ **Customer Support Agents**
   Intelligent chatbots with context retention and escalation handling

✅ **Lead Generation Systems**
   Automated sales qualification and CRM integration

✅ **Data Processing Pipelines**
   Multi-step workflows with AI-powered decision making

✅ **Knowledge Management**
   RAG-enabled question answering systems


Community
---------

- **GitHub**: https://github.com/yourusername/fluxgraph
- **Issues**: https://github.com/yourusername/fluxgraph/issues
- **Discord**: https://discord.gg/fluxgraph


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
