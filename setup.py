# setup.py
from setuptools import setup, find_packages
import os
from pathlib import Path

# Get the directory containing setup.py
here = Path(__file__).parent.resolve()

# Read PyPI-specific README with absolute path
readme_path = here / "pypi_readme.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    # Fallback to README.md if pypi_readme.md doesn't exist
    with open(here / "README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

# Read requirements
requirements_path = here / "requirements.txt"
with open(requirements_path, "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Optional dependencies for different feature sets
extras_require = {
    # v3.0 P0 Features - Graph workflows, advanced memory, caching
    'p0': [
        'networkx>=3.1',
        'sentence-transformers>=2.2.2',
        'numpy>=1.24.0',
        'faiss-cpu>=1.7.4',
    ],
    
    # Core production features
    'production': [
        'fastapi>=0.104.0',
        'uvicorn[standard]>=0.24.0',
        'gunicorn>=21.0.0',
        'sse-starlette>=1.6.5',
        'pydantic>=2.5.0',
        'python-multipart>=0.0.6',
    ],
    
    # Security features
    'security': [
        'pyjwt>=2.8.0',
        'cryptography>=41.0.0',
        'python-jose[cryptography]>=3.3.0',
        'passlib[bcrypt]>=1.7.4',
    ],
    
    # Advanced orchestration
    'orchestration': [
        'celery>=5.3.4',
        'redis>=5.0.0',
    ],
    
    # RAG capabilities
    'rag': [
        'chromadb>=0.4.18',
        'langchain>=0.1.0',
        'sentence-transformers>=2.2.2',
        'pypdf>=3.17.0',
        'docx2txt>=0.8',
        'unstructured>=0.11.0',
    ],
    
    # Memory persistence
    'postgres': [
        'psycopg2-binary>=2.9.9',
        'sqlalchemy>=2.0.23',
    ],
    
    # Analytics
    'analytics': [
        'prometheus-client>=0.19.0',
        'plotly>=5.18.0',
        'pandas>=2.0.0',
    ],
    
    # v3.2 LangChain Parity Features
    'chains': [
        'langchain-core>=0.1.0',
        'pydantic>=2.5.0',
    ],
    
    'tracing': [
        'opentelemetry-api>=1.20.0',
        'opentelemetry-sdk>=1.20.0',
    ],
    
    # Development tools
    'dev': [
        'pytest>=7.4.3',
        'pytest-asyncio>=0.21.1',
        'pytest-cov>=4.1.0',
        'black>=23.12.0',
        'ruff>=0.1.0',
        'mypy>=1.7.1',
        'isort>=5.13.0',
        'pre-commit>=3.5.0',
    ],
    
    # Testing utilities
    'test': [
        'pytest>=7.4.3',
        'pytest-asyncio>=0.21.1',
        'pytest-cov>=4.1.0',
        'httpx>=0.25.0',
        'faker>=20.1.0',
    ],
}

# 'full' includes all production features (recommended)
extras_require['full'] = list(set(
    extras_require['p0'] +
    extras_require['production'] +
    extras_require['orchestration'] +
    extras_require['rag'] +
    extras_require['postgres'] +
    extras_require['analytics'] +
    extras_require['chains'] +
    extras_require['tracing']
))

# 'all' includes everything including security, dev, and test
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    name="fluxgraph",
    version="2.1.0",
    author="Ihtesham Jahangir",
    author_email="ceo@alphanetwork.com.pk",
    description="Production-grade AI agent orchestration framework with graph workflows, semantic caching, and hybrid memory. The complete alternative to LangGraph, CrewAI, and AutoGen.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ihtesham-jahangir/fluxgraph",
    project_urls={
        "Homepage": "https://github.com/ihtesham-jahangir/fluxgraph",
        "Documentation": "https://fluxgraph.readthedocs.io",
        "Source Code": "https://github.com/ihtesham-jahangir/fluxgraph",
        "Bug Tracker": "https://github.com/ihtesham-jahangir/fluxgraph/issues",
        "Changelog": "https://github.com/ihtesham-jahangir/fluxgraph/blob/main/CHANGELOG.md",
        "LinkedIn": "https://linkedin.com/in/ihtesham-jahangir",
    },
    packages=find_packages(where=".", exclude=["tests*", "docs*", "examples*"]),
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        'fluxgraph': [
            'py.typed',
            '*.json',
            '*.yaml',
            '*.yml',
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Framework :: FastAPI",
        "Framework :: AsyncIO",
        "Typing :: Typed",
        "Natural Language :: English",
        "Environment :: Web Environment",
        "Environment :: Console",
    ],
    keywords=[
        "ai", "agents", "llm", "gpt", "orchestration", "multi-agent", "autonomous-agents",
        "langchain", "langgraph", "autogen", "crewai", "semantic-kernel", "haystack",
        "openai", "anthropic", "claude", "gemini", "groq", "ollama", "gpt-4", "gpt-3.5",
        "fastapi", "async", "rag", "vector-database", "embeddings", "chromadb",
        "workflow", "workflow-graph", "semantic-caching", "hybrid-memory", "episodic-memory",
        "graph-workflows", "dag", "state-machine",
        "langchain-parity", "lcel", "langsmith", "distributed-tracing", "batch-processing",
        "streaming", "chains",
        "enterprise", "production", "production-ready", "security", "audit", "compliance",
        "rbac", "pii-detection", "prompt-injection", "prompt-shield",
        "handoff", "hitl", "human-in-the-loop", "circuit-breaker", "cost-tracking",
        "agent-coordination", "multi-agent-systems",
        "caching", "performance", "optimization", "cost-reduction", "semantic-similarity",
        "mcp", "model-context-protocol", "session-management", "websocket", "sse",
        "observability", "monitoring", "analytics", "dashboard", "metrics",
        "agent-framework", "ai-orchestration", "llm-framework", "agent-platform"
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'flux=fluxgraph.core.app:main',
            'fluxgraph=fluxgraph.core.app:main',
        ],
    },
    zip_safe=False,
    platforms=['any'],
    license='MIT',
    license_files=['LICENSE'],
)
