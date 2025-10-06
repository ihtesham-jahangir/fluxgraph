# setup.py
from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Optional dependencies for different feature sets
extras_require = {
    # Core production features (Phase 1)
    'production': [
        'fastapi>=0.104.0',
        'uvicorn[standard]>=0.24.0',
        'sse-starlette>=1.6.5',
        'pydantic>=2.5.0',
    ],
    
    # Security features (Phase 2)
    'security': [
        'pyjwt>=2.8.0',
        'cryptography>=41.0.0',
        'python-jose[cryptography]>=3.3.0',
        'passlib[bcrypt]>=1.7.4',
    ],
    
    # Advanced orchestration (Phase 3)
    'orchestration': [
        'celery>=5.3.4',
        'redis>=5.0.0',
    ],
    
    # Ecosystem features (Phase 4)
    'ecosystem': [
        'pillow>=10.1.0',
        'opencv-python>=4.8.0',
    ],
    
    # RAG capabilities
    'rag': [
        'chromadb>=0.4.18',
        'langchain>=0.1.0',
        'sentence-transformers>=2.2.2',
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
    
    # P0 Features (NEW in v3.0) - Graph workflows, advanced memory, caching
    'workflows': [
        'networkx>=3.1',  # For graph visualization
    ],
    
    'advanced-memory': [
        'sentence-transformers>=2.2.2',  # For semantic memory
        'numpy>=1.24.0',
    ],
    
    'caching': [
        'sentence-transformers>=2.2.2',  # For semantic caching
        'numpy>=1.24.0',
    ],
    
    # Convenience bundles
    'p0': [  # All P0 features
        'sentence-transformers>=2.2.2',
        'numpy>=1.24.0',
        'networkx>=3.1',
    ],
    
    # Development tools
    'dev': [
        'pytest>=7.4.3',
        'pytest-asyncio>=0.21.1',
        'pytest-cov>=4.1.0',
        'black>=23.12.0',
        'flake8>=6.1.0',
        'mypy>=1.7.1',
        'isort>=5.13.0',
        'pre-commit>=3.5.0',
    ],
    
    # Testing utilities
    'test': [
        'pytest>=7.4.3',
        'pytest-asyncio>=0.21.1',
        'pytest-cov>=4.1.0',
        'httpx>=0.25.0',  # For FastAPI testing
        'faker>=20.1.0',  # For test data generation
    ],
}

# 'all' includes everything
extras_require['all'] = list(set(sum(extras_require.values(), [])))

# 'full' includes all production features (no dev/test)
extras_require['full'] = list(set(
    extras_require['production'] +
    extras_require['security'] +
    extras_require['orchestration'] +
    extras_require['ecosystem'] +
    extras_require['rag'] +
    extras_require['analytics'] +
    extras_require['p0']
))

setup(
    name="fluxgraph",
    version="3.0.0",  # Major version for P0 features
    author="Ihtesham Jahangir",
    author_email="ceo@alphanetwork.com.pk",
    description="Enterprise AI Agent Orchestration Framework v3.0 - Graph workflows, advanced memory, semantic caching. The production-ready alternative to LangGraph, AutoGen, and CrewAI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ihtesham-jahangir/fluxgraph",
    project_urls={
        "Bug Tracker": "https://github.com/ihtesham-jahangir/fluxgraph/issues",
        "Documentation": "https://fluxgraph.readthedocs.io",
        "Source Code": "https://github.com/ihtesham-jahangir/fluxgraph",
        "Discord": "https://discord.gg/Z9bAqjYvPc",
        "Changelog": "https://github.com/ihtesham-jahangir/fluxgraph/blob/main/CHANGELOG.md",
        "LinkedIn": "https://linkedin.com/in/ihtesham-jahangir",
    },
    packages=find_packages(where="."),
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        'fluxgraph': [
            'py.typed',  # PEP 561 marker for type checking
            '*.json',
            '*.yaml',
            '*.yml',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: FastAPI",
        "Framework :: AsyncIO",
        "Typing :: Typed",
        "Natural Language :: English",
        "Environment :: Web Environment",
    ],
    keywords=[
        # Core concepts
        "ai", "agents", "llm", "gpt", "orchestration", "multi-agent",
        
        # Framework comparisons
        "langchain", "langgraph", "autogen", "crewai", "semantic-kernel",
        
        # LLM providers
        "openai", "anthropic", "claude", "gemini", "groq", "ollama",
        
        # Key features
        "fastapi", "async", "rag", "vector-database", "embeddings",
        
        # Enterprise features
        "enterprise", "production", "security", "audit", "compliance",
        "rbac", "pii-detection", "prompt-injection",
        
        # Orchestration
        "workflow", "graph", "dag", "state-machine", "handoff", "hitl",
        "batch-processing", "circuit-breaker", "cost-tracking",
        
        # P0 Features (v3.0)
        "workflow-graph", "semantic-caching", "hybrid-memory", "episodic-memory",
        
        # Protocols & standards
        "mcp", "model-context-protocol", "streaming", "session-management",
        
        # Additional
        "agent-marketplace", "observability", "monitoring", "analytics"
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'flux=fluxgraph.core.app:main',
            'fluxgraph=fluxgraph.core.app:main',  # Alternative command
        ],
    },
    zip_safe=False,
    platforms=['any'],
    license='MIT',
    license_files=['LICENSE'],
)
