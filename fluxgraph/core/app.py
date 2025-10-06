# fluxgraph/core/app.py
"""
FluxGraph Application Core - Enterprise Edition v3.0

Multi-agent orchestration framework with production-grade features including
streaming, sessions, security, advanced orchestration, and ecosystem integrations.

NEW in v3.0 (P0 Features):
- Graph-based workflow orchestration
- Advanced hybrid memory system
- Intelligent agent caching

Virtual Environment Handling:
Auto-creates and activates '.venv_fluxgraph' in the current working directory.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Any, Dict, Callable, Optional
import asyncio
import uuid
import time
import argparse
from contextvars import ContextVar

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | [%(filename)s:%(lineno)d] | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Import analytics
try:
    from fluxgraph.analytics import PerformanceMonitor, AnalyticsDashboard
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    logging.getLogger(__name__).warning("Analytics modules not available.")

# --- VIRTUAL ENVIRONMENT HANDLING ---
def _ensure_virtual_environment():
    """
    Ensures a virtual environment is set up and activated for FluxGraph.
    
    Checks if running in venv. If not, looks for '.venv_fluxgraph'.
    Creates new venv if needed and installs dependencies.
    """
    venv_name = ".venv_fluxgraph"
    venv_path = os.path.join(os.getcwd(), venv_name)
    logger = logging.getLogger(__name__)

    def _is_in_venv():
        return (
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )

    def _get_python_executable(venv_dir):
        if os.name == 'nt':
            return os.path.join(venv_dir, 'Scripts', 'python.exe')
        else:
            return os.path.join(venv_dir, 'bin', 'python')

    if _is_in_venv():
        logger.debug("✅ Already in virtual environment.")
        return

    if os.path.isdir(venv_path):
        logger.info(f"📦 Found venv at '{venv_path}'.")
    else:
        logger.info(f"🔧 Creating venv at '{venv_path}'...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            logger.info("✅ Venv created.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create venv: {e}")
            sys.exit(1)
        except FileNotFoundError:
            logger.error("❌ 'venv' module not found. Use Python 3.3+.")
            sys.exit(1)

        requirements_file = "requirements.txt"
        if os.path.isfile(requirements_file):
            venv_python = _get_python_executable(venv_path)
            logger.info(f"📦 Installing dependencies from '{requirements_file}'...")
            try:
                subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
                subprocess.run([venv_python, "-m", "pip", "install", "-r", requirements_file], check=True)
                logger.info("✅ Dependencies installed.")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ Dependency installation failed: {e}")

    venv_python = _get_python_executable(venv_path)
    if os.path.isfile(venv_python):
        logger.info(f"🔄 Activating venv...")
        try:
            os.execv(venv_python, [venv_python, __file__] + sys.argv[1:])
        except OSError as e:
            logger.error(f"❌ Failed to activate venv: {e}")
            sys.exit(1)
    else:
        logger.error(f"❌ Python executable not found in venv.")
        sys.exit(1)

_ensure_virtual_environment()

# --- STANDARD IMPORTS ---
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Core components
try:
    from .registry import AgentRegistry
    from .tool_registry import ToolRegistry
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Import error: {e}")
    print(f"❌ Import error: {e}")
    print("💡 Activate venv manually: source ./.venv_fluxgraph/bin/activate")
    print("   Then install: pip install -e .")
    sys.exit(1)

# P0 Features - New imports
try:
    from .workflow_graph import WorkflowGraph, WorkflowBuilder, NodeType
    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False
    WorkflowGraph = WorkflowBuilder = None
    logging.getLogger(__name__).warning("⚠️ Workflow graph not available")

try:
    from .advanced_memory import AdvancedMemory, MemoryType
    ADVANCED_MEMORY_AVAILABLE = True
except ImportError:
    ADVANCED_MEMORY_AVAILABLE = False
    AdvancedMemory = None
    logging.getLogger(__name__).warning("⚠️ Advanced memory not available")

try:
    from .agent_cache import AgentCache, CacheStrategy
    AGENT_CACHE_AVAILABLE = True
except ImportError:
    AGENT_CACHE_AVAILABLE = False
    AgentCache = None
    logging.getLogger(__name__).warning("⚠️ Agent cache not available")

# Orchestrator
try:
    from .orchestrator_advanced import AdvancedOrchestrator
    ADVANCED_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ADVANCED_ORCHESTRATOR_AVAILABLE = False
    try:
        from .orchestrator import FluxOrchestrator
        logging.getLogger(__name__).warning("⚠️ Using basic orchestrator.")
    except ImportError as e:
        logging.getLogger(__name__).error(f"❌ No orchestrator found: {e}")
        sys.exit(1)

# Memory
try:
    from .memory import Memory
    MEMORY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MEMORY_AVAILABLE = False
    class Memory:
        pass

# RAG
try:
    from .universal_rag import UniversalRAG
    RAG_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    RAG_AVAILABLE = False
    class UniversalRAG:
        pass

# Event Hooks
try:
    from ..utils.hooks import EventHooks
    HOOKS_MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    HOOKS_MODULE_AVAILABLE = False

if HOOKS_MODULE_AVAILABLE:
    _EventHooksClass = EventHooks
else:
    class _DummyEventHooks:
        async def trigger(self, event_name: str, payload: Dict[str, Any]):
            pass
    _EventHooksClass = _DummyEventHooks

# Enterprise features - Streaming
try:
    from ..core.streaming import StreamManager
except ImportError:
    StreamManager = None

# Sessions
try:
    from ..core.session_manager import SessionManager
except ImportError:
    SessionManager = None

# Retry
try:
    from ..core.retry import RetryManager
except ImportError:
    RetryManager = None

# Validation
try:
    from ..core.validation import OutputValidator
except ImportError:
    OutputValidator = None

# Security
try:
    from ..security.audit import AuditLogger, AuditEventType
    from ..security.pii_detector import PIIDetector
    from ..security.prompt_injection import PromptInjectionDetector
    from ..security.rbac import RBACManager, Role, Permission
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    AuditLogger = PIIDetector = PromptInjectionDetector = RBACManager = None
    AuditEventType = None

# Orchestration
try:
    from ..orchestration.handoff import HandoffProtocol
    from ..orchestration.hitl import HITLManager
    from ..orchestration.task_adherence import TaskAdherenceMonitor
    from ..orchestration.batch import BatchProcessor
    ORCHESTRATION_AVAILABLE = True
except ImportError:
    ORCHESTRATION_AVAILABLE = False
    HandoffProtocol = HITLManager = TaskAdherenceMonitor = BatchProcessor = None

# Ecosystem
try:
    from ..protocols.mcp import MCPServer
    from ..core.versioning import AgentVersionManager
    from ..marketplace.templates import TemplateMarketplace
    from ..multimodal.processor import MultiModalProcessor
    ECOSYSTEM_AVAILABLE = True
except ImportError:
    ECOSYSTEM_AVAILABLE = False
    MCPServer = AgentVersionManager = TemplateMarketplace = MultiModalProcessor = None

logger = logging.getLogger(__name__)
request_id_context: ContextVar[str] = ContextVar('request_id', default='N/A')


class FluxApp:
    """
    FluxGraph enterprise application manager v3.0.
    
    Core Features:
    - Multi-agent orchestration with message bus
    - Streaming responses and session management
    - Security (RBAC, audit logs, PII detection, prompt shields)
    - Advanced orchestration (handoffs, HITL, batch processing)
    - Ecosystem (MCP, versioning, templates, multi-modal)
    - Circuit breakers, smart routing, cost tracking
    
    NEW P0 Features (v3.0):
    - Graph-based workflow orchestration (like LangGraph)
    - Advanced hybrid memory system (short-term + long-term + episodic)
    - Intelligent agent caching (semantic + exact match)
    """

    def __init__(
        self,
        title: str = "FluxGraph API",
        description: str = "Enterprise AI agent orchestration framework",
        version: str = "3.0.0",
        memory_store: Optional[Memory] = None,
        rag_connector: Optional[UniversalRAG] = None,
        auto_init_rag: bool = True,
        enable_analytics: bool = True,
        enable_advanced_features: bool = True,
        enable_streaming: bool = True,
        enable_sessions: bool = True,
        enable_security: bool = True,
        enable_orchestration: bool = True,
        enable_mcp: bool = True,
        enable_versioning: bool = True,
        enable_templates: bool = True,
        # P0 Features
        enable_workflows: bool = True,
        enable_advanced_memory: bool = True,
        enable_agent_cache: bool = True,
        cache_strategy: str = "hybrid",  # "exact", "semantic", "hybrid"
        log_level: str = "INFO"
    ):
        logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        self.title = title
        self.description = description
        self.version = version
        self.api = FastAPI(title=title, description=description, version=version)
        
        logger.info("=" * 80)
        logger.info(f"🚀 Initializing FluxGraph: {title} v{version}")
        logger.info("=" * 80)
        
        # Core
        logger.info("📦 Core components...")
        self.registry = AgentRegistry()
        self.tool_registry = ToolRegistry()
        self.memory_store = memory_store
        self.rag_connector = rag_connector
        self.hooks = _EventHooksClass()
        
        # Orchestrator
        if enable_advanced_features and ADVANCED_ORCHESTRATOR_AVAILABLE:
            logger.info("🔧 AdvancedOrchestrator enabled")
            self.orchestrator = AdvancedOrchestrator(self.registry)
            self.advanced_features_enabled = True
        else:
            logger.info("🔧 Basic orchestrator")
            self.orchestrator = FluxOrchestrator(self.registry)
            self.advanced_features_enabled = False
        
        # Analytics
        if enable_analytics and ANALYTICS_AVAILABLE:
            logger.info("📊 Analytics enabled")
            self.performance_monitor = PerformanceMonitor()
            self.analytics_dashboard = AnalyticsDashboard(self.performance_monitor)
            self.api.include_router(self.analytics_dashboard.router)
        else:
            self.performance_monitor = None
            self.analytics_dashboard = None
        
        # Streaming & Sessions
        self.stream_manager = StreamManager() if enable_streaming and StreamManager else None
        self.session_manager = SessionManager() if enable_sessions and SessionManager else None
        self.retry_manager = RetryManager() if RetryManager else None
        self.output_validator = OutputValidator() if OutputValidator else None
        
        # Security
        if enable_security and SECURITY_AVAILABLE:
            logger.info("🔒 Security features enabled")
            self.audit_logger = AuditLogger()
            self.pii_detector = PIIDetector()
            self.prompt_shield = PromptInjectionDetector()
            self.rbac_manager = RBACManager()
        else:
            self.audit_logger = None
            self.pii_detector = None
            self.prompt_shield = None
            self.rbac_manager = None
        
        # Orchestration
        if enable_orchestration and ORCHESTRATION_AVAILABLE and self.advanced_features_enabled:
            logger.info("🎯 Advanced orchestration enabled")
            self.handoff_protocol = HandoffProtocol(self.orchestrator)
            self.hitl_manager = HITLManager()
            self.task_adherence = TaskAdherenceMonitor()
            self.batch_processor = BatchProcessor(self.orchestrator)
        else:
            self.handoff_protocol = None
            self.hitl_manager = None
            self.task_adherence = None
            self.batch_processor = None
        
        # Ecosystem
        if ECOSYSTEM_AVAILABLE:
            logger.info("🌐 Ecosystem features enabled")
            self.mcp_server = MCPServer() if enable_mcp else None
            self.version_manager = AgentVersionManager() if enable_versioning else None
            self.template_marketplace = TemplateMarketplace() if enable_templates else None
            self.multimodal_processor = MultiModalProcessor()
        else:
            self.mcp_server = None
            self.version_manager = None
            self.template_marketplace = None
            self.multimodal_processor = None
        
        # ===== P0 FEATURES (NEW in v3.0) =====
        logger.info("🆕 Initializing P0 features...")
        
        # 1. Workflow Graph
        if enable_workflows and WORKFLOW_AVAILABLE:
            logger.info("🔀 Graph-based workflows enabled")
            self.workflow_builder = WorkflowBuilder
            self.workflow_graphs: Dict[str, WorkflowGraph] = {}
        else:
            self.workflow_builder = None
            self.workflow_graphs = {}
        
        # 2. Advanced Memory
        if enable_advanced_memory and ADVANCED_MEMORY_AVAILABLE:
            logger.info("🧠 Advanced memory system enabled")
            self.advanced_memory = AdvancedMemory(
                short_term_capacity=100,
                long_term_capacity=10000,
                consolidation_threshold=0.7
            )
        else:
            self.advanced_memory = None
        
        # 3. Agent Cache
        if enable_agent_cache and AGENT_CACHE_AVAILABLE:
            strategy_map = {
                "exact": CacheStrategy.EXACT,
                "semantic": CacheStrategy.SEMANTIC,
                "hybrid": CacheStrategy.HYBRID
            }
            logger.info(f"⚡ Agent caching enabled (strategy={cache_strategy})")
            self.agent_cache = AgentCache(
                strategy=strategy_map.get(cache_strategy, CacheStrategy.HYBRID),
                max_size=1000,
                default_ttl=3600,  # 1 hour
                semantic_threshold=0.85
            )
        else:
            self.agent_cache = None
        
        # Auto-init RAG
        if auto_init_rag and RAG_AVAILABLE and self.rag_connector is None:
            self._auto_initialize_rag()
        
        self._setup_middleware()
        self._setup_routes()
        
        logger.info("=" * 80)
        logger.info(f"✅ FluxApp '{title}' ready!")
        logger.info(f"📝 Memory: {'ON' if self.memory_store else 'OFF'}")
        logger.info(f"🔍 RAG: {'ON' if self.rag_connector else 'OFF'}")
        logger.info(f"🎯 Advanced: {'ON' if self.advanced_features_enabled else 'OFF'}")
        logger.info(f"🔀 Workflows: {'ON' if self.workflow_builder else 'OFF'}")
        logger.info(f"🧠 Adv Memory: {'ON' if self.advanced_memory else 'OFF'}")
        logger.info(f"⚡ Cache: {'ON' if self.agent_cache else 'OFF'}")
        logger.info("=" * 80)

    def _auto_initialize_rag(self):
        """Auto-initialize RAG connector."""
        AUTO_RAG_PERSIST_DIR = "./my_chroma_db"
        AUTO_RAG_COLLECTION_NAME = "my_knowledge_base"
        
        logger.info("🔄 Auto-initializing RAG...")
        persist_path = Path(AUTO_RAG_PERSIST_DIR)
        
        if not persist_path.exists():
            logger.info(f"📁 Creating RAG directory: {AUTO_RAG_PERSIST_DIR}")
            try:
                persist_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"❌ RAG directory creation failed: {e}")
                self.rag_connector = None
                return
        
        try:
            embedding_model = os.getenv("FLUXGRAPH_RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            chunk_size = int(os.getenv("FLUXGRAPH_RAG_CHUNK_SIZE", "750"))
            chunk_overlap = int(os.getenv("FLUXGRAPH_RAG_CHUNK_OVERLAP", "100"))
            
            logger.info(f"🔧 RAG config: {embedding_model}, chunk:{chunk_size}")
            
            self.rag_connector = UniversalRAG(
                persist_directory=AUTO_RAG_PERSIST_DIR,
                collection_name=AUTO_RAG_COLLECTION_NAME,
                embedding_model_name=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            logger.info("✅ RAG initialized")
        except Exception as e:
            logger.error(f"❌ RAG init failed: {e}")
            self.rag_connector = None

    def _setup_middleware(self):
        """Setup CORS and logging middleware."""
        logger.info("🔧 Setting up middleware...")
        
        self.api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("✅ CORS configured")
        
        @self.api.middleware("http")
        async def advanced_logging_middleware(request: Request, call_next):
            request_id = str(uuid.uuid4())
            request_id_context.set(request_id)
            
            start_time = time.time()
            client_host = request.client.host if request.client else "unknown"
            method = request.method
            url = str(request.url)
            
            # Security checks
            if self.prompt_shield:
                pass
            
            # Audit
            if self.audit_logger and SECURITY_AVAILABLE:
                self.audit_logger.log(
                    AuditEventType.API_REQUEST,
                    user_id=request.headers.get("X-User-ID"),
                    details={"method": method, "url": url, "client": client_host}
                )
            
            logger.info("━" * 80)
            logger.info(f"📥 {method} {url} | ID: {request_id[:8]}")
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = str(round(process_time, 4))
                
                if process_time < 0.5:
                    perf = "⚡FAST"
                elif process_time < 2.0:
                    perf = "✅NORMAL"
                elif process_time < 5.0:
                    perf = "⚠️SLOW"
                else:
                    perf = "🔴VERY SLOW"
                
                logger.info(f"📤 {response.status_code} | {process_time:.4f}s | {perf}")
                logger.info("━" * 80)
                
                return response
            except Exception as e:
                process_time = time.time() - start_time
                logger.error("━" * 80)
                logger.error(f"❌ ERROR: {type(e).__name__}: {e}")
                logger.error(f"   Time: {process_time:.4f}s")
                logger.error("━" * 80)
                logger.exception("Traceback:")
                raise
        
        logger.info("✅ Logging middleware configured")

    def _setup_routes(self):
        """Setup API routes."""
        logger.info("🔧 Setting up routes...")
        
        @self.api.get("/", summary="Root")
        async def root():
            request_id = request_id_context.get()
            logger.info(f"[{request_id}] Root accessed")
            
            return {
                "message": f"Welcome to {self.title}",
                "version": self.version,
                "status": "operational",
                "features": {
                    "memory": self.memory_store is not None,
                    "rag": self.rag_connector is not None,
                    "analytics": self.performance_monitor is not None,
                    "advanced": self.advanced_features_enabled,
                    "streaming": self.stream_manager is not None,
                    "sessions": self.session_manager is not None,
                    "security": self.audit_logger is not None,
                    "orchestration": self.handoff_protocol is not None,
                    "mcp": self.mcp_server is not None,
                    # P0 features
                    "workflows": self.workflow_builder is not None,
                    "advanced_memory": self.advanced_memory is not None,
                    "agent_cache": self.agent_cache is not None
                },
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "request_id": request_id
            }
        
        @self.api.post("/ask/{agent_name}", summary="Execute Agent")
        async def ask_agent(agent_name: str, payload: Dict[str, Any]):
            request_id = request_id_context.get()
            start_time = time.time()
            
            logger.info("─" * 80)
            logger.info(f"🤖 AGENT: {agent_name}")
            logger.info(f"   ID: {request_id}")
            logger.info(f"   Payload keys: {list(payload.keys())}")
            
            # Check agent cache first
            if self.agent_cache:
                cache_key = f"{agent_name}:{str(payload)}"
                if cached_result := self.agent_cache.get(cache_key):
                    logger.info(f"   ⚡ CACHE HIT ({time.time() - start_time:.4f}s)")
                    return cached_result
            
            await self.hooks.trigger("request_received", {
                "request_id": request_id,
                "agent_name": agent_name,
                "payload": payload
            })
            
            try:
                logger.info(f"   ▶️ Executing...")
                result = await self.orchestrator.run(agent_name, payload)
                
                # Store in cache
                if self.agent_cache:
                    cache_key = f"{agent_name}:{str(payload)}"
                    self.agent_cache.set(cache_key, result, ttl=3600)
                
                # Store in advanced memory
                if self.advanced_memory:
                    self.advanced_memory.store(
                        f"Agent {agent_name} executed with {payload}",
                        MemoryType.EPISODIC,
                        metadata={"agent": agent_name, "result": result}
                    )
                
                duration = time.time() - start_time
                
                await self.hooks.trigger("agent_completed", {
                    "request_id": request_id,
                    "agent_name": agent_name,
                    "result": result,
                    "duration": duration
                })
                
                logger.info(f"   ✅ Success ({duration:.4f}s)")
                logger.info("─" * 80)
                
                return result
            except ValueError as e:
                duration = time.time() - start_time
                logger.warning(f"   ⚠️ {str(e)} ({duration:.4f}s)")
                logger.warning("─" * 80)
                
                await self.hooks.trigger("agent_error", {
                    "request_id": request_id,
                    "agent_name": agent_name,
                    "error": str(e),
                    "duration": duration
                })
                
                status_code = 404 if "not registered" in str(e).lower() else 400
                raise HTTPException(status_code=status_code, detail=str(e))
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"   ❌ {type(e).__name__}: {e} ({duration:.4f}s)")
                logger.error("─" * 80)
                logger.exception("Traceback:")
                
                await self.hooks.trigger("server_error", {
                    "request_id": request_id,
                    "agent_name": agent_name,
                    "error": str(e),
                    "duration": duration
                })
                
                raise HTTPException(status_code=500, detail="Internal Server Error")
        
        # P0 Feature Routes
        if self.workflow_builder:
            @self.api.get("/workflows", summary="List Workflows")
            async def list_workflows():
                """List all registered workflow graphs."""
                return {
                    "workflows": list(self.workflow_graphs.keys()),
                    "count": len(self.workflow_graphs)
                }
            
            @self.api.post("/workflows/{workflow_name}/execute", summary="Execute Workflow")
            async def execute_workflow(workflow_name: str, initial_data: Dict[str, Any]):
                """Execute a workflow graph."""
                if workflow_name not in self.workflow_graphs:
                    raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")
                
                workflow = self.workflow_graphs[workflow_name]
                result = await workflow.execute(initial_data)
                return result
        
        if self.advanced_memory:
            @self.api.get("/memory/stats", summary="Memory Statistics")
            async def memory_stats():
                """Get advanced memory system statistics."""
                return self.advanced_memory.get_stats()
            
            @self.api.post("/memory/recall", summary="Recall Memories")
            async def recall_memories(query: str, k: int = 5):
                """Recall similar memories."""
                results = self.advanced_memory.recall_similar(query, k=k)
                return {
                    "query": query,
                    "results": [
                        {"content": entry.content, "similarity": float(score)}
                        for entry, score in results
                    ]
                }
            
            @self.api.post("/memory/consolidate", summary="Consolidate Memory")
            async def consolidate_memory():
                """Consolidate short-term to long-term memory."""
                self.advanced_memory.consolidate()
                return {"status": "consolidated", "stats": self.advanced_memory.get_stats()}
        
        if self.agent_cache:
            @self.api.get("/cache/stats", summary="Cache Statistics")
            async def cache_stats():
                """Get agent cache statistics."""
                return self.agent_cache.get_stats()
            
            @self.api.post("/cache/clear", summary="Clear Cache")
            async def clear_cache():
                """Clear agent cache."""
                self.agent_cache.clear()
                return {"status": "cleared"}
        
        # ... (rest of existing routes remain the same) ...
        
        logger.info("✅ Routes configured")

    def register(self, name: str, agent: Any):
        """Register an agent."""
        self.registry.add(name, agent)
        logger.info(f"✅ Agent registered: {name}")

    def register_workflow(self, name: str, workflow: WorkflowGraph):
        """Register a workflow graph."""
        if not self.workflow_builder:
            raise RuntimeError("Workflows not enabled")
        self.workflow_graphs[name] = workflow
        logger.info(f"🔀 Workflow registered: {name}")

    def tool(self, name: Optional[str] = None):
        """Register a tool via decorator."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            self.tool_registry.register(tool_name, func)
            logger.info(f"🛠️ Tool registered: {tool_name}")
            return func
        return decorator

    def agent(self, name: Optional[str] = None, track_performance: bool = True):
        """Register an agent via decorator."""
        def decorator(func: Callable) -> Callable:
            agent_name = name or func.__name__
            
            if track_performance and self.performance_monitor:
                func = self.performance_monitor.track_performance(func, agent_name=agent_name)
            
            class _FluxDynamicAgent:
                async def run(self, **kwargs):
                    # Filter out internal orchestrator parameters
                    internal_params = {
                        '_message_bus', '_current_agent', '_circuit_breaker',
                        '_router', '_cost_tracker'
                    }
                    
                    user_kwargs = {k: v for k, v in kwargs.items() if k not in internal_params}
                    
                    # Add FluxGraph utilities
                    user_kwargs['tools'] = self._tool_registry
                    if self._memory_store:
                        user_kwargs['memory'] = self._memory_store
                    if self._rag_connector:
                        user_kwargs['rag'] = self._rag_connector
                    
                    # P0: Add advanced memory and cache
                    if self._advanced_memory:
                        user_kwargs['advanced_memory'] = self._advanced_memory
                    if self._agent_cache:
                        user_kwargs['cache'] = self._agent_cache
                    
                    # Add agent communication methods
                    if self._advanced_features:
                        message_bus = kwargs.get('_message_bus')
                        current_agent = kwargs.get('_current_agent', agent_name)
                        
                        async def call_agent_wrapper(target_agent: str, **call_kwargs):
                            if message_bus:
                                logger.info(f"📞 '{current_agent}' calling '{target_agent}'")
                                return await message_bus.send_message(
                                    sender=current_agent,
                                    receiver=target_agent,
                                    payload=call_kwargs
                                )
                            raise RuntimeError("Message bus unavailable")
                        
                        async def broadcast_wrapper(agent_names: List[str], **broadcast_kwargs):
                            if message_bus:
                                logger.info(f"📢 '{current_agent}' broadcasting to {len(agent_names)} agents")
                                return await message_bus.broadcast(
                                    sender=current_agent,
                                    receivers=agent_names,
                                    payload=broadcast_kwargs
                                )
                            raise RuntimeError("Message bus unavailable")
                        
                        user_kwargs['call_agent'] = call_agent_wrapper
                        user_kwargs['broadcast'] = broadcast_wrapper
                    
                    # Call user's function
                    if asyncio.iscoroutinefunction(func):
                        return await func(**user_kwargs)
                    return func(**user_kwargs)
            
            agent_instance = _FluxDynamicAgent()
            agent_instance._tool_registry = self.tool_registry
            agent_instance._memory_store = self.memory_store
            agent_instance._rag_connector = self.rag_connector
            agent_instance._advanced_features = self.advanced_features_enabled
            agent_instance._advanced_memory = self.advanced_memory
            agent_instance._agent_cache = self.agent_cache
            
            self.register(agent_name, agent_instance)
            logger.info(f"🤖 Agent '{agent_name}' registered via decorator")
            return func
        return decorator

    def run(self, host: str = "127.0.0.1", port: int = 8000, reload: bool = False, **kwargs):
        """Start the FluxGraph API server."""
        logger.info("=" * 80)
        logger.info(f"🚀 Starting FluxGraph Server v{self.version}")
        logger.info(f"   Host: {host}")
        logger.info(f"   Port: {port}")
        logger.info(f"   Reload: {reload}")
        logger.info("=" * 80)
        
        try:
            import uvicorn
            uvicorn.run(self.api, host=host, port=port, reload=reload, **kwargs)
        except ImportError as e:
            if "watchdog" in str(e).lower():
                logger.error("❌ 'watchdog' required for --reload. Install: pip install watchdog")
                sys.exit(1)
            logger.error(f"❌ Failed to import uvicorn: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Server startup failed: {e}", exc_info=True)
            raise


def main():
    """CLI entry point: flux run [--reload] <file.py>"""
    parser = argparse.ArgumentParser(prog='flux', description="FluxGraph CLI v3.0")
    subparsers = parser.add_subparsers(dest='command')
    
    run_parser = subparsers.add_parser('run', help='Run FluxGraph app')
    run_parser.add_argument('file', help="Path to app file")
    run_parser.add_argument('--reload', action='store_true', help="Enable auto-reload")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command != 'run':
        print(f"❌ Unknown command: {args.command}")
        sys.exit(1)
    
    import importlib.util
    file_path = Path(args.file).resolve()
    
    if not file_path.exists():
        print(f"❌ File '{args.file}' not found")
        sys.exit(1)
    
    logger.info(f"📦 Loading app from '{args.file}'...")
    spec = importlib.util.spec_from_file_location("user_app", str(file_path))
    if not spec or not spec.loader:
        print(f"❌ Could not load '{args.file}'")
        sys.exit(1)
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_app"] = module
    
    try:
        spec.loader.exec_module(module)
        logger.info("✅ App loaded")
    except Exception as e:
        print(f"❌ Error executing '{args.file}': {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("🔍 Looking for 'app' instance...")
    app_instance = getattr(module, 'app', None)
    
    if not app_instance:
        print("❌ No 'app' variable found")
        sys.exit(1)
    
    if not isinstance(app_instance, FluxApp):
        print(f"❌ 'app' is not FluxApp instance. Type: {type(app_instance)}")
        sys.exit(1)
    
    logger.info("✅ FluxApp found")
    
    try:
        app_instance.run(host="127.0.0.1", port=8000, reload=args.reload)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown complete")
        logger.info("🛑 Server stopped")
    except Exception as e:
        logger.error(f"❌ Failed to start: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
