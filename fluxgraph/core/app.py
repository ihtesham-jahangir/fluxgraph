# fluxgraph/core/app.py
"""
FluxGraph Application Core - Enterprise Edition v3.2 (PRODUCTION READY)

Multi-agent orchestration framework with complete production-grade features.

NEW in v3.2 - LangChain Feature Parity:
✅ LCEL-style chain building with pipe operators
✅ LangSmith-style distributed tracing
✅ Optimized batch processing with adaptive concurrency
✅ Time-to-First-Token streaming optimization
✅ LangServe-style production deployment
✅ Complete integration with all existing features
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

logger = logging.getLogger(__name__)

# ===== VIRTUAL ENVIRONMENT HANDLING =====
def _ensure_virtual_environment():
    """Ensures a virtual environment is set up and activated for FluxGraph."""
    venv_name = ".venv_fluxgraph"
    venv_path = os.path.join(os.getcwd(), venv_name)

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

# ===== STANDARD IMPORTS =====
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import json

# Core components
try:
    from .registry import AgentRegistry
    from .tool_registry import ToolRegistry
    from .orchestrator import FluxOrchestrator
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    print(f"❌ Import error: {e}")
    print("💡 Activate venv manually: source ./.venv_fluxgraph/bin/activate")
    print("   Then install: pip install -e .")
    sys.exit(1)

# Analytics
try:
    from fluxgraph.analytics import PerformanceMonitor, AnalyticsDashboard
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    logger.warning("Analytics modules not available.")

# ===== V3.2 NEW IMPORTS (LangChain Feature Parity) =====
try:
    from fluxgraph.chains import (
        Runnable, RunnableSequence, RunnableParallel, RunnableLambda,
        chain, parallel, runnable, RunnableConfig
    )
    from fluxgraph.chains.prompts import PromptTemplate, ChatPromptTemplate
    from fluxgraph.chains.parsers import (
        JsonOutputParser, PydanticOutputParser, ListOutputParser
    )
    from fluxgraph.chains.models import create_llm_runnable, LLMRunnable
    from fluxgraph.chains.batch import BatchProcessor, BatchConfig, BatchStrategy
    from fluxgraph.chains.streaming import StreamOptimizer, optimize_stream, StreamMetrics
    from fluxgraph.tracing import (
        Tracer, configure_tracing, trace, span, RunType, TraceRun
    )
    from fluxgraph.serve import FluxServe, serve, deploy_multiple
    CHAINS_V32_AVAILABLE = True
    logger.info("✅ v3.2 chain features loaded")
except ImportError as e:
    CHAINS_V32_AVAILABLE = False
    logger.warning(f"⚠️ v3.2 chain features not available: {e}")

# Memory & RAG
try:
    from .memory import Memory
    MEMORY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MEMORY_AVAILABLE = False
    class Memory:
        pass

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

# Context for Request Tracking
request_id_context: ContextVar[str] = ContextVar('request_id', default='N/A')


# ===== V3.2 REQUEST/RESPONSE MODELS =====
class ChainInvokeRequest(BaseModel):
    """Request model for chain invocation"""
    input: Any = Field(..., description="Input to the chain")
    config: Optional[Dict[str, Any]] = None
    stream: bool = False


class ChainBatchRequest(BaseModel):
    """Request model for batch processing"""
    inputs: List[Any] = Field(..., description="List of inputs")
    config: Optional[Dict[str, Any]] = None
    max_concurrency: int = 10


class ChainResponse(BaseModel):
    """Response model for chain operations"""
    output: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ===== FLUXAPP CLASS v3.2 =====
class FluxApp:
    """
    FluxGraph enterprise application manager v3.2 - PRODUCTION READY.
    
    Complete feature set:
    - Multi-agent orchestration
    - LCEL-style chains with pipe operators
    - Distributed tracing
    - Batch processing with adaptive concurrency
    - Streaming optimization
    - Production deployment
    - Memory and RAG integration
    - Analytics and monitoring
    """

    def __init__(
        self,
        title: str = "FluxGraph API",
        description: str = "Enterprise AI agent orchestration framework v3.2",
        version: str = "3.2.0",
        memory_store: Optional[Memory] = None,
        rag_connector: Optional[UniversalRAG] = None,
        auto_init_rag: bool = True,
        enable_analytics: bool = True,
        # v3.2 NEW Features (LangChain Parity)
        enable_chains: bool = True,
        enable_tracing: bool = True,
        enable_batch_optimization: bool = True,
        enable_streaming_optimization: bool = True,
        enable_langserve_api: bool = True,
        tracing_export_path: str = "./traces",
        tracing_project_name: Optional[str] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize FluxGraph v3.2 application.
        
        Args:
            title: API title
            description: API description
            version: API version
            memory_store: Optional memory store instance
            rag_connector: Optional RAG connector instance
            auto_init_rag: Auto-initialize RAG if not provided
            enable_analytics: Enable analytics dashboard
            enable_chains: Enable LCEL-style chains (v3.2)
            enable_tracing: Enable distributed tracing (v3.2)
            enable_batch_optimization: Enable batch processing (v3.2)
            enable_streaming_optimization: Enable streaming optimization (v3.2)
            enable_langserve_api: Enable LangServe-style API (v3.2)
            tracing_export_path: Path for trace exports
            tracing_project_name: Project name for tracing
            log_level: Logging level
        """
        logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        self.title = title
        self.description = description
        self.version = version
        self.api = FastAPI(title=title, description=description, version=version)
        
        logger.info("=" * 80)
        logger.info(f"🚀 Initializing FluxGraph: {title} v{version}")
        logger.info("=" * 80)
        
        # Core components
        logger.info("📦 Core components...")
        self.registry = AgentRegistry()
        self.tool_registry = ToolRegistry()
        self.orchestrator = FluxOrchestrator(self.registry)
        self.memory_store = memory_store
        self.rag_connector = rag_connector
        self.hooks = _EventHooksClass()
        
        # Analytics
        if enable_analytics and ANALYTICS_AVAILABLE:
            logger.info("📊 Analytics enabled")
            self.performance_monitor = PerformanceMonitor()
            self.analytics_dashboard = AnalyticsDashboard(self.performance_monitor)
            self.api.include_router(self.analytics_dashboard.router)
        else:
            self.performance_monitor = None
            self.analytics_dashboard = None
        
        # ===== V3.2 FEATURES (LangChain Parity) =====
        logger.info("🚀 Initializing v3.2 features (LangChain Parity)...")
        
        # Chains
        if enable_chains and CHAINS_V32_AVAILABLE:
            logger.info("⛓️ LCEL-style chains enabled")
            self.chains: Dict[str, Runnable] = {}
            self.chain_registry = {}
            self.chains_enabled = True
        else:
            self.chains = None
            self.chain_registry = {}
            self.chains_enabled = False
        
        # Tracing
        if enable_tracing and CHAINS_V32_AVAILABLE:
            logger.info("🔍 Distributed tracing enabled")
            project_name = tracing_project_name or title
            self.tracer = configure_tracing(
                project_name=project_name,
                enabled=True,
                export_path=tracing_export_path,
                export_format="json"
            )
            self.tracing_enabled = True
        else:
            self.tracer = None
            self.tracing_enabled = False
        
        # Batch Optimization
        if enable_batch_optimization and CHAINS_V32_AVAILABLE:
            logger.info("📦 Batch optimization enabled")
            self.batch_optimizer_enabled = True
        else:
            self.batch_optimizer_enabled = False
        
        # Streaming Optimization
        if enable_streaming_optimization and CHAINS_V32_AVAILABLE:
            logger.info("⚡ Streaming optimization enabled")
            self.streaming_optimizer_enabled = True
        else:
            self.streaming_optimizer_enabled = False
        
        # LangServe-style API
        if enable_langserve_api and CHAINS_V32_AVAILABLE:
            logger.info("🌐 LangServe-style API enabled")
            self.langserve_enabled = True
        else:
            self.langserve_enabled = False
        
        # Auto-init RAG
        if auto_init_rag and RAG_AVAILABLE and self.rag_connector is None:
            self._auto_initialize_rag()
        
        self._setup_middleware()
        self._setup_routes()
        
        logger.info("=" * 80)
        logger.info(f"✅ FluxApp '{title}' ready!")
        logger.info(f"📝 Memory: {'ON' if self.memory_store else 'OFF'}")
        logger.info(f"🔍 RAG: {'ON' if self.rag_connector else 'OFF'}")
        logger.info(f"⛓️ Chains (v3.2): {'ON' if self.chains_enabled else 'OFF'}")
        logger.info(f"🔍 Tracing (v3.2): {'ON' if self.tracing_enabled else 'OFF'}")
        logger.info(f"📦 Batch Opt (v3.2): {'ON' if self.batch_optimizer_enabled else 'OFF'}")
        logger.info(f"⚡ Streaming Opt (v3.2): {'ON' if self.streaming_optimizer_enabled else 'OFF'}")
        logger.info(f"🌐 LangServe API (v3.2): {'ON' if self.langserve_enabled else 'OFF'}")
        logger.info("=" * 80)

    def _auto_initialize_rag(self):
        """Attempts to automatically initialize the UniversalRAG connector."""
        AUTO_RAG_PERSIST_DIR = "./my_chroma_db"
        AUTO_RAG_COLLECTION_NAME = "my_knowledge_base"
        logger.info("🔄 Attempting to auto-initialize UniversalRAG connector...")
        
        persist_path = Path(AUTO_RAG_PERSIST_DIR)
        if not persist_path.exists():
            logger.info(f"📁 Creating RAG persist directory '{AUTO_RAG_PERSIST_DIR}'")
            try:
                persist_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"❌ Failed to create RAG persist directory '{AUTO_RAG_PERSIST_DIR}': {e}")
                self.rag_connector = None
                return
        
        try:
            embedding_model = os.getenv("FLUXGRAPH_RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            chunk_size = int(os.getenv("FLUXGRAPH_RAG_CHUNK_SIZE", "750"))
            chunk_overlap = int(os.getenv("FLUXGRAPH_RAG_CHUNK_OVERLAP", "100"))
            
            self.rag_connector = UniversalRAG(
                persist_directory=AUTO_RAG_PERSIST_DIR,
                collection_name=AUTO_RAG_COLLECTION_NAME,
                embedding_model_name=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            logger.info("✅ Universal RAG connector auto-initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to auto-initialize RAG connector: {e}", exc_info=True)
            self.rag_connector = None

    def _setup_middleware(self):
        """Setup default middlewares (CORS, Logging)."""
        self.api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.api.middleware("http")
        async def log_and_context_middleware(request: Request, call_next):
            request_id = str(uuid.uuid4())
            request_id_context.set(request_id)
            start_time = time.time()
            client_host = request.client.host if request.client else "unknown"
            method = request.method
            url = str(request.url)
            logger.info(f"[Request ID: {request_id}] 🌐 Incoming request: {method} {url} from {client_host}")
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = str(round(process_time, 4))
                logger.info(f"[Request ID: {request_id}] ⬅️ Response status: {response.status_code} (Processing Time: {process_time:.4f}s)")
                return response
            except Exception as e:
                process_time = time.time() - start_time
                logger.error(f"[Request ID: {request_id}] ❌ Unhandled error during request processing (Processing Time: {process_time:.4f}s): {e}", exc_info=True)
                raise

    def _setup_routes(self):
        """Setup API routes including v3.2 chain endpoints."""
        
        @self.api.get("/")
        async def root():
            """Root endpoint providing API information."""
            request_id = request_id_context.get()
            logger.info(f"[Request ID: {request_id}] 📢 Root endpoint called.")
            return {
                "message": "Welcome to FluxGraph v3.2",
                "title": self.title,
                "version": self.version,
                "features": {
                    "memory": self.memory_store is not None,
                    "rag": self.rag_connector is not None,
                    "chains": self.chains_enabled,
                    "tracing": self.tracing_enabled,
                    "batch_optimization": self.batch_optimizer_enabled,
                    "streaming_optimization": self.streaming_optimizer_enabled,
                    "langserve_api": self.langserve_enabled
                },
                "endpoints": {
                    "agents": "/ask/{agent_name}",
                    "chains": "/chains",
                    "chain_invoke": "/chains/{chain_name}/invoke",
                    "chain_batch": "/chains/{chain_name}/batch",
                    "chain_stream": "/chains/{chain_name}/stream",
                    "tools": "/tools",
                    "docs": "/docs"
                }
            }
        
        @self.api.post("/ask/{agent_name}")
        async def ask_agent(agent_name: str, payload: Dict[str, Any]):
            """Execute a registered agent."""
            request_id = request_id_context.get()
            start_time = time.time()
            logger.info(f"[Request ID: {request_id}] 🤖 Executing agent '{agent_name}' with payload keys: {list(payload.keys())}")
            
            await self.hooks.trigger("request_received", {
                "request_id": request_id,
                "agent_name": agent_name,
                "payload": payload
            })
            
            try:
                result = await self.orchestrator.run(agent_name, payload)
                end_time = time.time()
                duration = end_time - start_time
                
                await self.hooks.trigger("agent_completed", {
                    "request_id": request_id,
                    "agent_name": agent_name,
                    "result": result,
                    "duration": duration
                })
                
                logger.info(f"[Request ID: {request_id}] ✅ Agent '{agent_name}' executed successfully (Total Duration: {duration:.4f}s).")
                return result
                
            except ValueError as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.warning(f"[Request ID: {request_id}] ⚠️ Agent '{agent_name}' error (Duration: {duration:.4f}s): {e}")
                
                await self.hooks.trigger("agent_error", {
                    "request_id": request_id,
                    "agent_name": agent_name,
                    "error": str(e),
                    "duration": duration
                })
                
                status_code = 404 if "not registered" in str(e).lower() or "not found" in str(e).lower() else 400
                raise HTTPException(status_code=status_code, detail=str(e))
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(f"[Request ID: {request_id}] ❌ Execution error for agent '{agent_name}' (Duration: {duration:.4f}s): {e}", exc_info=True)
                
                await self.hooks.trigger("server_error", {
                    "request_id": request_id,
                    "agent_name": agent_name,
                    "error": str(e),
                    "duration": duration
                })
                
                raise HTTPException(status_code=500, detail="Internal Server Error")
        
        # ===== V3.2 CHAIN ENDPOINTS =====
        
        if self.langserve_enabled and CHAINS_V32_AVAILABLE:
            
            @self.api.get("/chains")
            async def list_chains():
                """List all registered chains."""
                return {
                    "chains": list(self.chains.keys()),
                    "count": len(self.chains),
                    "registry": self.chain_registry
                }
            
            @self.api.post("/chains/{chain_name}/invoke", response_model=ChainResponse)
            async def invoke_chain(chain_name: str, request: ChainInvokeRequest):
                """Invoke a chain with single input."""
                if chain_name not in self.chains:
                    raise HTTPException(status_code=404, detail=f"Chain '{chain_name}' not found")
                
                chain = self.chains[chain_name]
                start_time = time.time()
                
                try:
                    if self.tracing_enabled:
                        async with self.tracer.span(f"chain_{chain_name}", run_type=RunType.CHAIN, inputs={"input": request.input}):
                            output = await chain.invoke(request.input, request.config)
                    else:
                        output = await chain.invoke(request.input, request.config)
                    
                    duration = time.time() - start_time
                    
                    return ChainResponse(
                        output=output,
                        metadata={
                            "chain_name": chain_name,
                            "duration_ms": duration * 1000,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    
                except Exception as e:
                    logger.error(f"Chain invocation error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            @self.api.post("/chains/{chain_name}/batch")
            async def batch_chain(chain_name: str, request: ChainBatchRequest):
                """Batch process multiple inputs."""
                if chain_name not in self.chains:
                    raise HTTPException(status_code=404, detail=f"Chain '{chain_name}' not found")
                
                chain = self.chains[chain_name]
                start_time = time.time()
                
                try:
                    if self.batch_optimizer_enabled:
                        config = RunnableConfig(max_concurrency=request.max_concurrency)
                        outputs = await chain.batch(request.inputs, config)
                    else:
                        outputs = await chain.batch(request.inputs)
                    
                    duration = time.time() - start_time
                    
                    return {
                        "outputs": outputs,
                        "metadata": {
                            "count": len(outputs),
                            "duration_ms": duration * 1000,
                            "avg_per_item_ms": (duration * 1000) / len(outputs) if outputs else 0
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            @self.api.websocket("/chains/{chain_name}/stream")
            async def stream_chain(websocket: WebSocket, chain_name: str):
                """WebSocket streaming endpoint."""
                await websocket.accept()
                
                try:
                    if chain_name not in self.chains:
                        await websocket.send_json({"error": f"Chain '{chain_name}' not found"})
                        await websocket.close()
                        return
                    
                    data = await websocket.receive_json()
                    input_data = data.get("input")
                    
                    chain = self.chains[chain_name]
                    
                    async for chunk in chain.stream(input_data):
                        await websocket.send_json({"chunk": chunk})
                    
                    await websocket.send_json({"done": True})
                    
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected")
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    await websocket.send_json({"error": str(e)})
                finally:
                    await websocket.close()
        
        # Tools endpoints
        @self.api.get("/tools")
        async def list_tools():
            """List registered tool names."""
            request_id = request_id_context.get()
            logger.info(f"[Request ID: {request_id}] 🛠️ Listing registered tools.")
            return {"tools": self.tool_registry.list_tools()}
        
        @self.api.get("/tools/{tool_name}")
        async def get_tool_info(tool_name: str):
            """Get information about a specific tool."""
            request_id = request_id_context.get()
            try:
                logger.info(f"[Request ID: {request_id}] 🔎 Fetching tool info for '{tool_name}'.")
                info = self.tool_registry.get_tool_info(tool_name)
                return info
            except ValueError as e:
                logger.warning(f"[Request ID: {request_id}] ⚠️ Tool '{tool_name}' not found: {e}")
                raise HTTPException(status_code=404, detail=str(e))
        
        # Memory & RAG endpoints
        if MEMORY_AVAILABLE and self.memory_store:
            @self.api.get("/memory/status")
            async def memory_status():
                """Check memory store status."""
                request_id = request_id_context.get()
                logger.info(f"[Request ID: {request_id}] 💾 Memory status requested.")
                return {"memory_enabled": True, "type": type(self.memory_store).__name__}
        
        if RAG_AVAILABLE and self.rag_connector:
            @self.api.get("/rag/status")
            async def rag_status():
                """Check RAG connector status."""
                request_id = request_id_context.get()
                logger.info(f"[Request ID: {request_id}] 🔍 RAG status requested.")
                try:
                    stats = self.rag_connector.get_collection_stats()
                    return {
                        "rag_enabled": True,
                        "type": type(self.rag_connector).__name__,
                        "collection_stats": stats
                    }
                except Exception as e:
                    logger.error(f"[Request ID: {request_id}] ❌ Error getting RAG stats: {e}", exc_info=True)
                    return {
                        "rag_enabled": True,
                        "type": type(self.rag_connector).__name__,
                        "stats_error": str(e)
                    }
        
        # Tracing endpoints
        if self.tracing_enabled:
            @self.api.get("/tracing/status")
            async def tracing_status():
                """Get tracing status and statistics."""
                stats = self.tracer.get_statistics()
                return {
                    "tracing_enabled": True,
                    "project": self.tracer.project_name,
                    "statistics": stats
                }
            
            @self.api.get("/tracing/traces")
            async def get_traces():
                """Get all root traces."""
                traces = self.tracer.get_root_traces()
                return {
                    "traces": [t.to_dict() for t in traces],
                    "count": len(traces)
                }

    # ===== V3.2 CHAIN METHODS (NEW) =====
    
    def register_chain(self, name: str, chain: Runnable, description: Optional[str] = None):
        """
        Register a chain for deployment.
        
        Example:
            chain = prompt | model | parser
            app.register_chain("summarizer", chain)
        """
        if not self.chains_enabled:
            raise RuntimeError("Chains not enabled. Set enable_chains=True")
        
        self.chains[name] = chain
        self.chain_registry[name] = {
            "type": type(chain).__name__,
            "description": description,
            "registered_at": datetime.utcnow().isoformat()
        }
        logger.info(f"⛓️ Chain '{name}' registered")
        return chain
    
    def chain(self, name: Optional[str] = None, description: Optional[str] = None):
        """
        Decorator to register a chain-building function.
        
        Example:
            @app.chain("summarizer")
            def create_summarizer():
                return prompt | model | parser
        """
        def decorator(func: Callable) -> Callable:
            chain_name = name if name is not None else func.__name__
            chain_instance = func()
            self.register_chain(chain_name, chain_instance, description)
            return func
        return decorator
    
    def deploy_chain(self, name: str, chain: Runnable, port: int = 8001):
        """
        Deploy a chain as standalone API server.
        
        Example:
            app.deploy_chain("summarizer", chain, port=8001)
        """
        if not CHAINS_V32_AVAILABLE:
            raise RuntimeError("Chain deployment requires v3.2 features")
        
        server = FluxServe(
            chain,
            name=name,
            enable_tracing=self.tracing_enabled
        )
        
        logger.info(f"🚀 Deploying chain '{name}' on port {port}")
        server.run(port=port)

    # ===== AGENT & TOOL MANAGEMENT =====
    
    def register(self, name: str, agent: Any):
        """Register an agent instance."""
        self.registry.add(name, agent)
        logger.info(f"✅ Agent '{name}' registered.")
    
    def tool(self, name: Optional[str] = None):
        """Decorator to register a tool function."""
        def decorator(func: Callable) -> Callable:
            tool_name = name if name is not None else func.__name__
            self.tool_registry.register(tool_name, func)
            logger.info(f"🛠️ Tool '{tool_name}' registered.")
            return func
        return decorator
    
    def agent(self, name: Optional[str] = None, track_performance: bool = True):
        """Decorator to define and register an agent function."""
        def decorator(func: Callable) -> Callable:
            agent_name = name if name is not None else func.__name__
            
            if track_performance and self.performance_monitor:
                func = self.performance_monitor.track_performance(func, agent_name=agent_name)
            
            class _FluxDynamicAgent:
                async def run(self, **kwargs):
                    kwargs['tools'] = self._tool_registry
                    if self._memory_store:
                        kwargs['memory'] = self._memory_store
                    if self._rag_connector:
                        kwargs['rag'] = self._rag_connector
                    
                    if asyncio.iscoroutinefunction(func):
                        return await func(**kwargs)
                    else:
                        return func(**kwargs)
            
            agent_instance = _FluxDynamicAgent()
            agent_instance._tool_registry = self.tool_registry
            agent_instance._memory_store = self.memory_store
            agent_instance._rag_connector = self.rag_connector
            
            self.register(agent_name, agent_instance)
            logger.info(f"🤖 Agent '{agent_name}' registered.")
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


# ===== CLI ENTRY POINT =====
def main():
    """CLI command entry point: `flux run [--reload] <file>`"""
    parser = argparse.ArgumentParser(prog='flux', description="FluxGraph CLI Runner")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    run_parser = subparsers.add_parser('run', help='Run a FluxGraph application file')
    run_parser.add_argument('file', help="Path to the Python file containing the FluxApp instance")
    run_parser.add_argument('--reload', action='store_true', help="Enable auto-reload on file changes")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command != 'run':
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)
    
    file_arg = args.file
    reload_flag = args.reload
    
    import importlib.util
    import pathlib
    
    file_path = pathlib.Path(file_arg).resolve()
    if not file_path.exists():
        print(f"❌ File '{file_arg}' not found.")
        sys.exit(1)
    
    logger.info(f"📦 Loading application from '{file_arg}'...")
    spec = importlib.util.spec_from_file_location("user_app", str(file_path))
    if spec is None or spec.loader is None:
        print(f"❌ Could not load module spec for '{file_arg}'.")
        sys.exit(1)
    
    user_module = importlib.util.module_from_spec(spec)
    sys.modules["user_app"] = user_module
    
    try:
        spec.loader.exec_module(user_module)
        logger.info("✅ Application file loaded successfully.")
    except Exception as e:
        print(f"❌ Error executing '{file_arg}': {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("🔍 Searching for FluxApp instance named 'app'...")
    app_instance = getattr(user_module, 'app', None)
    
    if app_instance is None:
        print("❌ No variable named 'app' found in the specified file.")
        sys.exit(1)
    
    if not isinstance(app_instance, FluxApp):
        print(f"❌ The 'app' variable found is not an instance of FluxApp. Type: {type(app_instance)}")
        sys.exit(1)
    
    logger.info("✅ FluxApp instance 'app' found.")
    
    reload_msg = " (with auto-reload)" if reload_flag else ""
    print(f"🚀 Starting FluxGraph app defined in '{file_arg}'{reload_msg}...")
    
    try:
        app_instance.run(host="127.0.0.1", port=8000, reload=reload_flag)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user.")
        logger.info("🛑 Server shutdown requested by user (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"❌ Failed to start the FluxGraph app: {e}", exc_info=True)
        print(f"❌ Failed to start the app: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
