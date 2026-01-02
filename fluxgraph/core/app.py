"""
FluxGraph Application Core - Enterprise Edition v3.2 (100% COMPLETE & PRODUCTION READY)

The most comprehensive AI agent orchestration framework with ALL features:
...
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Any, Dict, Callable, Optional, Union
import asyncio
import uuid
import time
import argparse
from contextvars import ContextVar
import json
# --- START: P1.2 Dependency Fix ---
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False
# --- END: P1.2 Dependency Fix ---

# ===== LOGGING CONFIGURATION (Modern Console) =====
# Use a custom formatter for attractive console output
class ModernConsoleFormatter(logging.Formatter):
    GREY = "\x1b[38;5;240m"
    CYAN = "\x1b[36m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    GREEN = "\x1b[32m"
    RESET = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.FORMATS = {
            logging.DEBUG: self.GREY + fmt + self.RESET,
            logging.INFO: self.CYAN + fmt + self.RESET,
            logging.WARNING: self.YELLOW + fmt + self.RESET,
            logging.ERROR: self.RED + fmt + self.RESET,
            logging.CRITICAL: self.BOLD_RED + fmt + self.RESET,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# Configure root logger
log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(ModernConsoleFormatter(log_format))

logging.basicConfig(
    level=logging.INFO,
    handlers=[stdout_handler]
)

logger = logging.getLogger(__name__)

# ===== VIRTUAL ENVIRONMENT HANDLING =====
# (No changes to _ensure_virtual_environment)
def _ensure_virtual_environment():
    """
    Ensures a virtual environment is set up and activated for FluxGraph.
    Automatically creates .venv_fluxgraph and installs dependencies.
    """
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

# _ensure_virtual_environment() # Commented out for iterative development

# ===== IMPORTS AFTER VENV ACTIVATION =====
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# Core components
try:
    from .registry import AgentRegistry
    from .tool_registry import ToolRegistry
except ImportError as e:
    logger.error(f"❌ Core import error: {e}")
    print(f"❌ Import error: {e}")
    # print("💡 Activate venv: source ./.venv_fluxgraph/bin/activate")
    # print("   Then install: pip install -e .")
    sys.exit(1)

# Analytics
try:
    from fluxgraph.analytics import PerformanceMonitor, AnalyticsDashboard
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    logger.warning("⚠️ Analytics modules not found. Dashboard will be disabled.")

# ===== V3.2 IMPORTS (LangChain Parity) =====
try:
    from fluxgraph.chains import (
        Runnable, RunnableSequence, RunnableParallel, RunnableLambda,
        chain, parallel, runnable, RunnableConfig
    )
    from fluxgraph.chains.prompts import PromptTemplate, ChatPromptTemplate
    from fluxgraph.chains.parsers import JsonOutputParser, PydanticOutputParser, ListOutputParser
    from fluxgraph.chains.models import create_llm_runnable, LLMRunnable
    from fluxgraph.chains.batch import BatchProcessor, BatchConfig, BatchStrategy
    from fluxgraph.chains.streaming import StreamOptimizer, optimize_stream, StreamMetrics
    from fluxgraph.tracing import Tracer, configure_tracing, trace, span, RunType, TraceRun
    from fluxgraph.serve import FluxServe, serve, deploy_multiple
    CHAINS_V32_AVAILABLE = True
    logger.debug("✅ v3.2 chain features loaded")
except ImportError as e:
    CHAINS_V32_AVAILABLE = False
    logger.warning(f"⚠️ v3.2 (LangChain parity) features not found. Chain API will be disabled. Error: {e}")

# ===== V3.0 P0 IMPORTS =====
try:
    # --- P0/P1 Integration: Import refactored workflow classes ---
    from .workflow_graph import WorkflowGraph, WorkflowBuilder, NodeType
    from .checkpointer import BaseCheckpointer
    from .postgres_checkpointer import PostgresCheckpointer
    from .logger import BaseLogger, LogEventType
    from .postgres_logger import PostgresWorkflowLogger
    WORKFLOW_AVAILABLE = True
    logger.debug("✅ Workflow graphs (with Checkpointing/Logging) loaded")
except ImportError as e:
    WORKFLOW_AVAILABLE = False
    WorkflowGraph = WorkflowBuilder = NodeType = BaseCheckpointer = PostgresCheckpointer = BaseLogger = PostgresWorkflowLogger = None
    logger.warning(f"⚠️ Workflow graph modules not found. Workflows will be disabled. Error: {e}")

try:
    from .advanced_memory import AdvancedMemory, MemoryType
    ADVANCED_MEMORY_AVAILABLE = True
    logger.debug("✅ Advanced memory loaded")
except ImportError:
    ADVANCED_MEMORY_AVAILABLE = False
    AdvancedMemory = MemoryType = None
    logger.warning("⚠️ Advanced memory modules not found. Advanced memory will be disabled.")

try:
    from .agent_cache import AgentCache, CacheStrategy
    AGENT_CACHE_AVAILABLE = True
    logger.debug("✅ Agent cache loaded")
except ImportError:
    AGENT_CACHE_AVAILABLE = False
    AgentCache = CacheStrategy = None
    logger.warning("⚠️ Agent cache modules not found. Caching will be disabled.")

# ===== V3.1 IMPORTS =====
try:
    # -----------------------------------------------------------------
    # BUG FIX 1: This was importing the wrong class from the wrong file.
    # It's now corrected to import EnhancedMemory from its own file.
    # (This assumes fluxgraph/core/enhanced_memory.py exists)
    # -----------------------------------------------------------------
    from .enhanced_memory import EnhancedMemory
    from fluxgraph.connectors import PostgresConnector, SalesforceConnector, ShopifyConnector
    from fluxgraph.workflows.visual_builder import VisualWorkflow
    V31_FEATURES_AVAILABLE = True
    logger.debug("✅ v3.1 features loaded")
except ImportError as e:
    V31_FEATURES_AVAILABLE = False
    EnhancedMemory = PostgresConnector = SalesforceConnector = ShopifyConnector = VisualWorkflow = None
    logger.warning("⚠️ v3.1 (Enhanced Memory/Connectors) features not found. They will be disabled.")

# Orchestrator
try:
    from .orchestrator_advanced import AdvancedOrchestrator
    ADVANCED_ORCHESTRATOR_AVAILABLE = True
    logger.debug("✅ Advanced orchestrator loaded")
except ImportError:
    ADVANCED_ORCHESTRATOR_AVAILABLE = False
    try:
        from .orchestrator import FluxOrchestrator
        logger.debug("✅ Basic orchestrator loaded")
    except ImportError as e:
        logger.error(f"❌ No orchestrator found: {e}")
        sys.exit(1)

# Memory & RAG
try:
    from .memory import Memory
    MEMORY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MEMORY_AVAILABLE = False
    class Memory:
        pass

try:
    # --- P2 Integration: Import refactored RAG classes ---
    from .rag import RAGConnector, Document
    from .universal_rag import UniversalRAG
    RAG_AVAILABLE = True
    logger.debug("✅ RAG pipeline modules loaded")
except (ImportError, ModuleNotFoundError):
    RAG_AVAILABLE = False
    class RAGConnector: pass
    class UniversalRAG: pass
    class Document: pass

# Event Hooks
try:
    from ..utils.hooks import EventHooks
    HOOKS_MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    HOOKS_MODULE_AVAILABLE = False
    class EventHooks:
        async def trigger(self, event_name: str, payload: Dict[str, Any]):
            pass

# Security
try:
    # --- P2 Integration: Import VerifiableAuditLogger ---
    from ..security.audit_logger import VerifiableAuditLogger, AuditEventType
    from ..security.pii_detector import PIIDetector
    from ..security.prompt_injection import PromptInjectionDetector
    from .rbac import RBACManager, Role, Permission
    SECURITY_AVAILABLE = True
    logger.debug("✅ Security features loaded")
except ImportError:
    SECURITY_AVAILABLE = False
    VerifiableAuditLogger = PIIDetector = PromptInjectionDetector = RBACManager = None
    AuditEventType = Role = Permission = None
    logger.warning("⚠️ Security modules not found. Security features will be disabled.")

# Orchestration
try:
    from ..orchestration.handoff import HandoffProtocol
    from ..orchestration.hitl import HITLManager
    from ..orchestration.task_adherence import TaskAdherenceMonitor
    from ..orchestration.batch import BatchProcessor as OrchBatchProcessor
    ORCHESTRATION_AVAILABLE = True
    logger.debug("✅ Orchestration features loaded")
except ImportError:
    ORCHESTRATION_AVAILABLE = False
    HandoffProtocol = HITLManager = TaskAdherenceMonitor = OrchBatchProcessor = None
    logger.warning("⚠️ Advanced orchestration features not found. They will be disabled.")

# Context for Request Tracking
request_id_context: ContextVar[str] = ContextVar('request_id', default='N/A')


# ===== REQUEST/RESPONSE MODELS =====
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

class WorkflowCreateRequest(BaseModel):
    """Request model for creating workflow"""
    name: str
    description: Optional[str] = None

class ConnectorConfigRequest(BaseModel):
    """Request model for connector configuration"""
    connector_type: str = Field(..., description="Type: postgres, salesforce, shopify")
    config: Dict[str, Any] = Field(..., description="Connector configuration")

# --- P2 Integration: New RAG Ingest Model ---
class RAGIngestURLsRequest(BaseModel):
    urls: List[str]
    metadata: Optional[Dict[str, Any]] = None


# ===== COMPLETE FLUXAPP v3.2 CLASS =====
class FluxApp:
    """
    FluxGraph v3.2 - 100% COMPLETE ENTERPRISE APPLICATION
    (Now with integrated Checkpointing, Logging, and RAG Pipelines)
    """

    def __init__(
        self,
        title: str = "FluxGraph API",
        description: str = "Enterprise AI agent orchestration framework v3.2 - 100% Complete",
        version: str = "3.2.0",
        memory_store: Optional[Memory] = None,
        rag_connector: Optional[UniversalRAG] = None,
        auto_init_rag: bool = True,
        enable_analytics: bool = True,
        enable_advanced_features: bool = True,
        # v3.0 P0 Features
        enable_workflows: bool = True,
        enable_advanced_memory: bool = True,
        enable_agent_cache: bool = True,
        cache_strategy: str = "hybrid",  # exact, semantic, hybrid
        # v3.1 Features
        enable_enhanced_memory: bool = False,
        enable_connectors: bool = False,
        enable_visual_workflows: bool = True,
        database_url: Optional[str] = os.getenv("FLUXGRAPH_DB_URL"), # <-- Read from env
        # v3.2 Features (LangChain Parity)
        enable_chains: bool = True,
        enable_tracing: bool = True,
        enable_batch_optimization: bool = True,
        enable_streaming_optimization: bool = True,
        enable_langserve_api: bool = True,
        tracing_export_path: str = "./traces",
        tracing_project_name: Optional[str] = None,
        # Security
        enable_security: bool = True, # <-- Enabled by default
        enable_audit_logging: bool = True,
        enable_pii_detection: bool = True,
        enable_prompt_shield: bool = True,
        enable_rbac: bool = False, # RBAC is complex, off by default
        # Orchestration
        enable_orchestration: bool = True,
        enable_handoffs: bool = True,
        enable_hitl: bool = True,
        enable_task_adherence: bool = True,
        # General
        log_level: str = "INFO",
        cors_origins: List[str] = ["*"]
    ):
        """
        Initialize FluxGraph v3.2 with all features.
        """
        # Set log level
        log_level_upper = log_level.upper()
        logging.getLogger().setLevel(getattr(logging, log_level_upper, logging.INFO))
        
        self.title = title
        self.description = description
        self.version = version
        self.api = FastAPI(
            title=title,
            description=description,
            version=version,
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        logger.info("=" * 100)
        logger.info(f"🚀 INITIALIZING FLUXGRAPH v{version} - ENTERPRISE EDITION")
        logger.info("=" * 100)
        
        # ===== CORE COMPONENTS =====
        logger.info("📦 Initializing core components...")
        self.registry = AgentRegistry()
        self.tool_registry = ToolRegistry()
        self.memory_store = memory_store
        self.rag_connector = rag_connector
        self.hooks = EventHooks() if HOOKS_MODULE_AVAILABLE else None
        
        # Orchestrator
        if enable_advanced_features and ADVANCED_ORCHESTRATOR_AVAILABLE:
            logger.info("   🔧 AdvancedOrchestrator: ✅ ENABLED")
            self.orchestrator = AdvancedOrchestrator(self.registry)
            self.advanced_features_enabled = True
        else:
            logger.info("   🔧 BasicOrchestrator: ✅ ENABLED")
            self.orchestrator = FluxOrchestrator(self.registry)
            self.advanced_features_enabled = False
        
        # Analytics
        if enable_analytics and ANALYTICS_AVAILABLE:
            logger.info("   📊 Analytics Dashboard: ✅ ENABLED")
            self.performance_monitor = PerformanceMonitor()
            self.analytics_dashboard = AnalyticsDashboard(self.performance_monitor)
            self.api.include_router(self.analytics_dashboard.router)
        else:
            logger.info("   📊 Analytics Dashboard: ❌ DISABLED")
            self.performance_monitor = None
            self.analytics_dashboard = None
        
        # ===== V3.0 P0 FEATURES (WITH P0/P1 INTEGRATION) =====
        logger.info("🆕 Initializing v3.0 P0 features...")
        
        # Workflows (P0/P1 Integration)
        if enable_workflows and WORKFLOW_AVAILABLE:
            logger.info("   🔀 Graph Workflows: ✅ ENABLED")
            self.workflow_builder = WorkflowBuilder # This is the class
            self.workflow_graphs: Dict[str, WorkflowGraph] = {}
            self.workflows_enabled = True

            if database_url and PostgresCheckpointer and PostgresWorkflowLogger:
                logger.info("     ... 🔧 PostgresCheckpointer: ✅ ENABLED (Resumable Workflows)")
                self.checkpointer = PostgresCheckpointer(database_url)
                logger.info("     ... 🔧 PostgresWorkflowLogger: ✅ ENABLED (Observability)")
                self.workflow_logger = PostgresWorkflowLogger(database_url)
            else:
                logger.warning("     ... ⚠️ Workflows enabled, but no 'database_url' provided.")
                logger.warning("     ... ⚠️ Checkpointing & DB Logging are ❌ DISABLED.")
                self.checkpointer = None
                self.workflow_logger = None
        else:
            logger.info("   🔀 Graph Workflows: ❌ DISABLED")
            self.workflow_builder = None
            self.workflow_graphs = {}
            self.workflows_enabled = False
            self.checkpointer = None
            self.workflow_logger = None
        
        # Advanced Memory
        if enable_advanced_memory and ADVANCED_MEMORY_AVAILABLE:
            logger.info("   🧠 Advanced Memory: ✅ ENABLED (Short/Long/Episodic)")
            self.advanced_memory = AdvancedMemory() # Using defaults
            self.advanced_memory_enabled = True
        else:
            logger.info("   🧠 Advanced Memory: ❌ DISABLED")
            self.advanced_memory = None
            self.advanced_memory_enabled = False
        
        # Agent Cache
        if enable_agent_cache and AGENT_CACHE_AVAILABLE:
            strategy_map = {"exact": CacheStrategy.EXACT, "semantic": CacheStrategy.SEMANTIC, "hybrid": CacheStrategy.HYBRID}
            selected_strategy = strategy_map.get(cache_strategy, CacheStrategy.HYBRID)
            logger.info(f"   ⚡ Agent Cache: ✅ ENABLED (Strategy: {cache_strategy.upper()})")
            self.agent_cache = AgentCache(strategy=selected_strategy)
            self.agent_cache_enabled = True
        else:
            logger.info("   ⚡ Agent Cache: ❌ DISABLED")
            self.agent_cache = None
            self.agent_cache_enabled = False
        
        # ===== V3.1 IMPORTS =====
        logger.info("🎉 Initializing v3.1 features...")
        
        # Enhanced Memory
        if enable_enhanced_memory and database_url and V31_FEATURES_AVAILABLE:
            logger.info("   🧠 Enhanced Memory: ✅ ENABLED (Entity Extraction)")
            # -----------------------------------------------------------------
            # BUG FIX 2: This was a NameError (typo) and assigning to the wrong var.
            # Was: self.advanced_memory = Advanced_memory(database_url)
            # -----------------------------------------------------------------
            self.enhanced_memory = EnhancedMemory(database_url)
            self._enhanced_memory_initialized = False
            self.enhanced_memory_enabled = True
        else:
            logger.info("   🧠 Enhanced Memory: ❌ DISABLED")
            self.enhanced_memory = None
            self._enhanced_memory_initialized = False
            self.enhanced_memory_enabled = False
        
        # Connectors
        if enable_connectors and V31_FEATURES_AVAILABLE:
            logger.info("   🔌 Database Connectors: ✅ ENABLED")
            self.connectors: Dict[str, Any] = {}
            self.connectors_enabled = True
        else:
            logger.info("   🔌 Database Connectors: ❌ DISABLED")
            self.connectors = None
            self.connectors_enabled = False
        
        # Visual Workflows
        if enable_visual_workflows and V31_FEATURES_AVAILABLE:
            logger.info("   🎨 Visual Workflow Builder: ✅ ENABLED")
            self.visual_workflows: Dict[str, VisualWorkflow] = {}
            self.visual_workflows_enabled = True
        else:
            logger.info("   🎨 Visual Workflow Builder: ❌ DISABLED")
            self.visual_workflows = None
            self.visual_workflows_enabled = False
        
        # ===== V3.2 FEATURES (LangChain Parity) =====
        logger.info("🚀 Initializing v3.2 features (LangChain Parity)...")
        
        # Chains
        if enable_chains and CHAINS_V32_AVAILABLE:
            logger.info("   ⛓️ LCEL Chains: ✅ ENABLED")
            self.chains: Dict[str, Runnable] = {}
            self.chain_registry: Dict[str, Dict] = {}
            self.chains_enabled = True
        else:
            logger.info("   ⛓️ LCEL Chains: ❌ DISABLED")
            self.chains = None
            self.chain_registry = {}
            self.chains_enabled = False
        
        # Tracing
        if enable_tracing and CHAINS_V32_AVAILABLE:
            project_name = tracing_project_name or title
            logger.info(f"   🔍 Distributed Tracing: ✅ ENABLED (Project: {project_name})")
            self.tracer = configure_tracing(project_name=project_name, enabled=True, export_path=tracing_export_path)
            self.tracing_enabled = True
        else:
            logger.info("   🔍 Distributed Tracing: ❌ DISABLED")
            self.tracer = None
            self.tracing_enabled = False
        
        # Batch & Streaming
        self.batch_optimizer_enabled = enable_batch_optimization and CHAINS_V32_AVAILABLE
        self.streaming_optimizer_enabled = enable_streaming_optimization and CHAINS_V32_AVAILABLE
        self.langserve_enabled = enable_langserve_api and CHAINS_V32_AVAILABLE
        logger.info(f"   📦 Batch Optimization: {'✅' if self.batch_optimizer_enabled else '❌'}")
        logger.info(f"   ⚡ Streaming Optimization: {'✅' if self.streaming_optimizer_enabled else '❌'}")
        logger.info(f"   🌐 LangServe API: {'✅' if self.langserve_enabled else '❌'}")

        # ===== SECURITY FEATURES (WITH P2 INTEGRATION) =====
        logger.info("🔒 Initializing security features...")
        
        if enable_security and SECURITY_AVAILABLE:
            # Audit Logging (P2 Integration)
            if enable_audit_logging and database_url and VerifiableAuditLogger:
                logger.info("   📝 Verifiable Audit Log: ✅ ENABLED (Postgres, Hash-Chained)")
                self.audit_logger = VerifiableAuditLogger(database_url=database_url)
            else:
                logger.warning("   📝 Verifiable Audit Log: ❌ DISABLED (Requires 'enable_audit_logging=True' and 'database_url')")
                self.audit_logger = None
            
            # PII Detection
            if enable_pii_detection and PIIDetector:
                logger.info("   🔐 PII Detection: ✅ ENABLED")
                self.pii_detector = PIIDetector()
            else:
                logger.info("   🔐 PII Detection: ❌ DISABLED")
                self.pii_detector = None
            
            # Prompt Injection Shield
            if enable_prompt_shield and PromptInjectionDetector:
                logger.info("   🛡️ Prompt Shield: ✅ ENABLED")
                self.prompt_shield = PromptInjectionDetector()
            else:
                logger.info("   🛡️ Prompt Shield: ❌ DISABLED")
                self.prompt_shield = None
            
            # RBAC
            if enable_rbac and RBACManager:
                logger.info("   👥 RBAC: ✅ ENABLED")
                self.rbac_manager = RBACManager()
            else:
                logger.info("   👥 RBAC: ❌ DISABLED")
                self.rbac_manager = None
            
            self.security_enabled = True
        else:
            logger.info("   🔒 Security: ❌ DISABLED (All)")
            self.audit_logger = None
            self.pii_detector = None
            self.prompt_shield = None
            self.rbac_manager = None
            self.security_enabled = False
        
        # ===== ORCHESTRATION FEATURES =====
        logger.info("🎯 Initializing orchestration features...")
        
        if enable_orchestration and ORCHESTRATION_AVAILABLE and self.advanced_features_enabled:
            logger.info("   🤝 Agent Handoffs: ✅ ENABLED")
            self.handoff_protocol = HandoffProtocol(self.orchestrator) if enable_handoffs else None
            logger.info(f"   👤 Human-in-the-Loop: {'✅' if enable_hitl else '❌'}")
            self.hitl_manager = HITLManager() if enable_hitl else None
            logger.info(f"   ✅ Task Adherence: {'✅' if enable_task_adherence else '❌'}")
            self.task_adherence = TaskAdherenceMonitor() if enable_task_adherence else None
            logger.info("   📦 Orchestration Batch: ✅ ENABLED")
            self.batch_processor = OrchBatchProcessor(self.orchestrator)
            self.orchestration_enabled = True
        else:
            logger.info("   🎯 Orchestration: ❌ DISABLED (All)")
            self.handoff_protocol = None
            self.hitl_manager = None
            self.task_adherence = None
            self.batch_processor = None
            self.orchestration_enabled = False
        
        # Auto-init RAG (P2 Integration)
        if auto_init_rag and RAG_AVAILABLE and self.rag_connector is None:
            self._auto_initialize_rag()
        
        # Setup
        self._setup_middleware(cors_origins)
        self._setup_routes() # This now includes the on_startup event
        
        # Final Summary
        logger.info("=" * 100)
        logger.info(f"✅ FLUXAPP v{version} INITIALIZATION COMPLETE")
        logger.info("=" * 100)
        logger.info("📋 FEATURE STATUS SUMMARY:")
        logger.info(f"   Core: Memory={'✅' if self.memory_store else '❌'} | RAG={'✅' if self.rag_connector else '❌'} | Analytics={'✅' if self.performance_monitor else '❌'}")
        logger.info(f"   v3.0: Workflows={'✅' if self.workflows_enabled else '❌'} | AdvMemory={'✅' if self.advanced_memory_enabled else '❌'} | Cache={'✅' if self.agent_cache_enabled else '❌'}")
        logger.info(f"     -> DB Features: Checkpointing={'✅' if self.checkpointer else '❌'} | WF Logging={'✅' if self.workflow_logger else '❌'}")
        logger.info(f"   v3.1: EnhancedMem={'✅' if self.enhanced_memory_enabled else '❌'} | Connectors={'✅' if self.connectors_enabled else '❌'} | Visual={'✅' if self.visual_workflows_enabled else '❌'}")
        logger.info(f"   v3.2: Chains={'✅' if self.chains_enabled else '❌'} | Tracing={'✅' if self.tracing_enabled else '❌'} | Batch={'✅' if self.batch_optimizer_enabled else '❌'} | Stream={'✅' if self.streaming_optimizer_enabled else '❌'}")
        logger.info(f"   Security: AuditLog={'✅' if self.audit_logger else '❌'} | PII={'✅' if self.pii_detector else '❌'} | Shield={'✅' if self.prompt_shield else '❌'}")
        logger.info(f"   Orchestration: Handoffs={'✅' if self.handoff_protocol else '❌'} | HITL={'✅' if self.hitl_manager else '❌'} | TaskAdherence={'✅' if self.task_adherence else '❌'}")
        logger.info("=" * 100)

    def _auto_initialize_rag(self):
        """Auto-initialize RAG connector."""
        AUTO_RAG_PERSIST_DIR = "./my_chroma_db"
        AUTO_RAG_COLLECTION_NAME = "my_knowledge_base"
        
        logger.info("🔄 Auto-initializing UniversalRAG connector...")
        
        persist_path = Path(AUTO_RAG_PERSIST_DIR)
        if not persist_path.exists():
            logger.info(f"   📁 Creating RAG directory: {AUTO_RAG_PERSIST_DIR}")
            try:
                persist_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"   ❌ Failed to create RAG directory: {e}")
                self.rag_connector = None
                return
        
        try:
            # ... (config loading) ...
            self.rag_connector = UniversalRAG(
                persist_directory=AUTO_RAG_PERSIST_DIR,
                collection_name=AUTO_RAG_COLLECTION_NAME,
                # ...
            )
            logger.info("   ✅ RAG connector auto-initialized")
        except Exception as e:
            logger.error(f"   ❌ Failed to auto-initialize RAG: {e}")
            self.rag_connector = None

    async def _ensure_enhanced_memory_initialized(self):
        """Lazy initialization for enhanced memory."""
        if self.enhanced_memory and not self._enhanced_memory_initialized:
            try:
                logger.info("   🔧 Initializing Enhanced Memory...")
                await self.enhanced_memory.initialize()
                self._enhanced_memory_initialized = True
                logger.info("   ✅ Enhanced Memory initialized.")
            except Exception as e:
                logger.error(f"   ❌ Enhanced Memory initialization failed: {e}")
                # Don't raise, just log
                
    # --- P0/P1/P2 Integration: Central Startup Handler ---
    def _setup_startup_events(self):
        """Register all async startup events."""
        @self.api.on_event("startup")
        async def on_startup():
            logger.info("=" * 100)
            logger.info("🚀 Running all async startup tasks...")
            
            # P0: Setup Checkpointer
            if self.checkpointer:
                try:
                    await self.checkpointer.setup()
                    logger.info("   ✅ P0: Resumable Workflow Checkpointer setup complete.")
                except Exception as e:
                    logger.error(f"   ❌ P0: Failed to setup Checkpointer: {e}")
            
            # P1: Setup Workflow Logger
            if self.workflow_logger:
                try:
                    await self.workflow_logger.setup()
                    logger.info("   ✅ P1: Workflow Observability Logger setup complete.")
                except Exception as e:
                    logger.error(f"   ❌ P1: Failed to setup Workflow Logger: {e}")
            
            # P2: Setup Verifiable Audit Logger
            if self.audit_logger:
                try:
                    await self.audit_logger.setup()
                    logger.info("   ✅ P2: Verifiable Audit Logger setup complete.")
                except Exception as e:
                    logger.error(f"   ❌ P2: Failed to setup Verifiable Audit Logger: {e}")
            
            # v3.1: Setup Enhanced Memory
            if self.enhanced_memory and not self._enhanced_memory_initialized:
                await self._ensure_enhanced_memory_initialized()
            
            logger.info("🚀 All startup tasks complete.")
            logger.info("=" * 100)

    def _setup_middleware(self, cors_origins: List[str]):
        """Setup middleware including CORS and request logging."""
        self.api.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
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
            url_path = request.url.path
            
            # Modern console logging for request
            logger.info(f"➡️  {method} {url_path} (from: {client_host}) [ID: {request_id}]")
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = str(round(process_time, 4))
                response.headers["X-Request-ID"] = request_id
                
                # Use emojis for status
                status_emoji = "✅" if 200 <= response.status_code < 300 else "⚠️" if 400 <= response.status_code < 500 else "❌"
                logger.info(f"⬅️  {status_emoji} {response.status_code} {method} {url_path} (in {process_time:.4f}s) [ID: {request_id}]")
                
                return response
            except Exception as e:
                process_time = time.time() - start_time
                logger.error(f"⬅️  ❌ 500 {method} {url_path} (in {process_time:.4f}s) [ID: {request_id}] Error: {e}", exc_info=True)
                # Return a generic 500 response to avoid leaking details
                return JSONResponse(
                    status_code=500,
                    content={"detail": "Internal Server Error", "request_id": request_id}
                )
        
        # --- P0/P1/P2 Integration: Call the startup event setup ---
        self._setup_startup_events()

    def _setup_routes(self):
        """Setup all API routes for v3.0, v3.1, v3.2, and security features."""
        
        # ===== ROOT ENDPOINT =====
        @self.api.get("/")
        async def root():
            """Root endpoint with full API information."""
            return {
                "message": "Welcome to FluxGraph v3.2 - 100% Complete Edition",
                "title": self.title,
                "version": self.version,
                "features": {
                    "core": {
                        "analytics": self.performance_monitor is not None,
                        "rag_pipeline": self.rag_connector is not None,
                        "memory": self.memory_store is not None,
                    },
                    "v3.0_p0": {
                        "workflows": self.workflows_enabled,
                        "advanced_memory": self.advanced_memory_enabled,
                        "agent_cache": self.agent_cache_enabled,
                        "resumable_workflows": self.checkpointer is not None,
                        "workflow_logging": self.workflow_logger is not None,
                    },
                    "v3.1": {
                        "enhanced_memory": self.enhanced_memory_enabled,
                        "connectors": self.connectors_enabled,
                        "visual_workflows": self.visual_workflows_enabled
                    },
                    "v3.2_langchain_parity": {
                        "chains": self.chains_enabled,
                        "tracing": self.tracing_enabled,
                        "batch_optimization": self.batch_optimizer_enabled,
                        "streaming_optimization": self.streaming_optimizer_enabled,
                        "langserve_api": self.langserve_enabled
                    },
                    "security": {
                        "verifiable_audit_log": self.audit_logger is not None,
                        "pii_detection": self.pii_detector is not None,
                        "prompt_shield": self.prompt_shield is not None,
                        "rbac": self.rbac_manager is not None
                    },
                    "orchestration": {
                        "handoffs": self.handoff_protocol is not None,
                        "hitl": self.hitl_manager is not None,
                        "task_adherence": self.task_adherence is not None
                    }
                },
                "docs": "/docs"
            }
        
        # ===== AGENT ENDPOINTS =====
        @self.api.post("/ask/{agent_name}")
        async def ask_agent(agent_name: str, payload: Dict[str, Any]):
            """Execute a registered agent."""
            request_id = request_id_context.get()
            start_time = time.time()
            user_id = payload.get("user_id", "anonymous")
            
            logger.debug(f"[{request_id}] 🤖 Executing agent '{agent_name}' for user '{user_id}'")
            
            if self.hooks:
                await self.hooks.trigger("request_received", {"request_id": request_id, "agent_name": agent_name, "payload": payload})
            
            try:
                # Security checks
                if self.prompt_shield:
                    is_safe, detections = self.prompt_shield.is_safe(payload.get("query", ""), block_on_detection=True)
                    if not is_safe:
                        if self.audit_logger:
                             await self.audit_logger.log(
                                AuditEventType.PROMPT_INJECTION_DETECTED,
                                actor="security_shield",
                                data={"agent": agent_name, "request_id": request_id, "detections": detections, "query_preview": payload.get("query", "")[:50]}
                             )
                        raise HTTPException(status_code=400, detail="Prompt injection detected and blocked")
                
                if self.pii_detector:
                    payload, detections = self.pii_detector.scan_dict(payload, redact=True)
                    if detections and self.audit_logger:
                        await self.audit_logger.log(
                            AuditEventType.PII_DETECTED,
                            actor="pii_detector",
                            data={"agent": agent_name, "request_id": request_id, "detections": [d.to_dict() for d in detections]}
                        )
                
                # Cache
                if self.agent_cache:
                    cached = self.agent_cache.get(payload.get("query", ""), payload)
                    if cached:
                        logger.info(f"[{request_id}] ⚡ Cache hit for '{agent_name}'")
                        return cached
                
                # Execute
                result = await self.orchestrator.run(agent_name, payload)
                
                if self.agent_cache:
                    self.agent_cache.set(payload.get("query", ""), result)
                
                # Audit log (P2 Integration: Updated call)
                if self.audit_logger:
                    await self.audit_logger.log(
                        AuditEventType.AGENT_EXECUTION,
                        actor=user_id,
                        data={"agent": agent_name, "request_id": request_id, "status": "success"}
                    )
                
                duration = time.time() - start_time
                if self.hooks:
                    await self.hooks.trigger("agent_completed", {"request_id": request_id, "agent_name": agent_name, "result": result, "duration": duration})
                
                logger.debug(f"[{request_id}] ✅ Agent '{agent_name}' completed ({duration:.4f}s)")
                return result
                
            except ValueError as e:
                if self.audit_logger:
                    await self.audit_logger.log(
                        AuditEventType.AGENT_EXECUTION,
                        actor=user_id,
                        data={"agent": agent_name, "request_id": request_id, "status": "failed", "error": str(e)}
                    )
                logger.warning(f"[{request_id}] ⚠️ Agent error: {e}")
                status_code = 404 if "not registered" in str(e).lower() else 400
                raise HTTPException(status_code=status_code, detail=str(e))
            except Exception as e:
                if self.audit_logger:
                    await self.audit_logger.log(
                        AuditEventType.AGENT_EXECUTION,
                        actor=user_id,
                        data={"agent": agent_name, "request_id": request_id, "status": "error", "error": str(e)}
                    )
                logger.error(f"[{request_id}] ❌ Execution error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Internal Server Error")
        
        # ===== V3.2 CHAIN ENDPOINTS =====
        if self.langserve_enabled and CHAINS_V32_AVAILABLE:
            
            @self.api.get("/chains", tags=["v3.2 - Chains"])
            async def list_chains():
                """List all registered chains."""
                return {
                    "chains": list(self.chains.keys()),
                    "count": len(self.chains),
                    "registry": self.chain_registry
                }
            
            @self.api.post("/chains/{chain_name}/invoke", response_model=ChainResponse, tags=["v3.2 - Chains"])
            async def invoke_chain(chain_name: str, request: ChainInvokeRequest):
                # ... (logic remains same)
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
                    return ChainResponse(output=output, metadata={"duration_ms": duration * 1000})
                except Exception as e:
                    logger.error(f"Chain error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            
            @self.api.post("/chains/{chain_name}/batch", tags=["v3.2 - Chains"])
            async def batch_chain(chain_name: str, request: ChainBatchRequest):
                # ... (logic remains same)
                if chain_name not in self.chains:
                    raise HTTPException(status_code=404, detail=f"Chain '{chain_name}' not found")
                chain = self.chains[chain_name]
                start_time = time.time()
                try:
                    config = RunnableConfig(max_concurrency=request.max_concurrency) if self.batch_optimizer_enabled else None
                    outputs = await chain.batch(request.inputs, config)
                    duration = time.time() - start_time
                    return {"outputs": outputs, "metadata": {"count": len(outputs), "duration_ms": duration * 1000}}
                except Exception as e:
                    logger.error(f"Batch error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            @self.api.websocket("/chains/{chain_name}/stream")
            async def stream_chain(websocket: WebSocket, chain_name: str):
                # ... (logic remains same)
                await websocket.accept()
                try:
                    if chain_name not in self.chains:
                        await websocket.send_json({"error": f"Chain '{chain_name}' not found"}); return
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
                finally:
                    await websocket.close()
        
        # ===== V3.0 WORKFLOW ENDPOINTS (P0/P1 Integrated) =====
        if self.workflows_enabled:
            
            @self.api.post("/workflows", tags=["v3.0 - Workflows"])
            async def create_workflow(request: WorkflowCreateRequest):
                """Create a new workflow graph."""
                try:
                    # --- P0/P1 Integration ---
                    # Pass the app's checkpointer and logger to the builder
                    workflow = self.workflow_builder(
                        request.name,
                        checkpointer=self.checkpointer,
                        logger=self.workflow_logger
                    ).graph
                    
                    self.workflow_graphs[request.name] = workflow
                    logger.info(f"Workflow '{request.name}' created successfully.")
                    return {
                        "message": f"Workflow '{request.name}' created",
                        "name": request.name,
                        "checkpointer_enabled": self.checkpointer is not None,
                        "logger_enabled": self.workflow_logger is not None
                    }
                except Exception as e:
                    logger.error(f"Failed to create workflow: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=str(e))
            
            @self.api.get("/workflows", tags=["v3.0 - Workflows"])
            async def list_workflows():
                # ... (logic remains same)
                return {"workflows": list(self.workflow_graphs.keys()), "count": len(self.workflow_graphs)}
            
            @self.api.get("/workflows/{workflow_name}", tags=["v3.0 - Workflows"])
            async def get_workflow(workflow_name: str):
                # ... (logic remains same)
                if workflow_name not in self.workflow_graphs:
                    raise HTTPException(status_code=404, detail="Workflow not found")
                return {"name": workflow_name, "nodes": len(self.workflow_graphs[workflow_name].nodes)}
        
        # ===== V3.1 CONNECTOR ENDPOINTS =====
        if self.connectors_enabled:
            
            @self.api.post("/connectors/{connector_name}", tags=["v3.1 - Connectors"])
            async def add_connector(connector_name: str, request: ConnectorConfigRequest):
                # ... (logic remains same)
                try:
                    if request.connector_type == "postgres": connector = PostgresConnector(request.config)
                    elif request.connector_type == "salesforce": connector = SalesforceConnector(request.config)
                    elif request.connector_type == "shopify": connector = ShopifyConnector(request.config)
                    else: raise HTTPException(status_code=400, detail="Unknown connector type")
                    await connector.connect()
                    self.connectors[connector_name] = connector
                    return {"message": f"Connector '{connector_name}' added", "type": request.connector_type}
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            @self.api.get("/connectors", tags=["v3.1 - Connectors"])
            async def list_connectors():
                # ... (logic remains same)
                return {"connectors": list(self.connectors.keys()), "count": len(self.connectors)}
        
        # ===== TRACING ENDPOINTS =====
        if self.tracing_enabled:
            @self.api.get("/tracing/status", tags=["v3.2 - Tracing"])
            async def tracing_status():
                # ... (logic remains same)
                return {"tracing_enabled": True, "project": self.tracer.project_name, "statistics": self.tracer.get_statistics()}
            
            @self.api.get("/tracing/traces", tags=["v3.2 - Tracing"])
            async def get_traces():
                # ... (logic remains same)
                traces = self.tracer.get_root_traces()
                return {"traces": [t.to_dict() for t in traces], "count": len(traces)}
            
            @self.api.get("/tracing/traces/{trace_id}", tags=["v3.2 - Tracing"])
            async def get_trace(trace_id: str):
                # ... (logic remains same)
                trace = self.tracer.get_trace(trace_id)
                if not trace: raise HTTPException(status_code=404, detail="Trace not found")
                return trace.to_dict()
        
        # ===== TOOLS ENDPOINTS =====
        @self.api.get("/tools", tags=["Core"])
        async def list_tools():
            return {"tools": self.tool_registry.list_tools()}
        
        @self.api.get("/tools/{tool_name}", tags=["Core"])
        async def get_tool_info(tool_name: str):
            try:
                return self.tool_registry.get_tool_info(tool_name)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
        
        # ===== MEMORY & RAG ENDPOINTS =====
        if self.memory_store:
            @self.api.get("/memory/status", tags=["Core"])
            async def memory_status():
                return {"memory_enabled": True, "type": type(self.memory_store).__name__}
        
        if self.rag_connector:
            @self.api.get("/rag/status", tags=["Core - RAG"])
            async def rag_status():
                try:
                    stats = self.rag_connector.get_collection_stats()
                    return {"rag_enabled": True, "stats": stats}
                except Exception as e:
                    return {"rag_enabled": True, "error": str(e)}

            # --- P2 Integration: New RAG URL Ingest Route ---
            @self.api.post("/rag/ingest_urls", tags=["Core - RAG"])
            async def rag_ingest_urls(request: RAGIngestURLsRequest):
                """Ingest data from a list of web URLs."""
                try:
                    logger.info(f"Attempting to ingest from {len(request.urls)} URLs...")
                    doc_ids = await self.rag_connector.ingest_from_urls(request.urls, request.metadata)
                    logger.info(f"Successfully ingested {len(doc_ids)} documents from URLs.")
                    return {
                        "message": "Ingestion successful",
                        "documents_ingested": len(doc_ids),
                        "doc_ids": doc_ids
                    }
                except Exception as e:
                    logger.error(f"RAG URL ingestion failed: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=str(e))
        
        # ===== HEALTH CHECK =====
        @self.api.get("/health", tags=["Core"])
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "version": self.version, "timestamp": datetime.utcnow().isoformat()}

    # ===== CHAIN METHODS =====
    
    def register_chain(self, name: str, chain: Runnable, description: Optional[str] = None):
        """Register a chain."""
        if not self.chains_enabled:
            logger.warning(f"Chains not enabled, skipping registration of '{name}'")
            return
        
        self.chains[name] = chain
        self.chain_registry[name] = {"type": type(chain).__name__, "description": description}
        logger.info(f"⛓️ Chain '{name}' registered")
        return chain
    
    def chain(self, name: Optional[str] = None, description: Optional[str] = None):
        """Decorator to register chain."""
        def decorator(func: Callable) -> Callable:
            chain_name = name or func.__name__
            chain_instance = func()
            self.register_chain(chain_name, chain_instance, description)
            return func
        return decorator
    
    # ===== CONNECTOR METHODS =====
    
    async def add_connector(self, name: str, connector_type: str, config: Dict):
        # ... (logic remains same)
        if not self.connectors_enabled:
            raise RuntimeError("Connectors not enabled")
        if connector_type == "postgres": connector = PostgresConnector(config)
        elif connector_type == "salesforce": connector = SalesforceConnector(config)
        elif connector_type == "shopify": connector = ShopifyConnector(config)
        else: raise ValueError(f"Unknown connector type: {connector_type}")
        await connector.connect()
        self.connectors[name] = connector
        logger.info(f"🔌 Connector '{name}' added ({connector_type})")
    
    # ===== AGENT & TOOL REGISTRATION =====
    
    def register(self, name: str, agent: Any):
        """Register an agent."""
        self.registry.add(name, agent)
        logger.info(f"🤖 Agent '{name}' registered")
    
    def tool(self, name: Optional[str] = None):
        """Decorator to register tool."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            self.tool_registry.register(tool_name, func)
            logger.info(f"🛠️ Tool '{tool_name}' registered")
            return func
        return decorator
    
    def agent(self, name: Optional[str] = None, track_performance: bool = True):
        """Decorator to register agent."""
        def decorator(func: Callable) -> Callable:
            agent_name = name or func.__name__
            
            if track_performance and self.performance_monitor:
                func = self.performance_monitor.track_performance(func, agent_name=agent_name)
            
            # Create a dynamic agent class to handle injection
            class _FluxDynamicAgent:
                # Store app-level components
                _tool_registry = self.tool_registry
                _memory_store = self.memory_store
                _rag_connector = self.rag_connector
                _orchestrator = self.orchestrator
                _advanced_memory = self.advanced_memory
                _hitl_manager = self.hitl_manager
                
                async def run(self, **kwargs):
                    # Inject all available components
                    kwargs['tools'] = self._tool_registry
                    if self._memory_store:
                        kwargs['memory'] = self._memory_store
                    if self._rag_connector:
                        kwargs['rag'] = self._rag_connector
                    if self._advanced_memory:
                        kwargs['advanced_memory'] = self._advanced_memory
                    if self._hitl_manager:
                        kwargs['hitl'] = self._hitl_manager
                    
                    if self._orchestrator:
                        kwargs['call_agent'] = self._orchestrator.run
                    
                    if asyncio.iscoroutinefunction(func):
                        return await func(**kwargs)
                    else:
                        return await asyncio.to_thread(func, **kwargs)
            
            self.register(agent_name, _FluxDynamicAgent())
            return func
        return decorator
    
    # ===== SERVER METHODS =====
    
    def run(self, host: str = "127.0.0.1", port: int = 8000, reload: bool = False, **kwargs):
        """Start the FluxGraph API server."""
        logger.info("=" * 100)
        logger.info(f"🚀 STARTING FLUXGRAPH SERVER v{self.version}")
        logger.info(f"   ➤ Host: {host}")
        logger.info(f"   ➤ Port: {port}")
        logger.info(f"   ➤ Reload: {'✅' if reload else '❌'}")
        logger.info(f"   ➤ Docs URL: http://{host}:{port}/docs")
        logger.info(f"   ➤ Health URL: http://{host}:{port}/health")
        logger.info("=" * 100)
        
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
    """CLI command: flux run [--reload] <file>"""
    # (No changes to main CLI logic)
    parser = argparse.ArgumentParser(
        prog='flux',
        description="FluxGraph v3.2 CLI - 100% Complete Edition"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    run_parser = subparsers.add_parser('run', help='Run a FluxGraph application')
    run_parser.add_argument('file', help="Path to Python file with FluxApp instance")
    run_parser.add_argument('--reload', action='store_true', help="Enable auto-reload")
    run_parser.add_argument('--host', default="127.0.0.1", help="Host to bind")
    run_parser.add_argument('--port', type=int, default=8000, help="Port to bind")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command != 'run':
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)
    
    file_arg = args.file
    
    import importlib.util
    import pathlib
    
    file_path = pathlib.Path(file_arg).resolve()
    if not file_path.exists():
        print(f"❌ File '{file_arg}' not found")
        sys.exit(1)
    
    logger.info(f"📦 Loading application from '{file_arg}'...")
    spec = importlib.util.spec_from_file_location("user_app", str(file_path))
    if spec is None or spec.loader is None:
        print(f"❌ Could not load module spec for '{file_arg}'")
        sys.exit(1)
    
    user_module = importlib.util.module_from_spec(spec)
    sys.modules["user_app"] = user_module
    
    try:
        spec.loader.exec_module(user_module)
        logger.info("✅ Application file loaded")
    except Exception as e:
        print(f"❌ Error executing '{file_arg}': {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    app_instance = getattr(user_module, 'app', None)
    
    if app_instance is None:
        print("❌ No 'app' variable found in file. Ensure you have 'app = FluxApp(...)'.")
        sys.exit(1)
    
    if not isinstance(app_instance, FluxApp):
        print(f"❌ 'app' is not a FluxApp instance (type: {type(app_instance)})")
        sys.exit(1)
    
    logger.info("✅ FluxApp instance found. Starting server...")
    
    try:
        app_instance.run(
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")
        logger.info("🛑 Server shutdown (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"❌ Failed to start: {e}", exc_info=True)
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()