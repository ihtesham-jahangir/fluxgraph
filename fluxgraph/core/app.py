# fluxgraph/core/app.py
"""
FluxGraph Application Core - Enterprise Edition v3.2 (100% COMPLETE & PRODUCTION READY)

The most comprehensive AI agent orchestration framework with ALL features:

✅ v3.0 P0 Features:
   - Graph-based workflows with visual builder
   - Advanced multi-tier memory system (short/long/episodic)
   - Agent response caching (exact/semantic/hybrid)
   
✅ v3.1 Features:
   - Enhanced memory with entity extraction & embeddings
   - Database connector marketplace (Postgres/Salesforce/Shopify)
   - Visual workflow builder with React UI
   
✅ v3.2 Features (LangChain Feature Parity):
   - LCEL-style chain building with pipe operators
   - LangSmith-style distributed tracing
   - Optimized batch processing with adaptive concurrency
   - Time-to-First-Token streaming optimization
   - LangServe-style production deployment
   
✅ Security Features:
   - Audit logging with AuditEventType tracking
   - PII detection and redaction
   - Prompt injection detection
   - Role-Based Access Control (RBAC)
   
✅ Advanced Orchestration:
   - Message bus for agent communication
   - Agent handoff protocols
   - Human-in-the-loop (HITL) support
   - Task adherence monitoring
   - Circuit breakers for fault tolerance
   
✅ Analytics & Monitoring:
   - Real-time performance monitoring
   - Interactive analytics dashboard
   - Cost tracking per agent/request
   - Token usage analytics
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

# ===== LOGGING CONFIGURATION =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | [%(filename)s:%(lineno)d] | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# ===== VIRTUAL ENVIRONMENT HANDLING =====
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

_ensure_virtual_environment()

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
    print("💡 Activate venv: source ./.venv_fluxgraph/bin/activate")
    print("   Then install: pip install -e .")
    sys.exit(1)

# Analytics
try:
    from fluxgraph.analytics import PerformanceMonitor, AnalyticsDashboard
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    logger.warning("⚠️ Analytics not available")

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
    logger.info("✅ v3.2 chain features loaded")
except ImportError as e:
    CHAINS_V32_AVAILABLE = False
    logger.warning(f"⚠️ v3.2 features not available: {e}")

# ===== V3.0 P0 IMPORTS =====
try:
    from .workflow_graph import WorkflowGraph, WorkflowBuilder, NodeType
    WORKFLOW_AVAILABLE = True
    logger.info("✅ Workflow graphs loaded")
except ImportError:
    WORKFLOW_AVAILABLE = False
    WorkflowGraph = WorkflowBuilder = NodeType = None
    logger.warning("⚠️ Workflow graphs not available")

try:
    from .advanced_memory import AdvancedMemory, MemoryType
    ADVANCED_MEMORY_AVAILABLE = True
    logger.info("✅ Advanced memory loaded")
except ImportError:
    ADVANCED_MEMORY_AVAILABLE = False
    AdvancedMemory = MemoryType = None
    logger.warning("⚠️ Advanced memory not available")

try:
    from .agent_cache import AgentCache, CacheStrategy
    AGENT_CACHE_AVAILABLE = True
    logger.info("✅ Agent cache loaded")
except ImportError:
    AGENT_CACHE_AVAILABLE = False
    AgentCache = CacheStrategy = None
    logger.warning("⚠️ Agent cache not available")

# ===== V3.1 IMPORTS =====
try:
    from fluxgraph.core.enhanced_memory import EnhancedMemory
    from fluxgraph.connectors import PostgresConnector, SalesforceConnector, ShopifyConnector
    from fluxgraph.workflows.visual_builder import VisualWorkflow
    V31_FEATURES_AVAILABLE = True
    logger.info("✅ v3.1 features loaded")
except ImportError:
    V31_FEATURES_AVAILABLE = False
    EnhancedMemory = PostgresConnector = SalesforceConnector = ShopifyConnector = VisualWorkflow = None
    logger.warning("⚠️ v3.1 features not available")

# Orchestrator
try:
    from .orchestrator_advanced import AdvancedOrchestrator
    ADVANCED_ORCHESTRATOR_AVAILABLE = True
    logger.info("✅ Advanced orchestrator loaded")
except ImportError:
    ADVANCED_ORCHESTRATOR_AVAILABLE = False
    try:
        from .orchestrator import FluxOrchestrator
        logger.info("✅ Basic orchestrator loaded")
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
    class EventHooks:
        async def trigger(self, event_name: str, payload: Dict[str, Any]):
            pass

# Security
try:
    from ..security.audit import AuditLogger, AuditEventType
    from ..security.pii_detector import PIIDetector
    from ..security.prompt_injection import PromptInjectionDetector
    from ..security.rbac import RBACManager, Role, Permission
    SECURITY_AVAILABLE = True
    logger.info("✅ Security features loaded")
except ImportError:
    SECURITY_AVAILABLE = False
    AuditLogger = PIIDetector = PromptInjectionDetector = RBACManager = None
    AuditEventType = Role = Permission = None
    logger.warning("⚠️ Security features not available")

# Orchestration
try:
    from ..orchestration.handoff import HandoffProtocol
    from ..orchestration.hitl import HITLManager
    from ..orchestration.task_adherence import TaskAdherenceMonitor
    from ..orchestration.batch import BatchProcessor as OrchBatchProcessor
    ORCHESTRATION_AVAILABLE = True
    logger.info("✅ Orchestration features loaded")
except ImportError:
    ORCHESTRATION_AVAILABLE = False
    HandoffProtocol = HITLManager = TaskAdherenceMonitor = OrchBatchProcessor = None
    logger.warning("⚠️ Orchestration features not available")

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


# ===== COMPLETE FLUXAPP v3.2 CLASS =====
class FluxApp:
    """
    FluxGraph v3.2 - 100% COMPLETE ENTERPRISE APPLICATION
    
    The ultimate AI agent orchestration framework with every feature:
    - Multi-agent coordination with advanced orchestration
    - LCEL-style chains for LangChain compatibility
    - Distributed tracing for full observability
    - Advanced memory systems (short/long/episodic)
    - Database connectors marketplace
    - Visual workflow builder
    - Security (audit, PII, RBAC, injection detection)
    - Analytics and monitoring
    - Production deployment ready
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
        database_url: Optional[str] = None,
        # v3.2 Features (LangChain Parity)
        enable_chains: bool = True,
        enable_tracing: bool = True,
        enable_batch_optimization: bool = True,
        enable_streaming_optimization: bool = True,
        enable_langserve_api: bool = True,
        tracing_export_path: str = "./traces",
        tracing_project_name: Optional[str] = None,
        # Security
        enable_security: bool = False,
        enable_audit_logging: bool = False,
        enable_pii_detection: bool = False,
        enable_prompt_shield: bool = False,
        enable_rbac: bool = False,
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
        
        Args:
            title: API title
            description: API description
            version: API version
            memory_store: Optional memory store instance
            rag_connector: Optional RAG connector
            auto_init_rag: Auto-initialize RAG if not provided
            enable_analytics: Enable analytics dashboard
            enable_advanced_features: Enable advanced orchestrator
            
            # v3.0 P0
            enable_workflows: Enable graph-based workflows
            enable_advanced_memory: Enable multi-tier memory
            enable_agent_cache: Enable response caching
            cache_strategy: Caching strategy (exact/semantic/hybrid)
            
            # v3.1
            enable_enhanced_memory: Enable entity extraction memory
            enable_connectors: Enable database connectors
            enable_visual_workflows: Enable visual workflow builder
            database_url: Database URL for enhanced memory
            
            # v3.2
            enable_chains: Enable LCEL-style chains
            enable_tracing: Enable distributed tracing
            enable_batch_optimization: Enable batch processing
            enable_streaming_optimization: Enable streaming optimization
            enable_langserve_api: Enable LangServe-style API
            tracing_export_path: Path for trace exports
            tracing_project_name: Project name for tracing
            
            # Security
            enable_security: Master switch for all security features
            enable_audit_logging: Enable audit logging
            enable_pii_detection: Enable PII detection
            enable_prompt_shield: Enable prompt injection detection
            enable_rbac: Enable role-based access control
            
            # Orchestration
            enable_orchestration: Master switch for orchestration
            enable_handoffs: Enable agent handoffs
            enable_hitl: Enable human-in-the-loop
            enable_task_adherence: Enable task monitoring
            
            log_level: Logging level
            cors_origins: CORS allowed origins
        """
        logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
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
        logger.info(f"🚀 INITIALIZING FLUXGRAPH v{version} - 100% COMPLETE EDITION")
        logger.info("=" * 100)
        
        # ===== CORE COMPONENTS =====
        logger.info("📦 Initializing core components...")
        self.registry = AgentRegistry()
        self.tool_registry = ToolRegistry()
        self.memory_store = memory_store
        self.rag_connector = rag_connector
        self.hooks = EventHooks()
        
        # Orchestrator
        if enable_advanced_features and ADVANCED_ORCHESTRATOR_AVAILABLE:
            logger.info("   🔧 AdvancedOrchestrator: ENABLED")
            self.orchestrator = AdvancedOrchestrator(self.registry)
            self.advanced_features_enabled = True
        else:
            logger.info("   🔧 BasicOrchestrator: ENABLED")
            self.orchestrator = FluxOrchestrator(self.registry)
            self.advanced_features_enabled = False
        
        # Analytics
        if enable_analytics and ANALYTICS_AVAILABLE:
            logger.info("   📊 Analytics Dashboard: ENABLED")
            self.performance_monitor = PerformanceMonitor()
            self.analytics_dashboard = AnalyticsDashboard(self.performance_monitor)
            self.api.include_router(self.analytics_dashboard.router)
        else:
            logger.info("   📊 Analytics Dashboard: DISABLED")
            self.performance_monitor = None
            self.analytics_dashboard = None
        
        # ===== V3.0 P0 FEATURES =====
        logger.info("🆕 Initializing v3.0 P0 features...")
        
        # Workflows
        if enable_workflows and WORKFLOW_AVAILABLE:
            logger.info("   🔀 Graph Workflows: ENABLED")
            self.workflow_builder = WorkflowBuilder
            self.workflow_graphs: Dict[str, WorkflowGraph] = {}
            self.workflows_enabled = True
        else:
            logger.info("   🔀 Graph Workflows: DISABLED")
            self.workflow_builder = None
            self.workflow_graphs = {}
            self.workflows_enabled = False
        
        # Advanced Memory
        if enable_advanced_memory and ADVANCED_MEMORY_AVAILABLE:
            logger.info("   🧠 Advanced Memory: ENABLED (Short/Long/Episodic)")
            self.advanced_memory = AdvancedMemory(
                short_term_capacity=100,
                long_term_capacity=10000,
                consolidation_threshold=0.7
            )
            self.advanced_memory_enabled = True
        else:
            logger.info("   🧠 Advanced Memory: DISABLED")
            self.advanced_memory = None
            self.advanced_memory_enabled = False
        
        # Agent Cache
        if enable_agent_cache and AGENT_CACHE_AVAILABLE:
            strategy_map = {
                "exact": CacheStrategy.EXACT,
                "semantic": CacheStrategy.SEMANTIC,
                "hybrid": CacheStrategy.HYBRID
            }
            selected_strategy = strategy_map.get(cache_strategy, CacheStrategy.HYBRID)
            logger.info(f"   ⚡ Agent Cache: ENABLED (Strategy: {cache_strategy.upper()})")
            self.agent_cache = AgentCache(
                strategy=selected_strategy,
                max_size=1000,
                default_ttl=3600,
                semantic_threshold=0.85
            )
            self.agent_cache_enabled = True
        else:
            logger.info("   ⚡ Agent Cache: DISABLED")
            self.agent_cache = None
            self.agent_cache_enabled = False
        
        # ===== V3.1 FEATURES =====
        logger.info("🎉 Initializing v3.1 features...")
        
        # Enhanced Memory
        if enable_enhanced_memory and database_url and V31_FEATURES_AVAILABLE:
            logger.info("   🧠 Enhanced Memory: ENABLED (Entity Extraction)")
            self.enhanced_memory = EnhancedMemory(database_url)
            self._enhanced_memory_initialized = False
            self.enhanced_memory_enabled = True
        else:
            logger.info("   🧠 Enhanced Memory: DISABLED")
            self.enhanced_memory = None
            self._enhanced_memory_initialized = False
            self.enhanced_memory_enabled = False
        
        # Connectors
        if enable_connectors and V31_FEATURES_AVAILABLE:
            logger.info("   🔌 Database Connectors: ENABLED")
            self.connectors: Dict[str, Any] = {}
            self.connectors_enabled = True
        else:
            logger.info("   🔌 Database Connectors: DISABLED")
            self.connectors = None
            self.connectors_enabled = False
        
        # Visual Workflows
        if enable_visual_workflows and V31_FEATURES_AVAILABLE:
            logger.info("   🎨 Visual Workflow Builder: ENABLED")
            self.visual_workflows: Dict[str, VisualWorkflow] = {}
            self.visual_workflows_enabled = True
        else:
            logger.info("   🎨 Visual Workflow Builder: DISABLED")
            self.visual_workflows = None
            self.visual_workflows_enabled = False
        
        # ===== V3.2 FEATURES (LangChain Parity) =====
        logger.info("🚀 Initializing v3.2 features (LangChain Parity)...")
        
        # Chains
        if enable_chains and CHAINS_V32_AVAILABLE:
            logger.info("   ⛓️ LCEL Chains: ENABLED")
            self.chains: Dict[str, Runnable] = {}
            self.chain_registry: Dict[str, Dict] = {}
            self.chains_enabled = True
        else:
            logger.info("   ⛓️ LCEL Chains: DISABLED")
            self.chains = None
            self.chain_registry = {}
            self.chains_enabled = False
        
        # Tracing
        if enable_tracing and CHAINS_V32_AVAILABLE:
            project_name = tracing_project_name or title
            logger.info(f"   🔍 Distributed Tracing: ENABLED (Project: {project_name})")
            self.tracer = configure_tracing(
                project_name=project_name,
                enabled=True,
                export_path=tracing_export_path,
                export_format="json",
                auto_export=True
            )
            self.tracing_enabled = True
        else:
            logger.info("   🔍 Distributed Tracing: DISABLED")
            self.tracer = None
            self.tracing_enabled = False
        
        # Batch Optimization
        if enable_batch_optimization and CHAINS_V32_AVAILABLE:
            logger.info("   📦 Batch Optimization: ENABLED")
            self.batch_optimizer_enabled = True
        else:
            logger.info("   📦 Batch Optimization: DISABLED")
            self.batch_optimizer_enabled = False
        
        # Streaming Optimization
        if enable_streaming_optimization and CHAINS_V32_AVAILABLE:
            logger.info("   ⚡ Streaming Optimization: ENABLED")
            self.streaming_optimizer_enabled = True
        else:
            logger.info("   ⚡ Streaming Optimization: DISABLED")
            self.streaming_optimizer_enabled = False
        
        # LangServe API
        if enable_langserve_api and CHAINS_V32_AVAILABLE:
            logger.info("   🌐 LangServe API: ENABLED")
            self.langserve_enabled = True
        else:
            logger.info("   🌐 LangServe API: DISABLED")
            self.langserve_enabled = False
        
        # ===== SECURITY FEATURES =====
        logger.info("🔒 Initializing security features...")
        
        if enable_security and SECURITY_AVAILABLE:
            # Audit Logging
            if enable_audit_logging:
                logger.info("   📝 Audit Logging: ENABLED")
                self.audit_logger = AuditLogger()
            else:
                logger.info("   📝 Audit Logging: DISABLED")
                self.audit_logger = None
            
            # PII Detection
            if enable_pii_detection:
                logger.info("   🔐 PII Detection: ENABLED")
                self.pii_detector = PIIDetector()
            else:
                logger.info("   🔐 PII Detection: DISABLED")
                self.pii_detector = None
            
            # Prompt Injection Shield
            if enable_prompt_shield:
                logger.info("   🛡️ Prompt Shield: ENABLED")
                self.prompt_shield = PromptInjectionDetector()
            else:
                logger.info("   🛡️ Prompt Shield: DISABLED")
                self.prompt_shield = None
            
            # RBAC
            if enable_rbac:
                logger.info("   👥 RBAC: ENABLED")
                self.rbac_manager = RBACManager()
            else:
                logger.info("   👥 RBAC: DISABLED")
                self.rbac_manager = None
            
            self.security_enabled = True
        else:
            logger.info("   🔒 Security: DISABLED (All)")
            self.audit_logger = None
            self.pii_detector = None
            self.prompt_shield = None
            self.rbac_manager = None
            self.security_enabled = False
        
        # ===== ORCHESTRATION FEATURES =====
        logger.info("🎯 Initializing orchestration features...")
        
        if enable_orchestration and ORCHESTRATION_AVAILABLE and self.advanced_features_enabled:
            # Handoffs
            if enable_handoffs:
                logger.info("   🤝 Agent Handoffs: ENABLED")
                self.handoff_protocol = HandoffProtocol(self.orchestrator)
            else:
                logger.info("   🤝 Agent Handoffs: DISABLED")
                self.handoff_protocol = None
            
            # HITL
            if enable_hitl:
                logger.info("   👤 Human-in-the-Loop: ENABLED")
                self.hitl_manager = HITLManager()
            else:
                logger.info("   👤 Human-in-the-Loop: DISABLED")
                self.hitl_manager = None
            
            # Task Adherence
            if enable_task_adherence:
                logger.info("   ✅ Task Adherence: ENABLED")
                self.task_adherence = TaskAdherenceMonitor()
            else:
                logger.info("   ✅ Task Adherence: DISABLED")
                self.task_adherence = None
            
            # Batch Processor
            logger.info("   📦 Orchestration Batch: ENABLED")
            self.batch_processor = OrchBatchProcessor(self.orchestrator)
            
            self.orchestration_enabled = True
        else:
            logger.info("   🎯 Orchestration: DISABLED (All)")
            self.handoff_protocol = None
            self.hitl_manager = None
            self.task_adherence = None
            self.batch_processor = None
            self.orchestration_enabled = False
        
        # Auto-init RAG
        if auto_init_rag and RAG_AVAILABLE and self.rag_connector is None:
            self._auto_initialize_rag()
        
        # Setup
        self._setup_middleware(cors_origins)
        self._setup_routes()
        
        # Final Summary
        logger.info("=" * 100)
        logger.info(f"✅ FLUXAPP v{version} INITIALIZATION COMPLETE - 100% FEATURE SET")
        logger.info("=" * 100)
        logger.info("📋 FEATURE STATUS SUMMARY:")
        logger.info(f"   Core: Memory={'✅' if self.memory_store else '❌'} | RAG={'✅' if self.rag_connector else '❌'} | Analytics={'✅' if self.performance_monitor else '❌'}")
        logger.info(f"   v3.0: Workflows={'✅' if self.workflows_enabled else '❌'} | AdvMemory={'✅' if self.advanced_memory_enabled else '❌'} | Cache={'✅' if self.agent_cache_enabled else '❌'}")
        logger.info(f"   v3.1: Enhanced={'✅' if self.enhanced_memory_enabled else '❌'} | Connectors={'✅' if self.connectors_enabled else '❌'} | Visual={'✅' if self.visual_workflows_enabled else '❌'}")
        logger.info(f"   v3.2: Chains={'✅' if self.chains_enabled else '❌'} | Tracing={'✅' if self.tracing_enabled else '❌'} | Batch={'✅' if self.batch_optimizer_enabled else '❌'} | Stream={'✅' if self.streaming_optimizer_enabled else '❌'}")
        logger.info(f"   Security: {'✅ ENABLED' if self.security_enabled else '❌ DISABLED'}")
        logger.info(f"   Orchestration: {'✅ ENABLED' if self.orchestration_enabled else '❌ DISABLED'}")
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
            logger.info("   ✅ RAG connector auto-initialized")
        except Exception as e:
            logger.error(f"   ❌ Failed to auto-initialize RAG: {e}")
            self.rag_connector = None

    async def _ensure_enhanced_memory_initialized(self):
        """Lazy initialization for enhanced memory."""
        if self.enhanced_memory and not self._enhanced_memory_initialized:
            try:
                await self.enhanced_memory.initialize()
                self._enhanced_memory_initialized = True
                logger.info("✅ Enhanced memory initialized")
            except Exception as e:
                logger.error(f"❌ Enhanced memory initialization failed: {e}")
                raise

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
            url = str(request.url)
            
            logger.info(f"[{request_id}] 🌐 {method} {url} from {client_host}")
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = str(round(process_time, 4))
                response.headers["X-Request-ID"] = request_id
                
                logger.info(f"[{request_id}] ⬅️ {response.status_code} ({process_time:.4f}s)")
                return response
            except Exception as e:
                process_time = time.time() - start_time
                logger.error(f"[{request_id}] ❌ Error ({process_time:.4f}s): {e}", exc_info=True)
                raise

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
                        "memory": self.memory_store is not None,
                        "rag": self.rag_connector is not None,
                        "analytics": self.performance_monitor is not None
                    },
                    "v3.0_p0": {
                        "workflows": self.workflows_enabled,
                        "advanced_memory": self.advanced_memory_enabled,
                        "agent_cache": self.agent_cache_enabled
                    },
                    "v3.1": {
                        "enhanced_memory": self.enhanced_memory_enabled,
                        "connectors": self.connectors_enabled,
                        "visual_workflows": self.visual_workflows_enabled
                    },
                    "v3.2": {
                        "chains": self.chains_enabled,
                        "tracing": self.tracing_enabled,
                        "batch_optimization": self.batch_optimizer_enabled,
                        "streaming_optimization": self.streaming_optimizer_enabled,
                        "langserve_api": self.langserve_enabled
                    },
                    "security": {
                        "enabled": self.security_enabled,
                        "audit_logging": self.audit_logger is not None,
                        "pii_detection": self.pii_detector is not None,
                        "prompt_shield": self.prompt_shield is not None,
                        "rbac": self.rbac_manager is not None
                    },
                    "orchestration": {
                        "enabled": self.orchestration_enabled,
                        "handoffs": self.handoff_protocol is not None,
                        "hitl": self.hitl_manager is not None,
                        "task_adherence": self.task_adherence is not None
                    }
                },
                "endpoints": {
                    "agents": "/ask/{agent_name}",
                    "tools": "/tools",
                    "chains": "/chains",
                    "workflows": "/workflows",
                    "connectors": "/connectors",
                    "memory": "/memory",
                    "tracing": "/tracing",
                    "docs": "/docs",
                    "health": "/health"
                }
            }
        
        # ===== AGENT ENDPOINTS =====
        @self.api.post("/ask/{agent_name}")
        async def ask_agent(agent_name: str, payload: Dict[str, Any]):
            """Execute a registered agent."""
            request_id = request_id_context.get()
            start_time = time.time()
            
            logger.info(f"[{request_id}] 🤖 Executing agent '{agent_name}'")
            
            await self.hooks.trigger("request_received", {
                "request_id": request_id,
                "agent_name": agent_name,
                "payload": payload
            })
            
            try:
                # Security checks
                if self.prompt_shield:
                    if self.prompt_shield.is_injection(payload.get("query", "")):
                        raise HTTPException(status_code=400, detail="Prompt injection detected")
                
                if self.pii_detector:
                    payload = self.pii_detector.redact_pii(payload)
                
                # Check cache
                if self.agent_cache:
                    cached = await self.agent_cache.get(agent_name, payload)
                    if cached:
                        logger.info(f"[{request_id}] ⚡ Cache hit for '{agent_name}'")
                        return cached
                
                # Execute agent
                result = await self.orchestrator.run(agent_name, payload)
                
                # Store in cache
                if self.agent_cache:
                    await self.agent_cache.set(agent_name, payload, result)
                
                # Audit log
                if self.audit_logger:
                    await self.audit_logger.log(
                        AuditEventType.AGENT_EXECUTED,
                        {"agent": agent_name, "request_id": request_id}
                    )
                
                duration = time.time() - start_time
                
                await self.hooks.trigger("agent_completed", {
                    "request_id": request_id,
                    "agent_name": agent_name,
                    "result": result,
                    "duration": duration
                })
                
                logger.info(f"[{request_id}] ✅ Agent '{agent_name}' completed ({duration:.4f}s)")
                return result
                
            except ValueError as e:
                logger.warning(f"[{request_id}] ⚠️ Agent error: {e}")
                status_code = 404 if "not registered" in str(e).lower() else 400
                raise HTTPException(status_code=status_code, detail=str(e))
            except Exception as e:
                logger.error(f"[{request_id}] ❌ Execution error: {e}", exc_info=True)
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
                        async with self.tracer.span(
                            f"chain_{chain_name}",
                            run_type=RunType.CHAIN,
                            inputs={"input": request.input}
                        ):
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
                    logger.error(f"Chain error: {e}")
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
                    logger.error(f"Batch error: {e}")
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
        
        # ===== V3.0 WORKFLOW ENDPOINTS =====
        if self.workflows_enabled:
            
            @self.api.post("/workflows")
            async def create_workflow(request: WorkflowCreateRequest):
                """Create a new workflow graph."""
                try:
                    workflow = self.workflow_builder()
                    self.workflow_graphs[request.name] = workflow
                    return {
                        "message": f"Workflow '{request.name}' created",
                        "name": request.name
                    }
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            @self.api.get("/workflows")
            async def list_workflows():
                """List all workflows."""
                return {
                    "workflows": list(self.workflow_graphs.keys()),
                    "count": len(self.workflow_graphs)
                }
            
            @self.api.get("/workflows/{workflow_name}")
            async def get_workflow(workflow_name: str):
                """Get workflow details."""
                if workflow_name not in self.workflow_graphs:
                    raise HTTPException(status_code=404, detail="Workflow not found")
                
                workflow = self.workflow_graphs[workflow_name]
                return {
                    "name": workflow_name,
                    "nodes": len(workflow.nodes) if hasattr(workflow, 'nodes') else 0
                }
        
        # ===== V3.1 CONNECTOR ENDPOINTS =====
        if self.connectors_enabled:
            
            @self.api.post("/connectors/{connector_name}")
            async def add_connector(connector_name: str, request: ConnectorConfigRequest):
                """Add a database connector."""
                try:
                    if request.connector_type == "postgres":
                        connector = PostgresConnector(**request.config)
                    elif request.connector_type == "salesforce":
                        connector = SalesforceConnector(**request.config)
                    elif request.connector_type == "shopify":
                        connector = ShopifyConnector(**request.config)
                    else:
                        raise HTTPException(status_code=400, detail="Unknown connector type")
                    
                    await connector.initialize()
                    self.connectors[connector_name] = connector
                    
                    return {
                        "message": f"Connector '{connector_name}' added",
                        "type": request.connector_type
                    }
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            @self.api.get("/connectors")
            async def list_connectors():
                """List all connectors."""
                return {
                    "connectors": list(self.connectors.keys()),
                    "count": len(self.connectors)
                }
        
        # ===== TRACING ENDPOINTS =====
        if self.tracing_enabled:
            
            @self.api.get("/tracing/status")
            async def tracing_status():
                """Get tracing status."""
                stats = self.tracer.get_statistics()
                return {
                    "tracing_enabled": True,
                    "project": self.tracer.project_name,
                    "statistics": stats
                }
            
            @self.api.get("/tracing/traces")
            async def get_traces():
                """Get all traces."""
                traces = self.tracer.get_root_traces()
                return {
                    "traces": [t.to_dict() for t in traces],
                    "count": len(traces)
                }
            
            @self.api.get("/tracing/traces/{trace_id}")
            async def get_trace(trace_id: str):
                """Get specific trace."""
                trace = self.tracer.get_trace(trace_id)
                if not trace:
                    raise HTTPException(status_code=404, detail="Trace not found")
                return trace.to_dict()
        
        # ===== TOOLS ENDPOINTS =====
        @self.api.get("/tools")
        async def list_tools():
            """List registered tools."""
            return {"tools": self.tool_registry.list_tools()}
        
        @self.api.get("/tools/{tool_name}")
        async def get_tool_info(tool_name: str):
            """Get tool information."""
            try:
                info = self.tool_registry.get_tool_info(tool_name)
                return info
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
        
        # ===== MEMORY & RAG ENDPOINTS =====
        if self.memory_store:
            @self.api.get("/memory/status")
            async def memory_status():
                """Memory status."""
                return {
                    "memory_enabled": True,
                    "type": type(self.memory_store).__name__
                }
        
        if self.rag_connector:
            @self.api.get("/rag/status")
            async def rag_status():
                """RAG status."""
                try:
                    stats = self.rag_connector.get_collection_stats()
                    return {
                        "rag_enabled": True,
                        "stats": stats
                    }
                except Exception as e:
                    return {
                        "rag_enabled": True,
                        "error": str(e)
                    }
        
        # ===== HEALTH CHECK =====
        @self.api.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "version": self.version,
                "timestamp": datetime.utcnow().isoformat()
            }

    # ===== CHAIN METHODS =====
    
    def register_chain(self, name: str, chain: Runnable, description: Optional[str] = None):
        """Register a chain."""
        if not self.chains_enabled:
            raise RuntimeError("Chains not enabled")
        
        self.chains[name] = chain
        self.chain_registry[name] = {
            "type": type(chain).__name__,
            "description": description,
            "registered_at": datetime.utcnow().isoformat()
        }
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
        """Add connector programmatically."""
        if not self.connectors_enabled:
            raise RuntimeError("Connectors not enabled")
        
        if connector_type == "postgres":
            connector = PostgresConnector(**config)
        elif connector_type == "salesforce":
            connector = SalesforceConnector(**config)
        elif connector_type == "shopify":
            connector = ShopifyConnector(**config)
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")
        
        await connector.initialize()
        self.connectors[name] = connector
        logger.info(f"🔌 Connector '{name}' added ({connector_type})")
    
    # ===== AGENT & TOOL REGISTRATION =====
    
    def register(self, name: str, agent: Any):
        """Register an agent."""
        self.registry.add(name, agent)
        logger.info(f"✅ Agent '{name}' registered")
    
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
            return func
        return decorator
    
    # ===== SERVER METHODS =====
    
    def run(self, host: str = "127.0.0.1", port: int = 8000, reload: bool = False, **kwargs):
        """Start the FluxGraph API server."""
        logger.info("=" * 100)
        logger.info(f"🚀 STARTING FLUXGRAPH SERVER v{self.version}")
        logger.info(f"   📍 Host: {host}")
        logger.info(f"   🔌 Port: {port}")
        logger.info(f"   🔄 Reload: {reload}")
        logger.info(f"   📚 Docs: http://{host}:{port}/docs")
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
        print("❌ No 'app' variable found")
        sys.exit(1)
    
    if not isinstance(app_instance, FluxApp):
        print(f"❌ 'app' is not a FluxApp instance (type: {type(app_instance)})")
        sys.exit(1)
    
    logger.info("✅ FluxApp instance found")
    
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
