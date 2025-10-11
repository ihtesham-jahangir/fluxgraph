"""
FluxGraph - Production-grade AI Agent Orchestration Framework
Cut AI costs by 70%+ with semantic caching
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)

__version__ = "2.1.0"

# Import with proper aliases for actual class names
try:
    from .core.app import FluxApp
except ImportError as e:
    logging.warning(f"FluxApp import failed: {e}")
    FluxApp = None

try:
    from .core.agent_cache import AgentCache
    # Create alias for API compatibility
    SemanticCache = AgentCache
except ImportError as e:
    logging.warning(f"AgentCache import failed: {e}")
    AgentCache = None
    SemanticCache = None

try:
    from .core.advanced_memory import EnhancedMemory
    # Create aliases for API compatibility
    AdvancedMemory = EnhancedMemory
    AdvancedMemorySystem = EnhancedMemory
except ImportError as e:
    logging.warning(f"Memory system import failed: {e}")
    EnhancedMemory = None
    AdvancedMemory = None
    AdvancedMemorySystem = None

try:
    from .workflows.visual_builder import VisualWorkflow
    # Create alias for API compatibility
    WorkflowBuilder = VisualWorkflow
except ImportError as e:
    logging.warning(f"VisualWorkflow import failed: {e}")
    VisualWorkflow = None
    WorkflowBuilder = None

# API module
try:
    from . import api
    logging.info("✅ FluxGraph API module loaded")
except ImportError as e:
    logging.warning(f"⚠️ FluxGraph API not available: {e}")
    api = None

# Export all available classes
__all__ = [
    # Core
    "FluxApp",
    # Caching (with alias)
    "AgentCache",
    "SemanticCache",  # Alias for AgentCache
    # Memory (with aliases)
    "EnhancedMemory",
    "AdvancedMemory",  # Alias for EnhancedMemory
    "AdvancedMemorySystem",  # Alias for EnhancedMemory
    # Workflows (with alias)
    "VisualWorkflow",
    "WorkflowBuilder",  # Alias for VisualWorkflow
    # API
    "api"
]

# Remove None values from exports
__all__ = [name for name in __all__ if globals().get(name) is not None]

logging.info(f"🚀 FluxGraph v{__version__} initialized with {len(__all__)} exports")
