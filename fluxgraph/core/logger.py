from abc import ABC, abstractmethod  # This import is correct
from typing import Dict, Any, Optional
from enum import Enum

class LogEventType(Enum):
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    NODE_START = "node_start"
    NODE_END = "node_end"
    NODE_ERROR = "node_error"
    # We can add LLM_CALL, TOOL_CALL, etc. later

class BaseLogger(ABC):
    """
    Abstract base class for a structured workflow logger.
    """

    @abstractmethod  # <-- FIX 1: Was @abstractmodule
    async def log_event(
        self,
        workflow_id: str,
        node_id: Optional[str],
        event_type: LogEventType,
        data: Optional[Dict[str, Any]] = None
    ):
        """Log a single event in the workflow execution."""
        pass

    @abstractmethod  # <-- FIX 2: Was @abstractmodule
    async def setup(self):
        """Optional method to set up any required resources, like DB tables."""
        pass