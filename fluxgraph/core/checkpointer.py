# fluxgraph/core/checkpointer.py
from abc import ABC, abstractmethod
from typing import Optional
import json

# Import the WorkflowState dataclass
from .workflow_graph import WorkflowState 

class BaseCheckpointer(ABC):
    """
    Abstract base class for a workflow state checkpointer.
    This allows workflows to be persistent and resumable.
    """

    @abstractmethod
    async def load_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load the most recent state for a given workflow ID."""
        pass

    @abstractmethod
    async def save_state(self, state: WorkflowState):
        """Save the current state of a workflow."""
        pass

    @abstractmethod
    async def setup(self):
        """Optional method to set up any required resources, like DB tables."""
        pass