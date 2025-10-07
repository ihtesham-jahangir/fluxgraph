from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseConnector(ABC):
    """Base connector interface."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
    
    @abstractmethod
    async def connect(self):
        """Establish connection."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection."""
        pass
    
    @abstractmethod
    async def execute(self, operation: str, **kwargs) -> Any:
        """Execute operation."""
        pass
