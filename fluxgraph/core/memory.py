# fluxgraph/core/memory.py
"""
Memory interface for FluxGraph agents.
Defines the abstract base class for storing and retrieving agent memories.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Memory(ABC):
    """
    Abstract base class for agent memory stores.
    Defines the interface for storing and retrieving information.
    """

    @abstractmethod
    async def add(self, agent_name: str, data: Dict[str, Any]) -> str:
        """
        Stores data associated with an agent.

        Args:
            agent_name (str): The name of the agent.
            data (Dict[str, Any]): The data to store (e.g., message, observation).

        Returns:
            str: A unique identifier for the stored memory item.
        """
        pass

    @abstractmethod
    async def get(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves recent memories for an agent.

        Args:
            agent_name (str): The name of the agent.
            limit (int): The maximum number of items to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of memory items, ordered by recency (newest first).
        """
        pass

    @abstractmethod
    async def delete(self, agent_name: str, memory_id: str) -> bool:
        """
        Deletes a specific memory item.

        Args:
            agent_name (str): The name of the agent.
            memory_id (str): The unique identifier of the memory item.

        Returns:
            bool: True if the item was deleted, False otherwise.
        """
        pass

    @abstractmethod
    async def clear(self, agent_name: str) -> int:
        """
        Clears all memories for a specific agent.

        Args:
            agent_name (str): The name of the agent.

        Returns:
            int: The number of items deleted.
        """
        pass
