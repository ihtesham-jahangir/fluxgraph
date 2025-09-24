# fluxgraph/core/memory.py
"""
Memory Interface for FluxGraph Agents.

This module defines the abstract `Memory` base class. It specifies the standard
interface that all concrete memory store implementations (e.g., PostgreSQL,
Redis, in-memory) must adhere to. This allows FluxGraph agents to interact
with different memory backends in a consistent way.

The memory interface is designed around storing and retrieving data associated
with specific agent sessions or identifiers.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# Use module-specific logger
logger = logging.getLogger(__name__)

class Memory(ABC):
    """
    Abstract base class for agent memory stores.

    Defines the standard interface for storing, retrieving, and managing
    information associated with agents or agent sessions. Concrete implementations
    (e.g., PostgresMemory, RedisMemory) must provide the logic for these methods.

    This interface focuses on basic CRUD (Create, Read, Delete, Clear) operations
    for agent-specific data.
    """

    @abstractmethod
    async def add(self, agent_session_id: str, data: Dict[str, Any]) -> str:
        """
        Stores data associated with an agent session.

        Args:
            agent_session_id (str): The unique identifier for the agent session
                                    or agent name. This scopes the memory.
            data (Dict[str, Any]): The data to store. This can be any JSON-serializable
                                   dictionary, such as a message, observation,
                                   or intermediate result.

        Returns:
            str: A unique identifier for the newly stored memory item.
                 This ID can be used for later retrieval or deletion.
        """
        # Subclasses should implement the storage logic here.
        # Example: Save to a database table with columns (id, agent_session_id, data, timestamp).
        # Return the generated unique ID (e.g., UUID).
        pass

    @abstractmethod
    async def get(self, agent_session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves recent memories for an agent session.

        Implementations should typically return the most recent items first,
        based on a timestamp or sequence order.

        Args:
            agent_session_id (str): The unique identifier for the agent session.
            limit (int, optional): The maximum number of items to retrieve.
                                   Defaults to 10.

        Returns:
            List[Dict[str, Any]]: A list of memory items. Each item is a dictionary
                                  as it was stored. The list is ordered by recency
                                  (newest first). Returns an empty list if no items
                                  are found for the session.
        """
        # Subclasses should implement the retrieval logic here.
        # Example: Query database for records matching agent_session_id,
        # order by timestamp descending, and limit results.
        pass

    @abstractmethod
    async def delete(self, agent_session_id: str, memory_id: str) -> bool:
        """
        Deletes a specific memory item.

        Args:
            agent_session_id (str): The unique identifier for the agent session.
                                    This might be used for validation or scoping.
            memory_id (str): The unique identifier of the specific memory item
                             to be deleted.

        Returns:
            bool: True if the item was successfully found and deleted.
                  False if the item was not found or deletion failed.
        """
        # Subclasses should implement the deletion logic here.
        # Example: Delete row from database where id=memory_id and agent_session_id matches.
        # Return True if a row was deleted, False otherwise.
        pass

    @abstractmethod
    async def clear(self, agent_session_id: str) -> int:
        """
        Clears all memories associated with a specific agent session.

        Args:
            agent_session_id (str): The unique identifier for the agent session
                                    whose memories are to be cleared.

        Returns:
            int: The number of memory items that were successfully deleted.
                 Returns 0 if no items were found for the session.
        """
        # Subclasses should implement the clear-all logic here.
        # Example: Delete all rows from database where agent_session_id matches.
        # Return the count of deleted rows.
        pass

    # --- Potential Future Extensions (Optional Abstract Methods) ---
    # These are not part of the core MVP but represent logical next steps.
    # They can be added as abstract methods here or implemented as concrete
    # methods that raise NotImplementedError.

    # @abstractmethod
    # async def update(self, agent_session_id: str, memory_id: str, data: Dict[str, Any]) -> bool:
    #     """
    #     Updates an existing memory item.
    #     (Not part of MVP core interface)
    #     """
    #     pass

    # @abstractmethod
    # async def search(self, agent_session_id: str, query: str) -> List[Dict[str, Any]]:
    #     """
    #     Searches memories for a given agent session based on content.
    #     (Not part of MVP core interface)
    #     """
    #     pass
