# fluxgraph/core/postgres_memory.py
"""
PostgreSQL Implementation of the Memory Interface for FluxGraph.

This module provides `PostgresMemory`, a concrete implementation of the
`fluxgraph.core.memory.Memory` abstract base class. It uses SQLAlchemy 2.x
for asynchronous database interactions with a PostgreSQL backend.

It allows agents to store, retrieve, and manage stateful information
persistently using a relational database.
"""
import uuid
import logging
from typing import List, Dict, Any

# Import SQLAlchemy components
from sqlalchemy import String, DateTime, func, select, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB

# Import FluxGraph components
from .memory import Memory
from ..utils.db import DatabaseManager

# Use module-specific logger
logger = logging.getLogger(__name__)

# --- SQLAlchemy Model Definition ---

class Base(DeclarativeBase):
    """
    Base class for SQLAlchemy declarative models within FluxGraph's PostgresMemory.
    """
    pass

class MemoryItem(Base):
    """
    SQLAlchemy ORM model representing an individual memory item stored in PostgreSQL.

    Attributes:
        id (uuid.UUID): Unique identifier for the memory item.
        agent_name (str): The name or session ID of the agent this memory belongs to.
        data (Dict[str, Any]): The actual data stored in the memory item (as JSONB).
        timestamp (DateTime): The UTC time the memory item was created.
    """
    __tablename__ = "agent_memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_name: Mapped[str] = mapped_column(String, index=True, nullable=False) # Make agent_name non-nullable
    data: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False) # Make data non-nullable
    # Use timezone-aware DateTime
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True, nullable=False)

    def __repr__(self) -> str:
        return f"<MemoryItem(id={self.id}, agent_name='{self.agent_name}', timestamp='{self.timestamp}')>"

# --- Memory Implementation ---

class PostgresMemory(Memory):
    """
    PostgreSQL-based memory store for FluxGraph agents.

    Implements the `Memory` interface using SQLAlchemy's asynchronous ORM.
    It provides persistent storage for agent state and conversation history.

    This class requires a `DatabaseManager` instance to handle database connections.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the PostgresMemory store.

        Args:
            db_manager (DatabaseManager): The database manager instance to use
                                          for acquiring asynchronous database sessions.
                                          It must be connected before this store is used.
        """
        self.db_manager = db_manager
        logger.debug("PostgresMemory store initialized.")

    async def create_tables(self):
        """
        Create database tables asynchronously.

        This method uses the async engine from the `DatabaseManager` to create
        all tables defined by the SQLAlchemy models (e.g., `MemoryItem`) within
        the `Base.metadata`. It should be called once during application setup
        after the `DatabaseManager` is connected.

        Raises:
            RuntimeError: If the `DatabaseManager` or its engine is not initialized.
        """
        if not self.db_manager or not self.db_manager.engine:
            raise RuntimeError(
                "DatabaseManager or its engine is not initialized. "
                "Ensure db_manager.connect() is called before creating tables."
            )
        logger.info("Creating/verifying agent memory tables in PostgreSQL...")
        async with self.db_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Agent memory tables (e.g., 'agent_memories') created/verified successfully.")

    async def add(self, agent_name: str, data: Dict[str, Any]) -> str:
        """
        Store a new memory item in the PostgreSQL database.

        Args:
            agent_name (str): The identifier for the agent (e.g., name or session ID).
            data (Dict[str, Any]): The dictionary data to store as the memory content.

        Returns:
            str: The UUID string of the newly created memory item.

        Raises:
            Exception: Propagates any exceptions raised during the database operation
                       (e.g., connection errors, serialization issues).
        """
        async with self.db_manager.get_session() as session: # Assumes get_session is correctly implemented in db manager
            try:
                memory_id = str(uuid.uuid4())
                logger.debug(f"[Agent: {agent_name}] Preparing to add memory item with ID {memory_id}.")
                
                new_memory = MemoryItem(
                    id=uuid.UUID(memory_id),
                    agent_name=agent_name,
                    data=data
                )
                session.add(new_memory)
                await session.commit()
                await session.refresh(new_memory) # Ensure the object is fully loaded post-commit if needed
                logger.info(f"[Agent: {agent_name}] Successfully added memory item with ID {memory_id}.")
                return memory_id
            except Exception as e:
                logger.error(f"[Agent: {agent_name}] Failed to add memory item: {e}", exc_info=True)
                await session.rollback() # Ensure rollback on error
                raise # Re-raise the exception

    async def get(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent memory items for an agent, ordered by timestamp (newest first).

        Args:
            agent_name (str): The identifier for the agent whose memories to retrieve.
            limit (int, optional): The maximum number of items to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: A list of memory data dictionaries, ordered from
                                  newest to oldest. Returns an empty list if no
                                  memories are found for the agent.

        Raises:
            Exception: Propagates any exceptions raised during the database query
                       (e.g., connection errors).
        """
        async with self.db_manager.get_session() as session:
            try:
                logger.debug(f"[Agent: {agent_name}] Retrieving up to {limit} recent memory items.")
                
                stmt = (
                    select(MemoryItem)
                    .where(MemoryItem.agent_name == agent_name)
                    .order_by(MemoryItem.timestamp.desc()) # Newest first
                    .limit(limit)
                )
                result = await session.execute(stmt)
                items = result.scalars().all()
                
                # Extract data dictionaries
                memory_data_list = [item.data for item in items]
                logger.info(f"[Agent: {agent_name}] Retrieved {len(memory_data_list)} memory item(s).")
                return memory_data_list
                
            except Exception as e:
                logger.error(f"[Agent: {agent_name}] Failed to retrieve memory items: {e}", exc_info=True)
                raise # Re-raise the exception

    async def delete(self, agent_name: str, memory_id: str) -> bool:
        """
        Delete a specific memory item by its ID and associated agent name.

        Args:
            agent_name (str): The identifier for the agent that owns the memory.
            memory_id (str): The UUID string of the memory item to delete.

        Returns:
            bool: True if the item was found and deleted, False otherwise.

        Raises:
            ValueError: If `memory_id` is not a valid UUID string.
            Exception: Propagates any other exceptions raised during the database operation.
        """
        # Validate UUID format early
        try:
            uuid.UUID(memory_id)
        except ValueError as ve:
            logger.error(f"[Agent: {agent_name}] Invalid UUID format for memory_id '{memory_id}': {ve}")
            raise ValueError(f"Invalid memory ID format: {memory_id}") from ve

        async with self.db_manager.get_session() as session:
            try:
                logger.debug(f"[Agent: {agent_name}] Attempting to delete memory item with ID {memory_id}.")
                
                stmt = delete(MemoryItem).where(
                    MemoryItem.id == uuid.UUID(memory_id),
                    MemoryItem.agent_name == agent_name
                )
                result = await session.execute(stmt)
                await session.commit()
                
                deleted_count = result.rowcount
                if deleted_count > 0:
                    logger.info(f"[Agent: {agent_name}] Successfully deleted memory item with ID {memory_id}.")
                else:
                    logger.info(f"[Agent: {agent_name}] Memory item with ID {memory_id} not found for deletion.")
                return deleted_count > 0
                
            except Exception as e:
                logger.error(f"[Agent: {agent_name}] Failed to delete memory item with ID {memory_id}: {e}", exc_info=True)
                await session.rollback() # Ensure rollback on error
                raise # Re-raise the exception

    async def clear(self, agent_name: str) -> int:
        """
        Clear all memory items associated with a specific agent.

        Args:
            agent_name (str): The identifier for the agent whose memories are to be cleared.

        Returns:
            int: The total number of memory items that were deleted.

        Raises:
            Exception: Propagates any exceptions raised during the database operation.
        """
        async with self.db_manager.get_session() as session:
            try:
                logger.debug(f"[Agent: {agent_name}] Clearing all memory items.")
                
                stmt = delete(MemoryItem).where(MemoryItem.agent_name == agent_name)
                result = await session.execute(stmt)
                await session.commit()
                
                deleted_count = result.rowcount
                logger.info(f"[Agent: {agent_name}] Cleared {deleted_count} memory item(s).")
                return deleted_count
                
            except Exception as e:
                logger.error(f"[Agent: {agent_name}] Failed to clear memory items: {e}", exc_info=True)
                await session.rollback() # Ensure rollback on error
                raise # Re-raise the exception
