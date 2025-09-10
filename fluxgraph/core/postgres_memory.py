
"""
PostgreSQL implementation of the Memory interface for FluxGraph.
Uses SQLAlchemy 2.x for async database interactions.
"""
import uuid
import logging
from typing import List, Dict, Any
from sqlalchemy import String, DateTime, func, select, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB
from .memory import Memory
from ..utils.db import DatabaseManager

logger = logging.getLogger(__name__)

# --- SQLAlchemy Model Definition ---
class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""
    pass

class MemoryItem(Base):
    """
    SQLAlchemy ORM model for storing agent memories in PostgreSQL.
    """
    __tablename__ = "agent_memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_name: Mapped[str] = mapped_column(String, index=True)
    data: Mapped[Dict[str, Any]] = mapped_column(JSONB)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    def __repr__(self) -> str:
        return f"<MemoryItem(id={self.id}, agent_name='{self.agent_name}', timestamp='{self.timestamp}')>"

# --- Memory Implementation ---
class PostgresMemory(Memory):
    """
    PostgreSQL-based memory store for FluxGraph agents.
    Implements the Memory interface using SQLAlchemy async ORM.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the PostgresMemory store.

        Args:
            db_manager (DatabaseManager): The database manager instance for async database connections.
        """
        self.db_manager = db_manager
        logger.debug("PostgresMemory initialized.")

    async def create_tables(self):
        """
        Create database tables asynchronously using the async engine from db_manager.
        Ensures the 'agent_memories' table is created if it does not exist.

        Raises:
            RuntimeError: If the database engine is not initialized.
        """
        if not self.db_manager.engine:
            raise RuntimeError("DatabaseManager.engine is not initialized. Call db_manager.connect() first.")
        async with self.db_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Agent memory tables (e.g., agent_memories) created/verified.")

    async def add(self, agent_name: str, data: Dict[str, Any]) -> str:
        """
        Store a new memory item in the PostgreSQL database.

        Args:
            agent_name (str): The name of the agent storing the memory.
            data (Dict[str, Any]): The memory data to store.

        Returns:
            str: The UUID of the stored memory item.

        Raises:
            Exception: If the database operation fails.
        """
        async with self.db_manager.get_session() as session:
            try:
                memory_id = str(uuid.uuid4())
                new_memory = MemoryItem(
                    id=uuid.UUID(memory_id),
                    agent_name=agent_name,
                    data=data
                )
                session.add(new_memory)
                await session.commit()
                await session.refresh(new_memory)
                logger.debug(f"Added memory for agent '{agent_name}' with ID {memory_id}")
                return memory_id
            except Exception as e:
                logger.error(f"Failed to add memory for agent '{agent_name}': {str(e)}")
                await session.rollback()
                raise

    async def get(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent memories for an agent, ordered by timestamp (newest first).

        Args:
            agent_name (str): The name of the agent whose memories are retrieved.
            limit (int): Maximum number of memories to return (default: 10).

        Returns:
            List[Dict[str, Any]]: A list of memory data dictionaries.

        Raises:
            Exception: If the database query fails.
        """
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    select(MemoryItem)
                    .where(MemoryItem.agent_name == agent_name)
                    .order_by(MemoryItem.timestamp.desc())
                    .limit(limit)
                )
                result = await session.execute(stmt)
                items = result.scalars().all()
                return [item.data for item in items]
            except Exception as e:
                logger.error(f"Failed to retrieve memories for agent '{agent_name}': {str(e)}")
                raise

    async def delete(self, agent_name: str, memory_id: str) -> bool:
        """
        Delete a specific memory item by ID and agent name.

        Args:
            agent_name (str): The name of the agent whose memory is to be deleted.
            memory_id (str): The UUID of the memory item to delete.

        Returns:
            bool: True if a memory item was deleted, False otherwise.

        Raises:
            ValueError: If the memory_id is not a valid UUID.
            Exception: If the database operation fails.
        """
        async with self.db_manager.get_session() as session:
            try:
                stmt = delete(MemoryItem).where(
                    MemoryItem.id == uuid.UUID(memory_id),
                    MemoryItem.agent_name == agent_name
                )
                result = await session.execute(stmt)
                await session.commit()
                deleted_count = result.rowcount
                logger.debug(f"Deleted {deleted_count} memory item(s) for agent '{agent_name}' with ID {memory_id}")
                return deleted_count > 0
            except ValueError as e:
                logger.error(f"Invalid UUID format for memory_id '{memory_id}': {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Failed to delete memory for agent '{agent_name}' with ID {memory_id}: {str(e)}")
                await session.rollback()
                raise

    async def clear(self, agent_name: str) -> int:
        """
        Clear all memories for a specific agent.

        Args:
            agent_name (str): The name of the agent whose memories are to be cleared.

        Returns:
            int: The number of memory items deleted.

        Raises:
            Exception: If the database operation fails.
        """
        async with self.db_manager.get_session() as session:
            try:
                stmt = delete(MemoryItem).where(MemoryItem.agent_name == agent_name)
                result = await session.execute(stmt)
                await session.commit()
                deleted_count = result.rowcount
                logger.debug(f"Cleared {deleted_count} memory item(s) for agent '{agent_name}'")
                return deleted_count
            except Exception as e:
                logger.error(f"Failed to clear memories for agent '{agent_name}': {str(e)}")
                await session.rollback()
                raise