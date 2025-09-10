# fluxgraph/core/postgres_memory.py
"""
PostgreSQL implementation of the Memory interface for FluxGraph.
Uses SQLAlchemy 2.x for async database interactions.
"""
import uuid
import logging
from typing import List, Dict, Any
from sqlalchemy import String, Text, DateTime, func, select, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB
from .memory import Memory
from ..utils.db import DatabaseManager

logger = logging.getLogger(__name__)

# --- SQLAlchemy Model Definition ---
class Base(DeclarativeBase):
    """Base class for declarative models."""
    pass

class MemoryItem(Base):
    """
    SQLAlchemy ORM model for storing agent memories in PostgreSQL.
    """
    __tablename__ = "agent_memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_name: Mapped[str] = mapped_column(String, index=True)
    data: Mapped[Dict[str, Any]] = mapped_column(JSONB) # Store arbitrary JSON data
    timestamp: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

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
        Initializes the PostgresMemory store.

        Args:
            db_manager (DatabaseManager): The database manager instance to use.
        """
        self.db_manager = db_manager
        logger.debug("PostgresMemory initialized.")

    async def add(self, agent_name: str, data: Dict[str, Any]) -> str:
        """
        Stores data in the PostgreSQL database.
        """
        async with self.db_manager.get_session() as session:
            memory_id = str(uuid.uuid4())
            new_memory = MemoryItem(
                id=uuid.UUID(memory_id),
                agent_name=agent_name,
                data=data
            )
            session.add(new_memory)
            await session.commit()
            await session.refresh(new_memory) # Ensure ID is set
            logger.debug(f"Added memory for agent '{agent_name}' with ID {memory_id}")
            return memory_id

    async def get(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves recent memories, ordered by timestamp descending (newest first).
        """
        async with self.db_manager.get_session() as session:
            stmt = (
                select(MemoryItem)
                .where(MemoryItem.agent_name == agent_name)
                .order_by(MemoryItem.timestamp.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            items = result.scalars().all()
            # Return data in a format suitable for agents
            return [item.data for item in items]

    async def delete(self, agent_name: str, memory_id: str) -> bool:
        """
        Deletes a specific memory item by ID and agent name.
        """
        async with self.db_manager.get_session() as session:
            stmt = delete(MemoryItem).where(
                MemoryItem.id == uuid.UUID(memory_id),
                MemoryItem.agent_name == agent_name
            )
            result = await session.execute(stmt)
            await session.commit()
            deleted_count = result.rowcount
            logger.debug(f"Deleted {deleted_count} memory item(s) for agent '{agent_name}' with ID {memory_id}")
            return deleted_count > 0

    async def clear(self, agent_name: str) -> int:
        """
        Clears all memories for a specific agent.
        """
        async with self.db_manager.get_session() as session:
            stmt = delete(MemoryItem).where(MemoryItem.agent_name == agent_name)
            result = await session.execute(stmt)
            await session.commit()
            deleted_count = result.rowcount
            logger.debug(f"Cleared {deleted_count} memory item(s) for agent '{agent_name}'")
            return deleted_count
