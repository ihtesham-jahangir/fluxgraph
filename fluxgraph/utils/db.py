# fluxgraph/utils/db.py
"""
Database connection utilities for FluxGraph.
Handles setup and management of the SQLAlchemy async engine and session.
"""
import logging
from typing import Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages the asynchronous connection to the PostgreSQL database.
    """

    def __init__(self, database_url: str):
        """
        Initializes the DatabaseManager.

        Args:
            database_url (str): The async PostgreSQL URL (e.g., 'postgresql+asyncpg://user:pass@localhost:5432/dbname').
        """
        self.database_url = database_url
        self.engine = None
        self.async_session_factory: Optional[async_sessionmaker] = None
        logger.debug("DatabaseManager initialized.")

    async def connect(self):
        """Creates the async engine and session factory."""
        if self.engine is None:
            logger.info(f"Connecting to database at {self.database_url}")
            # Create the async engine
            self.engine = create_async_engine(
                self.database_url,
                # echo=True, # Enable for SQL logging
                pool_pre_ping=True, # Check connections before use
                pool_recycle=3600, # Recycle connections after 1 hour
            )
            # Create the session factory
            self.async_session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False, # Recommended for async
            )
            logger.info("Database connection established.")
        else:
            logger.warning("DatabaseManager.connect() called, but engine already exists.")

    async def disconnect(self):
        """Disposes of the async engine."""
        if self.engine:
            logger.info("Closing database connection.")
            await self.engine.dispose()
            self.engine = None
            self.async_session_factory = None
            logger.info("Database connection closed.")
        else:
            logger.debug("DatabaseManager.disconnect() called, but no engine was active.")

    def get_session(self) -> AsyncSession:
        """
        Provides a new async session instance.

        Returns:
            AsyncSession: An instance of AsyncSession.

        Raises:
            RuntimeError: If the manager is not connected.
        """
        if self.async_session_factory is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        return self.async_session_factory()
