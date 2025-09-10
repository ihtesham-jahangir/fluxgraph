
"""
Database connection utilities for FluxGraph.
Handles setup and management of the SQLAlchemy async engine and session.
"""
import logging
from typing import Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import socket
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages the asynchronous connection to the PostgreSQL database.
    """

    def __init__(self, database_url: str):
        """
        Initialize the DatabaseManager.

        Args:
            database_url (str): The async PostgreSQL URL (e.g., 'postgresql+asyncpg://user:pass@host/dbname').
        """
        # Strip unsupported query parameters (e.g., sslmode, channel_binding)
        self.database_url = database_url.split("?")[0]
        self.engine = None
        self.async_session_factory: Optional[async_sessionmaker] = None
        logger.debug("DatabaseManager initialized with URL: %s", self.database_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((socket.gaierror, ConnectionError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying database connection (attempt {retry_state.attempt_number}/3)..."
        )
    )
    async def connect(self):
        """
        Create the async engine and session factory with retry logic for transient errors.

        Raises:
            Exception: If the connection fails after retries.
        """
        if self.engine is None:
            try:
                logger.info("Connecting to database at %s", self.database_url)
                # Create the async engine with SSL enabled for Neon PostgreSQL
                self.engine = create_async_engine(
                    self.database_url,
                    connect_args={"ssl": True},  # Enable SSL for Neon
                    pool_pre_ping=True,  # Check connections before use
                    pool_recycle=3600,  # Recycle connections after 1 hour
                    echo=False  # Set to True for debugging SQL queries
                )
                # Create the session factory
                self.async_session_factory = async_sessionmaker(
                    self.engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
                logger.info("Database connection established.")
            except Exception as e:
                logger.error("Failed to connect to database: %s", str(e))
                raise
        else:
            logger.warning("DatabaseManager.connect() called, but engine already exists.")

    async def disconnect(self):
        """
        Dispose of the async engine and clear the session factory.

        Raises:
            Exception: If disconnection fails.
        """
        if self.engine:
            try:
                logger.info("Closing database connection.")
                await self.engine.dispose()
                self.engine = None
                self.async_session_factory = None
                logger.info("Database connection closed.")
            except Exception as e:
                logger.error("Failed to disconnect from database: %s", str(e))
                raise
        else:
            logger.debug("DatabaseManager.disconnect() called, but no engine was active.")

    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """
        Provide an async session as a context manager.

        Yields:
            AsyncSession: An instance of AsyncSession for database operations.

        Raises:
            RuntimeError: If the manager is not connected.
        """
        if self.async_session_factory is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            logger.error("Session error: %s", str(e))
            await session.rollback()
            raise
        finally:
            await session.close()
