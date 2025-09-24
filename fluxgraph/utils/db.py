# fluxgraph/utils/db.py
"""
Database Connection Utilities for FluxGraph.

This module provides the `DatabaseManager`, responsible for setting up and
managing the asynchronous connection to a PostgreSQL database using SQLAlchemy 2.x.

It handles the creation of the async engine, the session factory, and provides
a context manager for acquiring and managing database sessions. It also includes
robust connection retry logic for transient network issues.
"""
import logging
from typing import Optional
from contextlib import asynccontextmanager

# Import SQLAlchemy components
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Import retry utilities
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import specific exceptions for retry logic
import socket
import asyncio

# Use module-specific logger
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages the asynchronous connection to a PostgreSQL database for FluxGraph.

    This class handles the lifecycle of the SQLAlchemy async engine and session
    factory. It provides methods to connect, disconnect, and acquire database
    sessions safely using an asynchronous context manager.

    Features:
    - Asynchronous database connectivity using `asyncpg` driver.
    - Configurable SSL settings (important for cloud DBs like Neon).
    - Automatic connection retries with exponential backoff for resilience.
    - Context manager for safe session acquisition and cleanup.
    - Connection pooling with pre-ping and recycling.

    Attributes:
        database_url (str): The cleaned PostgreSQL connection URL.
        engine (Optional[AsyncEngine]): The SQLAlchemy async engine instance.
        async_session_factory (Optional[async_sessionmaker]): Factory for creating async sessions.
    """

    def __init__(self, database_url: str):
        """
        Initialize the DatabaseManager.

        Args:
            database_url (str): The full PostgreSQL connection URL.
                               (e.g., 'postgresql+asyncpg://user:pass@host:port/dbname?sslmode=require').
        """
        # Store the original URL for logging/transparency
        self._original_url = database_url
        # Basic cleaning: Remove query parameters that might confuse the driver
        # A more robust solution might parse and reconstruct the URL.
        # For now, this simple split handles common cases like sslmode.
        self.database_url = database_url.split("?")[0]
        self.engine = None
        self.async_session_factory: Optional[async_sessionmaker] = None
        logger.debug("DatabaseManager initialized with base URL: %s", self.database_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=( # Retry on specific network/connection errors
            retry_if_exception_type((socket.gaierror, ConnectionError, asyncio.TimeoutError))
            # Add database-specific errors if needed, e.g., from asyncpg
        ),
        before_sleep=lambda retry_state: logger.warning(
            f"Database connection attempt {retry_state.attempt_number}/3 failed. Retrying in {retry_state.next_action.sleep} seconds..."
        )
    )
    async def connect(self):
        """
        Create the async engine and session factory with retry logic.

        This method initializes the SQLAlchemy async engine and the session factory.
        It includes retry logic to handle transient network issues during
        the initial connection attempt. It's recommended to call this during
        application startup.

        Raises:
            Exception: If the connection fails after all retry attempts.
                       This includes network errors, authentication failures,
                       or issues with the database URL.
        """
        if self.engine is not None:
            logger.warning("DatabaseManager.connect() called, but engine already exists. Skipping.")
            return # Or raise an error if re-connecting should be forbidden

        try:
            logger.info("Attempting to connect to database at %s", self.database_url)
            
            # --- Create the Async Engine ---
            self.engine = create_async_engine(
                self.database_url,
                # Enable SSL. Crucial for services like Neon.tech
                # Adjust connect_args based on your specific DB provider's requirements.
                connect_args={
                    "ssl": True,
                    # Example for requiring full verification (if cert is available):
                    # "ssl": {"required": True, "ca": "/path/to/ca.crt"}
                },
                # Pool settings for robustness and performance
                pool_pre_ping=True,      # Validate connections before use
                pool_recycle=3600,       # Recycle connections after 1 hour
                pool_timeout=30,         # Timeout for getting a connection from the pool
                max_overflow=10,         # Allow up to 10 overflow connections
                # echo=True, # Enable for SQL query debugging (very verbose)
                echo=False # Set to False for production
            )

            # --- Create the Session Factory ---
            self.async_session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False, # Recommended for async to prevent stale data issues
                # autoflush=False, # Consider if you want manual control over flushing
                # autocommit=False, # Explicitly false, sessions require explicit commits
            )

            logger.info("✅ Database connection established successfully.")
            
        except Exception as e:
            logger.error("❌ Failed to connect to database after retries: %s", e, exc_info=True)
            # Re-raise the exception so the caller knows the connection failed
            raise RuntimeError(f"Database connection failed: {e}") from e

    async def disconnect(self):
        """
        Gracefully dispose of the async engine and clear the session factory.

        This method should be called during application shutdown to release
        database connections cleanly.

        Raises:
            Exception: If an error occurs while disposing the engine.
                       The engine is still set to None in this case.
        """
        if self.engine:
            try:
                logger.info("🔌 Closing database connection pool...")
                await self.engine.dispose()
                logger.info("✅ Database connection pool closed successfully.")
            except Exception as e:
                logger.error("❌ Error occurred while closing database connection: %s", e, exc_info=True)
                # Re-raise to signal failure, but ensure cleanup state
                raise RuntimeError(f"Database disconnection failed: {e}") from e
            finally:
                # Ensure state is reset regardless of success/failure
                self.engine = None
                self.async_session_factory = None
        else:
            logger.debug("DatabaseManager.disconnect() called, but no active engine found. Nothing to do.")

    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """
        Provide an asynchronous database session via a context manager.

        This is the primary way to obtain a session for performing database
        operations. It ensures that the session is committed on success,
        rolled back on error, and closed afterwards, preventing resource leaks.

        Yields:
            AsyncSession: An instance of SQLAlchemy's AsyncSession for database interaction.

        Raises:
            RuntimeError: If the DatabaseManager is not connected (i.e., `connect()`
                          has not been called successfully).
            Exception: Propagates any exceptions raised during session operations
                       (e.g., database errors during commit/rollback).
        """
        if self.async_session_factory is None:
            error_msg = (
                "DatabaseManager is not connected. "
                "Ensure `await db_manager.connect()` is called successfully before acquiring sessions."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Acquire a session from the factory
        session: AsyncSession = self.async_session_factory()
        logger.debug("Acquired new database session.")

        try:
            # Yield the session to the caller's 'async with' block
            yield session

            # If no exception occurred, commit the transaction
            logger.debug("Committing database session transaction.")
            await session.commit()

        except Exception as e:
            # If an exception occurred, rollback the transaction
            logger.warning("Rolling back database session transaction due to error: %s", e)
            await session.rollback()
            raise # Re-raise the exception to the caller

        finally:
            # Ensure the session is always closed, even if commit/rollback fails
            logger.debug("Closing database session.")
            await session.close()
