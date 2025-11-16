# fluxgraph/core/postgres_logger.py
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import asyncpg

from .logger import BaseLogger, LogEventType

logger = logging.getLogger(__name__)

class PostgresWorkflowLogger(BaseLogger):
    """
    A Logger implementation that saves structured workflow events to PostgreSQL.
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        self.table_name = "workflow_events"

    async def setup(self):
        """Create the pool and the events table if it doesn't exist."""
        try:
            self.pool = await asyncpg.create_pool(dsn=self.database_url)
            async with self.pool.acquire() as conn:
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        event_id BIGSERIAL PRIMARY KEY,
                        workflow_id TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        node_id TEXT,
                        event_type TEXT NOT NULL,
                        event_data JSONB
                    );
                    CREATE INDEX IF NOT EXISTS idx_workflow_id ON {self.table_name} (workflow_id);
                """)
            logger.info("PostgresWorkflowLogger setup complete.")
        except Exception as e:
            logger.error(f"Failed to setup PostgresWorkflowLogger: {e}")
            self.pool = None

    async def log_event(
        self,
        workflow_id: str,
        node_id: Optional[str],
        event_type: LogEventType,
        data: Optional[Dict[str, Any]] = None
    ):
        """Log the event to the database."""
        if not self.pool:
            logger.warning("Logger pool not initialized. Attempting setup...")
            await self.setup()
            if not self.pool:
                logger.error("Failed to log event: Logger pool is not available.")
                return

        event_json = json.dumps(data) if data else None

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(f"""
                    INSERT INTO {self.table_name} (workflow_id, timestamp, node_id, event_type, event_data)
                    VALUES ($1, $2, $3, $4, $5);
                """, workflow_id, datetime.utcnow(), node_id, event_type.value, event_json)
        except Exception as e:
            logger.error(f"Failed to log event to Postgres: {e}")