import asyncio
import json
from datetime import datetime
import asyncpg # Assuming you use asyncpg or similar

# --- ADD THIS IMPORT ---
from typing import Optional
# -------------------------

from fluxgraph.core.checkpointer import BaseCheckpointer
from fluxgraph.core.workflow_graph import WorkflowState

class PostgresCheckpointer(BaseCheckpointer):
    """
    A Checkpointer implementation that saves workflow state to PostgreSQL.
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        self.table_name = "workflow_runs"

    async def setup(self):
        """Create the pool and the necessary table if it doesn't exist."""
        self.pool = await asyncpg.create_pool(dsn=self.database_url)
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    workflow_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                );
            """)

    async def load_state(self, workflow_id: str) -> Optional[WorkflowState]: # <- This line now works
        """Load the state from the database."""
        if not self.pool:
            await self.setup()
            
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT state_json FROM {self.table_name} WHERE workflow_id = $1",
                workflow_id
            )
        
        if row:
            state_data = json.loads(row['state_json'])
            # Re-hydrate the WorkflowState object
            state = WorkflowState(**state_data)
            return state
        return None

    async def save_state(self, state: WorkflowState):
        """Save the state to the database (upsert)."""
        if not self.pool:
            await self.setup()
            
        # Add workflow_id to state data if not present
        if not state.workflow_id:
            raise ValueError("WorkflowState must have a workflow_id to be saved.")

        # Serialize the state
        # Assuming your WorkflowState class has a .to_dict() method
        state_data = state.to_dict() 
        state_json = json.dumps(state_data)
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"""
                    INSERT INTO {self.table_name} (workflow_id, state_json, updated_at)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (workflow_id)
                    DO UPDATE SET state_json = $2, updated_at = $3;
                """, state.workflow_id, state_json, datetime.utcnow())