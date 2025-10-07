import asyncpg
from .base import BaseConnector

class PostgresConnector(BaseConnector):
    """PostgreSQL connector."""
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(self.config['url'], min_size=1, max_size=10)
        self.connected = True
    
    async def disconnect(self):
        await self.pool.close()
        self.connected = False
    
    async def execute(self, operation: str, **kwargs):
        """Execute database operation."""
        async with self.pool.acquire() as conn:
            if operation == 'query':
                return await conn.fetch(kwargs['sql'], *kwargs.get('params', []))
            elif operation == 'execute':
                return await conn.execute(kwargs['sql'], *kwargs.get('params', []))
            elif operation == 'fetchone':
                return await conn.fetchrow(kwargs['sql'], *kwargs.get('params', []))
            elif operation == 'fetchall':
                return await conn.fetch(kwargs['sql'], *kwargs.get('params', []))
            else:
                raise ValueError(f"Unsupported operation: {operation}") 