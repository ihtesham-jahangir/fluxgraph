"""
Enhanced Memory System with Entity Extraction
"""
import asyncpg
import json
import re
from datetime import datetime
from typing import List, Dict, Optional

class EnhancedMemory:
    """Enhanced memory with entity extraction and temporal decay."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None
    
    async def initialize(self):
        """Initialize database connection."""
        self.pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    agent_name VARCHAR(255),
                    user_message TEXT,
                    agent_response TEXT,
                    intent VARCHAR(100),
                    timestamp TIMESTAMP DEFAULT NOW(),
                    metadata JSONB,
                    importance_score FLOAT DEFAULT 1.0
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255),
                    entity_type VARCHAR(50),
                    entity_value TEXT,
                    context TEXT,
                    first_seen TIMESTAMP DEFAULT NOW(),
                    last_seen TIMESTAMP DEFAULT NOW(),
                    mention_count INT DEFAULT 1
                )
            """)
    
    async def store_conversation(
        self,
        session_id: str,
        agent_name: str,
        user_message: str,
        agent_response: str,
        intent: str = None,
        metadata: dict = None
    ):
        """Store conversation with entity extraction."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO conversations 
                (session_id, agent_name, user_message, agent_response, intent, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, session_id, agent_name, user_message, agent_response, intent, json.dumps(metadata or {}))
            
            # Extract entities
            entities = self._extract_entities(user_message + " " + agent_response)
            for entity_type, value in entities:
                await self._upsert_entity(conn, session_id, entity_type, value, user_message)
    
    def _extract_entities(self, text: str) -> List[tuple]:
        """Extract entities from text."""
        entities = []
        
        # Email
        emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
        for email in emails:
            entities.append(('email', email))
        
        # Phone
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        for phone in phones:
            entities.append(('phone', phone))
        
        return entities
    
    async def _upsert_entity(self, conn, session_id: str, entity_type: str, entity_value: str, context: str):
        """Insert or update entity."""
        existing = await conn.fetchrow("""
            SELECT id, mention_count FROM entities 
            WHERE session_id = $1 AND entity_type = $2 AND entity_value = $3
        """, session_id, entity_type, entity_value)
        
        if existing:
            await conn.execute("""
                UPDATE entities SET last_seen = NOW(), mention_count = mention_count + 1
                WHERE id = $1
            """, existing['id'])
        else:
            await conn.execute("""
                INSERT INTO entities (session_id, entity_type, entity_value, context)
                VALUES ($1, $2, $3, $4)
            """, session_id, entity_type, entity_value, context[:500])
    
    async def get_session_entities(self, session_id: str) -> Dict[str, List[str]]:
        """Get all entities for a session."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT entity_type, entity_value, mention_count
                FROM entities WHERE session_id = $1
                ORDER BY mention_count DESC
            """, session_id)
            
            entities = {}
            for row in rows:
                if row['entity_type'] not in entities:
                    entities[row['entity_type']] = []
                entities[row['entity_type']].append(row['entity_value'])
            
            return entities
    
    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT user_message, agent_response, intent, timestamp
                FROM conversations WHERE session_id = $1
                ORDER BY timestamp DESC LIMIT $2
            """, session_id, limit)
            
            return [dict(row) for row in rows]
    
    async def generate_summary(self, session_id: str) -> str:
        """Generate conversation summary."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            messages = await conn.fetch("""
                SELECT user_message, agent_response FROM conversations
                WHERE session_id = $1 ORDER BY timestamp DESC LIMIT 20
            """, session_id)
            
            key_topics = set()
            for msg in messages:
                words = msg['user_message'].split()
                key_topics.update([w for w in words if len(w) > 5])
            
            return f"Discussed: {', '.join(list(key_topics)[:10])}"
