# fluxgraph/core/session_manager.py
"""
Session Management System for FluxGraph.
Handles conversation state and history with SQLite backend.
"""

import sqlite3
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class ConversationSession:
    """Represents a single conversation session."""
    
    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}
        self.message_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata,
            "message_count": self.message_count
        }


class SessionManager:
    """
    Manages conversation sessions with persistent storage.
    Uses SQLite for lightweight, file-based persistence.
    """
    
    def __init__(self, db_path: str = "./sessions.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()
        logger.info(f"SessionManager initialized with database: {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_database(self):
        """Initialize database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                metadata TEXT,
                message_count INTEGER DEFAULT 0
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_messages 
            ON messages(session_id, timestamp)
        """)
        
        conn.commit()
        logger.info("Database schema initialized")
    
    def create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationSession:
        """Create a new conversation session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            INSERT INTO sessions (session_id, user_id, created_at, last_activity, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, user_id, now, now, json.dumps(metadata or {})))
        
        conn.commit()
        
        session = ConversationSession(session_id, user_id)
        session.metadata = metadata or {}
        
        logger.info(f"[Session:{session_id}] Created new session for user: {user_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve a session by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM sessions WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        session = ConversationSession(row['session_id'], row['user_id'])
        session.created_at = datetime.fromisoformat(row['created_at'])
        session.last_activity = datetime.fromisoformat(row['last_activity'])
        session.metadata = json.loads(row['metadata'] or '{}')
        session.message_count = row['message_count']
        
        return session
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to a session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            INSERT INTO messages (session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, role, content, now, json.dumps(metadata or {})))
        
        # Update session activity
        cursor.execute("""
            UPDATE sessions 
            SET last_activity = ?, message_count = message_count + 1
            WHERE session_id = ?
        """, (now, session_id))
        
        conn.commit()
        
        logger.debug(f"[Session:{session_id}] Added {role} message")
    
    def get_messages(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get messages for a session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT role, content, timestamp, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (session_id, limit, offset))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                "role": row['role'],
                "content": row['content'],
                "timestamp": row['timestamp'],
                "metadata": json.loads(row['metadata'] or '{}')
            })
        
        return list(reversed(messages))  # Return in chronological order
    
    def delete_session(self, session_id: str):
        """Delete a session and all its messages."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        logger.info(f"[Session:{session_id}] Deleted")
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Delete sessions older than specified hours."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
        
        cursor.execute("""
            SELECT session_id FROM sessions WHERE last_activity < ?
        """, (cutoff,))
        
        expired_sessions = [row['session_id'] for row in cursor.fetchall()]
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)
