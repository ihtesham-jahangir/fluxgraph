"""
Verifiable, Append-Only Audit Logging System for FluxGraph.
Replaces the 'Blockchain' concept with a production-grade
hash-chaining implementation for compliance (GDPR, HIPAA, SOC2).
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import asyncpg  # Uses async postgres, not sqlite
from enum import Enum
logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of auditable events."""
    AGENT_EXECUTION = "agent_execution"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_CHECK = "permission_check"
    PERMISSION_DENIED = "permission_denied"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_ALERT = "security_alert"
    PII_DETECTED = "pii_detected"
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"


class VerifiableAuditLogger:
    """
    Implements a verifiable, append-only audit log in PostgreSQL
    using hash-chaining to ensure integrity.
    """
    
    def __init__(self, database_url: str):
        """
        Args:
            database_url (str): The connection string for the PostgreSQL database.
        """
        if not database_url:
            raise ValueError("database_url is required for VerifiableAuditLogger")
        self.database_url = database_url
        self.pool = None
        self.table_name = "verifiable_audit_log"
        # The "genesis_hash" is the hash for the very first entry's "previous_hash"
        self.genesis_hash = "fluxgraph_genesis_v1"

    async def setup(self):
        """Create the pool and the verifiable_audit_log table."""
        try:
            self.pool = await asyncpg.create_pool(dsn=self.database_url)
            async with self.pool.acquire() as conn:
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        actor TEXT,
                        action TEXT NOT NULL,
                        event_data JSONB,
                        previous_hash TEXT NOT NULL,
                        current_hash TEXT NOT NULL UNIQUE
                    );
                    -- Index for fast retrieval of the latest hash
                    CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                        ON {self.table_name} (timestamp DESC);
                    -- Index for querying by action or actor
                    CREATE INDEX IF NOT EXISTS idx_audit_action 
                        ON {self.table_name} (action);
                    CREATE INDEX IF NOT EXISTS idx_audit_actor 
                        ON {self.table_name} (actor);
                """)
            logger.info("VerifiableAuditLogger setup complete.")
        except Exception as e:
            logger.error(f"Failed to setup VerifiableAuditLogger: {e}")
            self.pool = None

    async def _get_latest_hash(self, conn: asyncpg.Connection) -> str:
        """Fetch the hash of the most recent log entry."""
        latest_hash = await conn.fetchval(
            f"SELECT current_hash FROM {self.table_name} "
            "ORDER BY timestamp DESC LIMIT 1"
        )
        return latest_hash or self.genesis_hash

    def _calculate_hash(
        self,
        timestamp: str,
        actor: Optional[str],
        action: str,
        event_data_json: str,
        previous_hash: str
    ) -> str:
        """Create a deterministic hash for the log entry."""
        sha256 = hashlib.sha256()
        
        # We create a stable, ordered string representation
        data = f"{timestamp}|{actor}|{action}|{event_data_json}|{previous_hash}"
        
        sha256.update(data.encode('utf-8'))
        return sha256.hexdigest()

    async def log(
        self,
        action: AuditEventType,
        actor: Optional[str] = "system",
        data: Optional[Dict[str, Any]] = None,
        severity: str = "INFO"
    ) -> Optional[str]:
        """
        Append a new, verifiable event to the audit log.
        
        Args:
            action (AuditEventType): The type of event being logged.
            actor (str): ID of the user or system component triggering the event.
            data (Dict): Event-specific details.
            severity (str): Log severity (INFO, WARNING, ERROR, CRITICAL).

        Returns:
            The hash of the new log entry, or None if logging failed.
        """
        if not self.pool:
            logger.warning("Logger pool not initialized. Attempting setup...")
            await self.setup()
            if not self.pool:
                logger.error("Failed to log event: Logger pool is not available.")
                return None

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # 1. Get the hash of the previous log entry
                    previous_hash = await self._get_latest_hash(conn)
                    
                    # 2. Prepare the new entry's data
                    now = datetime.now(timezone.utc)
                    now_iso = now.isoformat()
                    
                    full_data = (data or {}).copy()
                    full_data["severity"] = severity
                    
                    try:
                        # Use compact, sorted JSON for stable hashing
                        event_data_json = json.dumps(full_data, separators=(',', ':'), sort_keys=True)
                    except TypeError:
                        event_data_json = json.dumps({"error": "unserializable_data", "details": str(full_data)})

                    # 3. Calculate the new entry's hash
                    current_hash = self._calculate_hash(
                        timestamp=now_iso,
                        actor=actor,
                        action=action.value,
                        event_data_json=event_data_json,
                        previous_hash=previous_hash
                    )
                    
                    # 4. Insert the new entry
                    await conn.execute(f"""
                        INSERT INTO {self.table_name}
                            (timestamp, actor, action, event_data, previous_hash, current_hash)
                        VALUES ($1, $2, $3, $4, $5, $6);
                    """, now, actor, action.value, event_data_json, previous_hash, current_hash)
                    
                    logger.info(f"[AUDIT] {action.value} | Actor: {actor or 'system'}")
                    return current_hash
                
                except asyncpg.exceptions.UniqueViolationError:
                    # This should almost never happen, but it protects against
                    # two logs being written at the exact same microsecond
                    logger.warning(f"Hash collision or race condition, retrying log for action: {action.value}")
                    # Let the transaction retry logic (if any) handle this
                    raise
                except Exception as e:
                    logger.error(f"Failed to write to audit log: {e}", exc_info=True)
                    raise

    async def verify_log_chain(self) -> Dict[str, Any]:
        """
        Verifies the integrity of the entire audit log chain.
        
        Returns:
            A dictionary with the verification result.
        """
        if not self.pool:
            await self.setup()
            if not self.pool:
                raise RuntimeError("Audit logger pool is not available.")
        
        logger.info("Starting audit log chain verification...")
        async with self.pool.acquire() as conn:
            entries = await conn.fetch(
                f"SELECT * FROM {self.table_name} ORDER BY timestamp ASC"
            )
            
        if not entries:
            logger.info("Verification complete: Log is empty.")
            return {"is_valid": True, "total_entries": 0, "errors": []}

        expected_prev_hash = self.genesis_hash
        errors = []
        
        for i, record in enumerate(entries):
            # 1. Check if the previous hash matches what we expect
            if record['previous_hash'] != expected_prev_hash:
                error_detail = {
                    "entry_id": record['id'],
                    "error": "Chain broken - previous_hash mismatch",
                    "expected": expected_prev_hash,
                    "actual": record['previous_hash']
                }
                logger.error(f"CHAIN BROKEN! {error_detail}")
                errors.append(error_detail)
                # Stop verification on first chain break
                break
            
            # 2. Re-calculate the current hash and check if it's intact
            # Ensure JSON is loaded and dumped identically for hash calculation
            event_data_json = json.dumps(json.loads(record['event_data']), separators=(',', ':'), sort_keys=True)
            
            recalculated_hash = self._calculate_hash(
                timestamp=record['timestamp'].isoformat(),
                actor=record['actor'],
                action=record['action'],
                event_data_json=event_data_json,
                previous_hash=record['previous_hash']
            )
            
            if recalculated_hash != record['current_hash']:
                error_detail = {
                    "entry_id": record['id'],
                    "error": "Hash mismatch - potential data tampering",
                    "expected": recalculated_hash,
                    "actual": record['current_hash']
                }
                logger.error(f"DATA TAMPERED! {error_detail}")
                errors.append(error_detail)
                # Stop verification on first data tamper
                break
            
            # 3. Set the hash for the next iteration
            expected_prev_hash = record['current_hash']

        is_valid = len(errors) == 0
        result = {
            "is_valid": is_valid,
            "total_entries": len(entries),
            "verified_entries": len(entries) - len(errors),
            "errors": errors
        }
        
        if is_valid:
            logger.info(f"Verification complete: All {len(entries)} log entries are intact.")
        else:
            logger.error(f"Verification FAILED: {len(errors)} error(s) found.")
            
        return result

    async def query(
        self,
        action: Optional[AuditEventType] = None,
        actor: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query audit logs with filters.
        """
        if not self.pool:
            await self.setup()
            if not self.pool:
                raise RuntimeError("Audit logger pool is not available.")
        
        query_parts = [f"SELECT id, timestamp, actor, action, event_data, current_hash FROM {self.table_name} WHERE 1=1"]
        params = []
        
        if action:
            query_parts.append(f"AND action = ${len(params) + 1}")
            params.append(action.value)
        
        if actor:
            query_parts.append(f"AND actor = ${len(params) + 1}")
            params.append(actor)
        
        if start_date:
            query_parts.append(f"AND timestamp >= ${len(params) + 1}")
            params.append(start_date)
        
        if end_date:
            query_parts.append(f"AND timestamp <= ${len(params) + 1}")
            params.append(end_date)
        
        query_parts.append(f"ORDER BY timestamp DESC LIMIT ${len(params) + 1}")
        params.append(limit)
        
        query = " ".join(query_parts)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"].isoformat(),
                "actor": row["actor"],
                "action": row["action"],
                "details": json.loads(row["event_data"]),
                "hash": row["current_hash"]
            } for row in rows
        ]