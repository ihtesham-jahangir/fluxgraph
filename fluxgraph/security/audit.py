# fluxgraph/security/audit.py
"""
Immutable Audit Logging System for FluxGraph.
Provides tamper-proof logs for compliance (GDPR, HIPAA, SOC2).
"""

import hashlib
import json
import logging
import sqlite3
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
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


class AuditEntry:
    """Represents a single immutable audit log entry."""
    
    def __init__(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        details: Dict[str, Any],
        previous_hash: Optional[str] = None
    ):
        self.timestamp = datetime.utcnow()
        self.event_type = event_type
        self.user_id = user_id
        self.details = details
        self.previous_hash = previous_hash
        
        # Calculate hash for blockchain-like immutability
        self.hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the entry."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "details": self.details,
            "previous_hash": self.previous_hash
        }
        
        data_string = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "details": json.dumps(self.details),
            "hash": self.hash,
            "previous_hash": self.previous_hash
        }


class AuditLogger:
    """
    Immutable audit logging system with blockchain-like verification.
    Logs are tamper-evident and suitable for compliance audits.
    """
    
    def __init__(self, db_path: str = "./audit_logs.db"):
        self.db_path = db_path
        self._init_database()
        self._last_hash: Optional[str] = self._get_last_hash()
        logger.info(f"AuditLogger initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize audit log database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Audit logs table with hash chain
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT,
                details TEXT NOT NULL,
                hash TEXT NOT NULL UNIQUE,
                previous_hash TEXT,
                FOREIGN KEY (previous_hash) REFERENCES audit_logs(hash)
            )
        """)
        
        # Indexes for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON audit_logs(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type 
            ON audit_logs(event_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_id 
            ON audit_logs(user_id)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("Audit database schema initialized")
    
    def _get_last_hash(self) -> Optional[str]:
        """Get the hash of the most recent audit entry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT hash FROM audit_logs 
            ORDER BY id DESC LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else None
    
    def log(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        details: Dict[str, Any],
        severity: str = "INFO"
    ):
        """
        Log an audit event.
        
        Args:
            event_type: Type of event being logged
            user_id: ID of user who triggered the event
            details: Event details and context
            severity: Log severity (INFO, WARNING, ERROR, CRITICAL)
        """
        # Create audit entry with hash chain
        entry = AuditEntry(
            event_type=event_type,
            user_id=user_id,
            details={**details, "severity": severity},
            previous_hash=self._last_hash
        )
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            entry_dict = entry.to_dict()
            cursor.execute("""
                INSERT INTO audit_logs 
                (timestamp, event_type, user_id, details, hash, previous_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entry_dict["timestamp"],
                entry_dict["event_type"],
                entry_dict["user_id"],
                entry_dict["details"],
                entry_dict["hash"],
                entry_dict["previous_hash"]
            ))
            
            conn.commit()
            self._last_hash = entry.hash
            
            # Log to standard logger as well
            log_msg = (
                f"[AUDIT] {event_type.value} | User: {user_id or 'system'} | "
                f"Details: {json.dumps(details)}"
            )
            
            if severity == "CRITICAL":
                logger.critical(log_msg)
            elif severity == "ERROR":
                logger.error(log_msg)
            elif severity == "WARNING":
                logger.warning(log_msg)
            else:
                logger.info(log_msg)
                
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to log audit entry: {e}")
            raise
        finally:
            conn.close()
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the audit log chain.
        
        Returns:
            Dictionary with verification results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, event_type, user_id, details, hash, previous_hash
            FROM audit_logs
            ORDER BY id ASC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        total_entries = len(rows)
        verified = 0
        errors = []
        
        for i, row in enumerate(rows):
            entry_id, timestamp, event_type, user_id, details, stored_hash, previous_hash = row
            
            # Recreate entry to verify hash
            details_dict = json.loads(details)
            entry = AuditEntry(
                event_type=AuditEventType(event_type),
                user_id=user_id,
                details=details_dict,
                previous_hash=previous_hash
            )
            entry.timestamp = datetime.fromisoformat(timestamp)
            entry.hash = entry._calculate_hash()
            
            # Verify hash matches
            if entry.hash != stored_hash:
                errors.append({
                    "entry_id": entry_id,
                    "error": "Hash mismatch - potential tampering detected",
                    "expected": entry.hash,
                    "actual": stored_hash
                })
            else:
                verified += 1
            
            # Verify chain integrity
            if i > 0 and previous_hash != rows[i-1][5]:  # rows[i-1][5] is previous entry's hash
                errors.append({
                    "entry_id": entry_id,
                    "error": "Chain broken - previous_hash doesn't match",
                    "expected": rows[i-1][5],
                    "actual": previous_hash
                })
        
        is_valid = len(errors) == 0
        
        result = {
            "is_valid": is_valid,
            "total_entries": total_entries,
            "verified_entries": verified,
            "errors": errors
        }
        
        if is_valid:
            logger.info(f"Audit log integrity verified: {verified}/{total_entries} entries")
        else:
            logger.error(f"Audit log integrity FAILED: {len(errors)} errors found")
        
        return result
    
    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query audit logs with filters.
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            start_date: Start of date range
            end_date: End of date range
            limit: Maximum number of results
        
        Returns:
            List of audit log entries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query_parts = ["SELECT * FROM audit_logs WHERE 1=1"]
        params = []
        
        if event_type:
            query_parts.append("AND event_type = ?")
            params.append(event_type.value)
        
        if user_id:
            query_parts.append("AND user_id = ?")
            params.append(user_id)
        
        if start_date:
            query_parts.append("AND timestamp >= ?")
            params.append(start_date.isoformat())
        
        if end_date:
            query_parts.append("AND timestamp <= ?")
            params.append(end_date.isoformat())
        
        query_parts.append("ORDER BY timestamp DESC LIMIT ?")
        params.append(limit)
        
        query = " ".join(query_parts)
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "event_type": row["event_type"],
                "user_id": row["user_id"],
                "details": json.loads(row["details"]),
                "hash": row["hash"]
            })
        
        conn.close()
        return results
    
    def export_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        output_path: str
    ):
        """
        Export compliance report for audit purposes.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            output_path: Path to save report
        """
        logs = self.query(start_date=start_date, end_date=end_date, limit=10000)
        
        report = {
            "report_generated": datetime.utcnow().isoformat(),
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_events": len(logs),
            "integrity_check": self.verify_integrity(),
            "events": logs
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Compliance report exported to: {output_path}")
