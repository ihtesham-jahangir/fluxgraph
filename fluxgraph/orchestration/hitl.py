# fluxgraph/orchestration/hitl.py
"""
Human-in-the-Loop (HITL) Workflow System for FluxGraph.
Enables human approval and intervention in agent workflows.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    MODIFIED = "modified"  # Approved with modifications


class ApprovalRequest:
    """Represents a request for human approval."""
    
    def __init__(
        self,
        agent_name: str,
        task_description: str,
        proposed_action: Dict[str, Any],
        risk_level: str,
        timeout_seconds: int = 300,
        approvers: Optional[List[str]] = None
    ):
        self.request_id = str(uuid.uuid4())
        self.agent_name = agent_name
        self.task_description = task_description
        self.proposed_action = proposed_action
        self.risk_level = risk_level
        self.timeout_seconds = timeout_seconds
        self.approvers = approvers or []
        
        self.status = ApprovalStatus.PENDING
        self.created_at = datetime.utcnow()
        self.resolved_at: Optional[datetime] = None
        self.approved_by: Optional[str] = None
        self.rejection_reason: Optional[str] = None
        self.modified_action: Optional[Dict[str, Any]] = None
        
        self._approval_event = asyncio.Event()
    
    def approve(self, approver: str, modified_action: Optional[Dict[str, Any]] = None):
        """Approve the request."""
        if modified_action:
            self.status = ApprovalStatus.MODIFIED
            self.modified_action = modified_action
        else:
            self.status = ApprovalStatus.APPROVED
        
        self.approved_by = approver
        self.resolved_at = datetime.utcnow()
        self._approval_event.set()
        
        logger.info(
            f"[HITL:{self.request_id[:8]}] Approved by {approver} "
            f"{'with modifications' if modified_action else ''}"
        )
    
    def reject(self, approver: str, reason: str):
        """Reject the request."""
        self.status = ApprovalStatus.REJECTED
        self.approved_by = approver
        self.rejection_reason = reason
        self.resolved_at = datetime.utcnow()
        self._approval_event.set()
        
        logger.warning(
            f"[HITL:{self.request_id[:8]}] Rejected by {approver}: {reason}"
        )
    
    async def wait_for_approval(self) -> bool:
        """
        Wait for approval or timeout.
        
        Returns:
            True if approved, False if rejected or timeout
        """
        try:
            await asyncio.wait_for(
                self._approval_event.wait(),
                timeout=self.timeout_seconds
            )
            return self.status in [ApprovalStatus.APPROVED, ApprovalStatus.MODIFIED]
        except asyncio.TimeoutError:
            self.status = ApprovalStatus.TIMEOUT
            self.resolved_at = datetime.utcnow()
            logger.error(
                f"[HITL:{self.request_id[:8]}] Timeout after {self.timeout_seconds}s"
            )
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "agent_name": self.agent_name,
            "task_description": self.task_description,
            "proposed_action": self.proposed_action,
            "risk_level": self.risk_level,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "approved_by": self.approved_by,
            "rejection_reason": self.rejection_reason,
            "modified_action": self.modified_action
        }


class HITLManager:
    """
    Manages human-in-the-loop workflows.
    Handles approval requests, escalations, and human intervention.
    """
    
    def __init__(self):
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.request_history: List[ApprovalRequest] = []
        self.approval_callbacks: Dict[str, Callable] = {}
        logger.info("HITLManager initialized")
    
    async def request_approval(
        self,
        agent_name: str,
        task_description: str,
        proposed_action: Dict[str, Any],
        risk_level: str = "MEDIUM",
        timeout_seconds: int = 300,
        approvers: Optional[List[str]] = None,
        auto_approve_low_risk: bool = True
    ) -> ApprovalRequest:
        """
        Request human approval for an action.
        
        Args:
            agent_name: Name of requesting agent
            task_description: Human-readable description
            proposed_action: Action to be approved
            risk_level: LOW, MEDIUM, HIGH, CRITICAL
            timeout_seconds: Timeout for approval
            approvers: List of user IDs who can approve
            auto_approve_low_risk: Automatically approve low-risk actions
        
        Returns:
            ApprovalRequest object
        """
        request = ApprovalRequest(
            agent_name=agent_name,
            task_description=task_description,
            proposed_action=proposed_action,
            risk_level=risk_level,
            timeout_seconds=timeout_seconds,
            approvers=approvers
        )
        
        self.pending_requests[request.request_id] = request
        
        logger.info(
            f"[HITL:{request.request_id[:8]}] New approval request from {agent_name} "
            f"(Risk: {risk_level}, Timeout: {timeout_seconds}s)"
        )
        
        # Auto-approve low-risk actions if enabled
        if auto_approve_low_risk and risk_level == "LOW":
            logger.info(f"[HITL:{request.request_id[:8]}] Auto-approved (low risk)")
            request.approve("system", None)
        
        # Notify approval callbacks
        await self._notify_approval_callbacks(request)
        
        return request
    
    async def _notify_approval_callbacks(self, request: ApprovalRequest):
        """Notify registered callbacks about new approval request."""
        for callback in self.approval_callbacks.values():
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(request)
                else:
                    callback(request)
            except Exception as e:
                logger.error(f"Error in approval callback: {e}")
    
    def register_approval_callback(self, name: str, callback: Callable):
        """
        Register callback for new approval requests.
        
        Args:
            name: Callback identifier
            callback: Function to call (can be async)
        """
        self.approval_callbacks[name] = callback
        logger.info(f"Registered approval callback: {name}")
    
    def approve_request(
        self,
        request_id: str,
        approver: str,
        modified_action: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Approve a pending request.
        
        Args:
            request_id: ID of request to approve
            approver: User ID of approver
            modified_action: Optional modified action
        
        Returns:
            True if successful, False if request not found
        """
        request = self.pending_requests.get(request_id)
        if not request:
            logger.warning(f"Approval request {request_id} not found")
            return False
        
        # Check if approver is authorized
        if request.approvers and approver not in request.approvers:
            logger.warning(
                f"User {approver} not authorized to approve request {request_id}"
            )
            return False
        
        request.approve(approver, modified_action)
        self.pending_requests.pop(request_id)
        self.request_history.append(request)
        
        return True
    
    def reject_request(self, request_id: str, approver: str, reason: str) -> bool:
        """
        Reject a pending request.
        
        Args:
            request_id: ID of request to reject
            approver: User ID of approver
            reason: Reason for rejection
        
        Returns:
            True if successful, False if request not found
        """
        request = self.pending_requests.get(request_id)
        if not request:
            logger.warning(f"Approval request {request_id} not found")
            return False
        
        request.reject(approver, reason)
        self.pending_requests.pop(request_id)
        self.request_history.append(request)
        
        return True
    
    def get_pending_requests(
        self,
        agent_name: Optional[str] = None,
        risk_level: Optional[str] = None
    ) -> List[ApprovalRequest]:
        """Get pending approval requests with optional filters."""
        requests = list(self.pending_requests.values())
        
        if agent_name:
            requests = [r for r in requests if r.agent_name == agent_name]
        
        if risk_level:
            requests = [r for r in requests if r.risk_level == risk_level]
        
        return requests
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get HITL statistics."""
        total = len(self.request_history)
        approved = sum(
            1 for r in self.request_history
            if r.status in [ApprovalStatus.APPROVED, ApprovalStatus.MODIFIED]
        )
        rejected = sum(
            1 for r in self.request_history
            if r.status == ApprovalStatus.REJECTED
        )
        timeout = sum(
            1 for r in self.request_history
            if r.status == ApprovalStatus.TIMEOUT
        )
        
        return {
            "total_requests": total,
            "approved": approved,
            "rejected": rejected,
            "timeout": timeout,
            "pending": len(self.pending_requests),
            "approval_rate": (approved / total * 100) if total > 0 else 0
        }
