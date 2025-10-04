# fluxgraph/orchestration/handoff.py
"""
Agent-to-Agent (A2A) Handoff Protocol for FluxGraph.
Implements structured agent delegation with context preservation.
Inspired by Microsoft Agent Framework A2A protocol.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HandoffReason(Enum):
    """Reasons for agent handoff."""
    DELEGATION = "delegation"  # Agent delegates to specialist
    ESCALATION = "escalation"  # Task complexity exceeds capability
    HUMAN_REQUIRED = "human_required"  # Human intervention needed
    COMPLETED = "completed"  # Task completed, return to originator
    FAILED = "failed"  # Task failed, escalate to supervisor
    TIMEOUT = "timeout"  # Agent exceeded time limit


class HandoffContext:
    """
    Context preserved during agent handoff.
    Contains conversation history, intermediate results, and metadata.
    """
    
    def __init__(
        self,
        task_id: str,
        origin_agent: str,
        conversation_history: List[Dict[str, Any]],
        intermediate_results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.task_id = task_id
        self.origin_agent = origin_agent
        self.conversation_history = conversation_history
        self.intermediate_results = intermediate_results
        self.metadata = metadata or {}
        self.handoff_chain: List[str] = [origin_agent]
        self.created_at = datetime.utcnow()
    
    def add_to_chain(self, agent_name: str):
        """Add agent to handoff chain."""
        self.handoff_chain.append(agent_name)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "origin_agent": self.origin_agent,
            "handoff_chain": self.handoff_chain,
            "conversation_history": self.conversation_history,
            "intermediate_results": self.intermediate_results,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class HandoffRequest:
    """Represents a handoff request from one agent to another."""
    
    def __init__(
        self,
        from_agent: str,
        to_agent: str,
        context: HandoffContext,
        reason: HandoffReason,
        priority: int = 0,
        timeout: Optional[float] = None
    ):
        self.handoff_id = str(uuid.uuid4())
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.context = context
        self.reason = reason
        self.priority = priority
        self.timeout = timeout
        self.timestamp = datetime.utcnow()
        self.status = "pending"
        self.result: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "handoff_id": self.handoff_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "reason": self.reason.value,
            "priority": self.priority,
            "status": self.status,
            "timestamp": self.timestamp.isoformat()
        }


class HandoffProtocol:
    """
    Manages agent-to-agent handoffs with context preservation.
    Implements A2A protocol for structured delegation.
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.active_handoffs: Dict[str, HandoffRequest] = {}
        self.handoff_history: List[HandoffRequest] = []
        self.handoff_callbacks: Dict[str, Callable] = {}
        logger.info("HandoffProtocol initialized")
    
    async def initiate_handoff(
        self,
        from_agent: str,
        to_agent: str,
        task_id: str,
        conversation_history: List[Dict[str, Any]],
        intermediate_results: Dict[str, Any],
        reason: HandoffReason,
        priority: int = 0,
        timeout: Optional[float] = 300.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HandoffRequest:
        """
        Initiate a handoff from one agent to another.
        
        Args:
            from_agent: Name of delegating agent
            to_agent: Name of target agent
            task_id: Unique task identifier
            conversation_history: Previous conversation messages
            intermediate_results: Results from previous agent work
            reason: Reason for handoff
            priority: Handoff priority (lower = higher priority)
            timeout: Maximum execution time for target agent
            metadata: Additional context metadata
        
        Returns:
            HandoffRequest object
        """
        # Create context
        context = HandoffContext(
            task_id=task_id,
            origin_agent=from_agent,
            conversation_history=conversation_history,
            intermediate_results=intermediate_results,
            metadata=metadata
        )
        context.add_to_chain(to_agent)
        
        # Create handoff request
        handoff = HandoffRequest(
            from_agent=from_agent,
            to_agent=to_agent,
            context=context,
            reason=reason,
            priority=priority,
            timeout=timeout
        )
        
        self.active_handoffs[handoff.handoff_id] = handoff
        
        logger.info(
            f"[Handoff:{handoff.handoff_id[:8]}] Initiated: {from_agent} → {to_agent} "
            f"(Reason: {reason.value}, Priority: {priority})"
        )
        
        # Execute handoff
        try:
            result = await self._execute_handoff(handoff)
            handoff.status = "completed"
            handoff.result = result
            
            logger.info(
                f"[Handoff:{handoff.handoff_id[:8]}] Completed: {to_agent} finished"
            )
            
        except asyncio.TimeoutError:
            handoff.status = "timeout"
            logger.error(
                f"[Handoff:{handoff.handoff_id[:8]}] Timeout: {to_agent} exceeded {timeout}s"
            )
            raise
            
        except Exception as e:
            handoff.status = "failed"
            logger.error(
                f"[Handoff:{handoff.handoff_id[:8]}] Failed: {to_agent} error: {e}"
            )
            raise
        
        finally:
            self.active_handoffs.pop(handoff.handoff_id, None)
            self.handoff_history.append(handoff)
        
        return handoff
    
    async def _execute_handoff(self, handoff: HandoffRequest) -> Any:
        """Execute the handoff by calling target agent."""
        payload = {
            "handoff_context": handoff.context.to_dict(),
            "task_id": handoff.context.task_id,
            "intermediate_results": handoff.context.intermediate_results,
            "conversation_history": handoff.context.conversation_history
        }
        
        # Execute with timeout
        if handoff.timeout:
            result = await asyncio.wait_for(
                self.orchestrator.run(handoff.to_agent, payload),
                timeout=handoff.timeout
            )
        else:
            result = await self.orchestrator.run(handoff.to_agent, payload)
        
        return result
    
    def register_handoff_callback(self, agent_name: str, callback: Callable):
        """
        Register a callback for when an agent receives a handoff.
        
        Args:
            agent_name: Name of agent to register callback for
            callback: Async function to call on handoff
        """
        self.handoff_callbacks[agent_name] = callback
        logger.info(f"Registered handoff callback for agent: {agent_name}")
    
    async def chain_handoff(
        self,
        agents: List[str],
        initial_payload: Dict[str, Any],
        task_id: Optional[str] = None
    ) -> List[Any]:
        """
        Execute a chain of handoffs across multiple agents.
        
        Args:
            agents: List of agent names in order
            initial_payload: Starting payload
            task_id: Optional task identifier
        
        Returns:
            List of results from each agent
        """
        task_id = task_id or str(uuid.uuid4())
        results = []
        conversation_history = []
        intermediate_results = initial_payload
        
        for i in range(len(agents)):
            current_agent = agents[i]
            next_agent = agents[i + 1] if i < len(agents) - 1 else None
            
            logger.info(
                f"[Chain:{task_id[:8]}] Step {i + 1}/{len(agents)}: Executing {current_agent}"
            )
            
            if next_agent:
                # Handoff to next agent
                handoff = await self.initiate_handoff(
                    from_agent=current_agent,
                    to_agent=next_agent,
                    task_id=task_id,
                    conversation_history=conversation_history,
                    intermediate_results=intermediate_results,
                    reason=HandoffReason.DELEGATION
                )
                
                result = handoff.result
            else:
                # Final agent in chain
                result = await self.orchestrator.run(current_agent, intermediate_results)
            
            results.append(result)
            
            # Update context for next handoff
            conversation_history.append({
                "agent": current_agent,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
            intermediate_results = result if isinstance(result, dict) else {"result": result}
        
        logger.info(f"[Chain:{task_id[:8]}] Completed all {len(agents)} agents")
        return results
    
    def get_handoff_statistics(self) -> Dict[str, Any]:
        """Get handoff statistics."""
        total = len(self.handoff_history)
        completed = sum(1 for h in self.handoff_history if h.status == "completed")
        failed = sum(1 for h in self.handoff_history if h.status == "failed")
        timeout = sum(1 for h in self.handoff_history if h.status == "timeout")
        
        return {
            "total_handoffs": total,
            "completed": completed,
            "failed": failed,
            "timeout": timeout,
            "success_rate": (completed / total * 100) if total > 0 else 0,
            "active_handoffs": len(self.active_handoffs)
        }
