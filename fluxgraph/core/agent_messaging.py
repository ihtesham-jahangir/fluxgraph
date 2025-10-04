# fluxgraph/core/agent_messaging.py
"""
Multi-Agent Communication System for FluxGraph.
Enables agents to communicate, delegate tasks, and coordinate actions.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class AgentMessage:
    """Represents a message between agents."""
    
    def __init__(
        self,
        sender: str,
        receiver: str,
        payload: Dict[str, Any],
        message_type: str = "task",
        priority: int = 0,
        correlation_id: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.payload = payload
        self.message_type = message_type
        self.priority = priority
        self.correlation_id = correlation_id or self.id
        self.timestamp = datetime.utcnow().isoformat()
        self.status = "pending"
        self.response = None


class AgentMessageBus:
    """Central message bus for inter-agent communication."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.message_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.message_history: List[AgentMessage] = []
        self.active_conversations: Dict[str, List[AgentMessage]] = {}
        logger.info("AgentMessageBus initialized")
    
    async def send_message(
        self,
        sender: str,
        receiver: str,
        payload: Dict[str, Any],
        message_type: str = "task",
        priority: int = 0,
        wait_for_response: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Send a message from one agent to another.
        
        Args:
            sender: Name of the sending agent
            receiver: Name of the receiving agent
            payload: Data to send
            message_type: Type of message (task, query, notification)
            priority: Message priority (lower = higher priority)
            wait_for_response: Whether to wait for the receiver's response
        
        Returns:
            Response from the receiver if wait_for_response=True
        """
        message = AgentMessage(
            sender=sender,
            receiver=receiver,
            payload=payload,
            message_type=message_type,
            priority=priority
        )
        
        logger.info(
            f"[MessageBus] {sender} → {receiver}: {message_type} "
            f"(ID: {message.id[:8]}...)"
        )
        
        # Store in conversation history
        conv_key = f"{sender}:{receiver}"
        if conv_key not in self.active_conversations:
            self.active_conversations[conv_key] = []
        self.active_conversations[conv_key].append(message)
        
        # Execute the receiver agent
        try:
            message.status = "processing"
            result = await self.orchestrator.run(receiver, payload)
            
            message.status = "completed"
            message.response = result
            self.message_history.append(message)
            
            logger.info(
                f"[MessageBus] {receiver} completed task from {sender} "
                f"(ID: {message.id[:8]}...)"
            )
            
            if wait_for_response:
                return result
            
        except Exception as e:
            message.status = "failed"
            message.response = {"error": str(e)}
            self.message_history.append(message)
            
            logger.error(
                f"[MessageBus] Failed to deliver message from {sender} "
                f"to {receiver}: {e}"
            )
            
            if wait_for_response:
                raise
        
        return None
    
    async def broadcast(
        self,
        sender: str,
        receivers: List[str],
        payload: Dict[str, Any],
        wait_for_all: bool = False
    ) -> Dict[str, Any]:
        """
        Broadcast a message to multiple agents.
        
        Args:
            sender: Name of the sending agent
            receivers: List of receiver agent names
            payload: Data to broadcast
            wait_for_all: Wait for all responses if True
        
        Returns:
            Dictionary mapping receiver names to their responses
        """
        logger.info(f"[MessageBus] {sender} broadcasting to {len(receivers)} agents")
        
        if wait_for_all:
            tasks = [
                self.send_message(sender, receiver, payload, wait_for_response=True)
                for receiver in receivers
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return dict(zip(receivers, results))
        else:
            # Fire and forget
            for receiver in receivers:
                asyncio.create_task(
                    self.send_message(sender, receiver, payload, wait_for_response=False)
                )
            return {"status": "broadcast_initiated", "receivers": receivers}
    
    def get_conversation_history(
        self,
        agent1: str,
        agent2: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get conversation history between two agents."""
        conv_key1 = f"{agent1}:{agent2}"
        conv_key2 = f"{agent2}:{agent1}"
        
        messages = (
            self.active_conversations.get(conv_key1, []) +
            self.active_conversations.get(conv_key2, [])
        )
        
        # Sort by timestamp and limit
        sorted_messages = sorted(
            messages,
            key=lambda m: m.timestamp,
            reverse=True
        )[:limit]
        
        return [
            {
                "id": m.id,
                "sender": m.sender,
                "receiver": m.receiver,
                "type": m.message_type,
                "timestamp": m.timestamp,
                "status": m.status,
                "payload": m.payload,
                "response": m.response
            }
            for m in sorted_messages
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        total_messages = len(self.message_history)
        completed = sum(1 for m in self.message_history if m.status == "completed")
        failed = sum(1 for m in self.message_history if m.status == "failed")
        
        return {
            "total_messages": total_messages,
            "completed": completed,
            "failed": failed,
            "success_rate": (completed / total_messages * 100) if total_messages > 0 else 0,
            "active_conversations": len(self.active_conversations)
        }
