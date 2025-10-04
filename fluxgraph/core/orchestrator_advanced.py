# fluxgraph/core/orchestrator_advanced.py
"""
Advanced Orchestrator with Multi-Agent Communication, Circuit Breakers,
Smart Routing, and Cost Tracking.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from .registry import AgentRegistry
from .agent_messaging import AgentMessageBus
from .circuit_breaker import CircuitBreakerManager
from .smart_router import AgentRouter
from .cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class AdvancedOrchestrator:
    """
    Enhanced orchestrator with enterprise-grade features.
    """
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        
        # Initialize advanced features
        self.message_bus = AgentMessageBus(self)
        self.circuit_breakers = CircuitBreakerManager()
        self.router = AgentRouter()
        self.cost_tracker = CostTracker()
        
        logger.info("AdvancedOrchestrator initialized with all features")
    
    async def run(
        self,
        agent_name: str,
        payload: Dict[str, Any],
        auto_route: bool = False
    ) -> Dict[str, Any]:
        """
        Execute an agent with advanced features.
        
        Args:
            agent_name: Name of the agent to execute
            payload: Arguments for the agent
            auto_route: Use smart routing if True
        
        Returns:
            Agent execution result
        """
        start_time = time.time()
        
        # Smart routing (if enabled)
        if auto_route and "query" in payload:
            agent_name = self.router.route(payload["query"])
            logger.info(f"[Router] Auto-routed to agent: {agent_name}")
        
        # Circuit breaker check
        breaker = self.circuit_breakers.get_breaker(agent_name)
        if breaker and not breaker.can_execute():
            fallback = self.circuit_breakers.get_fallback(agent_name)
            if fallback:
                logger.warning(
                    f"[CircuitBreaker] {agent_name} unavailable, "
                    f"routing to fallback: {fallback}"
                )
                agent_name = fallback
            else:
                error_msg = f"Agent '{agent_name}' is temporarily unavailable (circuit open)"
                logger.error(f"[CircuitBreaker] {error_msg}")
                if breaker:
                    breaker.record_failure()
                raise RuntimeError(error_msg)
        
        # Inject message bus into payload
        payload["_message_bus"] = self.message_bus
        payload["_current_agent"] = agent_name
        
        # Execute agent
        try:
            logger.info(f"Executing agent '{agent_name}'")
            agent = self.registry.get(agent_name)
            
            if asyncio.iscoroutinefunction(agent.run):
                result = await agent.run(**payload)
            else:
                result = agent.run(**payload)
            
            # Record success
            execution_time = time.time() - start_time
            
            if breaker:
                breaker.record_success()
            
            self.router.update_performance(agent_name, True, execution_time)
            
            # Track costs (if token usage in result)
            if isinstance(result, dict) and "usage" in result:
                usage = result["usage"]
                self.cost_tracker.track_usage(
                    agent_name=agent_name,
                    model=usage.get("model", "unknown"),
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0)
                )
            
            logger.info(
                f"Agent '{agent_name}' completed successfully "
                f"({execution_time:.3f}s)"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failure
            if breaker:
                breaker.record_failure()
            
            self.router.update_performance(agent_name, False, execution_time)
            
            logger.error(
                f"Agent '{agent_name}' failed: {e} ({execution_time:.3f}s)"
            )
            raise
    
    # Expose sub-component methods
    async def send_agent_message(
        self,
        sender: str,
        receiver: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send message between agents."""
        return await self.message_bus.send_message(sender, receiver, payload)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "circuit_breakers": self.circuit_breakers.get_all_statuses(),
            "routing_stats": self.router.get_routing_stats(),
            "cost_summary": self.cost_tracker.get_summary(),
            "message_bus_stats": self.message_bus.get_statistics()
        }
