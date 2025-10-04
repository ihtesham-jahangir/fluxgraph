# fluxgraph/core/circuit_breaker.py
"""
Circuit Breaker Pattern for FluxGraph.
Prevents cascading failures and provides automatic fallback routing.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failures detected, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker for agent execution.
    Automatically disables failing agents and routes to fallbacks.
    """
    
    def __init__(
        self,
        agent_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            agent_name: Name of the agent to protect
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            success_threshold: Consecutive successes needed to close circuit
        """
        self.agent_name = agent_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.utcnow()
        
        logger.info(f"Circuit breaker initialized for agent '{agent_name}'")
    
    def record_success(self):
        """Record a successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.info(
                f"[CircuitBreaker:{self.agent_name}] Success in HALF_OPEN "
                f"({self.success_count}/{self.success_threshold})"
            )
            
            if self.success_count >= self.success_threshold:
                self._transition_to_closed()
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record a failed execution."""
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test
            logger.warning(
                f"[CircuitBreaker:{self.agent_name}] Failed during recovery test"
            )
            self._transition_to_open()
        
        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            logger.warning(
                f"[CircuitBreaker:{self.agent_name}] Failure recorded "
                f"({self.failure_count}/{self.failure_threshold})"
            )
            
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()
    
    def can_execute(self) -> bool:
        """Check if the agent can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                time_since_failure = datetime.utcnow() - self.last_failure_time
                if time_since_failure > timedelta(seconds=self.recovery_timeout):
                    self._transition_to_half_open()
                    return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state."""
        self.state = CircuitState.OPEN
        self.last_state_change = datetime.utcnow()
        logger.error(
            f"🔴 [CircuitBreaker:{self.agent_name}] Circuit OPENED - "
            f"Agent disabled due to failures"
        )
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.last_state_change = datetime.utcnow()
        logger.info(
            f"🟡 [CircuitBreaker:{self.agent_name}] Circuit HALF_OPEN - "
            f"Testing recovery"
        )
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = datetime.utcnow()
        logger.info(
            f"🟢 [CircuitBreaker:{self.agent_name}] Circuit CLOSED - "
            f"Agent recovered"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "agent": self.agent_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat(),
            "can_execute": self.can_execute()
        }


class CircuitBreakerManager:
    """Manages circuit breakers for all agents."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_routes: Dict[str, str] = {}
        logger.info("CircuitBreakerManager initialized")
    
    def register_agent(
        self,
        agent_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        fallback_agent: Optional[str] = None
    ):
        """Register an agent with circuit breaker protection."""
        self.breakers[agent_name] = CircuitBreaker(
            agent_name=agent_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        
        if fallback_agent:
            self.fallback_routes[agent_name] = fallback_agent
            logger.info(
                f"Registered fallback route: {agent_name} → {fallback_agent}"
            )
    
    def get_breaker(self, agent_name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for an agent."""
        return self.breakers.get(agent_name)
    
    def get_fallback(self, agent_name: str) -> Optional[str]:
        """Get fallback agent name if circuit is open."""
        breaker = self.breakers.get(agent_name)
        if breaker and not breaker.can_execute():
            return self.fallback_routes.get(agent_name)
        return None
    
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self.breakers.items()
        }
