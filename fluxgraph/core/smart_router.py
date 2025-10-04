# fluxgraph/core/smart_router.py
"""
Smart Agent Router for FluxGraph.
AI-powered routing based on query classification and agent capabilities.
"""

import logging
from typing import Any, Dict, List, Optional
import re

logger = logging.getLogger(__name__)


class AgentRouter:
    """
    Intelligent router that selects the best agent for a given query.
    Uses keyword matching and agent performance history.
    """
    
    def __init__(self):
        self.agent_capabilities: Dict[str, Dict[str, Any]] = {}
        self.routing_history: List[Dict[str, Any]] = []
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        logger.info("AgentRouter initialized")
    
    def register_agent(
        self,
        agent_name: str,
        capabilities: List[str],
        keywords: List[str],
        priority: int = 0
    ):
        """
        Register an agent with its capabilities.
        
        Args:
            agent_name: Name of the agent
            capabilities: List of capability tags (e.g., ['code', 'python', 'debug'])
            keywords: List of keywords that trigger this agent
            priority: Routing priority (higher = preferred)
        """
        self.agent_capabilities[agent_name] = {
            "capabilities": [c.lower() for c in capabilities],
            "keywords": [k.lower() for k in keywords],
            "priority": priority
        }
        
        self.agent_performance[agent_name] = {
            "success_rate": 1.0,
            "avg_response_time": 0.0,
            "total_requests": 0
        }
        
        logger.info(
            f"Registered agent '{agent_name}' with capabilities: {capabilities}"
        )
    
    def route(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Route a query to the most appropriate agent.
        
        Args:
            query: User query or task description
            context: Optional context information
        
        Returns:
            Name of the selected agent
        """
        query_lower = query.lower()
        scores: Dict[str, float] = {}
        
        for agent_name, config in self.agent_capabilities.items():
            score = 0.0
            
            # Keyword matching
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    score += 10.0
            
            # Capability matching (partial word matches)
            for capability in config["capabilities"]:
                if capability in query_lower:
                    score += 5.0
            
            # Priority bonus
            score += config["priority"] * 2.0
            
            # Performance bonus (success rate)
            perf = self.agent_performance.get(agent_name, {})
            score += perf.get("success_rate", 0.5) * 3.0
            
            scores[agent_name] = score
        
        # Select agent with highest score
        if not scores:
            logger.warning("No agents registered for routing")
            raise ValueError("No agents available for routing")
        
        selected_agent = max(scores, key=scores.get)
        confidence = scores[selected_agent]
        
        # Log routing decision
        routing_entry = {
            "query": query[:100],
            "selected_agent": selected_agent,
            "confidence": confidence,
            "all_scores": scores
        }
        self.routing_history.append(routing_entry)
        
        logger.info(
            f"[Router] Query routed to '{selected_agent}' "
            f"(confidence: {confidence:.2f})"
        )
        
        return selected_agent
    
    def update_performance(
        self,
        agent_name: str,
        success: bool,
        response_time: float
    ):
        """Update agent performance metrics after execution."""
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = {
                "success_rate": 1.0,
                "avg_response_time": 0.0,
                "total_requests": 0
            }
        
        perf = self.agent_performance[agent_name]
        total = perf["total_requests"]
        
        # Update success rate (exponential moving average)
        perf["success_rate"] = (
            perf["success_rate"] * 0.9 + (1.0 if success else 0.0) * 0.1
        )
        
        # Update average response time
        perf["avg_response_time"] = (
            (perf["avg_response_time"] * total + response_time) / (total + 1)
        )
        
        perf["total_requests"] += 1
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "total_routes": len(self.routing_history),
            "registered_agents": len(self.agent_capabilities),
            "agent_performance": self.agent_performance
        }
