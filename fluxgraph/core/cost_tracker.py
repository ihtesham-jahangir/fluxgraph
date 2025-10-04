# fluxgraph/core/cost_tracker.py
"""
Cost Tracking System for FluxGraph.
Track LLM API costs per agent, per request, and set budget alerts.
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Track and monitor LLM API costs across agents.
    """
    
    # Pricing per 1M tokens (as of Oct 2025)
    PRICING = {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "gemini-pro": {"input": 0.50, "output": 1.50},
    }
    
    def __init__(self):
        self.cost_by_agent: Dict[str, float] = defaultdict(float)
        self.cost_by_model: Dict[str, float] = defaultdict(float)
        self.token_usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0}
        )
        self.daily_costs: Dict[str, float] = defaultdict(float)
        self.budget_limits: Dict[str, float] = {}
        self.cost_history: List[Dict[str, Any]] = []
        self.start_time = datetime.utcnow()
        
        logger.info("CostTracker initialized")
    
    def track_usage(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None
    ) -> float:
        """
        Track token usage and calculate cost.
        
        Args:
            agent_name: Name of the agent
            model: Model name (e.g., 'gpt-4')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            custom_pricing: Optional custom pricing override
        
        Returns:
            Cost of this request in USD
        """
        pricing = custom_pricing or self.PRICING
        model_pricing = pricing.get(model, {"input": 0.0, "output": 0.0})
        
        # Calculate cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        # Update tracking
        self.cost_by_agent[agent_name] += total_cost
        self.cost_by_model[model] += total_cost
        
        self.token_usage[agent_name]["input"] += input_tokens
        self.token_usage[agent_name]["output"] += output_tokens
        
        # Track daily costs
        today = datetime.utcnow().date().isoformat()
        self.daily_costs[today] += total_cost
        
        # Record in history
        self.cost_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent_name,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": total_cost
        })
        
        logger.info(
            f"[CostTracker] {agent_name} ({model}): "
            f"{input_tokens} in + {output_tokens} out = ${total_cost:.6f}"
        )
        
        # Check budget alerts
        self._check_budget_alerts(agent_name, total_cost)
        
        return total_cost
    
    def set_budget(self, agent_name: str, daily_limit: float):
        """Set daily budget limit for an agent."""
        self.budget_limits[agent_name] = daily_limit
        logger.info(f"Budget set for '{agent_name}': ${daily_limit:.2f}/day")
    
    def _check_budget_alerts(self, agent_name: str, cost: float):
        """Check if agent exceeded budget and log alert."""
        if agent_name in self.budget_limits:
            limit = self.budget_limits[agent_name]
            current_cost = self.cost_by_agent[agent_name]
            
            if current_cost > limit:
                logger.error(
                    f"🚨 [CostTracker] BUDGET EXCEEDED for '{agent_name}': "
                    f"${current_cost:.2f} / ${limit:.2f}"
                )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        total_cost = sum(self.cost_by_agent.values())
        runtime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "total_cost": round(total_cost, 4),
            "runtime_hours": round(runtime / 3600, 2),
            "cost_per_hour": round(total_cost / (runtime / 3600), 4) if runtime > 0 else 0,
            "cost_by_agent": {
                k: round(v, 4) for k, v in self.cost_by_agent.items()
            },
            "cost_by_model": {
                k: round(v, 4) for k, v in self.cost_by_model.items()
            },
            "total_requests": len(self.cost_history),
            "budget_status": {
                agent: {
                    "limit": limit,
                    "spent": round(self.cost_by_agent[agent], 4),
                    "remaining": round(limit - self.cost_by_agent[agent], 4),
                    "exceeded": self.cost_by_agent[agent] > limit
                }
                for agent, limit in self.budget_limits.items()
            }
        }
    
    def get_agent_report(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed cost report for a specific agent."""
        if agent_name not in self.cost_by_agent:
            return {"error": f"No data for agent '{agent_name}'"}
        
        return {
            "agent": agent_name,
            "total_cost": round(self.cost_by_agent[agent_name], 4),
            "input_tokens": self.token_usage[agent_name]["input"],
            "output_tokens": self.token_usage[agent_name]["output"],
            "budget_limit": self.budget_limits.get(agent_name),
            "requests": len([
                h for h in self.cost_history if h["agent"] == agent_name
            ])
        }
