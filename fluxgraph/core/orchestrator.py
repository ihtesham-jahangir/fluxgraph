# fluxgraph/core/orchestrator.py
from typing import Any, Dict
from .registry import AgentRegistry
import asyncio

class FluxOrchestrator:
    """
    Executes agent flows based on requests.
    """
    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    async def run(self, agent_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a registered agent with the given payload.
        """
        agent = self.registry.get(agent_name)
        try:
            if asyncio.iscoroutinefunction(agent.run):
                return await agent.run(**payload)
            else:
                return agent.run(**payload)
        except TypeError as e:
            raise ValueError(f"Agent '{agent_name}' execution failed: {e}")
