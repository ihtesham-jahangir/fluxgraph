# fluxgraph/core/registry.py
from typing import Dict, Any

class AgentRegistry:
    """
    Store and manage registered agents.
    """
    def __init__(self):
        self._agents: Dict[str, Any] = {}

    def add(self, name: str, agent: Any):
        """Register an agent instance under a name."""
        if not hasattr(agent, 'run'):
            raise ValueError(f"Agent '{name}' must have a 'run' method.")
        self._agents[name] = agent

    def get(self, name: str) -> Any:
        """Retrieve a registered agent by name."""
        agent = self._agents.get(name)
        if agent is None:
            raise ValueError(f"Agent '{name}' not found.")
        return agent

    def list_agents(self) -> list:
        """List all registered agent names."""
        return list(self._agents.keys())
