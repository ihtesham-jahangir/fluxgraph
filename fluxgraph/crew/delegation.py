# fluxgraph/crew/delegation.py
"""
Advanced task delegation with smart routing and load balancing.
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class DelegationStrategy(Enum):
    """Strategies for task delegation."""
    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"
    SKILL_MATCH = "skill_match"
    MANAGER_CHOICE = "manager_choice"


@dataclass
class DelegationManager:
    """
    Manages task delegation among agents.
    
    Example:
        manager = DelegationManager(agents=[agent1, agent2, agent3])
        agent = manager.select_agent_for_task(task, strategy=DelegationStrategy.SKILL_MATCH)
    """
    agents: List['RoleAgent']
    strategy: DelegationStrategy = DelegationStrategy.LEAST_BUSY
    
    # Tracking
    _agent_task_counts: Dict[str, int] = field(default_factory=dict)
    _delegation_history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize tracking."""
        for agent in self.agents:
            self._agent_task_counts[agent.role] = 0
    
    def select_agent_for_task(
        self,
        task: 'Task',
        strategy: Optional[DelegationStrategy] = None
    ) -> 'RoleAgent':
        """
        Select best agent for a task based on strategy.
        
        Args:
            task: Task to assign
            strategy: Override default strategy
            
        Returns:
            Selected agent
        """
        strategy = strategy or self.strategy
        
        if strategy == DelegationStrategy.ROUND_ROBIN:
            return self._round_robin()
        elif strategy == DelegationStrategy.LEAST_BUSY:
            return self._least_busy()
        elif strategy == DelegationStrategy.SKILL_MATCH:
            return self._skill_match(task)
        elif strategy == DelegationStrategy.MANAGER_CHOICE:
            return self._manager_choice(task)
        else:
            return self.agents[0]
    
    def _round_robin(self) -> 'RoleAgent':
        """Simple round-robin selection."""
        total_tasks = sum(self._agent_task_counts.values())
        agent_idx = total_tasks % len(self.agents)
        agent = self.agents[agent_idx]
        self._agent_task_counts[agent.role] += 1
        return agent
    
    def _least_busy(self) -> 'RoleAgent':
        """Select agent with fewest tasks."""
        min_tasks = min(self._agent_task_counts.values())
        for agent in self.agents:
            if self._agent_task_counts[agent.role] == min_tasks:
                self._agent_task_counts[agent.role] += 1
                return agent
        return self.agents[0]
    
    def _skill_match(self, task: 'Task') -> 'RoleAgent':
        """Match task to agent based on role keywords."""
        task_lower = task.description.lower()
        
        # Simple keyword matching
        keywords = {
            "research": ["research", "investigate", "study", "analyze data"],
            "write": ["write", "draft", "compose", "create content"],
            "analyze": ["analyze", "evaluate", "assess", "examine"],
            "code": ["code", "program", "develop", "implement"],
            "design": ["design", "create ui", "visual", "mockup"]
        }
        
        for agent in self.agents:
            agent_role_lower = agent.role.lower()
            for skill, words in keywords.items():
                if skill in agent_role_lower:
                    if any(word in task_lower for word in words):
                        self._agent_task_counts[agent.role] += 1
                        logger.info(f"🎯 Skill match: {task.description[:50]}... -> {agent.role}")
                        return agent
        
        # Default to least busy
        return self._least_busy()
    
    def _manager_choice(self, task: 'Task') -> 'RoleAgent':
        """Manager explicitly chooses (simplified: use least busy)."""
        return self._least_busy()
    
    def get_workload_report(self) -> Dict[str, Any]:
        """Get current workload distribution."""
        return {
            "agents": [
                {
                    "role": agent.role,
                    "tasks_assigned": self._agent_task_counts[agent.role]
                }
                for agent in self.agents
            ],
            "total_tasks": sum(self._agent_task_counts.values()),
            "strategy": self.strategy.value
        }
