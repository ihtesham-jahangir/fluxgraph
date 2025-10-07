# fluxgraph/crew/role_agent.py
"""
CrewAI-style role-based agents with personalities and backstories.
Agents have specific roles, goals, and can work together in teams.
"""
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Predefined agent roles (extensible)"""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    REVIEWER = "reviewer"
    MANAGER = "manager"
    DEVELOPER = "developer"
    DESIGNER = "designer"
    DATA_SCIENTIST = "data_scientist"
    CUSTOMER_SUPPORT = "customer_support"
    SALES = "sales"
    CUSTOM = "custom"


@dataclass
class Task:
    """
    A task to be executed by an agent.
    
    Example:
        task = Task(
            description="Research AI market trends for 2025",
            expected_output="Detailed 5-page report with statistics",
            agent=researcher_agent,
            context=[previous_task1, previous_task2]
        )
    """
    description: str
    expected_output: str
    agent: Optional['RoleAgent'] = None
    context: List['Task'] = field(default_factory=list)
    output: Optional[Any] = None
    tools: List[Callable] = field(default_factory=list)
    async_execution: bool = False
    callback: Optional[Callable] = None
    
    # Execution metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    
    async def execute(self, **kwargs) -> Any:
        """Execute the task using assigned agent."""
        if not self.agent:
            raise ValueError("Task must have an agent assigned")
        
        self.started_at = datetime.now()
        
        try:
            # Build context from previous tasks
            context_str = ""
            if self.context:
                context_str = "\n\n📋 Context from previous tasks:\n"
                for ctx_task in self.context:
                    if ctx_task.output:
                        context_str += f"- {ctx_task.description}: {ctx_task.output}\n"
            
            full_description = f"{self.description}{context_str}"
            
            # Execute with agent
            logger.info(f"📝 Executing task: {self.description[:50]}...")
            self.output = await self.agent.execute_task(full_description, **kwargs)
            
            self.completed_at = datetime.now()
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
            
            logger.info(f"✅ Task completed in {self.duration_seconds:.2f}s")
            
            # Callback
            if self.callback:
                await self.callback(self)
            
            return self.output
            
        except Exception as e:
            self.error = str(e)
            self.completed_at = datetime.now()
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
            logger.error(f"❌ Task failed: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "description": self.description,
            "expected_output": self.expected_output,
            "output": str(self.output) if self.output else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error
        }


@dataclass
class RoleAgent:
    """
    CrewAI-style role-based agent with personality.
    
    Example:
        researcher = RoleAgent(
            role="Senior Market Researcher",
            goal="Conduct thorough market research and provide actionable insights",
            backstory="Expert with 10 years in market analysis. Known for deep dives.",
            verbose=True,
            allow_delegation=True
        )
    """
    role: str
    goal: str
    backstory: str
    verbose: bool = False
    allow_delegation: bool = True
    max_iter: int = 10
    max_rpm: Optional[int] = None  # Rate limiting
    memory: bool = True
    tools: List[Callable] = field(default_factory=list)
    llm: Optional[str] = "gpt-4"  # Model to use
    
    # Internal state
    task_history: List[Task] = field(default_factory=list)
    delegation_history: List[Dict[str, Any]] = field(default_factory=list)
    _memory_store: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize agent."""
        if self.verbose:
            logger.info(f"🤖 Initialized agent: {self.role}")
            logger.info(f"   Goal: {self.goal}")
            logger.info(f"   Backstory: {self.backstory[:100]}...")
    
    async def execute_task(self, task_description: str, **kwargs) -> str:
        """
        Execute a task with this agent's expertise.
        
        Args:
            task_description: Full task description with context
            **kwargs: Additional parameters
            
        Returns:
            Task output
        """
        try:
            from fluxgraph.models import ask
        except ImportError:
            raise RuntimeError("FluxGraph models not available")
        
        # Build system prompt with role, goal, and backstory
        system_prompt = f"""You are a {self.role}.

YOUR GOAL:
{self.goal}

YOUR BACKGROUND:
{self.backstory}

INSTRUCTIONS:
- Execute the task using your expertise and experience
- Provide detailed, high-quality output
- Be thorough and professional
- Think step by step

Execute the following task:"""
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🤖 {self.role} is working...")
            print(f"{'='*60}")
            print(f"Task: {task_description[:200]}...")
            print(f"{'='*60}\n")
        
        # Execute with LLM
        result = await ask(
            task_description,
            system=system_prompt,
            model=self.llm
        )
        
        # Store in memory
        if self.memory:
            self._memory_store.append({
                "task": task_description,
                "output": result,
                "timestamp": datetime.now().isoformat()
            })
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✅ {self.role} completed task")
            print(f"{'='*60}\n")
        
        return result
    
    async def delegate_task(self, task: Task, to_agent: 'RoleAgent') -> Any:
        """
        Delegate a task to another agent.
        
        Example:
            result = await manager.delegate_task(
                task=research_task,
                to_agent=researcher
            )
        """
        if not self.allow_delegation:
            raise ValueError(f"{self.role} is not allowed to delegate tasks")
        
        if self.verbose:
            print(f"\n📤 {self.role} delegating to {to_agent.role}")
        
        # Assign task to target agent
        task.agent = to_agent
        result = await task.execute()
        
        # Record delegation
        self.delegation_history.append({
            "task": task.description,
            "delegated_to": to_agent.role,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def get_memory(self) -> List[Dict[str, Any]]:
        """Get agent's memory."""
        return self._memory_store
    
    def clear_memory(self):
        """Clear agent's memory."""
        self._memory_store.clear()


# ===== PROCESS TYPES =====

class ProcessType(Enum):
    """Types of crew execution processes."""
    SEQUENTIAL = "sequential"  # Tasks one after another
    HIERARCHICAL = "hierarchical"  # Manager coordinates
    PARALLEL = "parallel"  # All tasks at once
    CONSENSUS = "consensus"  # Agents vote/agree


@dataclass
class Crew:
    """
    A crew of agents working together.
    
    Example:
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            process=ProcessType.SEQUENTIAL,
            verbose=True
        )
        
        result = await crew.kickoff()
    """
    agents: List[RoleAgent]
    tasks: List[Task]
    process: ProcessType = ProcessType.SEQUENTIAL
    verbose: bool = True
    manager_llm: Optional[str] = None
    max_rpm: Optional[int] = None
    
    # Execution results
    results: List[Any] = field(default_factory=list)
    execution_time: Optional[float] = None
    
    async def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start crew execution.
        
        Args:
            inputs: Optional input parameters
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = datetime.now()
        
        if self.verbose:
            print("\n" + "="*80)
            print(f"🚀 Starting crew with {len(self.agents)} agents")
            print(f"📋 Process type: {self.process.value}")
            print(f"📝 Tasks: {len(self.tasks)}")
            print("="*80 + "\n")
        
        # Execute based on process type
        if self.process == ProcessType.SEQUENTIAL:
            results = await self._execute_sequential(inputs)
        elif self.process == ProcessType.PARALLEL:
            results = await self._execute_parallel(inputs)
        elif self.process == ProcessType.HIERARCHICAL:
            results = await self._execute_hierarchical(inputs)
        elif self.process == ProcessType.CONSENSUS:
            results = await self._execute_consensus(inputs)
        else:
            raise ValueError(f"Unknown process type: {self.process}")
        
        end_time = datetime.now()
        self.execution_time = (end_time - start_time).total_seconds()
        
        if self.verbose:
            print("\n" + "="*80)
            print(f"✅ Crew execution completed in {self.execution_time:.2f}s")
            print("="*80 + "\n")
        
        return {
            "results": results,
            "process": self.process.value,
            "agents": [agent.role for agent in self.agents],
            "execution_time": self.execution_time,
            "tasks_completed": len(results)
        }
    
    async def _execute_sequential(self, inputs: Optional[Dict] = None) -> List[Any]:
        """Execute tasks one after another."""
        results = []
        
        for i, task in enumerate(self.tasks):
            if self.verbose:
                print(f"\n📝 Task {i+1}/{len(self.tasks)}")
                print(f"   Description: {task.description[:80]}...")
                print(f"   Agent: {task.agent.role if task.agent else 'Unassigned'}\n")
            
            result = await task.execute(**(inputs or {}))
            results.append(result)
        
        self.results = results
        return results
    
    async def _execute_parallel(self, inputs: Optional[Dict] = None) -> List[Any]:
        """Execute all tasks in parallel."""
        if self.verbose:
            print(f"\n⚡ Executing {len(self.tasks)} tasks in parallel\n")
        
        tasks = [task.execute(**(inputs or {})) for task in self.tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.results = results
        return results
    
    async def _execute_hierarchical(self, inputs: Optional[Dict] = None) -> List[Any]:
        """Execute with manager coordinating."""
        if self.verbose:
            print("\n👔 Manager coordinating team execution\n")
        
        # Find manager or use first agent
        manager = next((a for a in self.agents if "manager" in a.role.lower()), self.agents[0])
        
        results = []
        for task in self.tasks:
            # Manager decides which agent to assign
            if not task.agent:
                # Simple assignment: round-robin
                agent_idx = len(results) % len(self.agents)
                task.agent = self.agents[agent_idx]
            
            if self.verbose:
                print(f"👔 Manager assigns task to {task.agent.role}")
            
            result = await task.execute(**(inputs or {}))
            results.append(result)
        
        self.results = results
        return results
    
    async def _execute_consensus(self, inputs: Optional[Dict] = None) -> List[Any]:
        """Execute with consensus among agents."""
        if self.verbose:
            print("\n🤝 Building consensus among agents\n")
        
        results = []
        
        for task in self.tasks:
            if self.verbose:
                print(f"🤝 All agents working on: {task.description[:60]}...")
            
            # Each agent contributes
            agent_results = []
            for agent in self.agents:
                task_copy = Task(
                    description=task.description,
                    expected_output=task.expected_output,
                    agent=agent
                )
                result = await task_copy.execute(**(inputs or {}))
                agent_results.append(result)
            
            # Consensus: combine or vote (simplified: use first)
            consensus_result = agent_results[0]
            results.append(consensus_result)
        
        self.results = results
        return results


# ===== CONVENIENCE FUNCTIONS =====

def create_agent(
    role: str,
    goal: str,
    backstory: str,
    **kwargs
) -> RoleAgent:
    """
    Quick agent creation.
    
    Example:
        researcher = create_agent(
            role="Senior Researcher",
            goal="Conduct thorough research",
            backstory="PhD in data science with 10 years experience",
            verbose=True
        )
    """
    return RoleAgent(role=role, goal=goal, backstory=backstory, **kwargs)


def create_task(
    description: str,
    expected_output: str,
    agent: Optional[RoleAgent] = None,
    **kwargs
) -> Task:
    """
    Quick task creation.
    
    Example:
        task = create_task(
            description="Research AI trends",
            expected_output="5-page report",
            agent=researcher
        )
    """
    return Task(description=description, expected_output=expected_output, agent=agent, **kwargs)


def create_crew(
    agents: List[RoleAgent],
    tasks: List[Task],
    process: ProcessType = ProcessType.SEQUENTIAL,
    **kwargs
) -> Crew:
    """
    Quick crew creation.
    
    Example:
        crew = create_crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            process=ProcessType.SEQUENTIAL,
            verbose=True
        )
    """
    return Crew(agents=agents, tasks=tasks, process=process, **kwargs)
