# fluxgraph/crew/__init__.py
"""
CrewAI-inspired features for FluxGraph v3.3

Exports:
- RoleAgent, Task, Crew, ProcessType
- DelegationManager, DelegationStrategy
- HumanInputHandler, enable_human_input
- KnowledgeBase, PDFKnowledge, WebKnowledge, etc.
- ConditionalRouter, Condition
"""

from .role_agent import (
    RoleAgent, Task, Crew, ProcessType, AgentRole,
    create_agent, create_task, create_crew
)

from .delegation import (
    DelegationManager, DelegationStrategy
)

from .human import (
    HumanInputHandler, enable_human_input, InputType
)

from .knowledge import (
    KnowledgeBase, KnowledgeSource, KnowledgeSourceType,
    PDFKnowledge, WebKnowledge, MarkdownKnowledge,
    TextKnowledge, CSVKnowledge
)

from .routing import (
    ConditionalRouter, Condition, ConditionType,
    create_condition, value_equals, value_greater_than,
    has_key, always_true
)

__all__ = [
    # Role-based agents
    'RoleAgent', 'Task', 'Crew', 'ProcessType', 'AgentRole',
    'create_agent', 'create_task', 'create_crew',
    
    # Delegation
    'DelegationManager', 'DelegationStrategy',
    
    # Human-in-the-loop
    'HumanInputHandler', 'enable_human_input', 'InputType',
    
    # Knowledge sources
    'KnowledgeBase', 'KnowledgeSource', 'KnowledgeSourceType',
    'PDFKnowledge', 'WebKnowledge', 'MarkdownKnowledge',
    'TextKnowledge', 'CSVKnowledge',
    
    # Conditional routing
    'ConditionalRouter', 'Condition', 'ConditionType',
    'create_condition', 'value_equals', 'value_greater_than',
    'has_key', 'always_true'
]

__version__ = "3.3.0"
