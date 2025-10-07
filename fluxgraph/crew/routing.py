# fluxgraph/crew/routing.py
"""
Conditional routing with or_ and and_ operators for complex workflows.
"""
from typing import List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class ConditionType(Enum):
    """Types of conditions."""
    OR = "or"
    AND = "and"
    NOT = "not"
    CUSTOM = "custom"


@dataclass
class Condition:
    """
    A routing condition.
    
    Example:
        condition = Condition(
            name="high_quality",
            check=lambda result: result.get("quality_score", 0) > 0.8
        )
    """
    name: str
    check: Callable[[Any], bool]
    description: Optional[str] = None
    
    def evaluate(self, input_data: Any) -> bool:
        """Evaluate the condition."""
        try:
            result = self.check(input_data)
            logger.debug(f"Condition '{self.name}': {result}")
            return result
        except Exception as e:
            logger.error(f"Condition '{self.name}' error: {e}")
            return False


class ConditionalRouter:
    """
    Route tasks based on conditions with or_/and_ operators.
    
    Example:
        router = ConditionalRouter()
        
        # OR condition: take first route that passes
        route = router.or_(
            (needs_research, research_agent),
            (needs_writing, writer_agent),
            (default_condition, default_agent)
        )
        
        # AND condition: all must pass
        route = router.and_(
            (is_approved, approved_condition),
            (is_complete, complete_condition)
        )
    """
    
    def __init__(self):
        self.routing_history: List[Dict] = []
    
    def or_(self, *routes: tuple) -> Callable:
        """
        OR routing: First matching condition wins.
        
        Args:
            *routes: Tuples of (condition, target)
            
        Returns:
            Routing function
            
        Example:
            route = router.or_(
                (is_urgent, urgent_agent),
                (is_complex, expert_agent),
                (always_true, default_agent)
            )
            
            agent = route(task_data)
        """
        def route_function(input_data: Any) -> Any:
            for condition, target in routes:
                if isinstance(condition, Condition):
                    if condition.evaluate(input_data):
                        logger.info(f"✅ OR route: '{condition.name}' matched -> {target}")
                        self._record_routing("OR", condition.name, target)
                        return target
                elif callable(condition):
                    if condition(input_data):
                        logger.info(f"✅ OR route: custom condition matched -> {target}")
                        self._record_routing("OR", "custom", target)
                        return target
            
            logger.warning("⚠️ No OR conditions matched, returning None")
            return None
        
        return route_function
    
    def and_(self, *conditions: tuple) -> Callable:
        """
        AND routing: All conditions must pass.
        
        Args:
            *conditions: Tuples of (condition, description)
            
        Returns:
            Boolean function
            
        Example:
            check = router.and_(
                (is_approved, "approved"),
                (is_complete, "complete"),
                (is_validated, "validated")
            )
            
            if check(data):
                # All conditions passed
                proceed()
        """
        def check_function(input_data: Any) -> bool:
            passed = []
            failed = []
            
            for condition, description in conditions:
                if isinstance(condition, Condition):
                    result = condition.evaluate(input_data)
                    if result:
                        passed.append(condition.name)
                    else:
                        failed.append(condition.name)
                elif callable(condition):
                    result = condition(input_data)
                    if result:
                        passed.append(description)
                    else:
                        failed.append(description)
            
            all_passed = len(failed) == 0
            
            if all_passed:
                logger.info(f"✅ AND conditions: All passed ({', '.join(passed)})")
            else:
                logger.warning(f"❌ AND conditions: Failed ({', '.join(failed)})")
            
            self._record_routing("AND", f"passed={len(passed)}, failed={len(failed)}", all_passed)
            return all_passed
        
        return check_function
    
    def not_(self, condition: Condition) -> Condition:
        """
        NOT routing: Invert condition.
        
        Example:
            not_urgent = router.not_(is_urgent)
        """
        return Condition(
            name=f"NOT_{condition.name}",
            check=lambda x: not condition.check(x),
            description=f"Inverted: {condition.description}"
        )
    
    def _record_routing(self, route_type: str, condition: str, target: Any):
        """Record routing decision."""
        self.routing_history.append({
            "type": route_type,
            "condition": condition,
            "target": str(target),
            "timestamp": asyncio.get_event_loop().time()
        })
    
    def get_routing_history(self) -> List[Dict]:
        """Get routing history."""
        return self.routing_history


# Predefined common conditions
def create_condition(name: str, check: Callable, description: Optional[str] = None) -> Condition:
    """Helper to create conditions."""
    return Condition(name=name, check=check, description=description)


# Common condition builders
def value_equals(key: str, value: Any) -> Condition:
    """Condition: key equals value."""
    return Condition(
        name=f"{key}_equals_{value}",
        check=lambda data: data.get(key) == value,
        description=f"Check if {key} == {value}"
    )


def value_greater_than(key: str, threshold: float) -> Condition:
    """Condition: value greater than threshold."""
    return Condition(
        name=f"{key}_gt_{threshold}",
        check=lambda data: data.get(key, 0) > threshold,
        description=f"Check if {key} > {threshold}"
    )


def has_key(key: str) -> Condition:
    """Condition: data has key."""
    return Condition(
        name=f"has_{key}",
        check=lambda data: key in data,
        description=f"Check if '{key}' exists"
    )


def always_true() -> Condition:
    """Condition that always passes (fallback)."""
    return Condition(
        name="always_true",
        check=lambda data: True,
        description="Always returns True (fallback)"
    )
