# fluxgraph/orchestration/task_adherence.py
"""
Task Adherence Monitoring for FluxGraph.
Ensures agents stay focused on assigned goals and don't drift.
Implements Microsoft Agent Framework-style goal tracking.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AdherenceLevel(Enum):
    """Task adherence levels."""
    EXCELLENT = "excellent"  # >90% adherence
    GOOD = "good"  # 70-90% adherence
    MODERATE = "moderate"  # 50-70% adherence
    POOR = "poor"  # 30-50% adherence
    CRITICAL = "critical"  # <30% adherence


class TaskGoal:
    """Represents a task goal with success criteria."""
    
    def __init__(
        self,
        goal_id: str,
        description: str,
        success_criteria: List[str],
        required_outputs: Optional[List[str]] = None,
        forbidden_actions: Optional[List[str]] = None
    ):
        self.goal_id = goal_id
        self.description = description
        self.success_criteria = success_criteria
        self.required_outputs = required_outputs or []
        self.forbidden_actions = forbidden_actions or []
        self.created_at = datetime.utcnow()


class AdherenceViolation:
    """Represents a detected adherence violation."""
    
    def __init__(
        self,
        violation_type: str,
        description: str,
        severity: str,
        agent_output: str
    ):
        self.violation_type = violation_type
        self.description = description
        self.severity = severity
        self.agent_output = agent_output
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type,
            "description": self.description,
            "severity": self.severity,
            "output_preview": self.agent_output[:200],
            "timestamp": self.timestamp.isoformat()
        }


class TaskAdherenceMonitor:
    """
    Monitors agent adherence to task goals.
    Detects drift, hallucination, and off-topic responses.
    """
    
    def __init__(self):
        self.active_goals: Dict[str, TaskGoal] = {}
        self.violations: List[AdherenceViolation] = []
        self.adherence_scores: Dict[str, List[float]] = {}
        logger.info("TaskAdherenceMonitor initialized")
    
    def register_goal(
        self,
        goal_id: str,
        description: str,
        success_criteria: List[str],
        required_outputs: Optional[List[str]] = None,
        forbidden_actions: Optional[List[str]] = None
    ):
        """
        Register a task goal for monitoring.
        
        Args:
            goal_id: Unique goal identifier
            description: Human-readable goal description
            success_criteria: List of criteria for success
            required_outputs: Expected output elements
            forbidden_actions: Actions that should not occur
        """
        goal = TaskGoal(
            goal_id=goal_id,
            description=description,
            success_criteria=success_criteria,
            required_outputs=required_outputs,
            forbidden_actions=forbidden_actions
        )
        
        self.active_goals[goal_id] = goal
        logger.info(f"[Adherence] Registered goal: {goal_id}")
    
    def evaluate_adherence(
        self,
        goal_id: str,
        agent_output: str,
        agent_actions: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate agent adherence to goal.
        
        Args:
            goal_id: Goal identifier
            agent_output: Agent's output text
            agent_actions: List of actions taken
            metadata: Additional evaluation context
        
        Returns:
            Adherence evaluation results
        """
        goal = self.active_goals.get(goal_id)
        if not goal:
            logger.warning(f"Goal {goal_id} not found for adherence check")
            return {"error": f"Goal {goal_id} not registered"}
        
        violations = []
        criteria_met = 0
        
        # Check success criteria
        for criterion in goal.success_criteria:
            if self._check_criterion(criterion, agent_output, metadata):
                criteria_met += 1
            else:
                violations.append(
                    AdherenceViolation(
                        violation_type="UNMET_CRITERION",
                        description=f"Success criterion not met: {criterion}",
                        severity="MEDIUM",
                        agent_output=agent_output
                    )
                )
        
        # Check required outputs
        for required in goal.required_outputs:
            if required.lower() not in agent_output.lower():
                violations.append(
                    AdherenceViolation(
                        violation_type="MISSING_OUTPUT",
                        description=f"Required output missing: {required}",
                        severity="HIGH",
                        agent_output=agent_output
                    )
                )
        
        # Check forbidden actions
        for forbidden in goal.forbidden_actions:
            if forbidden in agent_actions:
                violations.append(
                    AdherenceViolation(
                        violation_type="FORBIDDEN_ACTION",
                        description=f"Forbidden action detected: {forbidden}",
                        severity="CRITICAL",
                        agent_output=agent_output
                    )
                )
        
        # Calculate adherence score
        total_checks = len(goal.success_criteria) + len(goal.required_outputs)
        score = (criteria_met / total_checks * 100) if total_checks > 0 else 100
        
        # Adjust score based on violations
        violation_penalty = len(violations) * 10
        score = max(0, score - violation_penalty)
        
        # Determine adherence level
        level = self._determine_adherence_level(score)
        
        # Store violations
        self.violations.extend(violations)
        
        # Store score
        if goal_id not in self.adherence_scores:
            self.adherence_scores[goal_id] = []
        self.adherence_scores[goal_id].append(score)
        
        result = {
            "goal_id": goal_id,
            "adherence_score": score,
            "adherence_level": level.value,
            "criteria_met": criteria_met,
            "total_criteria": len(goal.success_criteria),
            "violations": [v.to_dict() for v in violations],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if violations:
            logger.warning(
                f"[Adherence:{goal_id}] Score: {score:.1f}% ({level.value}) "
                f"with {len(violations)} violations"
            )
        else:
            logger.info(
                f"[Adherence:{goal_id}] Score: {score:.1f}% ({level.value})"
            )
        
        return result
    
    def _check_criterion(
        self,
        criterion: str,
        output: str,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if a success criterion is met."""
        # Simple keyword-based check (can be enhanced with LLM evaluation)
        output_lower = output.lower()
        criterion_lower = criterion.lower()
        
        # Check for keyword presence
        keywords = criterion_lower.split()
        matches = sum(1 for keyword in keywords if keyword in output_lower)
        
        # Criterion met if >50% of keywords present
        return (matches / len(keywords)) > 0.5 if keywords else False
    
    def _determine_adherence_level(self, score: float) -> AdherenceLevel:
        """Determine adherence level from score."""
        if score >= 90:
            return AdherenceLevel.EXCELLENT
        elif score >= 70:
            return AdherenceLevel.GOOD
        elif score >= 50:
            return AdherenceLevel.MODERATE
        elif score >= 30:
            return AdherenceLevel.POOR
        else:
            return AdherenceLevel.CRITICAL
    
    def get_goal_statistics(self, goal_id: str) -> Dict[str, Any]:
        """Get adherence statistics for a goal."""
        scores = self.adherence_scores.get(goal_id, [])
        if not scores:
            return {"error": f"No data for goal {goal_id}"}
        
        goal_violations = [
            v for v in self.violations
            if goal_id in str(v.description)
        ]
        
        return {
            "goal_id": goal_id,
            "total_evaluations": len(scores),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "total_violations": len(goal_violations),
            "violation_types": {
                v_type: sum(1 for v in goal_violations if v.violation_type == v_type)
                for v_type in set(v.violation_type for v in goal_violations)
            }
        }
