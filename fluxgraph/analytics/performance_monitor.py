# File: fluxgraph/analytics/performance_monitor.py
import time
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from dataclasses import dataclass, asdict
import json
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Metrics for a single agent execution."""
    request_id: str
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    input_size: int = 0
    output_size: int = 0
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            **asdict(self),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
        }


class PerformanceMonitor:
    """
    Monitors and tracks performance metrics for FluxGraph agents.
    """
    
    def __init__(self, max_metrics_history: int = 10000):
        self.max_metrics_history = max_metrics_history
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.active_requests: Dict[str, AgentMetrics] = {}
        self.agent_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_requests': 0,
            'total_successes': 0,
            'total_failures': 0,
            'total_duration_ms': 0,
            'avg_duration_ms': 0,
            'min_duration_ms': float('inf'),
            'max_duration_ms': 0,
            'total_tokens': 0,
            'total_cost_usd': 0,
            'last_request': None
        })
    
    def start_tracking(self, 
                      agent_name: str, 
                      request_data: Any = None,
                      user_id: str = None,
                      session_id: str = None) -> str:
        """Start tracking an agent execution."""
        request_id = str(uuid.uuid4())
        
        # Calculate input size
        input_size = 0
        if request_data:
            try:
                input_size = len(json.dumps(request_data, default=str))
            except (TypeError, ValueError):
                input_size = len(str(request_data))
        
        metrics = AgentMetrics(
            request_id=request_id,
            agent_name=agent_name,
            start_time=datetime.utcnow(),
            input_size=input_size,
            user_id=user_id,
            session_id=session_id
        )
        
        self.active_requests[request_id] = metrics
        logger.debug(f"Started tracking agent '{agent_name}' with request_id: {request_id}")
        
        return request_id
    
    def finish_tracking(self, 
                       request_id: str, 
                       success: bool = True,
                       error_message: str = None,
                       output_data: Any = None,
                       tokens_used: int = None,
                       cost_usd: float = None) -> Optional[AgentMetrics]:
        """Finish tracking an agent execution."""
        if request_id not in self.active_requests:
            logger.warning(f"Request ID {request_id} not found in active requests")
            return None
        
        metrics = self.active_requests.pop(request_id)
        metrics.end_time = datetime.utcnow()
        metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
        metrics.success = success
        metrics.error_message = error_message
        metrics.tokens_used = tokens_used
        metrics.cost_usd = cost_usd
        
        # Calculate output size
        if output_data:
            try:
                metrics.output_size = len(json.dumps(output_data, default=str))
            except (TypeError, ValueError):
                metrics.output_size = len(str(output_data))
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Update agent stats
        self._update_agent_stats(metrics)
        
        logger.debug(f"Finished tracking agent '{metrics.agent_name}' - Duration: {metrics.duration_ms:.2f}ms")
        
        return metrics
    
    def _update_agent_stats(self, metrics: AgentMetrics):
        """Update aggregate statistics for an agent."""
        stats = self.agent_stats[metrics.agent_name]
        
        stats['total_requests'] += 1
        stats['last_request'] = metrics.end_time.isoformat()
        
        if metrics.success:
            stats['total_successes'] += 1
        else:
            stats['total_failures'] += 1
        
        if metrics.duration_ms:
            stats['total_duration_ms'] += metrics.duration_ms
            stats['avg_duration_ms'] = stats['total_duration_ms'] / stats['total_requests']
            stats['min_duration_ms'] = min(stats['min_duration_ms'], metrics.duration_ms)
            stats['max_duration_ms'] = max(stats['max_duration_ms'], metrics.duration_ms)
        
        if metrics.tokens_used:
            stats['total_tokens'] += metrics.tokens_used
        
        if metrics.cost_usd:
            stats['total_cost_usd'] += metrics.cost_usd
    
    def get_agent_stats(self, agent_name: str) -> Dict:
        """Get statistics for a specific agent."""
        stats = self.agent_stats.get(agent_name, {})
        if stats.get('min_duration_ms') == float('inf'):
            stats['min_duration_ms'] = 0
        return stats
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all agents."""
        result = {}
        for agent_name, stats in self.agent_stats.items():
            result[agent_name] = self.get_agent_stats(agent_name)
        return result
    
    def get_recent_metrics(self, limit: int = 100) -> List[Dict]:
        """Get recent metrics history."""
        recent = list(self.metrics_history)[-limit:]
        return [metric.to_dict() for metric in recent]
    
    def get_metrics_by_timerange(self, 
                               start_time: datetime, 
                               end_time: datetime) -> List[Dict]:
        """Get metrics within a specific time range."""
        filtered_metrics = []
        for metric in self.metrics_history:
            if start_time <= metric.start_time <= end_time:
                filtered_metrics.append(metric.to_dict())
        return filtered_metrics
    
    def track_performance(self, func: Callable = None, *, agent_name: str = None):
        """
        Decorator to automatically track performance of agent functions.
        
        Usage:
        @monitor.track_performance
        def my_agent(input_data):
            return process_data(input_data)
        
        @monitor.track_performance(agent_name="custom_name")
        async def my_async_agent(input_data):
            return await process_data_async(input_data)
        """
        def decorator(func):
            actual_agent_name = agent_name or func.__name__
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    request_id = self.start_tracking(
                        agent_name=actual_agent_name,
                        request_data=kwargs if kwargs else args,
                        user_id=kwargs.get('user_id'),
                        session_id=kwargs.get('session_id')
                    )
                    
                    try:
                        result = await func(*args, **kwargs)
                        self.finish_tracking(
                            request_id=request_id,
                            success=True,
                            output_data=result
                        )
                        return result
                    except Exception as e:
                        self.finish_tracking(
                            request_id=request_id,
                            success=False,
                            error_message=str(e)
                        )
                        raise
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    request_id = self.start_tracking(
                        agent_name=actual_agent_name,
                        request_data=kwargs if kwargs else args,
                        user_id=kwargs.get('user_id'),
                        session_id=kwargs.get('session_id')
                    )
                    
                    try:
                        result = func(*args, **kwargs)
                        self.finish_tracking(
                            request_id=request_id,
                            success=True,
                            output_data=result
                        )
                        return result
                    except Exception as e:
                        self.finish_tracking(
                            request_id=request_id,
                            success=False,
                            error_message=str(e)
                        )
                        raise
                
                return sync_wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def clear_metrics(self):
        """Clear all metrics history and stats."""
        self.metrics_history.clear()
        self.agent_stats.clear()
        self.active_requests.clear()
        logger.info("Cleared all performance metrics")
