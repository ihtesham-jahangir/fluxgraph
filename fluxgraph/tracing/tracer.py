# fluxgraph/tracing/tracer.py
"""
LangSmith-style distributed tracing for FluxGraph
Full observability with performance metrics and error tracking
"""
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json
import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RunStatus(Enum):
    """Status of a traced run"""
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


class RunType(Enum):
    """Types of traced operations"""
    CHAIN = "chain"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVER = "retriever"
    AGENT = "agent"
    EMBEDDING = "embedding"
    PARSER = "parser"
    PROMPT = "prompt"


@dataclass
class TokenUsage:
    """Token usage statistics"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def cost(self) -> float:
        """Estimate cost (OpenAI GPT-4 pricing)"""
        prompt_cost = self.prompt_tokens * 0.00003  # $0.03 per 1K tokens
        completion_cost = self.completion_tokens * 0.00006  # $0.06 per 1K tokens
        return prompt_cost + completion_cost


@dataclass
class TraceMetrics:
    """Performance metrics for a trace"""
    duration_ms: float = 0.0
    token_usage: Optional[TokenUsage] = None
    llm_calls: int = 0
    tool_calls: int = 0
    retrieval_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    
    @property
    def total_cost(self) -> float:
        """Total estimated cost"""
        return self.token_usage.cost if self.token_usage else 0.0


@dataclass
class TraceRun:
    """
    A single traced run/span in the execution tree.
    
    Captures comprehensive information about each operation:
    - Input/output data
    - Timing and performance metrics
    - Error information
    - Hierarchical relationships
    - Custom metadata and tags
    """
    run_id: str
    name: str
    run_type: RunType
    inputs: Dict[str, Any]
    status: RunStatus = RunStatus.RUNNING
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    parent_run_id: Optional[str] = None
    child_runs: List['TraceRun'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metrics: TraceMetrics = field(default_factory=TraceMetrics)
    
    @property
    def duration(self) -> timedelta:
        """Duration of the run"""
        if not self.end_time:
            return timedelta(0)
        return self.end_time - self.start_time
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds"""
        return self.duration.total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "run_id": self.run_id,
            "name": self.name,
            "run_type": self.run_type.value,
            "status": self.status.value,
            "inputs": self._serialize_data(self.inputs),
            "outputs": self._serialize_data(self.outputs) if self.outputs else None,
            "error": self.error,
            "error_type": self.error_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "parent_run_id": self.parent_run_id,
            "metadata": self.metadata,
            "tags": self.tags,
            "metrics": {
                "duration_ms": self.metrics.duration_ms,
                "llm_calls": self.metrics.llm_calls,
                "tool_calls": self.metrics.tool_calls,
                "total_cost": self.metrics.total_cost,
                "token_usage": {
                    "prompt_tokens": self.metrics.token_usage.prompt_tokens,
                    "completion_tokens": self.metrics.token_usage.completion_tokens,
                    "total_tokens": self.metrics.token_usage.total_tokens,
                } if self.metrics.token_usage else None
            },
            "child_runs": [c.to_dict() for c in self.child_runs]
        }
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON"""
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        else:
            return str(data)
    
    def to_langsmith_format(self) -> Dict[str, Any]:
        """Export in LangSmith-compatible format"""
        return {
            "id": self.run_id,
            "name": self.name,
            "run_type": self.run_type.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "parent_run_id": self.parent_run_id,
            "extra": {
                "metadata": self.metadata,
                "tags": self.tags,
                "metrics": self.metrics.__dict__
            }
        }


class Tracer:
    """
    Global tracer for comprehensive observability.
    
    Features:
    - Hierarchical trace trees
    - Performance metrics collection
    - Cost estimation
    - Export to multiple formats
    - Integration with existing logging
    
    Example:
        tracer = Tracer(
            project_name="my-project",
            enabled=True,
            export_path="./traces"
        )
        
        @tracer.trace("my_chain", run_type=RunType.CHAIN)
        async def my_chain(input):
            return await process(input)
    """
    
    def __init__(
        self,
        project_name: str = "fluxgraph",
        enabled: bool = True,
        export_path: Optional[str] = None,
        export_format: str = "json",
        auto_export: bool = True,
        sample_rate: float = 1.0
    ):
        self.project_name = project_name
        self.enabled = enabled
        self.export_path = Path(export_path) if export_path else None
        self.export_format = export_format
        self.auto_export = auto_export
        self.sample_rate = sample_rate
        
        self.runs: Dict[str, TraceRun] = {}
        self.current_run: Optional[TraceRun] = None
        self._context_stack: List[TraceRun] = []
        self._callbacks: List[Callable] = []
        
        if self.export_path:
            self.export_path.mkdir(parents=True, exist_ok=True)
    
    def trace(
        self,
        name: str,
        run_type: RunType = RunType.CHAIN,
        **metadata
    ):
        """
        Decorator to trace a function.
        
        Example:
            @tracer.trace("summarize", run_type=RunType.CHAIN, tags=["prod"])
            async def summarize(text):
                return await summarize_text(text)
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if not self.enabled or not self._should_sample():
                    return await func(*args, **kwargs)
                
                # Create run
                run = self._create_run(
                    name=name,
                    run_type=run_type,
                    inputs=self._capture_inputs(args, kwargs),
                    metadata=metadata
                )
                
                # Execute with context
                prev_run = self.current_run
                self.current_run = run
                self._context_stack.append(run)
                
                try:
                    # Execute function
                    result = await func(*args, **kwargs)
                    
                    # Record success
                    run.outputs = {"result": result}
                    run.status = RunStatus.SUCCESS
                    run.end_time = datetime.now()
                    run.metrics.duration_ms = run.duration_ms
                    
                    # Trigger callbacks
                    await self._trigger_callbacks("on_run_end", run)
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    run.error = str(e)
                    run.error_type = type(e).__name__
                    run.status = RunStatus.ERROR
                    run.end_time = datetime.now()
                    run.metrics.errors += 1
                    
                    # Trigger callbacks
                    await self._trigger_callbacks("on_run_error", run, e)
                    
                    raise
                    
                finally:
                    # Restore context
                    self._context_stack.pop()
                    self.current_run = prev_run
                    
                    # Auto export root runs
                    if self.auto_export and not run.parent_run_id:
                        self._export_run(run)
            
            return wrapper
        return decorator
    
    @asynccontextmanager
    async def span(
        self,
        name: str,
        run_type: RunType = RunType.CHAIN,
        inputs: Optional[Dict] = None,
        **metadata
    ):
        """
        Context manager for manual tracing.
        
        Example:
            async with tracer.span("processing", run_type=RunType.CHAIN):
                result = await process_data()
                tracer.current_run.outputs = {"result": result}
        """
        if not self.enabled or not self._should_sample():
            yield None
            return
        
        run = self._create_run(
            name=name,
            run_type=run_type,
            inputs=inputs or {},
            metadata=metadata
        )
        
        prev_run = self.current_run
        self.current_run = run
        
        try:
            await self._trigger_callbacks("on_run_start", run)
            yield run
            
            run.status = RunStatus.SUCCESS
            run.end_time = datetime.now()
            run.metrics.duration_ms = run.duration_ms
            
            await self._trigger_callbacks("on_run_end", run)
            
        except Exception as e:
            run.error = str(e)
            run.error_type = type(e).__name__
            run.status = RunStatus.ERROR
            run.end_time = datetime.now()
            run.metrics.errors += 1
            
            await self._trigger_callbacks("on_run_error", run, e)
            raise
            
        finally:
            self.current_run = prev_run
            
            if self.auto_export and not run.parent_run_id:
                self._export_run(run)
    
    def _create_run(
        self,
        name: str,
        run_type: RunType,
        inputs: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> TraceRun:
        """Create a new trace run"""
        run_id = str(uuid.uuid4())
        
        run = TraceRun(
            run_id=run_id,
            name=name,
            run_type=run_type,
            inputs=inputs,
            parent_run_id=self.current_run.run_id if self.current_run else None,
            metadata=metadata
        )
        
        self.runs[run_id] = run
        
        if self.current_run:
            self.current_run.child_runs.append(run)
        
        return run
    
    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled"""
        import random
        return random.random() < self.sample_rate
    
    def _capture_inputs(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Capture function inputs"""
        return {
            "args": [str(a)[:1000] for a in args],  # Truncate long strings
            "kwargs": {k: str(v)[:1000] for k, v in kwargs.items()}
        }
    
    async def _trigger_callbacks(self, event: str, *args):
        """Trigger registered callbacks"""
        for callback in self._callbacks:
            if hasattr(callback, event):
                try:
                    method = getattr(callback, event)
                    if asyncio.iscoroutinefunction(method):
                        await method(*args)
                    else:
                        method(*args)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")
    
    def _export_run(self, run: TraceRun):
        """Export run to file"""
        if not self.export_path:
            return
        
        try:
            filename = self.export_path / f"{run.run_id}.{self.export_format}"
            
            if self.export_format == "json":
                with open(filename, 'w') as f:
                    json.dump(run.to_dict(), f, indent=2)
            elif self.export_format == "langsmith":
                with open(filename, 'w') as f:
                    json.dump(run.to_langsmith_format(), f, indent=2)
            
            logger.debug(f"Exported trace to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export trace: {e}")
    
    def add_callback(self, callback: Callable):
        """Add a callback handler"""
        self._callbacks.append(callback)
    
    def get_trace(self, run_id: str) -> Optional[TraceRun]:
        """Get a specific trace by ID"""
        return self.runs.get(run_id)
    
    def get_root_traces(self) -> List[TraceRun]:
        """Get all root traces (no parent)"""
        return [r for r in self.runs.values() if not r.parent_run_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics"""
        root_traces = self.get_root_traces()
        
        total_duration = sum(t.duration_ms for t in root_traces)
        total_cost = sum(t.metrics.total_cost for t in root_traces)
        total_llm_calls = sum(t.metrics.llm_calls for t in root_traces)
        total_errors = sum(t.metrics.errors for t in root_traces)
        
        return {
            "total_traces": len(root_traces),
            "total_duration_ms": total_duration,
            "avg_duration_ms": total_duration / len(root_traces) if root_traces else 0,
            "total_cost": total_cost,
            "total_llm_calls": total_llm_calls,
            "total_errors": total_errors,
            "success_rate": (len(root_traces) - total_errors) / len(root_traces) if root_traces else 0
        }
    
    def clear(self):
        """Clear all traces"""
        self.runs.clear()
        self.current_run = None
        self._context_stack.clear()
    
    def export_all(self, format: str = "json") -> Path:
        """Export all traces to a single file"""
        if not self.export_path:
            raise ValueError("export_path not configured")
        
        filename = self.export_path / f"traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        
        root_traces = self.get_root_traces()
        data = {
            "project": self.project_name,
            "exported_at": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "traces": [t.to_dict() for t in root_traces]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(root_traces)} traces to {filename}")
        return filename


# Global tracer instance
_global_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get or create global tracer"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def configure_tracing(
    project_name: str = "fluxgraph",
    enabled: bool = True,
    export_path: Optional[str] = None,
    **kwargs
):
    """
    Configure global tracing.
    
    Example:
        configure_tracing(
            project_name="my-app",
            enabled=True,
            export_path="./traces"
        )
    """
    global _global_tracer
    _global_tracer = Tracer(
        project_name=project_name,
        enabled=enabled,
        export_path=export_path,
        **kwargs
    )
    return _global_tracer


def trace(name: str, run_type: RunType = RunType.CHAIN, **kwargs):
    """Convenience decorator using global tracer"""
    return get_tracer().trace(name, run_type, **kwargs)


async def span(name: str, run_type: RunType = RunType.CHAIN, **kwargs):
    """Convenience context manager using global tracer"""
    return get_tracer().span(name, run_type, **kwargs)
