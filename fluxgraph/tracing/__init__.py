# fluxgraph/tracing/__init__.py
"""
LangSmith-style distributed tracing for FluxGraph v3.2
Tracks execution, performance, and debugging across chains and agents.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field
from contextvars import ContextVar
import uuid
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Context variable for current trace
current_trace: ContextVar[Optional['TraceRun']] = ContextVar('current_trace', default=None)


class RunType(Enum):
    """Type of run being traced."""
    CHAIN = "chain"
    LLM = "llm"
    TOOL = "tool"
    AGENT = "agent"
    RETRIEVER = "retriever"
    EMBEDDING = "embedding"


@dataclass
class TraceRun:
    """
    Represents a single traced run/span.
    
    Example:
        run = TraceRun(
            name="my_chain",
            run_type=RunType.CHAIN,
            inputs={"question": "What is AI?"}
        )
    """
    name: str
    run_type: RunType
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_run_id: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    children: List['TraceRun'] = field(default_factory=list)
    
    def end(self, outputs: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """End the trace run."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.outputs = outputs
        self.error = error
    
    def add_child(self, child: 'TraceRun'):
        """Add a child run."""
        child.parent_run_id = self.run_id
        self.children.append(child)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "name": self.name,
            "run_type": self.run_type.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "tags": self.tags,
            "children": [child.to_dict() for child in self.children]
        }


class Tracer:
    """
    Main tracing interface - LangSmith-style tracer.
    
    Example:
        tracer = Tracer(project_name="my_project")
        
        with tracer.trace("my_operation", RunType.CHAIN) as run:
            result = do_something()
            run.end(outputs={"result": result})
    """
    
    def __init__(
        self,
        project_name: str = "fluxgraph",
        enabled: bool = True,
        export_path: Optional[str] = None,
        export_format: str = "json",
        auto_export: bool = True
    ):
        self.project_name = project_name
        self.enabled = enabled
        self.export_path = Path(export_path) if export_path else Path("./traces")
        self.export_format = export_format
        self.auto_export = auto_export
        self.runs: List[TraceRun] = []
        
        if self.export_path and self.auto_export:
            self.export_path.mkdir(parents=True, exist_ok=True)
    
    def trace(self, name: str, run_type: RunType, **kwargs) -> 'TraceContext':
        """
        Create a traced context.
        
        Args:
            name: Name of the operation
            run_type: Type of run
            **kwargs: Additional metadata
            
        Returns:
            TraceContext manager
        """
        return TraceContext(self, name, run_type, **kwargs)
    
    def start_run(self, name: str, run_type: RunType, **kwargs) -> TraceRun:
        """Start a new trace run."""
        if not self.enabled:
            return TraceRun(name=name, run_type=run_type)
        
        parent = current_trace.get()
        
        run = TraceRun(
            name=name,
            run_type=run_type,
            parent_run_id=parent.run_id if parent else None,
            metadata=kwargs.get('metadata', {}),
            tags=kwargs.get('tags', []),
            inputs=kwargs.get('inputs', {})
        )
        
        if parent:
            parent.add_child(run)
        else:
            self.runs.append(run)
        
        return run
    
    def end_run(self, run: TraceRun, outputs: Optional[Dict] = None, error: Optional[str] = None):
        """End a trace run."""
        if not self.enabled:
            return
        
        run.end(outputs=outputs, error=error)
        
        if self.auto_export and not run.parent_run_id:
            self._export_run(run)
    
    def _export_run(self, run: TraceRun):
        """Export a run to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.project_name}_{run.name}_{timestamp}.{self.export_format}"
            filepath = self.export_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(run.to_dict(), f, indent=2)
            
            logger.debug(f"Exported trace to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export trace: {e}")
    
    def get_runs(self) -> List[TraceRun]:
        """Get all traced runs."""
        return self.runs
    
    def clear_runs(self):
        """Clear all traced runs."""
        self.runs.clear()
    
    async def span(self, name: str, run_type: RunType = RunType.CHAIN, **kwargs):
        """
        Async context manager for tracing.
        
        Example:
            async with tracer.span("my_op", RunType.CHAIN, inputs={"x": 1}):
                result = await do_async_work()
        """
        return AsyncTraceContext(self, name, run_type, **kwargs)


class TraceContext:
    """
    Synchronous context manager for tracing.
    
    Example:
        with TraceContext(tracer, "operation", RunType.CHAIN) as run:
            result = do_work()
            run.end(outputs={"result": result})
    """
    
    def __init__(self, tracer: Tracer, name: str, run_type: RunType, **kwargs):
        self.tracer = tracer
        self.name = name
        self.run_type = run_type
        self.kwargs = kwargs
        self.run: Optional[TraceRun] = None
        self.token = None
    
    def __enter__(self) -> TraceRun:
        self.run = self.tracer.start_run(self.name, self.run_type, **self.kwargs)
        self.token = current_trace.set(self.run)
        return self.run
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run:
            error = str(exc_val) if exc_val else None
            self.tracer.end_run(self.run, error=error)
        
        if self.token:
            current_trace.reset(self.token)


class AsyncTraceContext:
    """
    Asynchronous context manager for tracing.
    
    Example:
        async with AsyncTraceContext(tracer, "async_op", RunType.LLM) as run:
            result = await async_work()
    """
    
    def __init__(self, tracer: Tracer, name: str, run_type: RunType, **kwargs):
        self.tracer = tracer
        self.name = name
        self.run_type = run_type
        self.kwargs = kwargs
        self.run: Optional[TraceRun] = None
        self.token = None
    
    async def __aenter__(self) -> TraceRun:
        self.run = self.tracer.start_run(self.name, self.run_type, **self.kwargs)
        self.token = current_trace.set(self.run)
        return self.run
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.run:
            error = str(exc_val) if exc_val else None
            self.tracer.end_run(self.run, error=error)
        
        if self.token:
            current_trace.reset(self.token)


# Global tracer instance
_global_tracer: Optional[Tracer] = None


def configure_tracing(
    project_name: str = "fluxgraph",
    enabled: bool = True,
    export_path: str = "./traces",
    export_format: str = "json",
    auto_export: bool = True
) -> Tracer:
    """
    Configure global tracing.
    
    Example:
        tracer = configure_tracing(
            project_name="my_app",
            enabled=True,
            export_path="./my_traces"
        )
    """
    global _global_tracer
    _global_tracer = Tracer(
        project_name=project_name,
        enabled=enabled,
        export_path=export_path,
        export_format=export_format,
        auto_export=auto_export
    )
    logger.info(f"Tracing configured: project={project_name}, enabled={enabled}")
    return _global_tracer


def get_tracer() -> Optional[Tracer]:
    """Get the global tracer instance."""
    return _global_tracer


def trace(name: str, run_type: RunType = RunType.CHAIN, **kwargs):
    """
    Decorator for tracing functions.
    
    Example:
        @trace("my_function", RunType.CHAIN)
        def my_function(x):
            return x * 2
    """
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            tracer = get_tracer()
            if not tracer or not tracer.enabled:
                return func(*args, **func_kwargs)
            
            with tracer.trace(name, run_type, **kwargs) as run:
                try:
                    result = func(*args, **func_kwargs)
                    run.end(outputs={"result": result})
                    return result
                except Exception as e:
                    run.end(error=str(e))
                    raise
        return wrapper
    return decorator


def span(name: str, run_type: RunType = RunType.CHAIN):
    """
    Context manager for manual tracing.
    
    Example:
        with span("operation", RunType.CHAIN):
            do_work()
    """
    tracer = get_tracer()
    if tracer:
        return tracer.trace(name, run_type)
    return _NoOpContext()


class _NoOpContext:
    """No-op context manager when tracing is disabled."""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def end(self, **kwargs):
        pass


# Export all public APIs
__all__ = [
    'Tracer',
    'TraceRun',
    'RunType',
    'TraceContext',
    'AsyncTraceContext',
    'configure_tracing',
    'get_tracer',
    'trace',
    'span'
]
