# fluxgraph/chains/streaming.py
"""
Time-to-First-Token (TTFT) optimization and streaming utilities
Minimize latency for real-time user experiences
"""
from typing import Any, AsyncIterator, Optional, List, Callable
import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
from fluxgraph.chains import Runnable, RunnableConfig

logger = logging.getLogger(__name__)


@dataclass
class StreamMetrics:
    """Metrics for streaming performance"""
    time_to_first_token: Optional[float] = None  # milliseconds
    tokens_per_second: Optional[float] = None
    total_tokens: int = 0
    total_duration: float = 0.0  # milliseconds
    
    @property
    def latency_grade(self) -> str:
        """Grade the latency performance"""
        if not self.time_to_first_token:
            return "N/A"
        
        ttft = self.time_to_first_token
        if ttft < 200:
            return "Excellent (<200ms)"
        elif ttft < 500:
            return "Good (200-500ms)"
        elif ttft < 1000:
            return "Fair (500-1000ms)"
        else:
            return "Poor (>1000ms)"


class StreamBuffer:
    """
    Buffer for optimizing streaming output.
    
    Features:
    - Automatic batching of small chunks
    - Backpressure handling
    - Smooth output rate
    """
    
    def __init__(
        self,
        min_chunk_size: int = 5,
        max_buffer_size: int = 1000,
        flush_interval: float = 0.05  # 50ms
    ):
        self.min_chunk_size = min_chunk_size
        self.max_buffer_size = max_buffer_size
        self.flush_interval = flush_interval
        self._buffer: List[str] = []
        self._buffer_size = 0
        self._last_flush = datetime.now()
    
    async def add(self, chunk: str) -> Optional[str]:
        """
        Add chunk to buffer, return flushed content if ready.
        
        Args:
            chunk: Text chunk to buffer
            
        Returns:
            Buffered content if ready to flush, else None
        """
        self._buffer.append(chunk)
        self._buffer_size += len(chunk)
        
        # Flush if buffer is large enough or time elapsed
        should_flush = (
            self._buffer_size >= self.min_chunk_size or
            (datetime.now() - self._last_flush).total_seconds() >= self.flush_interval or
            self._buffer_size >= self.max_buffer_size
        )
        
        if should_flush:
            return await self.flush()
        
        return None
    
    async def flush(self) -> str:
        """Flush buffer and return content"""
        if not self._buffer:
            return ""
        
        content = "".join(self._buffer)
        self._buffer.clear()
        self._buffer_size = 0
        self._last_flush = datetime.now()
        
        return content


class StreamOptimizer(Runnable):
    """
    Optimize streaming for minimal TTFT and smooth output.
    
    Features:
    - Parallel processing of prompt and initial generation
    - Smart buffering to reduce network overhead
    - Metrics collection for monitoring
    
    Example:
        optimizer = StreamOptimizer(chain)
        
        async for chunk in optimizer.stream(input):
            print(chunk, end="", flush=True)
    """
    
    def __init__(
        self,
        runnable: Runnable,
        buffer_config: Optional[dict] = None,
        collect_metrics: bool = True
    ):
        super().__init__(name=f"Optimized_{runnable.name}")
        self.runnable = runnable
        self.buffer = StreamBuffer(**(buffer_config or {}))
        self.collect_metrics = collect_metrics
        self.metrics = StreamMetrics()
    
    async def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Standard invocation (non-streaming)"""
        return await self.runnable.invoke(input, config)
    
    async def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[str]:
        """
        Optimized streaming with TTFT minimization.
        
        Strategy:
        1. Start generation immediately (no preprocessing delay)
        2. Buffer small chunks to reduce overhead
        3. Track metrics for monitoring
        """
        start_time = datetime.now()
        first_token_time = None
        token_count = 0
        
        try:
            # Stream from underlying runnable
            async for chunk in self.runnable.stream(input, config):
                # Record TTFT
                if first_token_time is None and chunk:
                    first_token_time = datetime.now()
                    ttft = (first_token_time - start_time).total_seconds() * 1000
                    
                    if self.collect_metrics:
                        self.metrics.time_to_first_token = ttft
                        logger.debug(f"TTFT: {ttft:.2f}ms")
                
                # Buffer and emit
                buffered = await self.buffer.add(chunk)
                if buffered:
                    token_count += len(buffered.split())
                    yield buffered
            
            # Flush remaining buffer
            final = await self.buffer.flush()
            if final:
                token_count += len(final.split())
                yield final
            
            # Calculate final metrics
            if self.collect_metrics:
                total_duration = (datetime.now() - start_time).total_seconds() * 1000
                self.metrics.total_duration = total_duration
                self.metrics.total_tokens = token_count
                
                if total_duration > 0:
                    self.metrics.tokens_per_second = (token_count / total_duration) * 1000
                
                logger.info(
                    f"Stream complete - TTFT: {self.metrics.time_to_first_token:.2f}ms, "
                    f"Rate: {self.metrics.tokens_per_second:.1f} tokens/s, "
                    f"Grade: {self.metrics.latency_grade}"
                )
        
        except Exception as e:
            logger.error(f"Stream error: {e}")
            raise


class ParallelStreamRunnable(Runnable):
    """
    Stream from multiple runnables in parallel and merge outputs.
    
    Use case: Get responses from multiple models simultaneously
    and present them as they arrive.
    
    Example:
        parallel = ParallelStreamRunnable({
            "gpt4": gpt4_chain,
            "claude": claude_chain
        })
        
        async for source, chunk in parallel.stream(input):
            print(f"[{source}] {chunk}", end="")
    """
    
    def __init__(self, branches: dict[str, Runnable]):
        super().__init__(name="ParallelStream")
        self.branches = branches
    
    async def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> dict:
        """Non-streaming: return all results"""
        results = {}
        for name, runnable in self.branches.items():
            results[name] = await runnable.invoke(input, config)
        return results
    
    async def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[tuple[str, str]]:
        """
        Stream from all branches in parallel.
        
        Yields:
            Tuples of (source_name, chunk)
        """
        # Create stream tasks for each branch
        streams = {
            name: runnable.stream(input, config)
            for name, runnable in self.branches.items()
        }
        
        # Convert to async iterators
        iterators = {
            name: stream.__aiter__()
            for name, stream in streams.items()
        }
        
        # Process chunks as they arrive
        pending = {
            asyncio.create_task(iterator.__anext__()): name
            for name, iterator in iterators.items()
        }
        
        while pending:
            done, pending_set = await asyncio.wait(
                pending.keys(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                name = pending.pop(task)
                
                try:
                    chunk = await task
                    yield name, chunk
                    
                    # Queue next chunk from this iterator
                    next_task = asyncio.create_task(iterators[name].__anext__())
                    pending[next_task] = name
                    
                except StopAsyncIteration:
                    # This stream is complete
                    pass
                except Exception as e:
                    logger.error(f"Stream error in {name}: {e}")


class StreamingCallbackHandler:
    """
    Callback handler for streaming events.
    
    Example:
        handler = StreamingCallbackHandler(
            on_token=lambda token: print(token, end=""),
            on_complete=lambda: print("\nDone!")
        )
    """
    
    def __init__(
        self,
        on_token: Optional[Callable[[str], None]] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        self.on_token = on_token
        self.on_start = on_start
        self.on_complete = on_complete
        self.on_error = on_error
    
    async def handle_stream(self, stream: AsyncIterator[str]):
        """Process a stream with callbacks"""
        try:
            if self.on_start:
                self.on_start()
            
            async for chunk in stream:
                if self.on_token:
                    self.on_token(chunk)
            
            if self.on_complete:
                self.on_complete()
                
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            raise


def optimize_stream(
    runnable: Runnable,
    buffer_size: int = 5,
    collect_metrics: bool = True
) -> StreamOptimizer:
    """
    Convenience function to create optimized streaming runnable.
    
    Example:
        optimized_chain = optimize_stream(chain)
    """
    return StreamOptimizer(
        runnable,
        buffer_config={"min_chunk_size": buffer_size},
        collect_metrics=collect_metrics
    )
