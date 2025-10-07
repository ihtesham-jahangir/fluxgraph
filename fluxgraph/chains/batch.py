# fluxgraph/chains/batch.py
"""
Optimized batch processing with intelligent concurrency control
"""
from typing import Any, List, Dict, Optional, Callable, TypeVar
import asyncio
from dataclasses import dataclass
from enum import Enum
import logging
from fluxgraph.chains import Runnable, RunnableConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BatchStrategy(Enum):
    """Strategies for batch processing"""
    CONCURRENT = "concurrent"  # Process all at once
    SEQUENTIAL = "sequential"  # One at a time
    ADAPTIVE = "adaptive"  # Adjust based on performance
    CHUNKED = "chunked"  # Process in chunks


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    max_concurrency: int = 10
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    chunk_size: Optional[int] = None
    timeout: Optional[float] = None
    return_exceptions: bool = False
    show_progress: bool = False


class BatchProcessor:
    """
    Intelligent batch processor with adaptive concurrency.
    
    Features:
    - Multiple processing strategies
    - Adaptive concurrency based on performance
    - Progress tracking
    - Error handling with partial results
    - Memory-efficient chunked processing
    
    Example:
        processor = BatchProcessor(chain, strategy=BatchStrategy.ADAPTIVE)
        results = await processor.process([input1, input2, ...])
    """
    
    def __init__(
        self,
        runnable: Runnable,
        config: Optional[BatchConfig] = None
    ):
        self.runnable = runnable
        self.config = config or BatchConfig()
        self._performance_history: List[float] = []
        self._optimal_concurrency: int = self.config.max_concurrency
    
    async def process(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig] = None
    ) -> List[Any]:
        """
        Process multiple inputs with optimal strategy.
        
        Args:
            inputs: List of inputs to process
            config: Optional runnable configuration
            
        Returns:
            List of results
        """
        if not inputs:
            return []
        
        logger.info(f"Batch processing {len(inputs)} inputs with {self.config.strategy.value} strategy")
        
        if self.config.strategy == BatchStrategy.CONCURRENT:
            return await self._process_concurrent(inputs, config)
        elif self.config.strategy == BatchStrategy.SEQUENTIAL:
            return await self._process_sequential(inputs, config)
        elif self.config.strategy == BatchStrategy.ADAPTIVE:
            return await self._process_adaptive(inputs, config)
        elif self.config.strategy == BatchStrategy.CHUNKED:
            return await self._process_chunked(inputs, config)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
    
    async def _process_concurrent(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig]
    ) -> List[Any]:
        """Process all inputs concurrently"""
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        
        async def process_with_semaphore(inp, idx):
            async with semaphore:
                try:
                    if self.config.show_progress:
                        logger.info(f"Processing {idx+1}/{len(inputs)}")
                    return await self.runnable.invoke(inp, config)
                except Exception as e:
                    if self.config.return_exceptions:
                        return e
                    raise
        
        tasks = [process_with_semaphore(inp, idx) for idx, inp in enumerate(inputs)]
        
        if self.config.timeout:
            return await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=self.config.return_exceptions),
                timeout=self.config.timeout
            )
        else:
            return await asyncio.gather(*tasks, return_exceptions=self.config.return_exceptions)
    
    async def _process_sequential(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig]
    ) -> List[Any]:
        """Process inputs one at a time"""
        results = []
        
        for idx, inp in enumerate(inputs):
            try:
                if self.config.show_progress:
                    logger.info(f"Processing {idx+1}/{len(inputs)}")
                result = await self.runnable.invoke(inp, config)
                results.append(result)
            except Exception as e:
                if self.config.return_exceptions:
                    results.append(e)
                else:
                    raise
        
        return results
    
    async def _process_adaptive(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig]
    ) -> List[Any]:
        """Adaptively adjust concurrency based on performance"""
        results = []
        remaining = inputs.copy()
        
        while remaining:
            # Process a batch with current optimal concurrency
            batch = remaining[:self._optimal_concurrency]
            remaining = remaining[self._optimal_concurrency:]
            
            start_time = asyncio.get_event_loop().time()
            
            batch_results = await self._process_concurrent(batch, config)
            results.extend(batch_results)
            
            # Measure performance
            elapsed = asyncio.get_event_loop().time() - start_time
            throughput = len(batch) / elapsed if elapsed > 0 else 0
            
            self._performance_history.append(throughput)
            
            # Adjust concurrency
            self._adjust_concurrency()
        
        return results
    
    async def _process_chunked(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig]
    ) -> List[Any]:
        """Process in fixed-size chunks"""
        chunk_size = self.config.chunk_size or self.config.max_concurrency
        results = []
        
        for i in range(0, len(inputs), chunk_size):
            chunk = inputs[i:i + chunk_size]
            
            if self.config.show_progress:
                logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(inputs) + chunk_size - 1)//chunk_size}")
            
            chunk_results = await self._process_concurrent(chunk, config)
            results.extend(chunk_results)
        
        return results
    
    def _adjust_concurrency(self):
        """Adjust optimal concurrency based on performance history"""
        if len(self._performance_history) < 3:
            return
        
        recent = self._performance_history[-3:]
        avg_throughput = sum(recent) / len(recent)
        
        # If throughput is decreasing, reduce concurrency
        if len(recent) >= 2 and recent[-1] < recent[-2] * 0.9:
            self._optimal_concurrency = max(1, self._optimal_concurrency - 1)
            logger.debug(f"Reducing concurrency to {self._optimal_concurrency}")
        
        # If throughput is stable/increasing, try increasing
        elif len(recent) >= 2 and recent[-1] >= recent[-2] * 1.1:
            self._optimal_concurrency = min(
                self.config.max_concurrency,
                self._optimal_concurrency + 1
            )
            logger.debug(f"Increasing concurrency to {self._optimal_concurrency}")


class BatchRunnable(Runnable):
    """
    Runnable wrapper with optimized batch processing.
    
    Example:
        batch_chain = BatchRunnable(chain, config=BatchConfig(max_concurrency=20))
        results = await batch_chain.batch([input1, input2, ...])
    """
    
    def __init__(
        self,
        runnable: Runnable,
        batch_config: Optional[BatchConfig] = None
    ):
        super().__init__(name=f"Batch_{runnable.name}")
        self.runnable = runnable
        self.batch_config = batch_config or BatchConfig()
        self.processor = BatchProcessor(runnable, self.batch_config)
    
    async def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Single invocation"""
        return await self.runnable.invoke(input, config)
    
    async def batch(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig] = None,
        return_exceptions: bool = False
    ) -> List[Any]:
        """Optimized batch processing"""
        return await self.processor.process(inputs, config)
