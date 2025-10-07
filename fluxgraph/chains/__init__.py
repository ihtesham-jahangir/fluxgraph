# fluxgraph/chains/__init__.py
"""
LangChain Expression Language (LCEL) inspired chains for FluxGraph
Declarative composition with pipe operators and streaming support
"""
from typing import Any, Dict, List, Callable, Optional, Union, AsyncIterator, Sequence
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class RunType(Enum):
    """Types of runnable components"""
    CHAIN = "chain"
    LLM = "llm"
    PROMPT = "prompt"
    PARSER = "parser"
    RETRIEVER = "retriever"
    TOOL = "tool"
    CUSTOM = "custom"


@dataclass
class RunnableConfig:
    """Configuration for runnable execution"""
    callbacks: List[Callable] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    run_name: Optional[str] = None
    max_concurrency: int = 5
    timeout: Optional[float] = None


class Runnable(ABC):
    """
    Base class for all runnable components in FluxGraph.
    Inspired by LangChain's Runnable but optimized for performance.
    
    Features:
    - Pipe operator (|) for chaining
    - Batch processing with configurable concurrency
    - Streaming support
    - Full async/await support
    - Type hints throughout
    
    Example:
        chain = prompt | model | parser
        result = await chain.invoke({"question": "What is AI?"})
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._run_type = RunType.CUSTOM
    
    @abstractmethod
    async def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """
        Execute the runnable with given input.
        
        Args:
            input: Input data
            config: Optional execution configuration
            
        Returns:
            Output data
        """
        pass
    
    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Async version of invoke (alias)"""
        return await self.invoke(input, config)
    
    def __or__(self, other: 'Runnable') -> 'RunnableSequence':
        """
        Pipe operator: chain this runnable with another.
        
        Example:
            chain = prompt | model | parser
        """
        if isinstance(other, RunnableSequence):
            return RunnableSequence([self] + other.steps)
        return RunnableSequence([self, other])
    
    def __ror__(self, other: Any) -> 'RunnableSequence':
        """Reverse pipe operator"""
        if isinstance(other, Runnable):
            return RunnableSequence([other, self])
        return RunnableSequence([RunnablePassthrough(lambda x: other), self])
    
    async def batch(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig] = None,
        return_exceptions: bool = False
    ) -> List[Any]:
        """
        Process multiple inputs in parallel with configurable concurrency.
        
        Args:
            inputs: List of inputs to process
            config: Execution configuration
            return_exceptions: If True, return exceptions instead of raising
            
        Returns:
            List of outputs
            
        Example:
            results = await chain.batch([input1, input2, input3])
        """
        config = config or RunnableConfig()
        semaphore = asyncio.Semaphore(config.max_concurrency)
        
        async def process_with_semaphore(inp):
            async with semaphore:
                try:
                    return await self.invoke(inp, config)
                except Exception as e:
                    if return_exceptions:
                        return e
                    raise
        
        return await asyncio.gather(
            *[process_with_semaphore(inp) for inp in inputs],
            return_exceptions=return_exceptions
        )
    
    async def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[Any]:
        """
        Stream output from the runnable.
        
        Args:
            input: Input data
            config: Execution configuration
            
        Yields:
            Output chunks
            
        Example:
            async for chunk in chain.stream(input):
                print(chunk, end="", flush=True)
        """
        # Default: yield the final result
        result = await self.invoke(input, config)
        yield result
    
    async def astream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[Any]:
        """Alias for stream"""
        async for chunk in self.stream(input, config):
            yield chunk
    
    def with_config(self, **kwargs) -> 'RunnableWithConfig':
        """
        Bind configuration to this runnable.
        
        Example:
            configured = chain.with_config(tags=["production"])
        """
        return RunnableWithConfig(self, RunnableConfig(**kwargs))
    
    def with_retry(
        self,
        retry_if_exception_type: tuple = (Exception,),
        wait_exponential_jitter: bool = True,
        stop_after_attempt: int = 3
    ) -> 'RunnableRetry':
        """
        Add retry logic to this runnable.
        
        Example:
            chain_with_retry = chain.with_retry(stop_after_attempt=5)
        """
        return RunnableRetry(
            self,
            retry_if_exception_type,
            wait_exponential_jitter,
            stop_after_attempt
        )
    
    def with_fallbacks(self, fallbacks: List['Runnable']) -> 'RunnableWithFallbacks':
        """
        Add fallback runnables if this fails.
        
        Example:
            chain = primary_model.with_fallbacks([backup_model1, backup_model2])
        """
        return RunnableWithFallbacks(self, fallbacks)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class RunnableSequence(Runnable):
    """
    Chain multiple runnables in sequence.
    
    Created automatically when using pipe operator (|).
    
    Example:
        chain = prompt | model | parser
        # Internally creates: RunnableSequence([prompt, model, parser])
    """
    
    def __init__(self, steps: List[Runnable]):
        super().__init__(name="RunnableSequence")
        self.steps = steps
        self._run_type = RunType.CHAIN
    
    async def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Execute all steps in sequence"""
        result = input
        for i, step in enumerate(self.steps):
            logger.debug(f"Step {i+1}/{len(self.steps)}: {step.name}")
            result = await step.invoke(result, config)
        return result
    
    async def stream(self, input: Any, config: Optional[RunnableConfig] = None) -> AsyncIterator[Any]:
        """Stream through the chain"""
        result = input
        
        # Process all steps except the last
        for step in self.steps[:-1]:
            result = await step.invoke(result, config)
        
        # Stream the last step
        if hasattr(self.steps[-1], 'stream'):
            async for chunk in self.steps[-1].stream(result, config):
                yield chunk
        else:
            yield await self.steps[-1].invoke(result, config)
    
    def __or__(self, other: Runnable) -> 'RunnableSequence':
        """Add another step to the chain"""
        if isinstance(other, RunnableSequence):
            return RunnableSequence(self.steps + other.steps)
        return RunnableSequence(self.steps + [other])


class RunnableParallel(Runnable):
    """
    Execute multiple runnables in parallel and return combined results.
    
    Example:
        parallel = RunnableParallel({
            "summary": summarize_chain,
            "sentiment": sentiment_chain,
            "keywords": keyword_chain
        })
        
        result = await parallel.invoke(text)
        # Returns: {"summary": "...", "sentiment": "...", "keywords": [...]}
    """
    
    def __init__(self, branches: Dict[str, Runnable]):
        super().__init__(name="RunnableParallel")
        self.branches = branches
    
    async def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Execute all branches in parallel"""
        config = config or RunnableConfig()
        
        async def run_branch(key: str, runnable: Runnable):
            try:
                return key, await runnable.invoke(input, config)
            except Exception as e:
                logger.error(f"Branch {key} failed: {e}")
                raise
        
        # Run all branches concurrently
        results = await asyncio.gather(
            *[run_branch(key, runnable) for key, runnable in self.branches.items()],
            return_exceptions=False
        )
        
        return dict(results)
    
    async def stream(self, input: Any, config: Optional[RunnableConfig] = None) -> AsyncIterator[Dict[str, Any]]:
        """Stream results as they complete"""
        pending = {
            asyncio.create_task(runnable.invoke(input, config)): key
            for key, runnable in self.branches.items()
        }
        
        results = {}
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            
            for task in done:
                key = pending.pop(task)
                results[key] = await task
                yield {key: results[key]}


class RunnableLambda(Runnable):
    """
    Convert any function into a Runnable.
    
    Example:
        uppercase = RunnableLambda(lambda x: x.upper())
        extract_text = RunnableLambda(lambda x: x["text"])
        
        chain = extract_text | uppercase
    """
    
    def __init__(self, func: Callable, name: Optional[str] = None):
        super().__init__(name=name or func.__name__)
        self.func = func
    
    async def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Execute the function"""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(input)
        else:
            # Run sync function in executor to not block event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.func, input)


class RunnablePassthrough(Runnable):
    """
    Pass input through unchanged (optionally with assignment).
    
    Example:
        # Simple passthrough
        passthrough = RunnablePassthrough()
        
        # With assignment
        passthrough = RunnablePassthrough(assign={"original": lambda x: x})
    """
    
    def __init__(self, assign: Optional[Dict[str, Callable]] = None):
        super().__init__(name="RunnablePassthrough")
        self.assign = assign or {}
    
    async def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Pass through input"""
        if not self.assign:
            return input
        
        if isinstance(input, dict):
            result = input.copy()
            for key, func in self.assign.items():
                if asyncio.iscoroutinefunction(func):
                    result[key] = await func(input)
                else:
                    result[key] = func(input)
            return result
        
        return input


class RunnableWithConfig(Runnable):
    """Runnable with bound configuration"""
    
    def __init__(self, runnable: Runnable, config: RunnableConfig):
        super().__init__(name=f"{runnable.name}_with_config")
        self.runnable = runnable
        self.default_config = config
    
    async def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Merge configs and invoke"""
        merged_config = self._merge_configs(config)
        return await self.runnable.invoke(input, merged_config)
    
    def _merge_configs(self, config: Optional[RunnableConfig]) -> RunnableConfig:
        """Merge default and provided configs"""
        if not config:
            return self.default_config
        
        return RunnableConfig(
            callbacks=self.default_config.callbacks + config.callbacks,
            tags=self.default_config.tags + config.tags,
            metadata={**self.default_config.metadata, **config.metadata},
            run_name=config.run_name or self.default_config.run_name,
            max_concurrency=config.max_concurrency,
            timeout=config.timeout or self.default_config.timeout
        )


class RunnableRetry(Runnable):
    """Runnable with automatic retry logic"""
    
    def __init__(
        self,
        runnable: Runnable,
        retry_if_exception_type: tuple,
        wait_exponential_jitter: bool,
        stop_after_attempt: int
    ):
        super().__init__(name=f"{runnable.name}_with_retry")
        self.runnable = runnable
        self.retry_if_exception_type = retry_if_exception_type
        self.wait_exponential_jitter = wait_exponential_jitter
        self.stop_after_attempt = stop_after_attempt
    
    async def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Invoke with retry logic"""
        import random
        
        for attempt in range(self.stop_after_attempt):
            try:
                return await self.runnable.invoke(input, config)
            except self.retry_if_exception_type as e:
                if attempt == self.stop_after_attempt - 1:
                    raise
                
                # Calculate wait time
                wait = 2 ** attempt
                if self.wait_exponential_jitter:
                    wait = wait * (0.5 + random.random() * 0.5)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait:.2f}s..."
                )
                await asyncio.sleep(wait)


class RunnableWithFallbacks(Runnable):
    """Runnable with fallback alternatives"""
    
    def __init__(self, runnable: Runnable, fallbacks: List[Runnable]):
        super().__init__(name=f"{runnable.name}_with_fallbacks")
        self.runnable = runnable
        self.fallbacks = fallbacks
    
    async def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Try main runnable, then fallbacks"""
        try:
            return await self.runnable.invoke(input, config)
        except Exception as e:
            logger.warning(f"Primary runnable failed: {e}. Trying fallbacks...")
            
            for i, fallback in enumerate(self.fallbacks):
                try:
                    logger.info(f"Trying fallback {i+1}/{len(self.fallbacks)}")
                    return await fallback.invoke(input, config)
                except Exception as fallback_error:
                    if i == len(self.fallbacks) - 1:
                        logger.error(f"All fallbacks exhausted")
                        raise
                    logger.warning(f"Fallback {i+1} failed: {fallback_error}")


# Convenience functions
def chain(*runnables: Runnable) -> RunnableSequence:
    """
    Create a chain from multiple runnables.
    
    Example:
        my_chain = chain(prompt, model, parser)
    """
    return RunnableSequence(list(runnables))


def parallel(**branches: Runnable) -> RunnableParallel:
    """
    Create parallel execution branches.
    
    Example:
        my_parallel = parallel(
            summary=summarize_chain,
            sentiment=sentiment_chain
        )
    """
    return RunnableParallel(branches)


def runnable(func: Callable) -> RunnableLambda:
    """
    Convert function to Runnable.
    
    Example:
        @runnable
        def uppercase(text):
            return text.upper()
    """
    return RunnableLambda(func)


def passthrough(assign: Optional[Dict[str, Callable]] = None) -> RunnablePassthrough:
    """
    Create a passthrough runnable.
    
    Example:
        pass_with_original = passthrough(assign={"original": lambda x: x})
    """
    return RunnablePassthrough(assign)
