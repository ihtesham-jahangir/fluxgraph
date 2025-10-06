# fluxgraph/core/retry.py
"""
Retry Logic with Exponential Backoff for FluxGraph.
Handles transient failures in LLM API calls and agent execution.
"""

import asyncio
import logging
import time
from typing import Callable, Any, Optional, Type, Tuple
from functools import wraps
from typing import Dict

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on_exceptions = retry_on_exceptions


class RetryManager:
    """Manages retry logic for agent operations."""
    
    def __init__(self, default_config: Optional[RetryConfig] = None):
        self.default_config = default_config or RetryConfig()
        self.retry_stats: dict = {}
        logger.info("RetryManager initialized")
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for the next retry attempt."""
        delay = min(
            config.initial_delay * (config.exponential_base ** attempt),
            config.max_delay
        )
        
        if config.jitter:
            import random
            delay = delay * (0.5 + random.random())
        
        return delay
    
    async def retry_async(
        self,
        func: Callable,
        *args,
        config: Optional[RetryConfig] = None,
        operation_name: str = "operation",
        **kwargs
    ) -> Any:
        """
        Retry an async function with exponential backoff.
        
        Args:
            func: Async function to retry
            *args: Positional arguments for func
            config: Retry configuration (uses default if None)
            operation_name: Name for logging purposes
            **kwargs: Keyword arguments for func
        
        Returns:
            Result of func
        
        Raises:
            Last exception if all retries fail
        """
        config = config or self.default_config
        last_exception = None
        
        for attempt in range(config.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt - 1, config)
                    logger.info(
                        f"[Retry:{operation_name}] Attempt {attempt + 1}/{config.max_retries + 1} "
                        f"after {delay:.2f}s delay"
                    )
                    await asyncio.sleep(delay)
                
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"[Retry:{operation_name}] Succeeded on attempt {attempt + 1}")
                
                # Track success
                self._record_retry_stat(operation_name, attempt, success=True)
                
                return result
                
            except config.retry_on_exceptions as e:
                last_exception = e
                logger.warning(
                    f"[Retry:{operation_name}] Attempt {attempt + 1} failed: {type(e).__name__}: {e}"
                )
                
                if attempt == config.max_retries:
                    logger.error(
                        f"[Retry:{operation_name}] All {config.max_retries + 1} attempts failed"
                    )
                    self._record_retry_stat(operation_name, attempt, success=False)
                    raise
        
        raise last_exception
    
    def retry_sync(
        self,
        func: Callable,
        *args,
        config: Optional[RetryConfig] = None,
        operation_name: str = "operation",
        **kwargs
    ) -> Any:
        """Synchronous version of retry logic."""
        config = config or self.default_config
        last_exception = None
        
        for attempt in range(config.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt - 1, config)
                    logger.info(
                        f"[Retry:{operation_name}] Attempt {attempt + 1}/{config.max_retries + 1} "
                        f"after {delay:.2f}s delay"
                    )
                    time.sleep(delay)
                
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"[Retry:{operation_name}] Succeeded on attempt {attempt + 1}")
                
                self._record_retry_stat(operation_name, attempt, success=True)
                return result
                
            except config.retry_on_exceptions as e:
                last_exception = e
                logger.warning(
                    f"[Retry:{operation_name}] Attempt {attempt + 1} failed: {type(e).__name__}: {e}"
                )
                
                if attempt == config.max_retries:
                    logger.error(
                        f"[Retry:{operation_name}] All {config.max_retries + 1} attempts failed"
                    )
                    self._record_retry_stat(operation_name, attempt, success=False)
                    raise
        
        raise last_exception
    
    def _record_retry_stat(self, operation_name: str, attempts: int, success: bool):
        """Record retry statistics."""
        if operation_name not in self.retry_stats:
            self.retry_stats[operation_name] = {
                "total_operations": 0,
                "total_retries": 0,
                "successes": 0,
                "failures": 0
            }
        
        stats = self.retry_stats[operation_name]
        stats["total_operations"] += 1
        stats["total_retries"] += attempts
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return self.retry_stats.copy()


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    operation_name: Optional[str] = None
):
    """
    Decorator to add retry logic to async functions.
    
    Usage:
        @with_retry(max_retries=5, initial_delay=2.0)
        async def my_function():
            # Your code here
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_retries=max_retries,
                initial_delay=initial_delay
            )
            retry_manager = RetryManager(config)
            op_name = operation_name or func.__name__
            
            return await retry_manager.retry_async(
                func, *args, operation_name=op_name, **kwargs
            )
        return wrapper
    return decorator
