# fluxgraph/utils/hooks.py
"""
Event Hooks for FluxGraph.

Provides a simple event-driven system for debugging and monitoring.
(MVP Implementation)
"""
import asyncio
import logging
from typing import Dict, Any, Callable, List

# Use module-specific logger
logger = logging.getLogger(__name__)

class EventHooks:
    """
    Transparent debugging and execution tracking.

    Allows registering callbacks for specific events during agent lifecycles.
    Callbacks can be either synchronous or asynchronous functions.
    """

    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {}
        logger.debug("EventHooks instance created.")

    def on(self, event: str, callback: Callable):
        """
        Register a callback function for an event.

        Args:
            event (str): The name of the event.
            callback (Callable): The function to call when the event is triggered.
                                Can be sync or async.
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
        logger.debug(f"Callback registered for event '{event}'.")

    async def trigger(self, event: str, data: Dict[str, Any]): # <-- CHANGED TO async def
        """
        Asynchronously trigger an event, calling all registered callbacks.

        This method ensures that both synchronous and asynchronous callbacks
        are handled correctly. Async callbacks are awaited, sync callbacks
        are called directly.

        Args:
            event (str): The name of the event to trigger.
            data (Dict[str, Any]): Data to pass to the callback functions.
        """
        callbacks = self._hooks.get(event, [])
        logger.debug(f"Triggering event '{event}' for {len(callbacks)} callback(s).")
        
        for callback in callbacks:
            try:
                # Check if the callback is a coroutine function (async)
                if asyncio.iscoroutinefunction(callback):
                    logger.debug(f"Awaiting async callback '{callback.__name__}' for event '{event}'.")
                    await callback(data) # Await async callbacks
                else:
                    logger.debug(f"Calling sync callback '{callback.__name__}' for event '{event}'.")
                    callback(data) # Call sync callbacks normally
            except Exception as e:
                # Handle errors in individual hooks gracefully to prevent
                # one bad hook from breaking the whole process
                logger.error(f"Error in hook '{event}' callback '{getattr(callback, '__name__', 'Unknown')}': {e}", exc_info=True)
        # Implicitly returns None, which is fine for an async function
        # that doesn't need to produce a value.
        logger.debug(f"Finished triggering event '{event}'.")

# Global instance for easy access within the framework
# This assumes this file is imported somewhere during FluxApp initialization.
global_hooks = EventHooks()
