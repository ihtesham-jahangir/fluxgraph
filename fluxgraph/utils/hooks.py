# fluxgraph/utils/hooks.py
"""
Event Hooks for FluxGraph.

Provides a simple event-driven system for debugging, monitoring, and extending
agent lifecycles. Callbacks can be synchronous or asynchronous.
"""
import asyncio
import logging
from typing import Dict, Any, Callable, List, Union

logger = logging.getLogger(__name__)

class EventHooks:
    """
    Transparent debugging and execution tracking.

    Allows registering callbacks for specific events during the FluxGraph lifecycle.
    Callbacks can be standard functions or async functions.
    """

    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {}
        logger.debug("EventHooks instance created.")

    def on(self, event_name: str, callback: Callable):
        """
        Register a callback function for an event.

        Args:
            event_name (str): The name of the event (e.g., 'agent_started', 'request_received').
            callback (Callable): The function to call when the event is triggered.
                                Can be sync (def) or async (async def).
        """
        if event_name not in self._hooks:
            self._hooks[event_name] = []
        self._hooks[event_name].append(callback)
        logger.debug(f"Hook callback '{callback.__name__}' registered for event '{event_name}'.")

    async def trigger(self, event_name: str, data: Dict[str, Any]):
        """
        Asynchronously trigger an event, calling all registered callbacks.

        This method ensures that both synchronous and asynchronous callbacks
        are handled correctly. Async callbacks are awaited, sync callbacks
        are called directly.

        Args:
            event_name (str): The name of the event to trigger.
            data (Dict[str, Any]): Data to pass to the callback functions.
                                   This often includes context like agent name,
                                   request ID, payload, or results.
        """
        callbacks = self._hooks.get(event_name, [])
        logger.debug(f"Triggering event '{event_name}' for {len(callbacks)} callback(s).")
        
        for callback in callbacks:
            try:
                # Check if the callback is a coroutine function (async)
                if asyncio.iscoroutinefunction(callback):
                    logger.debug(f"Awaiting async callback '{callback.__name__}' for event '{event_name}'.")
                    await callback(data) # Await async callbacks
                else:
                    logger.debug(f"Calling sync callback '{callback.__name__}' for event '{event_name}'.")
                    callback(data) # Call sync callbacks normally
            except Exception as e:
                # Handle errors in individual hooks gracefully to prevent
                # one bad hook from breaking the whole process or request flow.
                logger.error(
                    f"Error in hook '{event_name}' callback '{getattr(callback, '__name__', 'Unknown')}': {e}",
                    exc_info=True # Log full traceback for debugging
                )
        logger.debug(f"Finished triggering event '{event_name}'.")

# Global instance for easy access within the framework
# This assumes this file is imported during FluxApp initialization.
global_hooks = EventHooks()
