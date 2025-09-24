# fluxgraph/core/orchestrator.py
"""
Flux Orchestrator for FluxGraph.

This module contains the `FluxOrchestrator`, responsible for executing
registered agents within the FluxGraph framework.

The orchestrator retrieves agents from the `AgentRegistry` using their name
and then manages the invocation of the agent's `run` method. It provides
structured error handling and integrates with the framework's logging context.
"""
import asyncio
import logging
from typing import Any, Dict

from .registry import AgentRegistry

# Use module-specific logger
logger = logging.getLogger(__name__)

class FluxOrchestrator:
    """
    Executes agent flows based on requests (MVP Implementation).

    The FluxOrchestrator is the core execution engine of FluxGraph. It retrieves
    agent instances from the `AgentRegistry` using their name and then invokes
    the agent's `run` method with the provided payload.

    Features:
    - Asynchronous execution support for agents.
    - Synchronous agent compatibility.
    - Structured error handling and logging.
    - Integration with FluxGraph's request context (e.g., request_id).

    Attributes:
        registry (AgentRegistry): The registry used to look up agents by name.
    """

    def __init__(self, registry: AgentRegistry):
        """
        Initializes the FluxOrchestrator.

        Args:
            registry (AgentRegistry): The agent registry instance to use for
                                      looking up agents to execute.
        """
        if not isinstance(registry, AgentRegistry):
            raise TypeError("registry must be an instance of AgentRegistry")
        self.registry = registry
        logger.debug("FluxOrchestrator initialized.")

    async def run(self, agent_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a registered agent asynchronously with the given payload.

        This method performs the following steps:
        1. Retrieves the agent instance from the registry.
        2. Logs the start of execution.
        3. Calls the agent's `run` method (handling sync/async).
        4. Logs successful completion.
        5. Wraps and re-raises specific errors.
        6. Handles unexpected errors gracefully.

        Args:
            agent_name (str): The unique name of the agent to execute.
            payload (Dict[str, Any]): A dictionary of arguments to pass to
                                      the agent's `run` method. This typically
                                      comes from the JSON body of the API request.

        Returns:
            Dict[str, Any]: The result returned by the agent's `run` method.
                            This result is expected to be a dictionary that
                            can be serialized to JSON by FastAPI.

        Raises:
            ValueError: If the agent is not found in the registry, or if
                        the agent's `run` method raises a TypeError
                        (e.g., due to incorrect arguments).
            Exception: Propagates any other unexpected errors raised by the agent's
                       `run` method after wrapping them with context.
                       The API layer (FluxApp) is responsible for catching these
                       and converting them into appropriate HTTP responses.
        """
        # While FluxApp's middleware sets request_id_context, accessing it here
        # directly couples the orchestrator to that specific context var.
        # For now, we'll log without it, relying on the API layer's context logs.
        # If deeper integration is needed, the context var can be passed explicitly
        # or accessed via a utility function.
        logger.info(f"Orchestrator received request to run agent '{agent_name}'.")

        # 1. Retrieve the agent from the registry
        try:
            agent = self.registry.get(agent_name)
        except ValueError as e:
            # Agent not found in registry
            logger.warning(f"Agent '{agent_name}' not found in registry: {e}")
            # Re-raise as ValueError for the API layer to handle (e.g., 404)
            raise ValueError(f"Agent '{agent_name}' is not registered.") from e

        # 2. Execute the agent's run method
        try:
            logger.debug(f"Executing agent '{agent_name}' with payload keys: {list(payload.keys())}")
            
            # 3. Determine if the agent's run method is async or sync
            if asyncio.iscoroutinefunction(agent.run):
                logger.debug(f"Agent '{agent_name}'.run is asynchronous.")
                result = await agent.run(**payload)
            else:
                logger.debug(f"Agent '{agent_name}'.run is synchronous.")
                # For CPU-bound sync tasks, consider running in a thread pool
                # to prevent blocking the event loop. However, for I/O-bound
                # or simple sync tasks, direct execution is often fine in async contexts.
                result = agent.run(**payload)

            logger.info(f"Agent '{agent_name}' executed successfully.")
            return result # Return the result from the agent

        except TypeError as e:
            # This typically indicates an argument mismatch in the agent's run method call
            error_msg = f"Failed to execute agent '{agent_name}': Argument error in agent.run(). {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e # Raise as ValueError for API layer

        except ValueError as e:
            # Re-raise ValueErrors from the agent (e.g., validation logic within the agent)
            logger.error(f"Agent '{agent_name}' raised a ValueError: {e}")
            raise # Re-raise as-is

        except Exception as e:
            # Catch any other unexpected errors during agent execution
            error_msg = f"Unexpected error during execution of agent '{agent_name}': {e}"
            logger.error(error_msg, exc_info=True) # Log with full traceback
            # It's often good practice to wrap unexpected errors to provide context
            # without exposing internal details directly.
            # The API layer can decide how much detail to send to the client.
            raise RuntimeError(error_msg) from e # Wrap in RuntimeError and re-raise
