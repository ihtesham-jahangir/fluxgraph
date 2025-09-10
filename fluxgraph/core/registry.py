# fluxgraph/core/registry.py
"""
Agent Registry for FluxGraph.

This module implements the `AgentRegistry`, a core component of the FluxGraph
framework responsible for storing, managing, and retrieving agent instances
by their unique names.

The registry ensures that all registered agents conform to the basic FluxGraph
agent interface (i.e., they possess a `run` method) and provides methods for
listing and accessing them.
"""
import logging
from typing import Dict, Any, List, Optional
import inspect

# Use module-specific logger
logger = logging.getLogger(__name__)

class AgentRegistry:
    """
    Central registry for storing and managing FluxGraph agents (MVP Implementation).

    The AgentRegistry acts as a directory for all agents available within
    a FluxGraph application. It ensures that agents are registered with a
    unique name and conform to the expected interface (having a `run` method).

    Features:
    - Register agents with unique names.
    - Retrieve agents by name.
    - List all registered agent names.
    - Validate agent interface upon registration.
    - Prevent accidental overwriting of agents.
    - Provide basic information about registered agents.

    Attributes:
        _agents (Dict[str, Any]): A dictionary mapping agent names (str) to
                                  agent instances (Any).
    """

    def __init__(self):
        """Initializes an empty agent registry."""
        self._agents: Dict[str, Any] = {}
        logger.debug("AgentRegistry initialized.")

    def add(self, name: str, agent: Any, overwrite: bool = False) -> None:
        """
        Registers an agent instance under a unique name.

        This method performs basic validation to ensure the agent object
        has a callable `run` method, which is the fundamental requirement
        for a FluxGraph agent.

        Args:
            name (str): The unique identifier for the agent within the registry.
            agent (Any): The agent instance to register. It must have a `run`
                         attribute that is callable.
            overwrite (bool, optional): If True, allows overwriting an existing
                                        agent with the same name. Defaults to False.

        Raises:
            ValueError: If `name` is empty, if `agent` is None, if `agent` does
                        not have a callable `run` method, or if an agent with
                        the same `name` already exists and `overwrite` is False.
            TypeError: If `name` is not a string.
        """
        if not isinstance(name, str):
            raise TypeError("Agent name must be a string.")
        if not name:
            raise ValueError("Agent name cannot be empty.")
        if agent is None:
            raise ValueError("Agent instance cannot be None.")

        if not hasattr(agent, 'run'):
            raise ValueError(
                f"Agent '{name}' is invalid: it must have a 'run' method."
            )
        if not callable(getattr(agent, 'run')):
             raise ValueError(
                f"Agent '{name}' is invalid: 'run' attribute must be callable."
            )

        if name in self._agents and not overwrite:
            raise ValueError(
                f"An agent with the name '{name}' is already registered. "
                f"Use overwrite=True to replace it."
            )

        self._agents[name] = agent
        logger.info(f"Agent '{name}' registered successfully.")

    def get(self, name: str) -> Any:
        """
        Retrieves a registered agent instance by its name.

        Args:
            name (str): The name of the agent to retrieve.

        Returns:
            Any: The agent instance associated with the given name.

        Raises:
            ValueError: If no agent is registered under the specified `name`.
            TypeError: If `name` is not a string.
        """
        if not isinstance(name, str):
            raise TypeError("Agent name must be a string.")

        agent = self._agents.get(name)
        if agent is None:
            logger.warning(f"Attempted to retrieve non-existent agent '{name}'.")
            raise ValueError(f"Agent '{name}' not found in the registry.")
        logger.debug(f"Agent '{name}' retrieved from registry.")
        return agent

    def remove(self, name: str) -> bool:
        """
        Removes an agent from the registry by its name.

        Args:
            name (str): The name of the agent to remove.

        Returns:
            bool: True if the agent was successfully removed, False if it
                  was not found.
        """
        if name in self._agents:
            del self._agents[name]
            logger.info(f"Agent '{name}' removed from registry.")
            return True
        logger.warning(f"Attempted to remove non-existent agent '{name}'.")
        return False

    def list_agents(self) -> List[str]:
        """
        Lists the names of all currently registered agents.

        Returns:
            List[str]: A list of strings, where each string is the name of
                       a registered agent.
        """
        agent_names = list(self._agents.keys())
        logger.debug(f"Listing {len(agent_names)} registered agents.")
        return agent_names

    def get_agent_info(self, name: str) -> Dict[str, Any]:
        """
        Retrieves detailed information about a registered agent.

        This method provides metadata about the agent, which can be useful
        for debugging, monitoring, or introspection.

        Args:
            name (str): The name of the agent.

        Returns:
            Dict[str, Any]: A dictionary containing information about the agent.
                            Keys may include 'name', 'type', 'run_signature'.
                            Returns an empty dict if the agent is not found.

        Raises:
            ValueError: If the agent is not found.
        """
        agent = self.get(name) # Use get to handle not found and log
        info: Dict[str, Any] = {
            "name": name,
            "type": type(agent).__name__,
            "module": getattr(type(agent), '__module__', 'unknown')
        }
        try:
            run_method = getattr(agent, 'run')
            info["run_signature"] = str(inspect.signature(run_method))
        except Exception as e:
             logger.warning(f"Could not inspect 'run' signature for agent '{name}': {e}")
             info["run_signature"] = "Unknown"
        logger.debug(f"Provided info for agent '{name}'.")
        return info

    def get_all_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves information for all registered agents.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping agent names to
                                       their respective info dictionaries.
        """
        all_info = {name: self.get_agent_info(name) for name in self.list_agents()}
        logger.debug("Provided info for all registered agents.")
        return all_info

    def __len__(self) -> int:
        """Returns the number of agents currently registered."""
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        """Checks if an agent with the given name is registered."""
        return name in self._agents
