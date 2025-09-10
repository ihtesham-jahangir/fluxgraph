# fluxgraph/core/tool_registry.py
"""
Tool Registry for FluxGraph.

This module implements the `ToolRegistry`, responsible for storing and managing
reusable Python functions (tools) that can be utilized by agents within the
FluxGraph framework.
"""
import logging
from typing import Dict, Callable, Any, List, Optional
import inspect
import functools

logger = logging.getLogger(__name__)

class ToolRegistry:
    """
    Central registry for storing and managing tools (MVP Implementation).

    Tools are typically Python functions that perform specific actions
    (e.g., calculations, API calls, data processing). Registering them
    allows for centralized management and potential discovery/introspection
    by agents or the orchestrator in more advanced setups.

    Attributes:
        _tools (Dict[str, Callable]): A dictionary mapping tool names (str) to
                                      tool functions (Callable).
    """

    def __init__(self):
        """Initializes an empty tool registry."""
        self._tools: Dict[str, Callable] = {}
        logger.debug("ToolRegistry initialized.")

    def register(self, name: str, func: Callable, overwrite: bool = False) -> None:
        """
        Registers a tool function under a unique name.

        Args:
            name (str): The unique identifier for the tool.
            func (Callable): The Python function to register as a tool.
            overwrite (bool, optional): Allow overwriting an existing tool.
                                       Defaults to False.

        Raises:
            TypeError: If `name` is not a string or `func` is not callable.
            ValueError: If `name` is empty or tool exists and `overwrite` is False.
        """
        if not isinstance(name, str):
            raise TypeError("Tool name must be a string.")
        if not name:
            raise ValueError("Tool name cannot be empty.")
        if not callable(func):
            raise TypeError("Tool must be a callable function.")

        if name in self._tools and not overwrite:
            raise ValueError(
                f"A tool with the name '{name}' is already registered. "
                f"Use overwrite=True to replace it."
            )

        self._tools[name] = func
        logger.info(f"Tool '{name}' registered successfully.")

    def get(self, name: str) -> Callable:
        """
        Retrieves a registered tool function by its name.

        Args:
            name (str): The name of the tool to retrieve.

        Returns:
            Callable: The tool function.

        Raises:
            ValueError: If no tool is registered under the specified `name`.
            TypeError: If `name` is not a string.
        """
        if not isinstance(name, str):
            raise TypeError("Tool name must be a string.")

        tool = self._tools.get(name)
        if tool is None:
            logger.warning(f"Attempted to retrieve non-existent tool '{name}'.")
            raise ValueError(f"Tool '{name}' not found in the registry.")
        logger.debug(f"Tool '{name}' retrieved from registry.")
        return tool

    def list_tools(self) -> List[str]:
        """
        Lists the names of all currently registered tools.

        Returns:
            List[str]: A list of registered tool names.
        """
        tool_names = list(self._tools.keys())
        logger.debug(f"Listing {len(tool_names)} registered tools.")
        return tool_names

    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """
        Retrieves detailed information about a registered tool.

        Args:
            name (str): The name of the tool.

        Returns:
            Dict[str, Any]: Information about the tool (name, signature, docstring).
        """
        tool_func = self.get(name) # Handles not found and logging
        info: Dict[str, Any] = {
            "name": name,
            "doc": tool_func.__doc__ or "No description available.",
        }
        try:
            info["signature"] = str(inspect.signature(tool_func))
        except Exception as e:
             logger.warning(f"Could not inspect signature for tool '{name}': {e}")
             info["signature"] = "Unknown"
        logger.debug(f"Provided info for tool '{name}'.")
        return info

    def __contains__(self, name: str) -> bool:
        """Checks if a tool with the given name is registered."""
        return name in self._tools

    def __len__(self) -> int:
        """Returns the number of tools currently registered."""
        return len(self._tools)
