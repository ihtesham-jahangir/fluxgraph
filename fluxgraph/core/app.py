# fluxgraph/core/app.py
"""
FluxGraph Application Core.

This module defines the `FluxApp` class, the central manager for a FluxGraph application.
It integrates the Agent Registry, Orchestrator, Tooling Layer, and provides hooks for
LangGraph adapters and optional Memory stores.

FluxApp is built on FastAPI, offering immediate REST API deployment capabilities.
"""
import asyncio
import logging
from typing import Any, Dict, Callable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import core components
from .registry import AgentRegistry
from .orchestrator import FluxOrchestrator
from .tool_registry import ToolRegistry
# Import utils for hooks
from ..utils.hooks import EventHooks # Import the hooks utility

# Attempt to import Memory interface for optional features
try:
    from .memory import Memory # Try relative import
    MEMORY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        from fluxgraph.core.memory import Memory # Try absolute import
        MEMORY_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        MEMORY_AVAILABLE = False
        Memory = Any # Type hint fallback
        logging.getLogger(__name__).debug("Memory interface not found. Memory features will be disabled.")

logger = logging.getLogger(__name__)

class FluxApp:
    """
    Main application manager, built on FastAPI.

    Integrates the Agent Registry, Flux Orchestrator, Tool Registry,
    and provides integration points for LangGraph Adapters, Memory, and Event Hooks.

    Core Components (from MVP Documentation):
    - Agent Registry: Tracks available agents.
    - Flux Orchestrator: Executes agent flows.
    - LangGraph Adapter: Plug-in for LangGraph workflows (used via registration).
    - Event Hooks: Transparent debugging and execution tracking.
    - Tooling Layer: Extendable Python functions (via tool registry).
    - LLM Providers: Integrated via agent logic.
    - Persistence/Memory: Optional integration point.

    Attributes:
        title (str): The title of the API.
        description (str): The description of the API.
        version (str): The version of the API.
        api (FastAPI): The underlying FastAPI instance.
        registry (AgentRegistry): Manages registered agents.
        orchestrator (FluxOrchestrator): Executes agent workflows.
        tool_registry (ToolRegistry): Manages registered tools.
        memory_store (Optional[Memory]): Optional memory store instance.
        hooks (EventHooks): Utility for event-driven debugging/monitoring.
    """

    def __init__(
        self,
        title: str = "FluxGraph API",
        description: str = "A lightweight Python framework for building, orchestrating, and deploying Agentic AI systems.",
        version: str = "0.1.0",
        memory_store: Optional[Memory] = None
    ):
        """
        Initializes the FluxGraph application.

        Args:
            title (str): Title for the FastAPI application.
            description (str): Description for the FastAPI application.
            version (str): Version string.
            memory_store (Optional[Memory]): An optional memory store instance
                                             implementing the Memory interface.
        """
        self.title = title
        self.description = description
        self.version = version
        self.api = FastAPI(title=self.title, description=self.description, version=self.version)
        
        # --- Core Components (MVP Alignment) ---
        self.registry = AgentRegistry()
        self.orchestrator = FluxOrchestrator(self.registry)
        self.tool_registry = ToolRegistry()
        self.memory_store = memory_store
        
        # --- Utilities ---
        self.hooks = EventHooks() # Initialize event hooks utility
        
        self._setup_routes()
        self._setup_middleware()
        logger.info(f"FluxApp '{self.title}' (v{self.version}) initialized.")

    def _setup_middleware(self):
        """Setup default middlewares (e.g., CORS)."""
        self.api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.debug("Default middlewares configured.")

    def _setup_routes(self):
        """Setup default API routes."""
        @self.api.get("/", summary="Root Endpoint", description="Welcome message and API status.")
        async def root():
            """Root endpoint providing API information."""
            return {
                "message": "Welcome to FluxGraph MVP",
                "title": self.title,
                "version": self.version,
                "memory_enabled": self.memory_store is not None
            }

        @self.api.post(
            "/ask/{agent_name}",
            summary="Ask Agent",
            description="Execute a registered agent by name with a JSON payload."
        )
        async def ask_agent(agent_name: str, payload: Dict[str, Any]):
            """
            Endpoint to interact with registered agents.

            The payload JSON is passed as keyword arguments to the agent's `run` method.
            """
            try:
                logger.info(f"Received request for agent '{agent_name}' with payload keys: {list(payload.keys())}")
                # Trigger 'request_received' hook
                await self.hooks.trigger("request_received", {"agent_name": agent_name, "payload": payload})

                result = await self.orchestrator.run(agent_name, payload)
                
                # Trigger 'agent_completed' hook
                await self.hooks.trigger("agent_completed", {"agent_name": agent_name, "result": result})
                logger.info(f"Agent '{agent_name}' executed successfully.")
                return result
            except ValueError as e: # Agent not found or execution logic error
                logger.warning(f"Agent '{agent_name}' error: {e}")
                status_code = 404 if "not registered" in str(e).lower() or "not found" in str(e).lower() else 400
                # Trigger 'agent_error' hook
                await self.hooks.trigger("agent_error", {"agent_name": agent_name, "error": str(e), "status_code": status_code})
                raise HTTPException(status_code=status_code, detail=str(e))
            except Exception as e: # Unexpected server error
                logger.error(f"Execution error for agent '{agent_name}': {e}", exc_info=True)
                # Trigger 'server_error' hook
                await self.hooks.trigger("server_error", {"agent_name": agent_name, "error": str(e)})
                raise HTTPException(status_code=500, detail="Internal Server Error")

        # --- Tooling Layer Endpoints ---
        @self.api.get("/tools", summary="List Tools", description="Get a list of all registered tool names.")
        async def list_tools():
            """Endpoint to list registered tool names."""
            return {"tools": self.tool_registry.list_tools()}

        @self.api.get("/tools/{tool_name}", summary="Get Tool Info", description="Get detailed information about a specific tool.")
        async def get_tool_info(tool_name: str):
            """Endpoint to get information about a specific tool."""
            try:
                info = self.tool_registry.get_tool_info(tool_name)
                return info
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        # --- Memory Status Endpoint (if memory is enabled) ---
        if MEMORY_AVAILABLE and self.memory_store:
            @self.api.get("/memory/status", summary="Memory Status", description="Check if a memory store is configured.")
            async def memory_status():
                """Endpoint to check memory store status."""
                return {"memory_enabled": True, "type": type(self.memory_store).__name__}
        else:
            logger.debug("Memory endpoints not added as memory store is not configured.")

    # --- Agent Management ---
    def register(self, name: str, agent: Any):
        """
        Register an agent instance with the Agent Registry.

        Args:
            name (str): The unique name for the agent.
            agent (Any): The agent instance (must have a `run` method).
        """
        self.registry.add(name, agent)
        logger.info(f"Agent '{name}' registered with the Agent Registry.")

    # --- Tool Management ---
    def tool(self, name: Optional[str] = None):
        """
        Decorator to define and register a tool function with the Tool Registry.

        Args:
            name (Optional[str]): The name to register the tool under.
                                  Defaults to the function's `__name__`.

        Usage:
            @app.tool()
            def my_utility_function(x: int, y: int) -> int:
                return x + y
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name if name is not None else func.__name__
            self.tool_registry.register(tool_name, func)
            logger.info(f"Tool '{tool_name}' registered via @app.tool decorator.")
            return func # Return the original function
        return decorator

    # --- Agent Definition Decorator ---
    def agent(self, name: Optional[str] = None):
        """
        Decorator to define and register an agent function.

        The decorated function becomes the `run` method of a dynamically created
        agent class. The `tools` registry and `memory` store (if configured)
        are automatically injected as keyword arguments when the agent is executed.

        This simplifies agent creation for logic that doesn't require a full class.

        Args:
            name (Optional[str]): The name to register the agent under.
                                  Defaults to the function's `__name__`.

        Usage:
            @app.agent()
            async def my_agent(query: str, tools, memory):
                # Use tools.get('tool_name') to access tools
                # Use await memory.add(...) if memory is available
                return {"response": f"Processed: {query}"}
        """
        def decorator(func: Callable) -> Callable:
            agent_name = name if name is not None else func.__name__

            # Dynamically create an agent class
            class _FluxDynamicAgent:
                async def run(self, **kwargs):
                    # Inject core dependencies
                    kwargs['tools'] = self._tool_registry
                    if self._memory_store:
                        kwargs['memory'] = self._memory_store
                    
                    # Execute the user-defined function
                    if asyncio.iscoroutinefunction(func):
                        return await func(**kwargs)
                    else:
                        return func(**kwargs)
            
            # Instantiate the dynamic agent and inject FluxApp dependencies
            agent_instance = _FluxDynamicAgent()
            agent_instance._tool_registry = self.tool_registry
            agent_instance._memory_store = self.memory_store

            # Register the instance with the FluxApp
            self.register(agent_name, agent_instance)
            logger.info(f"Agent '{agent_name}' registered via @app.agent decorator.")
            return func # Return the original function
        return decorator

    def run(self, host: str = "127.0.0.1", port: int = 8000, **kwargs):
        """
        Run the FluxGraph API using Uvicorn.

        This is a convenience method. For production, it's recommended to use
        `uvicorn` command line tool directly (e.g., `uvicorn my_app:app.api`).

        Args:
            host (str): The host to bind to. Defaults to "127.0.0.1".
            port (int): The port to bind to. Defaults to 8000.
            **kwargs: Additional arguments passed to `uvicorn.run`.
        """
        logger.info(f"Starting FluxGraph API server on {host}:{port}")
        import uvicorn
        uvicorn.run(self.api, host=host, port=port, **kwargs)
