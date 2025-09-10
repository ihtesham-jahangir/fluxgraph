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
import sys
from typing import Any, Dict, Callable, Optional
import argparse # For better CLI parsing

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Import core components
from .registry import AgentRegistry
from .orchestrator import FluxOrchestrator
from .tool_registry import ToolRegistry

# --- Safe Import for Memory ---
try:
    from .memory import Memory  # Try relative import
    MEMORY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        from fluxgraph.core.memory import Memory  # Try absolute import
        MEMORY_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        MEMORY_AVAILABLE = False
        # Create a dummy type for type hints if Memory is not available
        class Memory:
            pass
        logging.getLogger(__name__).debug("Memory interface not found. Memory features will be disabled.")

# --- Safe Import and Setup for Event Hooks ---
# Attempt to import the EventHooks class
try:
    from ..utils.hooks import EventHooks # Try relative import first
    HOOKS_MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        from fluxgraph.utils.hooks import EventHooks # Try absolute import
        HOOKS_MODULE_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        HOOKS_MODULE_AVAILABLE = False
        logging.getLogger(__name__).warning("Event hooks module not found. Event hooks disabled.")

# If the module was found, use the real EventHooks class.
# If not, define a minimal dummy class to prevent AttributeError.
if HOOKS_MODULE_AVAILABLE:
    _EventHooksClass = EventHooks
else:
    class _DummyEventHooks:
        # Crucially, this MUST be async def to be awaitable
        async def trigger(self, event_name: str, payload: Dict[str, Any]):
            # Do nothing silently.
            pass

    _EventHooksClass = _DummyEventHooks

# --- Logger Setup ---
# Configure logger for this module
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
        # Ensure self.hooks is always an instance with an async 'trigger' method
        self.hooks = _EventHooksClass()
        self._setup_middleware()
        self._setup_routes()
        logger.info(f"✅ FluxApp '{self.title}' (v{self.version}) initialized.")

    def _setup_middleware(self):
        """Setup default middlewares (e.g., CORS, Logging)."""
        # CORS Middleware
        self.api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Custom Logging Middleware
        @self.api.middleware("http")
        async def log_requests(request: Request, call_next):
            logger.info(f"🌐 Incoming request: {request.method} {request.url}")
            response = await call_next(request)
            logger.info(f"⬅️ Response status: {response.status_code}")
            return response
            
        logger.debug("Default middlewares configured.")

    def _setup_routes(self):
        """Setup default API routes."""
        @self.api.get("/", summary="Root Endpoint", description="Welcome message and API status.")
        async def root():
            """Root endpoint providing API information."""
            logger.info("📢 Root endpoint called.")
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
                logger.info(f"🤖 Executing agent '{agent_name}' with payload keys: {list(payload.keys())}")
                # Trigger 'request_received' hook
                await self.hooks.trigger("request_received", {"agent_name": agent_name, "payload": payload})

                result = await self.orchestrator.run(agent_name, payload)
                
                # Trigger 'agent_completed' hook
                await self.hooks.trigger("agent_completed", {"agent_name": agent_name, "result": result})
                logger.info(f"✅ Agent '{agent_name}' executed successfully.")
                return result
            except ValueError as e: # Agent not found or execution logic error
                logger.warning(f"⚠️ Agent '{agent_name}' error: {e}")
                # Trigger 'agent_error' hook
                await self.hooks.trigger("agent_error", {"agent_name": agent_name, "error": str(e)})
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e: # Unexpected server error
                logger.error(f"❌ Execution error for agent '{agent_name}': {e}", exc_info=True)
                # Trigger 'server_error' hook
                await self.hooks.trigger("server_error", {"agent_name": agent_name, "error": str(e)})
                raise HTTPException(status_code=500, detail="Internal Server Error")

        # --- Tooling Layer Endpoints ---
        @self.api.get("/tools", summary="List Tools", description="Get a list of all registered tool names.")
        async def list_tools():
            """Endpoint to list registered tool names."""
            logger.info("🛠️ Listing registered tools.")
            return {"tools": self.tool_registry.list_tools()}

        @self.api.get("/tools/{tool_name}", summary="Get Tool Info", description="Get detailed information about a specific tool.")
        async def get_tool_info(tool_name: str):
            """Endpoint to get information about a specific tool."""
            try:
                logger.info(f"🔎 Fetching tool info for '{tool_name}'.")
                info = self.tool_registry.get_tool_info(tool_name)
                return info
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        # --- Memory Status Endpoint (if memory is enabled) ---
        if MEMORY_AVAILABLE and self.memory_store:
            @self.api.get("/memory/status", summary="Memory Status", description="Check if a memory store is configured.")
            async def memory_status():
                """Endpoint to check memory store status."""
                logger.info("💾 Memory status requested.")
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
        logger.info(f"✅ Agent '{name}' registered with the Agent Registry.")

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
            logger.info(f"🛠️ Tool '{tool_name}' registered via @app.tool decorator.")
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
            logger.info(f"🤖 Agent '{agent_name}' registered via @app.agent decorator.")
            return func # Return the original function
        return decorator

    def run(self, host: str = "127.0.0.1", port: int = 8000, reload: bool = False, **kwargs):
        """
        Run the FluxGraph API using Uvicorn.

        This is a convenience method. For production, it's recommended to use
        `uvicorn` command line tool directly.

        Args:
            host (str): The host to bind to. Defaults to "127.0.0.1".
            port (int): The port to bind to. Defaults to 8000.
            reload (bool): Enable auto-reload using Uvicorn's built-in reloader (requires `watchdog`).
                           Note: For reload to work effectively, starting via `uvicorn` CLI or
                           `flux run --reload` is recommended. Defaults to False.
            **kwargs: Additional arguments passed to `uvicorn.run`.
        """
        logger.info(f"🚀 Starting FluxGraph API server on {host}:{port}" + (" (with reload)" if reload else ""))
        
        try:
            import uvicorn
            # Uvicorn's reload feature requires the 'watchdog' package.
            # Passing reload=True tells uvicorn to handle it.
            uvicorn.run(
                self.api,        # Pass the FastAPI instance
                host=host,
                port=port,
                reload=reload,   # Enable/disable reload
                **kwargs         # Pass any other uvicorn arguments
            )
        except ImportError as e:
            if "watchdog" in str(e).lower():
                logger.error("❌ 'watchdog' is required for the --reload feature but not found.")
                print("❌ 'watchdog' is required for the --reload feature but not found. Install it with `pip install watchdog`.")
                sys.exit(1)
            else:
                logger.error(f"❌ Failed to import uvicorn or a dependency: {e}")
                raise
        except Exception as e:
            logger.error(f"❌ Failed to start the server with uvicorn: {e}")
            raise


# --- CLI Entry Point for `flux run [--reload] main.py` ---
def main():
    """
    CLI command entry point: `flux run [--reload] <file.py>`
    
    This function is intended to be called when the user runs `flux run my_app.py`.
    It loads the specified Python file, finds the `FluxApp` instance named `app`,
    and calls its `run` method, potentially with auto-reload enabled.
    """
    # Use argparse for robust command-line parsing
    parser = argparse.ArgumentParser(
        prog='flux',
        description="FluxGraph CLI Runner"
    )
    # Subcommand (currently only 'run' is supported)
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 'run' subcommand
    run_parser = subparsers.add_parser('run', help='Run a FluxGraph application file')
    run_parser.add_argument('file', help="Path to the Python file containing the FluxApp instance (e.g., my_app.py)")
    run_parser.add_argument('--reload', action='store_true', help="Enable auto-reload on file changes (requires `watchdog`)")

    # Parse arguments
    args = parser.parse_args()

    # Check if a command was provided
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command != 'run':
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

    # --- Argument Extraction ---
    file_arg = args.file
    reload_flag = args.reload

    # --- File Handling ---
    import importlib.util
    import pathlib

    file_path = pathlib.Path(file_arg).resolve() # Get absolute path

    if not file_path.exists():
        print(f"❌ File '{file_arg}' not found.")
        sys.exit(1)

    # --- Load the User's Application File ---
    logger.info(f"📦 Loading application from '{file_arg}'...")
    spec = importlib.util.spec_from_file_location("user_app", str(file_path))
    if spec is None or spec.loader is None:
         print(f"❌ Could not load module spec for '{file_arg}'.")
         sys.exit(1)

    user_module = importlib.util.module_from_spec(spec)
    # Crucial: Add to sys.modules to allow relative imports within the user file
    sys.modules["user_app"] = user_module 
    try:
        spec.loader.exec_module(user_module)
        logger.info("✅ Application file loaded successfully.")
    except Exception as e:
         print(f"❌ Error executing '{file_arg}': {e}")
         import traceback
         traceback.print_exc()
         sys.exit(1)

    # --- Find the FluxApp Instance ---
    logger.info("🔍 Searching for FluxApp instance named 'app'...")
    app_instance = getattr(user_module, 'app', None)
    if app_instance is None:
        print("❌ No variable named 'app' found in the specified file.")
        sys.exit(1)
        
    if not isinstance(app_instance, FluxApp):
        print(f"❌ The 'app' variable found is not an instance of FluxApp. Type: {type(app_instance)}")
        sys.exit(1)
    logger.info("✅ FluxApp instance 'app' found.")

    # --- Run the Application ---
    reload_msg = " (with auto-reload)" if reload_flag else ""
    print(f"🚀 Starting FluxGraph app defined in '{file_arg}'{reload_msg}...")
    try:
        # Pass the reload flag to the app's run method
        # Uvicorn will handle the reloading logic if reload=True
        app_instance.run(host="127.0.0.1", port=8000, reload=reload_flag)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user.")
        logger.info("🛑 Server shutdown requested by user (KeyboardInterrupt).")
    except ImportError as e:
        if "watchdog" in str(e).lower():
             logger.error("❌ 'watchdog' is required for --reload but not found.")
             print("❌ 'watchdog' is required for the --reload feature but not found. Install it with `pip install watchdog`.")
        else:
            logger.error(f"❌ Import error while starting app: {e}")
            print(f"❌ Import error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to start the FluxGraph app: {e}", exc_info=True)
        print(f"❌ Failed to start the app: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Handle direct execution of this script (e.g., `python -m fluxgraph.core.app`)
# This is standard practice for modules that can be run as scripts.
# Primarily useful for the `flux run` command via setup.py console_scripts.
if __name__ == "__main__":
    main()
