
# fluxgraph/core/app.py
"""
FluxGraph Application Core.

This module defines the `FluxApp` class, the central manager for a FluxGraph application.
It integrates the Agent Registry, Orchestrator, Tooling Layer, and provides hooks for
LangGraph adapters and optional Memory/RAG stores.

FluxApp is built on FastAPI, offering immediate REST API deployment capabilities.

Virtual Environment Handling:
This script includes logic to automatically create and/or activate a virtual environment
named '.venv_fluxgraph' in the current working directory. This happens before any
other imports or application logic is executed. This ensures dependencies are isolated.
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# --- 1. VIRTUAL ENVIRONMENT HANDLING (AUTO-CREATE/ACTIVATE) ---
def _ensure_virtual_environment():
    """
    Ensures a virtual environment is set up and activated for FluxGraph.

    This function checks if the script is running inside a virtual environment.
    If not, it looks for an existing '.venv_fluxgraph' directory.
    - If it exists, it activates it by modifying `sys.path` and `os.environ`.
    - If it doesn't exist, it creates a new one using the current Python executable
      and installs dependencies from 'requirements.txt' (if found).
    Finally, it re-executes the script within the activated virtual environment.
    """
    venv_name = ".venv_fluxgraph"
    venv_path = os.path.join(os.getcwd(), venv_name)

    def _is_in_venv():
        """Check if the current Python process is running inside a virtual environment."""
        # Common indicators
        return (
            hasattr(sys, 'real_prefix') or # For older venv/virtualenv
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) # For venv
        )

    def _get_python_executable(venv_dir):
        """Get the path to the Python executable inside a virtual environment."""
        if os.name == 'nt':  # Windows
            return os.path.join(venv_dir, 'Scripts', 'python.exe')
        else:  # POSIX (Linux, macOS)
            return os.path.join(venv_dir, 'bin', 'python')

    # 1. Check if already inside a venv
    if _is_in_venv():
        logging.getLogger(__name__).debug("Already inside a virtual environment. Skipping auto-setup.")
        return # Already in a venv, nothing to do

    # 2. Check if the target venv directory exists
    if os.path.isdir(venv_path):
        logging.getLogger(__name__).info(f"Found existing virtual environment at '{venv_path}'. Activating...")
    else:
        # 3. Create the virtual environment if it doesn't exist
        logging.getLogger(__name__).info(f"Creating new virtual environment at '{venv_path}'...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            logging.getLogger(__name__).info("Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("❌ 'venv' module not found. Please ensure you are using Python 3.3+.")
            sys.exit(1)

        # 4. Install dependencies (if requirements.txt exists)
        #    Note: Installing here might fail if fluxgraph itself isn't installed yet.
        #    This is a basic attempt. A more robust setup would use setup.py install/editable.
        requirements_file = "requirements.txt"
        if os.path.isfile(requirements_file):
            venv_python = _get_python_executable(venv_path)
            logging.getLogger(__name__).info(f"Installing dependencies from '{requirements_file}'...")
            try:
                # Upgrade pip first
                subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
                # Install requirements
                subprocess.run([venv_python, "-m", "pip", "install", "-r", requirements_file], check=True)
                logging.getLogger(__name__).info("Dependencies installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Warning: Failed to install dependencies from '{requirements_file}': {e}")
                print("    You might need to activate the venv and install dependencies manually.")
        else:
             logging.getLogger(__name__).debug(f"'{requirements_file}' not found. Skipping dependency installation.")

    # 5. Activate the virtual environment by re-executing the script
    #    This ensures all subsequent imports use the venv's packages.
    venv_python = _get_python_executable(venv_path)
    if os.path.isfile(venv_python):
        logging.getLogger(__name__).info(f"Activating virtual environment and re-executing with '{venv_python}'...")
        try:
            os.execv(venv_python, [venv_python, __file__] + sys.argv[1:])
        except OSError as e:
            print(f"❌ Failed to re-execute script in virtual environment: {e}")
            sys.exit(1)
    else:
        print(f"❌ Could not find Python executable in virtual environment at '{venv_path}'.")
        sys.exit(1)

# --- Automatically run the venv setup/activation logic ---
# This runs before any other imports.
_ensure_virtual_environment()

# --- 2. STANDARD FLUXGRAPH IMPORTS AND CODE ---
# Import standard library modules needed for the venv logic
# (These were already imported above for the venv function)

# Now import asyncio, logging etc. for the main app logic
import asyncio
import uuid  # For generating request IDs
import time   # For timing requests
from typing import Any, Dict, Callable, Optional
import argparse # For better CLI parsing
from contextvars import ContextVar # For request context

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Import core components
# Note: Relative imports might behave differently after re-execution.
# Ensure the package structure is correct.
try:
    from .registry import AgentRegistry
    from .orchestrator import FluxOrchestrator
    from .tool_registry import ToolRegistry
except ImportError as e:
    print(f"❌ Import error after venv activation: {e}")
    # This might happen if fluxgraph isn't installed in the venv yet.
    # Guide the user.
    print("💡 It seems FluxGraph or its dependencies are not installed in the virtual environment.")
    print("💡 Please activate the venv manually and install the package:")
    print(f"   Unix/macOS:")
    print(f"     source ./{'.venv_fluxgraph'}/bin/activate")
    print(f"     pip install -e .  # If running from source")
    print(f"     # or pip install fluxgraph")
    print(f"   Windows:")
    print(f"     .\\{'.venv_fluxgraph'}\\Scripts\\activate")
    print(f"     pip install -e .")
    sys.exit(1)

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

# --- Safe Import for RAG ---
RAG_AVAILABLE = False
try:
    from .universal_rag import UniversalRAG # Try relative import first
    RAG_AVAILABLE = True
    logging.getLogger(__name__).debug("UniversalRAG found via relative import.")
except (ImportError, ModuleNotFoundError):
    try:
        from fluxgraph.core.universal_rag import UniversalRAG # Try absolute import
        RAG_AVAILABLE = True
        logging.getLogger(__name__).debug("UniversalRAG found via absolute import.")
    except (ImportError, ModuleNotFoundError):
        # RAG is optional, log at debug level
        logging.getLogger(__name__).debug("UniversalRAG not found. RAG features will be disabled.")
        # Create a dummy class for type hints if RAG is not available
        class UniversalRAG: pass

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

# --- Context for Request Tracking ---
# Store request ID in context for use across async calls within a request
request_id_context: ContextVar[str] = ContextVar('request_id', default='N/A')

class FluxApp:
    """
    Main application manager, built on FastAPI.

    Integrates the Agent Registry, Flux Orchestrator, Tool Registry,
    and provides integration points for LangGraph Adapters, Memory, Event Hooks, and RAG.

    Core Components (from MVP Documentation):
    - Agent Registry: Tracks available agents.
    - Flux Orchestrator: Executes agent flows.
    - LangGraph Adapter: Plug-in for LangGraph workflows (used via registration).
    - Event Hooks: Transparent debugging and execution tracking.
    - Tooling Layer: Extendable Python functions (via tool registry).
    - LLM Providers: Integrated via agent logic.
    - Persistence/Memory: Optional integration point.
    - Retrieval Augmented Generation (RAG): Optional integration point.
    """

    def __init__(
        self,
        title: str = "FluxGraph API",
        description: str = "A lightweight Python framework for building, orchestrating, and deploying Agentic AI systems.",
        version: str = "0.1.0",
        memory_store: Optional[Memory] = None,
        rag_connector: Optional[UniversalRAG] = None, # Accept an external RAG connector
        auto_init_rag: bool = True # Flag to enable/disable auto-initialization
    ):
        """
        Initializes the FluxGraph application.

        Args:
            title (str): Title for the FastAPI application.
            description (str): Description for the FastAPI application.
            version (str): Version string.
            memory_store (Optional[Memory]): An optional memory store instance
                                             implementing the Memory interface.
            rag_connector (Optional[UniversalRAG]): An optional RAG connector instance.
                                                   If None and auto_init_rag is True,
                                                   an attempt will be made to create one.
            auto_init_rag (bool): If True and rag_connector is None, automatically
                                  initialize a UniversalRAG instance.
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
        self.rag_connector = rag_connector # Store the RAG connector
        
        # --- Utilities ---
        # Ensure self.hooks is always an instance with an async 'trigger' method
        self.hooks = _EventHooksClass()
        
        # --- Auto-Initialize RAG (if enabled and not provided) ---
        if auto_init_rag and RAG_AVAILABLE and self.rag_connector is None:
            self._auto_initialize_rag()
            
        self._setup_middleware()
        self._setup_routes()
        logger.info(f"✅ FluxApp '{self.title}' (v{self.version}) initialized.")

    def _auto_initialize_rag(self):
        """Attempts to automatically initialize the UniversalRAG connector."""
        AUTO_RAG_PERSIST_DIR = "./my_chroma_db"
        AUTO_RAG_COLLECTION_NAME = "my_knowledge_base"
        
        logger.info("🔄 Attempting to auto-initialize UniversalRAG connector...")
        
        # Ensure the persist directory exists
        persist_path = Path(AUTO_RAG_PERSIST_DIR)
        if not persist_path.exists():
            logger.info(f"📁 Creating RAG persist directory '{AUTO_RAG_PERSIST_DIR}'")
            try:
                persist_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"❌ Failed to create RAG persist directory '{AUTO_RAG_PERSIST_DIR}': {e}")
                self.rag_connector = None
                return
        else:
             logger.debug(f"📂 RAG persist directory '{AUTO_RAG_PERSIST_DIR}' already exists.")

        try:
            # Check for embedding model preference (optional env var)
            embedding_model = os.getenv("FLUXGRAPH_RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            chunk_size = int(os.getenv("FLUXGRAPH_RAG_CHUNK_SIZE", "750"))
            chunk_overlap = int(os.getenv("FLUXGRAPH_RAG_CHUNK_OVERLAP", "100"))

            logger.info(f"🔧 Initializing UniversalRAG with model '{embedding_model}', chunk_size {chunk_size}, overlap {chunk_overlap}...")
            self.rag_connector = UniversalRAG(
                persist_directory=AUTO_RAG_PERSIST_DIR,
                collection_name=AUTO_RAG_COLLECTION_NAME,
                embedding_model_name=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            logger.info("✅ Universal RAG connector auto-initialized successfully.")
            
        except Exception as e:
            logger.error(f"❌ Failed to auto-initialize RAG connector: {e}", exc_info=True)
            self.rag_connector = None # Ensure it's None on failure

    def _setup_middleware(self):
        """Setup default middlewares (e.g., CORS, Logging with Context)."""
        # CORS Middleware
        self.api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Custom Logging Middleware with Request Context and Timing
        @self.api.middleware("http")
        async def log_and_context_middleware(request: Request, call_next):
            # 1. Generate and set a unique Request ID for this request
            request_id = str(uuid.uuid4())
            request_id_context.set(request_id)
            
            # 2. Log incoming request details
            start_time = time.time()
            client_host = request.client.host if request.client else "unknown"
            method = request.method
            url = str(request.url)
            
            logger.info(
                f"[Request ID: {request_id}] 🌐 Incoming request: {method} {url} from {client_host}"
            )

            try:
                # 3. Process the request
                response = await call_next(request)
                
                # 4. Calculate and log processing time
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = str(round(process_time, 4))
                
                logger.info(
                    f"[Request ID: {request_id}] ⬅️ Response status: {response.status_code} "
                    f"(Processing Time: {process_time:.4f}s)"
                )
                return response
                
            except Exception as e:
                # 5. Log unhandled errors during request processing
                process_time = time.time() - start_time
                logger.error(
                    f"[Request ID: {request_id}] ❌ Unhandled error during request processing "
                    f"(Processing Time: {process_time:.4f}s): {e}",
                    exc_info=True # Include full traceback for debugging
                )
                raise # Re-raise to let FastAPI handle the HTTP response

            
        logger.debug("Default middlewares configured.")

    def _setup_routes(self):
        """Setup default API routes."""
        @self.api.get("/", summary="Root Endpoint", description="Welcome message and API status.")
        async def root():
            """Root endpoint providing API information."""
            request_id = request_id_context.get() # Get request ID from context
            logger.info(f"[Request ID: {request_id}] 📢 Root endpoint called.")
            return {
                "message": "Welcome to FluxGraph MVP",
                "title": self.title,
                "version": self.version,
                "memory_enabled": self.memory_store is not None,
                "rag_enabled": self.rag_connector is not None # Report RAG status
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
            # Get request context
            request_id = request_id_context.get()
            start_time = time.time()
            
            logger.info(
                f"[Request ID: {request_id}] 🤖 Executing agent '{agent_name}' with payload keys: {list(payload.keys())}"
            )
            
            # --- Trigger 'request_received' Event Hook ---
            hook_start = time.time()
            await self.hooks.trigger("request_received", {
                "request_id": request_id,
                "agent_name": agent_name,
                "payload": payload,
                "hook_duration": time.time() - hook_start
            })
            logger.debug(f"[Request ID: {request_id}] Hook 'request_received' triggered.")

            try:
                # --- Execute Agent via Orchestrator ---
                result = await self.orchestrator.run(agent_name, payload)
                
                end_time = time.time()
                duration = end_time - start_time
                
                # --- Trigger 'agent_completed' Event Hook ---
                hook_start = time.time()
                await self.hooks.trigger("agent_completed", {
                    "request_id": request_id,
                    "agent_name": agent_name,
                    "result": result,
                    "duration": duration,
                    "hook_duration": time.time() - hook_start
                })
                logger.debug(f"[Request ID: {request_id}] Hook 'agent_completed' triggered.")
                
                logger.info(
                    f"[Request ID: {request_id}] ✅ Agent '{agent_name}' executed successfully "
                    f"(Total Duration: {duration:.4f}s)."
                )
                return result
                
            except ValueError as e: # Agent not found or execution logic error from agent
                end_time = time.time()
                duration = end_time - start_time
                logger.warning(
                    f"[Request ID: {request_id}] ⚠️ Agent '{agent_name}' error (Duration: {duration:.4f}s): {e}"
                )
                
                # --- Trigger 'agent_error' Event Hook ---
                hook_start = time.time()
                await self.hooks.trigger("agent_error", {
                    "request_id": request_id,
                    "agent_name": agent_name,
                    "error": str(e),
                    "duration": duration,
                    "hook_duration": time.time() - hook_start
                })
                logger.debug(f"[Request ID: {request_id}] Hook 'agent_error' triggered.")
                
                # Differentiate status code based on error type if needed
                status_code = 404 if "not registered" in str(e).lower() or "not found" in str(e).lower() else 400
                raise HTTPException(status_code=status_code, detail=str(e))
                
            except Exception as e: # Unexpected server error
                end_time = time.time()
                duration = end_time - start_time
                logger.error(
                    f"[Request ID: {request_id}] ❌ Execution error for agent '{agent_name}' (Duration: {duration:.4f}s): {e}",
                    exc_info=True # Log full traceback
                )
                
                # --- Trigger 'server_error' Event Hook ---
                hook_start = time.time()
                await self.hooks.trigger("server_error", {
                    "request_id": request_id,
                    "agent_name": agent_name,
                    "error": str(e),
                    "duration": duration,
                    "hook_duration": time.time() - hook_start
                })
                logger.debug(f"[Request ID: {request_id}] Hook 'server_error' triggered.")
                
                raise HTTPException(status_code=500, detail="Internal Server Error")

        # --- Tooling Layer Endpoints ---
        @self.api.get("/tools", summary="List Tools", description="Get a list of all registered tool names.")
        async def list_tools():
            """Endpoint to list registered tool names."""
            request_id = request_id_context.get()
            logger.info(f"[Request ID: {request_id}] 🛠️ Listing registered tools.")
            return {"tools": self.tool_registry.list_tools()}

        @self.api.get("/tools/{tool_name}", summary="Get Tool Info", description="Get detailed information about a specific tool.")
        async def get_tool_info(tool_name: str):
            """Endpoint to get information about a specific tool."""
            request_id = request_id_context.get()
            try:
                logger.info(f"[Request ID: {request_id}] 🔎 Fetching tool info for '{tool_name}'.")
                info = self.tool_registry.get_tool_info(tool_name)
                return info
            except ValueError as e:
                logger.warning(f"[Request ID: {request_id}] ⚠️ Tool '{tool_name}' not found: {e}")
                raise HTTPException(status_code=404, detail=str(e))

        # --- Memory Status Endpoint (if memory is enabled) ---
        if MEMORY_AVAILABLE and self.memory_store:
            @self.api.get("/memory/status", summary="Memory Status", description="Check if a memory store is configured.")
            async def memory_status():
                """Endpoint to check memory store status."""
                request_id = request_id_context.get()
                logger.info(f"[Request ID: {request_id}] 💾 Memory status requested.")
                return {"memory_enabled": True, "type": type(self.memory_store).__name__}
        else:
            # Log this only once during setup, not on every request
            # logger.debug("Memory endpoints not added as memory store is not configured.")
            pass # Already logged in __init__ if needed

        # --- NEW: RAG Status Endpoint (if RAG is enabled) ---
        if RAG_AVAILABLE and self.rag_connector:
            @self.api.get("/rag/status", summary="RAG Status", description="Check if a RAG connector is configured and get stats.")
            async def rag_status():
                """Endpoint to check RAG connector status."""
                request_id = request_id_context.get()
                logger.info(f"[Request ID: {request_id}] 🔍 RAG status requested.")
                try:
                    stats = self.rag_connector.get_collection_stats()
                    return {
                        "rag_enabled": True,
                        "type": type(self.rag_connector).__name__,
                        "collection_stats": stats
                    }
                except Exception as e:
                    logger.error(f"[Request ID: {request_id}] ❌ Error getting RAG stats: {e}", exc_info=True)
                    return {
                        "rag_enabled": True,
                        "type": type(self.rag_connector).__name__,
                        "stats_error": str(e)
                    }
        # --- END OF NEW RAG ENDPOINT ---

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

    # --- Agent Definition Decorator (UPDATED to inject RAG) ---
    def agent(self, name: Optional[str] = None):
        """
        Decorator to define and register an agent function.

        The decorated function becomes the `run` method of a dynamically created
        agent class. The `tools` registry, `memory` store, and `rag` connector (if configured)
        are automatically injected as keyword arguments when the agent is executed.

        This simplifies agent creation for logic that doesn't require a full class.

        Args:
            name (Optional[str]): The name to register the agent under.
                                  Defaults to the function's `__name__`.

        Usage:
            @app.agent()
            async def my_agent(query: str, tools, memory, rag):
                # Use tools.get('tool_name') to access tools
                # Use await memory.add(...) if memory is available
                # Use await rag.query(...) if rag is available
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
                    if self._rag_connector: # <-- INJECT RAG CONNECTOR
                        kwargs['rag'] = self._rag_connector
                    
                    # Execute the user-defined function
                    if asyncio.iscoroutinefunction(func):
                        return await func(**kwargs)
                    else:
                        return func(**kwargs)
            
            # Instantiate the dynamic agent and inject FluxApp dependencies
            agent_instance = _FluxDynamicAgent()
            agent_instance._tool_registry = self.tool_registry
            agent_instance._memory_store = self.memory_store
            agent_instance._rag_connector = self.rag_connector # <-- ASSIGN RAG CONNECTOR

            # Register the instance with the FluxApp
            self.register(agent_name, agent_instance)
            logger.info(f"🤖 Agent '{agent_name}' registered via @app.agent decorator.")
            return func # Return the original function
        return decorator
    # --- END OF UPDATE ---

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

