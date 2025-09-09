# fluxgraph/core/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, Callable, Optional
import logging
import functools
from .registry import AgentRegistry
from .orchestrator import SimpleOrchestrator

logger = logging.getLogger(__name__)

class FluxApp:
    """
    Main application manager, built on FastAPI.
    Integrates the Agent Registry and Orchestrator.
    """
    def __init__(self, title: str = "FluxGraph API", description: str = "", version: str = "0.1.0"):
        self.title = title
        self.description = description
        self.version = version
        self.api = FastAPI(title=self.title, description=self.description, version=self.version)
        self.registry = AgentRegistry()
        self.orchestrator = SimpleOrchestrator(self.registry)
        self._setup_routes()
        self._setup_middleware()

    def _setup_middleware(self):
        """Setup default middlewares."""
        self.api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup default API routes."""
        @self.api.get("/")
        async def root():
            return {
                "message": "Welcome to FluxGraph MVP",
                "title": self.title,
                "version": self.version
            }

        @self.api.post("/ask/{agent_name}")
        async def ask_agent(agent_name: str, payload: Dict[str, Any]):
            try:
                result = await self.orchestrator.run(agent_name, payload)
                return result
            except ValueError as e: # Agent not found or execution error from agent logic
                logger.error(f"Agent '{agent_name}' error: {e}")
                raise HTTPException(status_code=404 if "not found" in str(e).lower() else 400, detail=str(e))
            except Exception as e: # Unexpected server error
                logger.error(f"Execution error for agent '{agent_name}': {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

    def register(self, name: str, agent: Any):
        """Register an agent with the registry."""
        self.registry.add(name, agent)

    def run(self, host: str = "127.0.0.1", port: int = 8000, **kwargs):
        """Placeholder for run. Use `uvicorn.run(self.api, ...)` externally."""
        import uvicorn
        uvicorn.run(self.api, host=host, port=port, **kwargs)

    # --- NEW: Decorator for easy agent definition ---
    def agent(self, name: Optional[str] = None):
        """
        Decorator to define and register an agent function.

        The decorated function becomes the `run` method of a simple agent class.
        The agent is automatically registered with the app using the function's name
        (or a provided `name`).

        Usage:
            @app.agent()
            async def my_agent(query: str):
                return {"response": f"Echo: {query}"}

            # Or with a custom name
            @app.agent(name="custom_echo")
            async def another_func(query: str):
                return {"response": f"Custom Echo: {query}"}
        """
        def decorator(func: Callable) -> Callable:
            # Determine the agent name
            agent_name = name if name is not None else func.__name__

            # Create a simple agent class on the fly
            class _DynamicAgent:
                async def run(self, **kwargs):
                    # Call the original function with the provided kwargs
                    if asyncio.iscoroutinefunction(func):
                        return await func(**kwargs)
                    else:
                        return func(**kwargs)

            # Register the agent instance with the app's registry
            self.register(agent_name, _DynamicAgent())
            logger.info(f"Agent '{agent_name}' registered via @app.agent decorator.")
            return func # Return the original function, unchanged
        return decorator

# Import asyncio here as it's used in the decorator logic
import asyncio
