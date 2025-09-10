# fluxgraph/core/adapter.py
"""
LangGraph Adapter for FluxGraph.

This module provides the `LangGraphAdapter` class, which allows integrating
LangGraph workflows as first-class agents within the FluxGraph framework.

By using the adapter, developers can wrap their compiled LangGraph state machines
and register them with the FluxGraph `AgentRegistry`. These workflows can then
be invoked via the standard `/ask/{agent_name}` REST API endpoint, just like
any other FluxGraph agent.

This enables structured, multi-step agent logic defined using LangGraph's
powerful state machine abstractions to be easily orchestrated and deployed
within the FluxGraph ecosystem.
"""
import logging
from typing import Any, Dict
# Import LangGraph types if available, otherwise set flag
try:
    # Attempt to import the specific type needed
    from langgraph.graph import CompiledGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # If langgraph is not installed, we cannot use this adapter
    LANGGRAPH_AVAILABLE = False
    # Define a placeholder for type hints
    class CompiledGraph: pass # pylint: disable=unused-variable

logger = logging.getLogger(__name__)

class LangGraphAdapter:
    """
    Adapter to integrate LangGraph workflows as FluxGraph agents (MVP Implementation).

    This adapter bridges the gap between LangGraph's asynchronous workflow execution
    (`CompiledGraph.ainvoke`) and FluxGraph's agent interface.

    Features:
    - Wraps compiled `langgraph.StateGraph` objects.
    - Registers LangGraph workflows as standard FluxGraph agents.
    - Invokes workflows via the `/ask/{agent_name}` endpoint.
    - Passes the JSON request payload directly as the initial state to the workflow.
    - Returns the final workflow state as the JSON response.
    - Handles execution errors gracefully.

    Usage:
        # 1. Define your LangGraph workflow (my_workflow.py)
        from langgraph.graph import StateGraph, END
        from typing import TypedDict

        class AgentState(TypedDict):
            input: str
            output: str

        def my_node(state: AgentState) -> AgentState:
            return {"output": f"Processed input: {state['input']}"}

        workflow = StateGraph(AgentState)
        workflow.add_node("process", my_node)
        workflow.set_entry_point("process")
        workflow.add_edge("process", END)
        app = workflow.compile()

        # 2. In your FluxGraph app setup (e.g., run_server.py)
        from fluxgraph import FluxApp
        from fluxgraph.core.adapter import LangGraphAdapter
        # Import your compiled workflow
        from my_workflow import app as compiled_langgraph_app

        flux_app = FluxApp()

        # 3. Wrap and register the LangGraph workflow
        adapter = LangGraphAdapter()
        lg_agent = adapter.wrap("my_langgraph_agent", compiled_langgraph_app)
        flux_app.register("my_langgraph_agent", lg_agent)

        # 4. The agent is now available at POST /ask/my_langgraph_agent
        #    with the request JSON body passed as the initial state.
    """

    def __init__(self):
        """
        Initializes the LangGraphAdapter.

        Raises:
            ImportError: If the 'langgraph' library is not installed.
                         This makes the langgraph dependency optional for
                         the core FluxGraph framework.
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "The 'langgraph' library is required for LangGraphAdapter but is not installed. "
                "Please install it using 'pip install langgraph' to use this feature."
            )
        logger.info("LangGraphAdapter initialized successfully.")

    def wrap(self, name: str, workflow: 'CompiledGraph') -> '_LangGraphFluxAgent':
        """
        Wraps a compiled LangGraph workflow into a FluxGraph agent.

        Args:
            name (str): The unique name to register the agent under in FluxGraph.
            workflow (langgraph.graph.CompiledGraph): The compiled LangGraph
                StateGraph object to be wrapped.

        Returns:
            _LangGraphFluxAgent: An internal wrapper object that conforms to
                the FluxGraph agent interface (i.e., has a `run` method).
                This object should be passed to `FluxApp.register()`.

        Raises:
            TypeError: If the provided `workflow` is not an instance of
                       `langgraph.graph.CompiledGraph`.
        """
        if not isinstance(workflow, CompiledGraph):
            raise TypeError(
                f"The 'workflow' argument must be a compiled LangGraph object "
                f"(langgraph.graph.CompiledGraph). "
                f"Got {type(workflow).__name__}. "
                f"Please ensure you are passing the result of `StateGraph.compile()`."
            )

        logger.debug(f"Wrapping LangGraph workflow '{workflow.get_name() if hasattr(workflow, 'get_name') else 'N/A'}' as FluxGraph agent '{name}'.")
        # Create and return the internal wrapper instance
        return _LangGraphFluxAgent(name, workflow)


class _LangGraphFluxAgent:
    """
    Internal wrapper class making a LangGraph workflow compatible with FluxGraph.

    This class is the output of `LangGraphAdapter.wrap` and is what gets
    registered with the FluxGraph `AgentRegistry`. It provides the standard
    `run(**kwargs)` method that the FluxGraph orchestrator expects.

    It is not intended to be instantiated directly by user code.
    """

    def __init__(self, name: str, workflow: 'CompiledGraph'):
        """
        Initializes the wrapper.

        Args:
            name (str): The name of the agent.
            workflow (langgraph.graph.CompiledGraph): The compiled workflow to wrap.
        """
        self.name = name
        self.workflow = workflow # The compiled langgraph StateGraph

    async def run(self, **payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the wrapped LangGraph workflow asynchronously.

        This method is called by the FluxGraph `SimpleOrchestrator` when the
        agent is invoked via the `/ask/{agent_name}` endpoint.

        Args:
            **payload (Dict[str, Any]): The JSON payload from the incoming
                HTTP request. This dictionary is passed directly as the
                initial state to the LangGraph workflow's `ainvoke` method.
                It's crucial that the structure of this payload matches the
                state schema defined in the LangGraph workflow.

        Returns:
            Dict[str, Any]: The final state dictionary of the LangGraph
                workflow after it has completed execution. This dictionary
                is returned as the JSON response body by the FluxGraph API.
                If an error occurs during execution, a dictionary containing
                error information is returned.
        """
        try:
            logger.info(f"Starting execution of LangGraph agent '{self.name}' with initial state keys: {list(payload.keys())}")
            
            # --- Core Execution ---
            # The payload from the FluxGraph API call (JSON body) is used as the
            # initial state for the LangGraph workflow.
            # LangGraph's `ainvoke` method is used for asynchronous execution.
            final_state = await self.workflow.ainvoke(input=payload)

            logger.info(f"LangGraph agent '{self.name}' completed successfully.")
            
            # Ensure the result is a dictionary for JSON serialization.
            # If the state is a Pydantic model, convert it.
            if hasattr(final_state, 'model_dump'): # Pydantic v2
                return final_state.model_dump()
            elif hasattr(final_state, 'dict'): # Pydantic v1
                return final_state.dict()
            elif isinstance(final_state, dict):
                return final_state
            else:
                # Fallback: attempt dict conversion
                return dict(final_state)

        except Exception as e:
            error_msg = f"Error executing LangGraph agent '{self.name}': {e}"
            logger.error(error_msg, exc_info=True) # Log full traceback
            # Return a structured error response to the FluxGraph API caller
            return {
                "error": "LangGraph Agent Execution Failed",
                "agent_name": self.name,
                "details": str(e)
                # Consider adding traceback info in debug mode if needed
            }
