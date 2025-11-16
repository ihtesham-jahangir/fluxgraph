"""
Graph-Based Workflow Orchestration
Enables complex agent workflows with conditional routing, loops, and state management
(Now with Checkpointing and Observability support)
"""

import asyncio
import logging
from typing import Dict, Any, Callable, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json

# Import the new logger and checkpointer interfaces
try:
    from .checkpointer import BaseCheckpointer
except ImportError:
    # Define a placeholder if the import fails, to allow type hinting
    class BaseCheckpointer: pass

try:
    from .logger import BaseLogger, LogEventType
except ImportError:
    # Define placeholders if the import fails
    class BaseLogger: pass
    class LogEventType:
        WORKFLOW_START = "workflow_start"
        WORKFLOW_END = "workflow_end"
        NODE_START = "node_start"
        NODE_END = "node_end"
        NODE_ERROR = "node_error"

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in workflow graph."""
    AGENT = "agent"
    CONDITION = "condition" 
    LOOP = "loop" 
    PARALLEL = "parallel"
    START = "start"
    END = "end"


@dataclass
class WorkflowState:
    """State container for workflow execution. (Now checkpoint-ready)"""
    workflow_id: str  # Add a unique ID for saving/loading
    data: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_node: Optional[str] = None # This will now store the *next* node to run
    current_node_result: Optional[Any] = None 
    iteration_count: int = 0
    max_iterations: int = 100
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def update(self, key: str, value: Any):
        """Update state and add to history."""
        self.data[key] = value
        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "node": self.current_node,
            "action": "update",
            "key": key,
            # Truncate value if it's too large for history
            "value": str(value)[:200] + "..." if isinstance(value, str) and len(value) > 200 else value
        })
    
    def get(self, key: str, default=None) -> Any:
        """Get value from state."""
        return self.data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict."""
        return {
            "workflow_id": self.workflow_id, # Add ID
            "data": self.data,
            "history": [h for h in self.history if isinstance(h, (dict, list, str, int, float, bool))], # Ensure history is JSON-serializable
            "current_node": self.current_node,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations, # Add max_iterations
            "created_at": self.created_at.isoformat()
        }

    # Add a classmethod to re-hydrate the state easily
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """Re-hydrate state from dict."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        # Handle potential unserializable items in history (if any)
        if 'current_node_result' in data:
            del data['current_node_result'] # This is transient, don't restore
        return cls(**data)


@dataclass
class WorkflowNode:
    """Node in workflow graph."""
    name: str
    node_type: NodeType
    handler: Optional[Callable] = None
    next_nodes: List[str] = field(default_factory=list)
    condition: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"WorkflowNode(name={self.name}, type={self.node_type.value})"


class WorkflowGraph:
    """
    Graph-based workflow orchestration system.
    """
    
    def __init__(
        self, 
        name: str = "workflow", 
        checkpointer: Optional[BaseCheckpointer] = None,
        logger: Optional[BaseLogger] = None  # Add logger
    ):
        self.name = name
        self.checkpointer = checkpointer
        self.logger = logger  # Store the logger
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: Dict[str, List[str]] = {}
        self.conditional_edges: Dict[str, Dict[str, Any]] = {} 
        self.entry_point: Optional[str] = None
        
        # Add special end node
        self.add_node("__end__", NodeType.END)
    
    def add_node(
        self,
        name: str,
        node_type: NodeType,
        handler: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a node to the workflow graph."""
        if name == "__end__" and node_type != NodeType.END:
            raise ValueError("Node '__end__' is reserved for NodeType.END")
        
        self.nodes[name] = WorkflowNode(
            name=name,
            node_type=node_type,
            handler=handler,
            metadata=metadata or {}
        )
        logger.debug(f"Added node: {name} ({node_type.value})")
    
    def add_edge(self, from_node: str, to_node: str):
        """Add a simple edge between two nodes."""
        if from_node not in self.edges:
            self.edges[from_node] = []
        # Prevent duplicate edges
        if to_node not in self.edges[from_node]:
            self.edges[from_node].append(to_node)
        
        logger.debug(f"Added edge: {from_node} -> {to_node}")
    
    def add_conditional_edge(
        self,
        from_node: str,
        condition: Callable[[WorkflowState], str],
        routes: Dict[str, str]
    ):
        """
        Add a conditional edge with routing logic.
        """
        # Ensure only one conditional edge per node
        if from_node in self.conditional_edges or (from_node in self.edges and self.edges[from_node]):
             raise ValueError(f"Node '{from_node}' already has a simple or conditional edge defined.")
        
        # Check that all route targets exist
        for target_node in routes.values():
            if target_node != "__end__" and target_node not in self.nodes:
                raise ValueError(f"Target node '{target_node}' in routes not found.")
        
        self.conditional_edges[from_node] = {
            "condition": condition,
            "routes": routes
        }
        logger.debug(f"Added conditional edge from {from_node} with {len(routes)} routes")
    
    def set_entry_point(self, node_name: str):
        """Set the entry point for workflow execution."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")
        if self.nodes[node_name].node_type in [NodeType.END, NodeType.CONDITION]:
             raise ValueError("Entry point cannot be END or CONDITION type.")
             
        self.entry_point = node_name
        logger.debug(f"Set entry point: {node_name}")
    
    async def execute(
        self,
        workflow_id: str, # Require a workflow_id
        initial_data: Optional[Dict[str, Any]] = None,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Execute the workflow graph step-by-step with checkpointing and logging.
        """
        if not self.entry_point:
            raise ValueError("Entry point not set")

        state: Optional[WorkflowState] = None

        # 1. Load or initialize state
        if self.checkpointer:
            state = await self.checkpointer.load_state(workflow_id)
            
        if state is None:
            # New workflow
            state = WorkflowState(
                workflow_id=workflow_id,
                data=initial_data or {},
                max_iterations=max_iterations,
                current_node=self.entry_point
            )
            logger.info(f"Starting new workflow '{workflow_id}' from '{self.entry_point}'")
            # LOG WORKFLOW START
            if self.logger:
                await self.logger.log_event(workflow_id, None, LogEventType.WORKFLOW_START, initial_data)
        else:
            # Resumed workflow
            logger.info(f"Resuming workflow '{workflow_id}' from node '{state.current_node}'")

        current_node = state.current_node
        
        # 3. Run the loop
        while current_node != "__end__":
            # Check iteration limit
            state.iteration_count += 1
            if state.iteration_count > state.max_iterations:
                error_msg = f"Max iterations ({state.max_iterations}) exceeded"
                if self.logger:
                    await self.logger.log_event(workflow_id, current_node, LogEventType.NODE_ERROR, {"error": error_msg})
                raise RuntimeError(error_msg)
            
            # Get node
            if current_node not in self.nodes:
                error_msg = f"Node '{current_node}' not found"
                if self.logger:
                    await self.logger.log_event(workflow_id, current_node, LogEventType.NODE_ERROR, {"error": error_msg})
                raise ValueError(error_msg)
            
            node = self.nodes[current_node]
            state.current_node = current_node # State reflects the node *being* run

            # LOG NODE START
            if self.logger:
                # Log state *before* execution
                await self.logger.log_event(workflow_id, current_node, LogEventType.NODE_START, state.to_dict())

            result = None
            error = None

            if node.node_type not in [NodeType.START, NodeType.END]:
                try:
                    # Execute node
                    result = await self._execute_node(node, state) 
                    state.update(f"{current_node}_result", result)
                    state.current_node_result = result
                    
                    # LOG NODE END (Success)
                    if self.logger:
                        # Truncate large results for logging
                        log_data = {"result": str(result)[:500] + "..." if isinstance(result, str) and len(str(result)) > 500 else result} 
                        await self.logger.log_event(workflow_id, current_node, LogEventType.NODE_END, log_data)

                except Exception as e:
                    error = str(e)
                    logger.error(f"Error in node '{current_node}': {e}", exc_info=True)
                    state.update(f"{current_node}_error", error)
                    state.current_node_result = {"error": error} 
                    
                    # LOG NODE ERROR
                    if self.logger:
                        await self.logger.log_event(workflow_id, current_node, LogEventType.NODE_ERROR, {"error": error})
            else:
                 state.current_node_result = None # Clear result for non-executing nodes
            
            # Determine next node
            next_node = self._get_next_node(current_node, state) # Pass state to router
            
            if next_node is None:
                logger.warning(f"No valid next node defined from '{current_node}', ending workflow")
                current_node = "__end__"
            else:
                current_node = next_node
                
            state.current_node = current_node # State now points to the *next* node to run

            # 4. Save state to checkpointer
            if self.checkpointer:
                logger.debug(f"Saving state for workflow '{workflow_id}' at next node '{current_node}'")
                await self.checkpointer.save_state(state)
        
        logger.info(f"Workflow '{self.name}' completed in {state.iteration_count} steps")
        # LOG WORKFLOW END
        if self.logger:
            await self.logger.log_event(workflow_id, "__end__", LogEventType.WORKFLOW_END, state.to_dict())
        
        return {
            "result": state.data,
            "history": state.history,
            "iterations": state.iteration_count
        }
    
    async def _execute_node(self, node: WorkflowNode, state: WorkflowState) -> Any:
        """Execute a single node."""
        if node.node_type in [NodeType.AGENT, NodeType.CONDITION]:
            if not node.handler:
                return {"status": "skipped", "message": f"Node {node.name} has no handler."}
            
            # Execute agent or condition handler (handler should return data or routing key)
            if asyncio.iscoroutinefunction(node.handler):
                return await node.handler(state) # Pass state
            # Run sync functions in a thread to prevent blocking the async event loop
            return await asyncio.to_thread(node.handler, state) # Pass state
        
        elif node.node_type == NodeType.PARALLEL:
            # Execute multiple handlers in parallel
            tasks = []
            for handler in node.metadata.get("handlers", []):
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(state)) # Pass state
                else:
                    # Run sync functions in a thread
                    tasks.append(asyncio.to_thread(handler, state)) # Pass state
            # Return list of results from parallel branches
            return await asyncio.gather(*tasks)
        
        return {"status": "no_action", "node_type": node.node_type.value}
    
    def _get_next_node(self, current_node: str, state: WorkflowState) -> Optional[str]:
        """Determine next node based on edges and conditions."""
        
        # 1. Check Conditional Edges (Takes precedence)
        if current_node in self.conditional_edges:
            condition_config = self.conditional_edges[current_node]
            condition_func = condition_config["condition"]
            routes = condition_config["routes"]
            
            # Evaluate condition function - expects a string key matching a route
            try:
                # The condition is executed with the current state object
                route_key = condition_func(state) # Pass state
                if not isinstance(route_key, str):
                    logger.error(f"Condition function for '{current_node}' returned non-string type: {type(route_key)}")
                    route_key = "error" # Fallback key on invalid type
            except Exception as e:
                logger.error(f"Error evaluating condition for '{current_node}': {e}", exc_info=True)
                route_key = "error" # Fallback key on error
            
            target = routes.get(route_key)
            if target:
                logger.debug(f"Conditional routing from {current_node}: Key '{route_key}' -> {target}")
                return target
            else:
                # If key not found, check for a 'default' route
                default_target = routes.get("default")
                if default_target:
                    logger.warning(f"Conditional route key '{route_key}' not found. Using default route: {default_target}")
                    return default_target
                
                logger.error(f"Conditional route key '{route_key}' not found and no default defined for node '{current_node}'.")
                return "__end__"
        
        # 2. Check Simple Edges
        if current_node in self.edges:
            next_nodes = self.edges[current_node]
            if next_nodes:
                # For simple edge, always follow the first defined path
                return next_nodes[0]
        
        # 3. Default to End
        return "__end__"

    def visualize(self) -> str:
        """Generate a simple text visualization of the workflow."""
        lines = [f"Workflow: {self.name}", "=" * 50]
        
        for node_name, node in self.nodes.items():
            if node_name == "__end__":
                continue
            
            line = f"[{node.node_type.value}] {node_name}"
            
            # Add edges
            if node_name in self.edges:
                next_nodes = ", ".join(self.edges[node_name])
                line += f" -> {next_nodes}"
            
            # Add conditional routes
            if node_name in self.conditional_edges:
                routes = self.conditional_edges[node_name]["routes"]
                route_str = ", ".join([f"{k}→{v}" for k, v in routes.items()])
                line += f" [?:{route_str}]"
            
            lines.append(line)
        
        return "\n".join(lines)


class WorkflowBuilder:
    """Fluent builder for creating workflow graphs."""
    
    def __init__(
        self, 
        name: str = "workflow", 
        checkpointer: Optional[BaseCheckpointer] = None,
        logger: Optional[BaseLogger] = None # Add logger
    ):
        self.graph = WorkflowGraph(name, checkpointer, logger) # Pass logger
    
    def add_agent(self, name: str, handler: Callable, metadata: Optional[Dict] = None):
        """Add an agent node."""
        self.graph.add_node(name, NodeType.AGENT, handler, metadata)
        return self
    
    def add_condition(self, name: str, handler: Callable):
        """Add a condition node."""
        self.graph.add_node(name, NodeType.CONDITION, handler)
        return self
    
    def connect(self, from_node: str, to_node: str):
        """Connect two nodes."""
        self.graph.add_edge(from_node, to_node)
        return self
    
    def branch(self, from_node: str, condition: Callable, routes: Dict[str, str]):
        """Add conditional branching."""
        self.graph.add_conditional_edge(from_node, condition, routes)
        return self
    
    def start_from(self, node_name: str):
        """Set entry point."""
        self.graph.set_entry_point(node_name)
        return self
    
    def build(self) -> WorkflowGraph:
        """Build and return the workflow graph."""
        return self.graph