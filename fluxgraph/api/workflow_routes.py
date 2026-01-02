"""
Workflow API Routes
Handle CRUD operations for visual workflows and execution
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime
import asyncio
import json
import logging

# Import your existing FluxGraph components with CORRECT class names
try:
    from fluxgraph.core.app import FluxApp
    from fluxgraph.workflows.visual_builder import VisualWorkflow  # ✅ Changed from WorkflowBuilder
    try:
        from fluxgraph.core.orchestrator import AgentOrchestrator
    except ImportError:
        AgentOrchestrator = None
    FLUXGRAPH_AVAILABLE = True
    logging.info("✅ FluxGraph core components imported successfully")
except ImportError as e:
    logging.warning(f"⚠️ FluxGraph core imports not available: {e}")
    FLUXGRAPH_AVAILABLE = False
    # Create mock classes for development
    class FluxApp:
        def __init__(self, **kwargs):
            self.orchestrator = None
    class VisualWorkflow:  # ✅ Changed from WorkflowBuilder
        pass
    class AgentOrchestrator:
        pass

router = APIRouter(prefix="/api", tags=["workflows"])

# Pydantic models for API requests/responses
class NodeData(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any]

class EdgeData(BaseModel):
    id: str
    source: str
    target: str
    data: Optional[Dict[str, Any]] = {}

class WorkflowCreateRequest(BaseModel):
    name: str = Field(..., description="Workflow name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Workflow description", max_length=1000)
    nodes: List[NodeData] = Field(..., description="Workflow nodes", min_items=1)
    edges: List[EdgeData] = Field(..., description="Workflow edges")
    tags: Optional[List[str]] = Field(default=[], description="Workflow tags")
    config: Optional[Dict[str, Any]] = Field(default={}, description="Additional configuration")

class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    nodes: List[NodeData]
    edges: List[EdgeData]
    tags: List[str]
    status: str
    version: int
    created_at: datetime
    updated_at: datetime

class WorkflowExecuteRequest(BaseModel):
    input_data: Dict[str, Any] = Field(..., description="Input data for workflow execution")
    enable_checkpointing: bool = Field(True, description="Enable state checkpointing")
    timeout: Optional[int] = Field(300, description="Execution timeout in seconds")

class WorkflowExecutionResponse(BaseModel):
    execution_id: str
    status: str
    started_at: datetime
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_trace: List[Dict[str, Any]] = []

# In-memory storage (replace with your database later)
workflows_db: Dict[str, Dict] = {}
executions_db: Dict[str, Dict] = {}

# Initialize FluxApp instance for execution
flux_app = None
if FLUXGRAPH_AVAILABLE:
    try:
        flux_app = FluxApp(
            enable_agent_cache=True,
            enable_advanced_memory=True,
            cache_strategy="semantic"
        )
        logging.info("✅ FluxApp initialized for workflow execution")
    except Exception as e:
        logging.warning(f"⚠️ Could not initialize FluxApp: {e}")
        flux_app = None

@router.post("/workflows", response_model=WorkflowResponse, summary="Create a new workflow")
async def create_workflow(workflow: WorkflowCreateRequest):
    """
    Create a new visual workflow.
    
    - **name**: Workflow name (required)
    - **description**: Optional description
    - **nodes**: List of workflow nodes with positions and data
    - **edges**: Connections between nodes
    - **tags**: Optional tags for organization
    """
    workflow_id = str(uuid4())
    
    # Validate nodes and edges
    if not workflow.nodes:
        raise HTTPException(status_code=400, detail="Workflow must have at least one node")
    
    # Check for start node
    start_nodes = [n for n in workflow.nodes if n.data.get("type") == "start" or n.data.get("isStart", False)]
    if not start_nodes:
        raise HTTPException(status_code=400, detail="Workflow must have a start node")
    
    # Validate edges reference existing nodes
    node_ids = {node.id for node in workflow.nodes}
    for edge in workflow.edges:
        if edge.source not in node_ids:
            raise HTTPException(status_code=400, detail=f"Edge source '{edge.source}' not found in nodes")
        if edge.target not in node_ids:
            raise HTTPException(status_code=400, detail=f"Edge target '{edge.target}' not found in nodes")
    
    # Store workflow
    workflow_data = {
        "id": workflow_id,
        "name": workflow.name,
        "description": workflow.description,
        "nodes": [n.dict() for n in workflow.nodes],
        "edges": [e.dict() for e in workflow.edges],
        "tags": workflow.tags,
        "config": workflow.config,
        "status": "draft",
        "version": 1,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    workflows_db[workflow_id] = workflow_data
    logging.info(f"✅ Workflow created: {workflow_id} - {workflow.name}")
    
    return WorkflowResponse(**workflow_data)

@router.get("/workflows", response_model=List[WorkflowResponse], summary="List all workflows")
async def list_workflows(
    skip: int = 0,
    limit: int = 50,
    tags: Optional[str] = None,
    status: Optional[str] = None
):
    """
    List all workflows with optional filtering.
    
    - **skip**: Number of workflows to skip (pagination)
    - **limit**: Maximum number of workflows to return
    - **tags**: Comma-separated list of tags to filter by
    - **status**: Filter by workflow status (draft, published, archived)
    """
    filtered_workflows = list(workflows_db.values())
    
    # Filter by tags
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        filtered_workflows = [
            wf for wf in filtered_workflows
            if any(tag in wf.get("tags", []) for tag in tag_list)
        ]
    
    # Filter by status
    if status:
        filtered_workflows = [
            wf for wf in filtered_workflows 
            if wf.get("status") == status
        ]
    
    # Sort by updated_at descending
    filtered_workflows.sort(key=lambda x: x.get("updated_at", datetime.min), reverse=True)
    
    # Apply pagination
    filtered_workflows = filtered_workflows[skip:skip + limit]
    
    return [WorkflowResponse(**wf) for wf in filtered_workflows]

@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse, summary="Get workflow by ID")
async def get_workflow(workflow_id: str):
    """Get a specific workflow by ID."""
    if workflow_id not in workflows_db:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow = workflows_db[workflow_id]
    return WorkflowResponse(**workflow)

@router.put("/workflows/{workflow_id}", response_model=WorkflowResponse, summary="Update workflow")
async def update_workflow(workflow_id: str, workflow_update: WorkflowCreateRequest):
    """Update an existing workflow. This increments the version number."""
    if workflow_id not in workflows_db:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    existing_workflow = workflows_db[workflow_id]
    
    # Update fields
    updated_data = {
        **existing_workflow,
        "name": workflow_update.name,
        "description": workflow_update.description,
        "nodes": [n.dict() for n in workflow_update.nodes],
        "edges": [e.dict() for e in workflow_update.edges],
        "tags": workflow_update.tags,
        "config": workflow_update.config,
        "version": existing_workflow["version"] + 1,
        "updated_at": datetime.utcnow()
    }
    
    workflows_db[workflow_id] = updated_data
    logging.info(f"✅ Workflow updated: {workflow_id} - v{updated_data['version']}")
    
    return WorkflowResponse(**updated_data)

@router.delete("/workflows/{workflow_id}", summary="Delete workflow")
async def delete_workflow(workflow_id: str):
    """Delete a workflow (soft delete by archiving)."""
    if workflow_id not in workflows_db:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Soft delete by changing status
    workflows_db[workflow_id]["status"] = "archived"
    workflows_db[workflow_id]["updated_at"] = datetime.utcnow()
    
    logging.info(f"🗑️ Workflow archived: {workflow_id}")
    return {"message": "Workflow archived successfully", "workflow_id": workflow_id}

@router.post("/workflows/{workflow_id}/duplicate", response_model=WorkflowResponse, summary="Duplicate workflow")
async def duplicate_workflow(workflow_id: str):
    """
    Duplicate an existing workflow with new IDs for all nodes and edges.
    """
    if workflow_id not in workflows_db:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    original = workflows_db[workflow_id]
    new_id = str(uuid4())
    
    # Generate new IDs for nodes and edges to avoid conflicts
    id_mapping = {}
    new_nodes = []
    
    for node in original["nodes"]:
        old_id = node["id"]
        new_node_id = f"{old_id}_copy_{new_id[:8]}"
        id_mapping[old_id] = new_node_id
        
        new_node = {**node, "id": new_node_id}
        new_nodes.append(new_node)
    
    # Update edge references
    new_edges = []
    for edge in original["edges"]:
        new_edge = {
            **edge,
            "id": f"{edge['id']}_copy_{new_id[:8]}",
            "source": id_mapping.get(edge["source"], edge["source"]),
            "target": id_mapping.get(edge["target"], edge["target"])
        }
        new_edges.append(new_edge)
    
    # Create duplicate
    duplicate_data = {
        **original,
        "id": new_id,
        "name": f"{original['name']} (Copy)",
        "nodes": new_nodes,
        "edges": new_edges,
        "status": "draft",
        "version": 1,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    workflows_db[new_id] = duplicate_data
    logging.info(f"📋 Workflow duplicated: {workflow_id} → {new_id}")
    
    return WorkflowResponse(**duplicate_data)

@router.post("/workflows/{workflow_id}/execute", response_model=WorkflowExecutionResponse, summary="Execute workflow")
async def execute_workflow(
    workflow_id: str, 
    execute_request: WorkflowExecuteRequest,
    background_tasks: BackgroundTasks
):
    """
    Execute a workflow (non-blocking).
    
    Use WebSocket endpoint `/api/workflows/ws/{workflow_id}` for real-time execution updates.
    """
    if workflow_id not in workflows_db:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow = workflows_db[workflow_id]
    execution_id = str(uuid4())
    
    # Create execution record
    execution_data = {
        "execution_id": execution_id,
        "workflow_id": workflow_id,
        "status": "running",
        "started_at": datetime.utcnow(),
        "input_data": execute_request.input_data,
        "output_data": None,
        "error_message": None,
        "execution_trace": []
    }
    
    executions_db[execution_id] = execution_data
    
    # Start background execution
    background_tasks.add_task(
        _execute_workflow_background, 
        workflow, 
        execution_id, 
        execute_request
    )
    
    logging.info(f"🚀 Workflow execution started: {execution_id}")
    return WorkflowExecutionResponse(**execution_data)

async def _execute_workflow_background(
    workflow: Dict, 
    execution_id: str, 
    execute_request: WorkflowExecuteRequest
):
    """Background task for workflow execution"""
    try:
        # Convert visual workflow to executable format
        execution_graph = _compile_workflow_graph(workflow["nodes"], workflow["edges"])
        
        # Execute the workflow
        result = await _execute_graph(
            execution_graph, 
            execute_request.input_data,
            execution_id
        )
        
        # Update execution record
        executions_db[execution_id].update({
            "status": "completed",
            "output_data": result,
            "completed_at": datetime.utcnow(),
            "duration_ms": int((datetime.utcnow() - executions_db[execution_id]["started_at"]).total_seconds() * 1000)
        })
        
        logging.info(f"✅ Workflow execution completed: {execution_id}")
        
    except Exception as e:
        # Handle execution errors
        executions_db[execution_id].update({
            "status": "failed",
            "error_message": str(e),
            "completed_at": datetime.utcnow()
        })
        logging.error(f"❌ Workflow execution failed for {execution_id}: {e}")

def _compile_workflow_graph(nodes: List[Dict], edges: List[Dict]) -> Dict:
    """Convert React Flow nodes/edges to executable workflow graph"""
    graph = {}
    node_map = {node["id"]: node for node in nodes}
    
    # Build adjacency list
    for node in nodes:
        node_id = node["id"]
        node_type = node["data"].get("type", "agent")
        
        graph[node_id] = {
            "type": node_type,
            "data": node["data"],
            "next": []
        }
    
    # Add edges
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        
        if source in graph:
            graph[source]["next"].append({
                "target": target,
                "condition": edge.get("data", {}).get("condition")
            })
    
    # Find start node
    start_nodes = [n for n in nodes if n["data"].get("type") == "start" or n["data"].get("isStart", False)]
    if not start_nodes:
        raise ValueError("Workflow must have a start node")
    
    return {
        "start": start_nodes[0]["id"],
        "graph": graph
    }

async def _execute_graph(execution_graph: Dict, input_data: Dict, execution_id: str) -> Dict:
    """Execute the compiled workflow graph"""
    current_state = {"input": input_data}
    current_node_id = execution_graph["start"]
    graph = execution_graph["graph"]
    
    execution_trace = []
    max_iterations = 100  # Prevent infinite loops
    iteration_count = 0
    
    while current_node_id and iteration_count < max_iterations:
        iteration_count += 1
        node_info = graph.get(current_node_id)
        if not node_info:
            break
            
        node_type = node_info["type"]
        node_data = node_info["data"]
        
        # Add to execution trace
        trace_entry = {
            "node_id": current_node_id,
            "node_type": node_type,
            "timestamp": datetime.utcnow().isoformat(),
            "input_state": current_state.copy()
        }
        
        try:
            # Execute node based on type
            if node_type == "agent":
                result = await _execute_agent_node(node_data, current_state)
            elif node_type == "conditional":
                result = await _execute_conditional_node(node_data, current_state)
            elif node_type == "end":
                result = current_state
                current_node_id = None  # End execution
            else:
                result = current_state  # Pass through unknown nodes
            
            # Update state
            if isinstance(result, dict):
                current_state.update(result)
            
            trace_entry["output_state"] = current_state.copy()
            trace_entry["status"] = "completed"
            
            # Determine next node
            if current_node_id and node_type != "end":
                next_nodes = node_info["next"]
                
                if not next_nodes:
                    break
                
                # Handle conditional branching
                if node_type == "conditional" and "branch" in result:
                    condition_result = result["branch"]
                    next_node_found = False
                    
                    for next_node in next_nodes:
                        if next_node.get("condition") == condition_result:
                            current_node_id = next_node["target"]
                            next_node_found = True
                            break
                    
                    if not next_node_found:
                        # Default to first next node
                        current_node_id = next_nodes[0]["target"] if next_nodes else None
                else:
                    current_node_id = next_nodes[0]["target"] if next_nodes else None
            
        except Exception as e:
            trace_entry["status"] = "failed"
            trace_entry["error"] = str(e)
            execution_trace.append(trace_entry)
            raise
        
        execution_trace.append(trace_entry)
    
    # Update execution trace
    if execution_id in executions_db:
        executions_db[execution_id]["execution_trace"] = execution_trace
    
    return current_state

async def _execute_agent_node(node_data: Dict, state: Dict) -> Dict:
    """Execute an agent node using FluxGraph orchestrator"""
    agent_name = node_data.get("agent_name", "default_agent")
    agent_config = node_data.get("config", {})
    
    # Simulate processing time
    await asyncio.sleep(0.5)
    
    # Mock response for now - replace with actual FluxGraph agent execution
    response = {
        "agent_result": f"Agent {agent_name} processed: {state.get('input', '')}",
        "cached": False,  # Will be set by actual caching system
        "cost": 0.001,  # Mock cost
        "agent_name": agent_name
    }
    
    # If FluxGraph is available, use real execution
    if flux_app and hasattr(flux_app, 'orchestrator') and flux_app.orchestrator:
        try:
            real_result = await flux_app.orchestrator.run_agent(
                agent_name,
                input_data={**state, **agent_config}
            )
            response.update(real_result)
            logging.info(f"✅ Real agent executed: {agent_name}")
        except Exception as e:
            logging.warning(f"⚠️ Could not execute real agent: {e}")
    
    return response

async def _execute_conditional_node(node_data: Dict, state: Dict) -> Dict:
    """Execute a conditional node"""
    condition_type = node_data.get("condition_type", "expression")
    
    if condition_type == "expression":
        expression = node_data.get("expression", "True")
        # Safely evaluate expression (in production, use a proper expression evaluator)
        try:
            # Simple evaluation for demo - replace with safe evaluator in production
            if "true" in expression.lower():
                result = "true"
            elif "false" in expression.lower():
                result = "false"
            else:
                # Basic evaluation for numbers/comparisons using safe AST evaluation
                import re
                import ast
                import operator as op

                # Define safe operators
                safe_operators = {
                    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                    ast.Div: op.truediv, ast.Mod: op.mod,
                    ast.Eq: op.eq, ast.NotEq: op.ne,
                    ast.Lt: op.lt, ast.LtE: op.le,
                    ast.Gt: op.gt, ast.GtE: op.ge
                }

                def safe_eval(expr_str):
                    """Safely evaluate simple mathematical and comparison expressions."""
                    try:
                        node = ast.parse(expr_str, mode='eval').body

                        def _eval(node):
                            if isinstance(node, ast.Constant):  # Python 3.8+
                                return node.value
                            elif isinstance(node, ast.Num):  # Python 3.7 compatibility
                                return node.n
                            elif isinstance(node, ast.BinOp):
                                left = _eval(node.left)
                                right = _eval(node.right)
                                return safe_operators[type(node.op)](left, right)
                            elif isinstance(node, ast.Compare):
                                left = _eval(node.left)
                                for op_node, comparator in zip(node.ops, node.comparators):
                                    right = _eval(comparator)
                                    if not safe_operators[type(op_node)](left, right):
                                        return False
                                    left = right
                                return True
                            else:
                                raise ValueError(f"Unsupported expression type: {type(node)}")

                        return _eval(node)
                    except Exception:
                        return None

                if re.search(r'\d+\s*[><=]+\s*\d+', expression):
                    eval_result = safe_eval(expression)
                    result = "true" if eval_result else "false"
                else:
                    result = "true"
        except Exception:
            result = "false"
        
        return {"branch": result}
    
    elif condition_type == "threshold":
        field = node_data.get("field", "value")
        threshold = node_data.get("threshold", 0)
        operator = node_data.get("operator", ">")
        
        value = state.get(field, 0)
        if isinstance(value, str):
            try:
                value = float(value)
            except:
                value = 0
        
        if operator == ">":
            result = value > threshold
        elif operator == "<":
            result = value < threshold
        elif operator == "==":
            result = value == threshold
        elif operator == ">=":
            result = value >= threshold
        elif operator == "<=":
            result = value <= threshold
        else:
            result = False
        
        return {"branch": "true" if result else "false"}
    
    return {"branch": "default"}

@router.get("/workflows/{workflow_id}/executions", summary="Get execution history")
async def get_workflow_executions(workflow_id: str):
    """Get execution history for a specific workflow."""
    executions = [
        exec_data for exec_data in executions_db.values()
        if exec_data["workflow_id"] == workflow_id
    ]
    
    # Sort by started_at descending
    executions.sort(key=lambda x: x["started_at"], reverse=True)
    
    return {"executions": executions, "total": len(executions)}

@router.get("/executions/{execution_id}", response_model=WorkflowExecutionResponse, summary="Get execution details")
async def get_execution(execution_id: str):
    """Get specific execution details including trace."""
    if execution_id not in executions_db:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    return WorkflowExecutionResponse(**executions_db[execution_id])

@router.get("/workflows/stats", summary="Get workflow statistics")
async def get_workflows_stats():
    """Get comprehensive workflow and execution statistics."""
    total_workflows = len(workflows_db)
    total_executions = len(executions_db)
    
    # Count by status
    status_counts = {}
    for workflow in workflows_db.values():
        status = workflow.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Execution stats
    execution_status_counts = {}
    for execution in executions_db.values():
        status = execution.get("status", "unknown")
        execution_status_counts[status] = execution_status_counts.get(status, 0) + 1
    
    return {
        "workflows": {
            "total": total_workflows,
            "by_status": status_counts
        },
        "executions": {
            "total": total_executions,
            "by_status": execution_status_counts
        },
        "system": {
            "fluxgraph_available": FLUXGRAPH_AVAILABLE,
            "flux_app_initialized": flux_app is not None
        }
    }
