# In fluxgraph/api/designer_routes.py
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, List

# Import your visual builder class
from fluxgraph.workflows.visual_builder import VisualWorkflow

# --- Mock Database ---
# In a real app, you'd save/load this from Postgres or Redis.
WORKFLOW_DB: Dict[str, str] = {}
# ---------------------

router = APIRouter(prefix="/designer", tags=["Workflow Designer"])

class WorkflowSaveRequest(BaseModel):
    """Pydantic model for receiving workflow JSON from the frontend."""
    id: str
    workflow_json: Dict[str, Any]

@router.get("/ui", response_class=HTMLResponse)
async def get_designer_ui():
    """
    Serves the main HTML file for the visual designer.
    We'll create this 'designer.html' file in the next step.
    """
    try:
        # Assumes designer.html is in a 'static' or 'templates' folder
        with open("fluxgraph/templates/designer.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Designer UI not found.</h1>", status_code=404)

@router.get("/api/workflows")
async def list_workflows() -> List[str]:
    """Get a list of all saved workflow IDs."""
    return list(WORKFLOW_DB.keys())

@router.get("/api/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Load a specific workflow's JSON representation."""
    if workflow_id not in WORKFLOW_DB:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Load the JSON string from our 'db'
    workflow_json_str = WORKFLOW_DB[workflow_id]
    
    # Use your class to validate and return
    # This ensures we only return valid workflow JSON
    workflow = VisualWorkflow.from_json(workflow_json_str)
    return workflow.to_dict() # Or just return the JSON string

@router.post("/api/workflows")
async def save_workflow(save_request: WorkflowSaveRequest):
    """
    Save a new or existing workflow.
    The frontend will send the complete workflow structure as JSON.
    """
    try:
        # Create a VisualWorkflow instance from the received JSON
        # This validates the structure using your existing class
        workflow = VisualWorkflow(save_request.id)
        workflow.start_node = save_request.workflow_json.get('start_node')
        
        # Re-create nodes from the JSON data
        for node_id, node_data in save_request.workflow_json.get('nodes', {}).items():
            workflow.nodes[node_id] = node_data # Simplified for this example
            # A more robust implementation would use a Node.from_dict() method

        # Convert it back to a clean JSON string for saving
        json_str_to_save = workflow.to_json()
        
        # Save to our 'db'
        WORKFLOW_DB[save_request.id] = json_str_to_save
        
        return {"status": "success", "workflow_id": save_request.id}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save workflow: {e}")