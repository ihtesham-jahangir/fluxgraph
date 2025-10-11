"""
WebSocket API Routes
Real-time workflow execution and monitoring via WebSockets
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import json
import asyncio
from datetime import datetime
import uuid
import logging

router = APIRouter(tags=["websockets"])

# Active WebSocket connections
active_connections: List[WebSocket] = []
execution_connections: Dict[str, List[WebSocket]] = {}
connection_metadata: Dict[WebSocket, Dict] = {}

class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.execution_subscriptions: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        connection_metadata[websocket] = {
            "connection_id": connection_id,
            "connected_at": datetime.utcnow().isoformat(),
            "message_count": 0
        }
        logging.info(f"WebSocket connected: {connection_id}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from all execution subscriptions
        for exec_id, subscribers in list(self.execution_subscriptions.items()):
            if websocket in subscribers:
                subscribers.remove(websocket)
                if not subscribers:
                    del self.execution_subscriptions[exec_id]
        
        if websocket in connection_metadata:
            conn_id = connection_metadata[websocket].get("connection_id", "unknown")
            del connection_metadata[websocket]
            logging.info(f"WebSocket disconnected: {conn_id}")
    
    async def subscribe_to_execution(self, websocket: WebSocket, execution_id: str):
        """Subscribe a connection to execution updates."""
        if execution_id not in self.execution_subscriptions:
            self.execution_subscriptions[execution_id] = set()
        self.execution_subscriptions[execution_id].add(websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific connection."""
        try:
            await websocket.send_json(message)
            if websocket in connection_metadata:
                connection_metadata[websocket]["message_count"] += 1
        except Exception as e:
            logging.error(f"Failed to send message: {e}")
    
    async def broadcast_to_execution(self, execution_id: str, message: dict):
        """Broadcast a message to all subscribers of an execution."""
        if execution_id not in self.execution_subscriptions:
            return
        
        disconnected = []
        for websocket in self.execution_subscriptions[execution_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logging.error(f"Failed to broadcast to execution {execution_id}: {e}")
                disconnected.append(websocket)
        
        # Clean up disconnected websockets
        for ws in disconnected:
            self.execution_subscriptions[execution_id].discard(ws)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast a message to all active connections."""
        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logging.error(f"Failed to broadcast: {e}")
                disconnected.append(websocket)
        
        # Clean up disconnected websockets
        for ws in disconnected:
            self.disconnect(ws)

# Global connection manager
manager = ConnectionManager()

@router.websocket("/api/workflows/ws/{workflow_id}")
async def websocket_execute_workflow(websocket: WebSocket, workflow_id: str):
    """
    WebSocket endpoint for real-time workflow execution monitoring.
    
    Clients can connect to this endpoint to receive live updates during
    workflow execution including:
    - Node start/completion events
    - Execution progress
    - Intermediate results
    - Error notifications
    """
    connection_id = str(uuid.uuid4())
    await manager.connect(websocket, connection_id)
    
    try:
        # Send welcome message
        await manager.send_personal_message({
            "type": "connected",
            "data": {
                "connection_id": connection_id,
                "workflow_id": workflow_id,
                "message": "Connected to workflow execution stream"
            }
        }, websocket)
        
        while True:
            # Wait for messages from client
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            
            if message_type == "execute":
                # Start workflow execution with real-time updates
                execution_id = str(uuid.uuid4())
                await manager.subscribe_to_execution(websocket, execution_id)
                
                # Start execution in background
                asyncio.create_task(
                    _execute_workflow_with_updates(
                        workflow_id,
                        execution_id,
                        data.get("input", {}),
                        manager
                    )
                )
                
                # Confirm execution started
                await manager.send_personal_message({
                    "type": "execution_started",
                    "data": {
                        "execution_id": execution_id,
                        "workflow_id": workflow_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }, websocket)
            
            elif message_type == "subscribe":
                # Subscribe to existing execution
                execution_id = data.get("execution_id")
                if execution_id:
                    await manager.subscribe_to_execution(websocket, execution_id)
                    await manager.send_personal_message({
                        "type": "subscribed",
                        "data": {
                            "execution_id": execution_id,
                            "message": "Subscribed to execution updates"
                        }
                    }, websocket)
            
            elif message_type == "ping":
                # Respond to ping with pong
                await manager.send_personal_message({
                    "type": "pong",
                    "data": {"timestamp": datetime.utcnow().isoformat()}
                }, websocket)
            
            elif message_type == "cancel":
                # Cancel execution (future implementation)
                execution_id = data.get("execution_id")
                await manager.send_personal_message({
                    "type": "cancel_acknowledged",
                    "data": {
                        "execution_id": execution_id,
                        "message": "Cancellation requested (not yet implemented)"
                    }
                }, websocket)
            
            else:
                # Unknown message type
                await manager.send_personal_message({
                    "type": "error",
                    "data": {
                        "message": f"Unknown message type: {message_type}",
                        "received_data": data
                    }
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        try:
            await manager.send_personal_message({
                "type": "error",
                "data": {"message": str(e)}
            }, websocket)
        except:
            pass
        manager.disconnect(websocket)

async def _execute_workflow_with_updates(
    workflow_id: str, 
    execution_id: str,
    input_data: Dict,
    manager: ConnectionManager
):
    """Execute workflow and send real-time updates via WebSocket."""
    from .workflow_routes import workflows_db, _compile_workflow_graph
    
    if workflow_id not in workflows_db:
        await manager.broadcast_to_execution(execution_id, {
            "type": "error",
            "data": {"message": "Workflow not found"}
        })
        return
    
    workflow = workflows_db[workflow_id]
    
    try:
        # Compile workflow
        execution_graph = _compile_workflow_graph(workflow["nodes"], workflow["edges"])
        
        # Send compilation success
        await manager.broadcast_to_execution(execution_id, {
            "type": "workflow_compiled",
            "data": {
                "execution_id": execution_id,
                "node_count": len(workflow["nodes"]),
                "edge_count": len(workflow["edges"]),
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
        # Execute with real-time updates
        await _execute_graph_with_updates(
            execution_graph, 
            input_data, 
            execution_id,
            manager
        )
        
        # Send completion event
        await manager.broadcast_to_execution(execution_id, {
            "type": "execution_completed",
            "data": {
                "execution_id": execution_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logging.error(f"Workflow execution failed: {e}")
        await manager.broadcast_to_execution(execution_id, {
            "type": "execution_failed",
            "data": {
                "execution_id": execution_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        })

async def _execute_graph_with_updates(
    execution_graph: Dict, 
    input_data: Dict, 
    execution_id: str,
    manager: ConnectionManager
):
    """Execute workflow graph with WebSocket updates."""
    current_state = {"input": input_data}
    current_node_id = execution_graph["start"]
    graph = execution_graph["graph"]
    
    max_iterations = 100
    iteration_count = 0
    
    while current_node_id and iteration_count < max_iterations:
        iteration_count += 1
        node_info = graph.get(current_node_id)
        
        if not node_info:
            break
        
        node_type = node_info["type"]
        node_data = node_info["data"]
        
        # Send node start event
        await manager.broadcast_to_execution(execution_id, {
            "type": "node_start",
            "data": {
                "node_id": current_node_id,
                "node_type": node_type,
                "node_label": node_data.get("label", ""),
                "iteration": iteration_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
        try:
            # Execute node based on type
            if node_type == "agent":
                result = await _execute_agent_with_updates(
                    node_data, 
                    current_state, 
                    execution_id,
                    manager
                )
            elif node_type == "conditional":
                result = await _execute_conditional_with_updates(
                    node_data, 
                    current_state, 
                    execution_id,
                    manager
                )
            elif node_type == "end":
                result = current_state
                current_node_id = None
            else:
                result = current_state
            
            # Update state
            if isinstance(result, dict):
                current_state.update(result)
            
            # Send node completion event
            await manager.broadcast_to_execution(execution_id, {
                "type": "node_complete",
                "data": {
                    "node_id": current_node_id,
                    "node_type": node_type,
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
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
                        current_node_id = next_nodes[0]["target"] if next_nodes else None
                    
                    # Send branch taken event
                    await manager.broadcast_to_execution(execution_id, {
                        "type": "branch_taken",
                        "data": {
                            "condition_result": condition_result,
                            "next_node": current_node_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    })
                else:
                    current_node_id = next_nodes[0]["target"] if next_nodes else None
            
            # Small delay for demo purposes
            await asyncio.sleep(0.5)
        
        except Exception as e:
            # Send node error event
            await manager.broadcast_to_execution(execution_id, {
                "type": "node_error",
                "data": {
                    "node_id": current_node_id,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            raise
    
    # Send final state
    await manager.broadcast_to_execution(execution_id, {
        "type": "final_state",
        "data": {
            "state": current_state,
            "total_iterations": iteration_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    })

async def _execute_agent_with_updates(
    node_data: Dict, 
    state: Dict,
    execution_id: str,
    manager: ConnectionManager
) -> Dict:
    """Execute agent node with WebSocket updates."""
    agent_name = node_data.get("agent_name", "demo_agent")
    
    # Send processing update
    await manager.broadcast_to_execution(execution_id, {
        "type": "agent_processing",
        "data": {
            "agent_name": agent_name,
            "message": f"Processing with agent: {agent_name}",
            "timestamp": datetime.utcnow().isoformat()
        }
    })
    
    # Simulate processing time
    await asyncio.sleep(1.0)
    
    # Mock agent response
    response = {
        "agent_result": f"Agent {agent_name} processed: {state.get('input', '')}",
        "cost": 0.001,
        "cached": False,
        "agent_name": agent_name
    }
    
    return response

async def _execute_conditional_with_updates(
    node_data: Dict, 
    state: Dict,
    execution_id: str,
    manager: ConnectionManager
) -> Dict:
    """Execute conditional node with WebSocket updates."""
    condition_type = node_data.get("condition_type", "expression")
    
    await manager.broadcast_to_execution(execution_id, {
        "type": "condition_evaluation",
        "data": {
            "condition_type": condition_type,
            "message": f"Evaluating {condition_type} condition",
            "timestamp": datetime.utcnow().isoformat()
        }
    })
    
    # Simple condition evaluation
    result = "true"  # Default for demo
    
    # Simulate evaluation time
    await asyncio.sleep(0.3)
    
    return {"branch": result}

@router.websocket("/api/playground/ws")
async def websocket_playground(websocket: WebSocket):
    """
    WebSocket endpoint for playground real-time interaction.
    Provides live feedback during query processing.
    """
    connection_id = str(uuid.uuid4())
    await manager.connect(websocket, connection_id)
    
    try:
        # Send welcome
        await manager.send_personal_message({
            "type": "connected",
            "data": {
                "connection_id": connection_id,
                "message": "Connected to FluxGraph playground"
            }
        }, websocket)
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "query":
                query = data.get("query", "")
                
                # Send typing indicator
                await manager.send_personal_message({
                    "type": "typing",
                    "data": {
                        "message": "FluxGraph is processing your query...",
                        "query": query
                    }
                }, websocket)
                
                # Simulate processing with progress updates
                await asyncio.sleep(0.3)
                await manager.send_personal_message({
                    "type": "progress",
                    "data": {
                        "stage": "checking_cache",
                        "message": "Checking semantic cache...",
                        "progress": 0.3
                    }
                }, websocket)
                
                await asyncio.sleep(0.3)
                await manager.send_personal_message({
                    "type": "progress",
                    "data": {
                        "stage": "processing",
                        "message": "Processing query...",
                        "progress": 0.6
                    }
                }, websocket)
                
                await asyncio.sleep(0.4)
                
                # Send final response
                response = f"Processed query: {query} (via WebSocket)"
                await manager.send_personal_message({
                    "type": "response",
                    "data": {
                        "response": response,
                        "cached": False,
                        "latency_ms": 1000,
                        "progress": 1.0
                    }
                }, websocket)
            
            elif data.get("type") == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "data": {"timestamp": datetime.utcnow().isoformat()}
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"Playground WebSocket error: {e}")
        manager.disconnect(websocket)

@router.websocket("/api/monitor")
async def websocket_system_monitor(websocket: WebSocket):
    """
    WebSocket endpoint for system monitoring and metrics.
    Broadcasts system stats in real-time.
    """
    connection_id = str(uuid.uuid4())
    await manager.connect(websocket, connection_id)
    
    try:
        await manager.send_personal_message({
            "type": "connected",
            "data": {
                "connection_id": connection_id,
                "message": "Connected to system monitor"
            }
        }, websocket)
        
        # Send periodic updates
        while True:
            # Send system stats every 5 seconds
            stats = {
                "active_connections": len(manager.active_connections),
                "active_executions": len(manager.execution_subscriptions),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await manager.send_personal_message({
                "type": "system_stats",
                "data": stats
            }, websocket)
            
            await asyncio.sleep(5)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"Monitor WebSocket error: {e}")
        manager.disconnect(websocket)
