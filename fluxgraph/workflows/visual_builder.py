import json
import uuid
from typing import Dict, List

class WorkflowNode:
    """Workflow node."""
    def __init__(self, node_id: str, node_type: str, config: Dict):
        self.id = node_id
        self.type = node_type
        self.config = config
        self.next_nodes = []
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'config': self.config,
            'next_nodes': self.next_nodes
        }

class VisualWorkflow:
    """Visual workflow builder."""
    
    def __init__(self, workflow_id: str = None):
        self.id = workflow_id or str(uuid.uuid4())
        self.nodes = {}
        self.start_node = None
    
    def add_node(self, node_type: str, config: Dict) -> str:
        """Add node to workflow."""
        node_id = str(uuid.uuid4())
        node = WorkflowNode(node_id, node_type, config)
        self.nodes[node_id] = node
        
        if not self.start_node:
            self.start_node = node_id
        
        return node_id
    
    def connect_nodes(self, from_node: str, to_node: str):
        """Connect two nodes."""
        if from_node in self.nodes:
            self.nodes[from_node].next_nodes.append(to_node)
    
    def to_json(self) -> str:
        """Export to JSON."""
        return json.dumps({
            'id': self.id,
            'start_node': self.start_node,
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()}
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str):
        """Import from JSON."""
        data = json.loads(json_str)
        workflow = cls(data['id'])
        workflow.start_node = data['start_node']
        
        for node_id, node_data in data['nodes'].items():
            node = WorkflowNode(node_id, node_data['type'], node_data['config'])
            node.next_nodes = node_data['next_nodes']
            workflow.nodes[node_id] = node
        
        return workflow
    
    async def execute(self, initial_data: Dict, app):
        """Execute workflow."""
        current_node_id = self.start_node
        context = initial_data.copy()
        
        while current_node_id:
            node = self.nodes[current_node_id]
            
            if node.type == 'agent':
                result = await app.orchestrator.run(node.config['agent_name'], context)
                context.update(result)
            
            current_node_id = node.next_nodes[0] if node.next_nodes else None
        
        return context
