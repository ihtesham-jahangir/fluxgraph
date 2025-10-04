# fluxgraph/protocols/mcp.py
"""
Model Context Protocol (MCP) Support for FluxGraph.
Implements Anthropic's MCP standard for tool/data integration.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MCPResourceType(Enum):
    """MCP resource types."""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    MEMORY = "memory"
    CUSTOM = "custom"


class MCPCapability(Enum):
    """MCP capabilities."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    STREAM = "stream"


class MCPResource:
    """Represents an MCP-compatible resource."""
    
    def __init__(
        self,
        resource_id: str,
        resource_type: MCPResourceType,
        name: str,
        description: str,
        capabilities: List[MCPCapability],
        schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.schema = schema or {}
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
    
    def to_mcp_format(self) -> Dict[str, Any]:
        """Convert to MCP standard format."""
        return {
            "id": self.resource_id,
            "type": self.resource_type.value,
            "name": self.name,
            "description": self.description,
            "capabilities": [c.value for c in self.capabilities],
            "schema": self.schema,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class MCPTool:
    """MCP-compatible tool definition."""
    
    def __init__(
        self,
        tool_id: str,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable,
        output_schema: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None
    ):
        self.tool_id = tool_id
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema or {}
        self.handler = handler
        self.examples = examples or []
    
    def to_mcp_format(self) -> Dict[str, Any]:
        """Convert to MCP tool format (OpenAI-compatible)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
                "returns": self.output_schema
            },
            "examples": self.examples
        }


class MCPServer:
    """
    MCP Server implementation for FluxGraph.
    Exposes tools and resources via Model Context Protocol.
    """
    
    def __init__(self):
        self.resources: Dict[str, MCPResource] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.protocol_version = "1.0"
        logger.info("MCPServer initialized (Protocol v1.0)")
    
    def register_resource(self, resource: MCPResource):
        """Register an MCP resource."""
        self.resources[resource.resource_id] = resource
        logger.info(f"[MCP] Registered resource: {resource.name} ({resource.resource_type.value})")
    
    def register_tool(self, tool: MCPTool):
        """Register an MCP tool."""
        self.tools[tool.tool_id] = tool
        logger.info(f"[MCP] Registered tool: {tool.name}")
    
    def register_fluxgraph_tool(
        self,
        tool_name: str,
        tool_func: Callable,
        description: str,
        input_schema: Dict[str, Any]
    ):
        """
        Register a FluxGraph tool as MCP-compatible tool.
        
        Args:
            tool_name: Name of the tool
            tool_func: Tool function
            description: Tool description
            input_schema: JSON schema for inputs
        """
        tool = MCPTool(
            tool_id=f"fluxgraph_{tool_name}",
            name=tool_name,
            description=description,
            input_schema=input_schema,
            handler=tool_func
        )
        self.register_tool(tool)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get MCP server capabilities."""
        return {
            "protocol_version": self.protocol_version,
            "server_info": {
                "name": "FluxGraph MCP Server",
                "version": "1.0.0"
            },
            "capabilities": {
                "tools": len(self.tools),
                "resources": len(self.resources),
                "streaming": True
            }
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all MCP tools."""
        return [tool.to_mcp_format() for tool in self.tools.values()]
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List all MCP resources."""
        return [resource.to_mcp_format() for resource in self.resources.values()]
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an MCP tool.
        
        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments
        
        Returns:
            Tool execution result
        """
        tool = None
        for t in self.tools.values():
            if t.name == tool_name:
                tool = t
                break
        
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        logger.info(f"[MCP] Executing tool: {tool_name}")
        
        try:
            import asyncio
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)
            
            return {
                "success": True,
                "result": result,
                "tool": tool_name
            }
        
        except Exception as e:
            logger.error(f"[MCP] Tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
    
    def read_resource(
        self,
        resource_id: str,
        query: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Read data from an MCP resource."""
        resource = self.resources.get(resource_id)
        if not resource:
            raise ValueError(f"Resource '{resource_id}' not found")
        
        if MCPCapability.READ not in resource.capabilities:
            raise PermissionError(f"Resource '{resource_id}' does not support READ")
        
        logger.info(f"[MCP] Reading resource: {resource.name}")
        
        # Implementation depends on resource type
        # This is a placeholder - actual implementation would delegate to resource handler
        return {
            "resource_id": resource_id,
            "data": {},
            "metadata": resource.metadata
        }
