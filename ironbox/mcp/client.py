"""
MCP client for IronBox.
"""
import logging
import json
import subprocess
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union

from ironbox.config import config

# Configure logging
logger = logging.getLogger(__name__)


class MCPClient:
    """Client for interacting with MCP servers."""
    
    def __init__(self, servers_config: Optional[Dict[str, Any]] = None):
        """
        Initialize MCPClient.
        
        Args:
            servers_config: Optional configuration for MCP servers
        """
        self.servers_config = servers_config or config["mcp"]["servers"]
        self.servers = {}
        self.enabled = config["mcp"]["enabled"]
    
    async def initialize(self):
        """Initialize MCP client."""
        if not self.enabled:
            logger.info("MCP is disabled")
            return
        
        for server_name, server_config in self.servers_config.items():
            if server_config.get("enabled", True):
                logger.info(f"Initializing MCP server: {server_name}")
                self.servers[server_name] = server_config
    
    async def list_servers(self) -> List[Dict[str, Any]]:
        """
        List available MCP servers.
        
        Returns:
            List of server information
        """
        if not self.enabled:
            return []
        
        result = []
        for server_name, server_config in self.servers.items():
            try:
                # Get server information
                info = await self._send_request(server_name, "server.info", {})
                result.append(info)
            except Exception as e:
                logger.error(f"Error getting server info for {server_name}: {e}")
                # Add basic information
                result.append({
                    "name": server_name,
                    "version": "unknown",
                    "capabilities": {},
                })
        
        return result
    
    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """
        List tools for a specific server.
        
        Args:
            server_name: Server name
            
        Returns:
            List of tool information
        """
        if not self.enabled:
            return []
        
        if server_name not in self.servers:
            raise ValueError(f"Unknown MCP server: {server_name}")
        
        try:
            result = await self._send_request(server_name, "tool.list", {})
            return result.get("tools", [])
        except Exception as e:
            logger.error(f"Error listing tools for {server_name}: {e}")
            return []
    
    async def list_resources(self, server_name: str) -> List[Dict[str, Any]]:
        """
        List resources for a specific server.
        
        Args:
            server_name: Server name
            
        Returns:
            List of resource information
        """
        if not self.enabled:
            return []
        
        if server_name not in self.servers:
            raise ValueError(f"Unknown MCP server: {server_name}")
        
        try:
            result = await self._send_request(server_name, "resource.list", {})
            return result.get("resources", [])
        except Exception as e:
            logger.error(f"Error listing resources for {server_name}: {e}")
            return []
    
    async def list_resource_templates(self, server_name: str) -> List[Dict[str, Any]]:
        """
        List resource templates for a specific server.
        
        Args:
            server_name: Server name
            
        Returns:
            List of resource template information
        """
        if not self.enabled:
            return []
        
        if server_name not in self.servers:
            raise ValueError(f"Unknown MCP server: {server_name}")
        
        try:
            result = await self._send_request(server_name, "resource.listTemplates", {})
            return result.get("resourceTemplates", [])
        except Exception as e:
            logger.error(f"Error listing resource templates for {server_name}: {e}")
            return []
    
    async def use_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Use a tool.
        
        Args:
            server_name: Server name
            tool_name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        if not self.enabled:
            raise ValueError("MCP is disabled")
        
        if server_name not in self.servers:
            raise ValueError(f"Unknown MCP server: {server_name}")
        
        try:
            result = await self._send_request(server_name, "tool.call", {
                "name": tool_name,
                "arguments": arguments,
            })
            
            # Process result
            if result.get("isError"):
                raise ValueError(result.get("content", [{"text": "Unknown error"}])[0].get("text"))
            
            # Extract content
            content = result.get("content", [])
            if len(content) == 1 and content[0].get("type") == "text":
                return content[0].get("text", "")
            
            return content
        except Exception as e:
            logger.error(f"Error using tool {tool_name} on {server_name}: {e}")
            raise
    
    async def access_resource(self, server_name: str, uri: str) -> Dict[str, Any]:
        """
        Access a resource.
        
        Args:
            server_name: Server name
            uri: Resource URI
            
        Returns:
            Resource content
        """
        if not self.enabled:
            raise ValueError("MCP is disabled")
        
        if server_name not in self.servers:
            raise ValueError(f"Unknown MCP server: {server_name}")
        
        try:
            result = await self._send_request(server_name, "resource.read", {
                "uri": uri,
            })
            
            # Process result
            if not result.get("contents"):
                raise ValueError("No content returned")
            
            # Return first content
            return result["contents"][0]
        except Exception as e:
            logger.error(f"Error accessing resource {uri} on {server_name}: {e}")
            raise
    
    async def _send_request(self, server_name: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request to an MCP server.
        
        Args:
            server_name: Server name
            method: Method name
            params: Method parameters
            
        Returns:
            Response from server
        """
        server_config = self.servers.get(server_name)
        if not server_config:
            raise ValueError(f"Unknown MCP server: {server_name}")
        
        # Create request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        
        # Get command and arguments
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})
        
        if not command:
            raise ValueError(f"No command specified for MCP server: {server_name}")
        
        # Create environment
        full_env = {**env}
        
        # Create process
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        
        # Send request
        request_json = json.dumps(request) + "\n"
        stdout, stderr = await process.communicate(request_json.encode())
        
        if stderr:
            logger.error(f"MCP server {server_name} stderr: {stderr.decode()}")
        
        if not stdout:
            raise ValueError(f"No response from MCP server: {server_name}")
        
        # Parse response
        try:
            response = json.loads(stdout.decode())
            
            if "error" in response:
                raise ValueError(f"MCP server error: {response['error']}")
            
            return response.get("result", {})
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response from MCP server: {server_name}")


# Create default MCP client
default_mcp_client = MCPClient()
