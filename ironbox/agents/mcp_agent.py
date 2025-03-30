"""
MCP agent for IronBox.
"""
import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union

from sqlalchemy.ext.asyncio import AsyncSession
from langchain.schema import AIMessage

from ironbox.core.graph import AgentState, AgentType
from ironbox.core.llm import default_llm
from ironbox.mcp.client import MCPClient

# Configure logging
logger = logging.getLogger(__name__)


class MCPAgent:
    """Agent for interacting with MCP servers."""
    
    def __init__(self, llm=default_llm, mcp_client: Optional[MCPClient] = None):
        """
        Initialize MCPAgent.
        
        Args:
            llm: LLM instance
            mcp_client: MCP client
        """
        self.llm = llm
        self.mcp_client = mcp_client
        self.system_prompt = """
        You are an MCP (Model Context Protocol) agent for the IronBox system. Your job is to interact with MCP servers to access external tools and resources.
        
        You can:
        - List available MCP servers
        - List tools and resources provided by MCP servers
        - Use MCP tools
        - Access MCP resources
        
        Extract the user's request and use the appropriate MCP functionality.
        """
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Process an MCP request.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            # Check if we have an MCP client
            if not self.mcp_client:
                state.error = "MCP client not available"
                state.agent_outputs[AgentType.MCP] = {
                    "response": "I'm unable to access MCP servers at the moment.",
                    "next": None,
                    "error": state.error,
                }
                return state
            
            # Determine the type of MCP request
            request_type, server_name, details = await self._analyze_request(state.input)
            
            if request_type == "list_servers":
                # List available MCP servers
                response = await self._list_servers()
            elif request_type == "list_tools" and server_name:
                # List tools for a specific server
                response = await self._list_tools(server_name)
            elif request_type == "list_resources" and server_name:
                # List resources for a specific server
                response = await self._list_resources(server_name)
            elif request_type == "use_tool" and server_name and details:
                # Use a tool
                response = await self._use_tool(server_name, details.get("tool_name"), details.get("arguments", {}))
            elif request_type == "access_resource" and server_name and details:
                # Access a resource
                response = await self._access_resource(server_name, details.get("uri"))
            else:
                # Unknown request
                response = "I'm not sure what MCP operation you're looking for. You can ask to list servers, list tools or resources for a server, use a tool, or access a resource."
            
            state.agent_outputs[AgentType.MCP] = {
                "response": response,
                "next": None,
            }
            
            return state
        except Exception as e:
            logger.error(f"Error in MCP agent: {e}")
            state.error = f"MCP agent error: {str(e)}"
            state.agent_outputs[AgentType.MCP] = {
                "response": "An error occurred while interacting with MCP servers.",
                "next": None,
                "error": state.error,
            }
            return state
    
    async def _analyze_request(self, user_input: str) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """
        Analyze the user's request to determine the type, server name, and details.
        
        Args:
            user_input: User input text
            
        Returns:
            Tuple of (request_type, server_name, details)
        """
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": """
            Analyze the user's request for MCP operations. Determine:
            1. The type of request: "list_servers", "list_tools", "list_resources", "use_tool", or "access_resource"
            2. The server name (if applicable)
            3. Additional details:
               - For "use_tool": tool_name and arguments
               - For "access_resource": uri
            
            Return a JSON object with the following fields:
            - request_type: "list_servers", "list_tools", "list_resources", "use_tool", or "access_resource"
            - server_name: The name of the server (or null if not applicable)
            - details: Additional details (or null if not applicable)
              - For "use_tool": { "tool_name": "...", "arguments": { ... } }
              - For "access_resource": { "uri": "..." }
            """},
            {"role": "user", "content": user_input},
        ]
        
        # Get response from LLM
        response = await self.llm.ainvoke(messages)
        
        # Extract response text
        if hasattr(response, 'generations'):
            # ChatResult object
            response_text = response.generations[0].message.content
        elif isinstance(response, AIMessage):
            # AIMessage object
            response_text = response.content
        else:
            # Fallback
            response_text = str(response)
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return data.get("request_type", "unknown"), data.get("server_name"), data.get("details")
            
            # If no JSON block, try to parse the entire response
            data = json.loads(response_text)
            return data.get("request_type", "unknown"), data.get("server_name"), data.get("details")
        except Exception as e:
            logger.error(f"Error analyzing request: {e}")
            
            # Fallback to simple text analysis
            if "list" in user_input.lower() and "server" in user_input.lower():
                return "list_servers", None, None
            
            # Extract server name if present
            server_match = re.search(r'server[:\s]+(["\']?)([^"\']+)\1', user_input, re.IGNORECASE)
            server_name = server_match.group(2) if server_match else None
            
            if "list" in user_input.lower() and "tool" in user_input.lower():
                return "list_tools", server_name, None
            elif "list" in user_input.lower() and "resource" in user_input.lower():
                return "list_resources", server_name, None
            
            # Extract tool name if present
            tool_match = re.search(r'tool[:\s]+(["\']?)([^"\']+)\1', user_input, re.IGNORECASE)
            tool_name = tool_match.group(2) if tool_match else None
            
            if tool_name and server_name:
                return "use_tool", server_name, {"tool_name": tool_name, "arguments": {}}
            
            # Extract URI if present
            uri_match = re.search(r'uri[:\s]+(["\']?)([^"\']+)\1', user_input, re.IGNORECASE)
            uri = uri_match.group(2) if uri_match else None
            
            if uri and server_name:
                return "access_resource", server_name, {"uri": uri}
            
            return "unknown", None, None
    
    async def _list_servers(self) -> str:
        """
        List available MCP servers.
        
        Returns:
            Response text
        """
        servers = await self.mcp_client.list_servers()
        
        if not servers:
            return "No MCP servers are currently available."
        
        response = "# Available MCP Servers\n\n"
        for server in servers:
            response += f"## {server['name']}\n\n"
            response += f"**Version:** {server.get('version', 'Unknown')}\n"
            
            if server.get('description'):
                response += f"**Description:** {server['description']}\n"
            
            capabilities = server.get('capabilities', {})
            if capabilities.get('tools'):
                response += f"**Tools:** {len(capabilities['tools'])}\n"
            
            if capabilities.get('resources'):
                response += f"**Resources:** {len(capabilities['resources'])}\n"
            
            response += "\n"
        
        response += "You can list tools or resources for a specific server by name."
        
        return response
    
    async def _list_tools(self, server_name: str) -> str:
        """
        List tools for a specific server.
        
        Args:
            server_name: Server name
            
        Returns:
            Response text
        """
        try:
            tools = await self.mcp_client.list_tools(server_name)
            
            if not tools:
                return f"The server '{server_name}' doesn't provide any tools."
            
            response = f"# Tools for {server_name}\n\n"
            for tool in tools:
                response += f"## {tool['name']}\n\n"
                
                if tool.get('description'):
                    response += f"{tool['description']}\n\n"
                
                if tool.get('inputSchema'):
                    response += "**Input Parameters:**\n\n"
                    
                    properties = tool['inputSchema'].get('properties', {})
                    required = tool['inputSchema'].get('required', [])
                    
                    for param_name, param_info in properties.items():
                        is_required = "✓" if param_name in required else "○"
                        response += f"- `{param_name}` {is_required}: {param_info.get('description', 'No description')}\n"
                
                response += "\n"
            
            response += "You can use a tool by specifying the server name, tool name, and arguments."
            
            return response
        except Exception as e:
            return f"Error listing tools for server '{server_name}': {str(e)}"
    
    async def _list_resources(self, server_name: str) -> str:
        """
        List resources for a specific server.
        
        Args:
            server_name: Server name
            
        Returns:
            Response text
        """
        try:
            resources = await self.mcp_client.list_resources(server_name)
            resource_templates = await self.mcp_client.list_resource_templates(server_name)
            
            if not resources and not resource_templates:
                return f"The server '{server_name}' doesn't provide any resources."
            
            response = f"# Resources for {server_name}\n\n"
            
            if resources:
                response += "## Static Resources\n\n"
                for resource in resources:
                    response += f"### {resource.get('name', 'Unnamed Resource')}\n\n"
                    response += f"**URI:** `{resource['uri']}`\n"
                    
                    if resource.get('description'):
                        response += f"**Description:** {resource['description']}\n"
                    
                    if resource.get('mimeType'):
                        response += f"**MIME Type:** {resource['mimeType']}\n"
                    
                    response += "\n"
            
            if resource_templates:
                response += "## Resource Templates\n\n"
                for template in resource_templates:
                    response += f"### {template.get('name', 'Unnamed Template')}\n\n"
                    response += f"**URI Template:** `{template['uriTemplate']}`\n"
                    
                    if template.get('description'):
                        response += f"**Description:** {template['description']}\n"
                    
                    if template.get('mimeType'):
                        response += f"**MIME Type:** {template['mimeType']}\n"
                    
                    response += "\n"
            
            response += "You can access a resource by specifying the server name and URI."
            
            return response
        except Exception as e:
            return f"Error listing resources for server '{server_name}': {str(e)}"
    
    async def _use_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Use a tool.
        
        Args:
            server_name: Server name
            tool_name: Tool name
            arguments: Tool arguments
            
        Returns:
            Response text
        """
        if not tool_name:
            return "Please specify a tool name."
        
        try:
            result = await self.mcp_client.use_tool(server_name, tool_name, arguments)
            
            response = f"# Result of {tool_name}\n\n"
            
            if isinstance(result, str):
                response += result
            elif isinstance(result, dict):
                response += "```json\n"
                response += json.dumps(result, indent=2)
                response += "\n```"
            else:
                response += str(result)
            
            return response
        except Exception as e:
            return f"Error using tool '{tool_name}' on server '{server_name}': {str(e)}"
    
    async def _access_resource(self, server_name: str, uri: str) -> str:
        """
        Access a resource.
        
        Args:
            server_name: Server name
            uri: Resource URI
            
        Returns:
            Response text
        """
        if not uri:
            return "Please specify a resource URI."
        
        try:
            resource = await self.mcp_client.access_resource(server_name, uri)
            
            response = f"# Resource: {uri}\n\n"
            
            if resource.get('mimeType'):
                response += f"**MIME Type:** {resource['mimeType']}\n\n"
            
            if resource.get('text'):
                if resource.get('mimeType') == 'application/json':
                    try:
                        json_data = json.loads(resource['text'])
                        response += "```json\n"
                        response += json.dumps(json_data, indent=2)
                        response += "\n```"
                    except:
                        response += resource['text']
                else:
                    response += resource['text']
            
            return response
        except Exception as e:
            return f"Error accessing resource '{uri}' on server '{server_name}': {str(e)}"
