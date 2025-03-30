"""
Core module for IronBox agent system.

This module provides the main entry point for processing user queries.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from ironbox.core.llm import default_llm
from ironbox.core.graph import AgentState, AgentType
from ironbox.core.agent_framework import (
    AgentFramework, 
    RouteAgentFramework, 
    ReactAgentFramework, 
    PlanAgentFramework, 
    FrameworkSelector
)
from ironbox.mcp.client import default_mcp_client

# Configure logging
logger = logging.getLogger(__name__)


class AgentCore:
    """
    Core agent system that decides which framework to use based on the query type.
    """
    
    def __init__(self, llm=default_llm):
        """
        Initialize AgentCore.
        
        Args:
            llm: LLM instance
        """
        self.llm = llm
        self.framework_selector = FrameworkSelector(llm=llm)
        self.frameworks = {}
        self.tools = {}  # Repository of both local and MCP tools
        self.agents = {}
        self.mcp_tools_initialized = False
    
    def register_framework(self, framework_type: str, framework: AgentFramework):
        """
        Register a framework.
        
        Args:
            framework_type: Framework type
            framework: Framework instance
        """
        self.frameworks[framework_type] = framework
        logger.debug(f"Registered framework: {framework_type}")
    
    def register_tool(self, tool_name: str, tool_func: Callable):
        """
        Register a tool.
        
        Args:
            tool_name: Tool name
            tool_func: Tool function
        """
        self.tools[tool_name] = tool_func
        logger.debug(f"Registered tool: {tool_name}")
    
    def register_agent(self, agent_type: str, agent: Callable):
        """
        Register an agent.
        
        Args:
            agent_type: Agent type
            agent: Agent function
        """
        self.agents[agent_type] = agent
        logger.debug(f"Registered agent: {agent_type}")
    
    def setup_route_framework(self):
        """
        Set up the route framework with registered agents.
        """
        if not self.agents:
            logger.warning("No agents registered for route framework")
            return
        
        # Create route framework with registered agents
        route_framework = RouteAgentFramework(llm=self.llm, agents=self.agents)
        self.register_framework("route", route_framework)
        logger.debug("Set up route framework with agents: %s", list(self.agents.keys()))
    
    def setup_react_framework(self):
        """
        Set up the react framework with registered tools.
        """
        if not self.tools:
            logger.warning("No tools registered for react framework")
            return
        
        # Create react framework with registered tools
        react_framework = ReactAgentFramework(llm=self.llm, tools=self.tools)
        self.register_framework("react", react_framework)
        logger.debug("Set up react framework with tools: %s", list(self.tools.keys()))
    
    def setup_plan_framework(self):
        """
        Set up the plan framework with registered tools.
        """
        if not self.tools:
            logger.warning("No tools registered for plan framework")
            return
        
        # Create plan framework with registered tools
        plan_framework = PlanAgentFramework(llm=self.llm, tools=self.tools)
        self.register_framework("plan", plan_framework)
        logger.debug("Set up plan framework with tools: %s", list(self.tools.keys()))
    
    async def process_query(self, query: str, session_id: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a user query.
        
        Args:
            query: User query
            session_id: Optional session ID
            chat_history: Optional chat history
            
        Returns:
            Processing result
        """
        try:
            logger.debug("Processing query: %s", query)
            
            # Select framework
            framework_type = await self.framework_selector.select_framework(query)
            logger.debug("Selected framework: %s", framework_type)
            
            # Create initial state
            state = AgentState(
                input=query,
                session_id=session_id,
                chat_history=chat_history or [],
            )
            
            # Process query with selected framework
            if framework_type == "direct":
                # Direct LLM response without using any framework
                state = await self._process_direct(state)
            elif framework_type in self.frameworks:
                # Process with selected framework
                framework = self.frameworks[framework_type]
                state = await framework.process(state)
            else:
                # Fallback to route framework
                logger.warning(f"Framework {framework_type} not found, falling back to route")
                if "route" in self.frameworks:
                    framework = self.frameworks["route"]
                    state = await framework.process(state)
                else:
                    # Direct LLM response as last resort
                    state = await self._process_direct(state)
            
            # Extract response
            response = self._extract_response(state)
            
            return {
                "response": response,
                "framework": framework_type,
                "state": state,
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"Error processing query: {str(e)}",
                "error": str(e),
            }
    
    async def _process_direct(self, state: AgentState) -> AgentState:
        """
        Process a query directly with LLM.
        
        Args:
            state: Agent state
            
        Returns:
            Updated state
        """
        try:
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": """
                You are an AI assistant for the IronBox Kubernetes management platform.
                
                Provide helpful, accurate, and concise responses to user queries.
                """},
                {"role": "user", "content": state.input},
            ]
            
            # Add chat history for context
            for message in state.chat_history[-5:]:  # Last 5 messages
                messages.append(message)
            
            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            
            # Extract response text
            from langchain.schema import AIMessage
            if hasattr(response, 'generations'):
                response_text = response.generations[0].message.content
            elif isinstance(response, AIMessage):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Update state
            state.current_agent = "direct"
            state.agent_outputs["direct"] = {
                "response": response_text,
                "next": None,
            }
            
            return state
        except Exception as e:
            logger.error(f"Error in direct processing: {e}")
            state.error = f"Direct processing error: {str(e)}"
            return state
    
    def _extract_response(self, state: AgentState) -> str:
        """
        Extract response from agent state.
        
        Args:
            state: Agent state
            
        Returns:
            Response text
        """
        # Check if we have a response in state
        if state.response:
            return state.response
        
        # Check if we have agent outputs
        if state.agent_outputs:
            # Try to get the current agent
            if state.current_agent and state.current_agent in state.agent_outputs:
                agent_output = state.agent_outputs[state.current_agent]
                if isinstance(agent_output, dict) and "response" in agent_output:
                    return agent_output["response"]
            
            # Try to find any agent output with a response
            for agent_name, agent_output in state.agent_outputs.items():
                if isinstance(agent_output, dict) and "response" in agent_output:
                    return agent_output["response"]
        
        # Fallback
        return "I couldn't process your request."


    async def initialize_tools(self):
        """Initialize all tools (both local and MCP)."""
        # Register local tools
        self.register_local_tools()
        
        # Register MCP tools
        await self.register_mcp_tools()
        
        # Set up frameworks with all registered tools
        self.setup_react_framework()
        self.setup_plan_framework()
        
        logger.debug("Initialized all tools and frameworks")
    
    def register_local_tools(self):
        """Register local tools with the agent core."""
        # This method can be overridden or extended to register specific local tools
        logger.debug("Registered local tools")
    
    async def register_mcp_tools(self):
        """Register MCP tools with the agent core."""
        if self.mcp_tools_initialized:
            logger.debug("MCP tools already initialized")
            return
        
        try:
            # Initialize MCP client
            await default_mcp_client.initialize()
            
            # Get list of servers
            servers = await default_mcp_client.list_servers()
            
            for server in servers:
                server_name = server.get("name")
                if not server_name:
                    continue
                
                # Get tools for this server
                tools = await default_mcp_client.list_tools(server_name)
                
                for tool in tools:
                    tool_name = tool.get("name")
                    if not tool_name:
                        continue
                    
                    # Create a wrapper function for this tool
                    tool_wrapper = self._create_mcp_tool_wrapper(server_name, tool_name, tool.get("description"))
                    
                    # Register the wrapper
                    registered_name = f"mcp_{server_name}_{tool_name}"
                    self.register_tool(registered_name, tool_wrapper)
            
            self.mcp_tools_initialized = True
            logger.debug(f"Registered MCP tools from {len(servers)} servers")
        except Exception as e:
            logger.error(f"Error registering MCP tools: {e}")
    
    def _create_mcp_tool_wrapper(self, server_name: str, tool_name: str, description: Optional[str] = None):
        """
        Create a wrapper function for an MCP tool.
        
        Args:
            server_name: Server name
            tool_name: Tool name
            description: Optional tool description
            
        Returns:
            Wrapper function
        """
        async def mcp_tool_wrapper(**kwargs):
            """MCP tool wrapper."""
            try:
                return await default_mcp_client.use_tool(
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=kwargs
                )
            except Exception as e:
                return f"Error using MCP tool {tool_name} on server {server_name}: {str(e)}"
        
        # Set function name and docstring
        mcp_tool_wrapper.__name__ = f"mcp_{server_name}_{tool_name}"
        mcp_tool_wrapper.__doc__ = description or f"MCP tool {tool_name} from server {server_name}"
        
        return mcp_tool_wrapper


# Create default agent core
default_agent_core = AgentCore()

# Initialize tools asynchronously
async def initialize_default_agent_core():
    """Initialize the default agent core."""
    await default_agent_core.initialize_tools()

# Run initialization in background if needed
# This can be called from the API server startup
