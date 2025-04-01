"""
Core module for IronBox agent system.

This module provides the main entry point for processing user queries.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from ironbox.core.llm import default_llm
from ironbox.core.graph import AgentState, AgentType
from ironbox.core.framework_selector import FrameworkSelector
from ironbox.core.langchain_frameworks import (
    BaseLCAgentFramework,
    LCRouteAgentFramework,
    LCReactAgentFramework,
    LCPlanAgentFramework,
    FrameworkRegistry
)
from ironbox.core.toolkit import Toolkit
from ironbox.mcp.client import default_mcp_client
from ironbox.config import load_config

# Configure logging
logger = logging.getLogger(__name__)


class AgentCore:
    """
    Core agent system that decides which framework to use based on the query type.
    """
    
    def __init__(self, config=None, llm=None):
        """
        Initialize AgentCore.
        
        Args:
            config: Optional configuration dictionary
            llm: Optional LLM instance
        """
        self.config = config or load_config()
        self.llm = llm or default_llm
        self.toolkit = Toolkit(config=self.config)  # Unified toolkit for tools and agents
        self.mcp_tools_initialized = False
        
        # Create framework registry
        self.framework_registry = FrameworkRegistry(config=self.config)
        
        # Create framework selector
        self.framework_selector = FrameworkSelector(llm=self.llm, config=self.config, toolkit=self.toolkit)
    
    def register_framework(self, framework_type: str, framework: BaseLCAgentFramework):
        """
        Register a framework.
        
        Args:
            framework_type: Framework type
            framework: Framework instance
        """
        self.framework_registry.register_framework_type(framework_type, framework.__class__)
        logger.debug(f"Registered framework type: {framework_type}")
    
    def register_tool(self, tool_name: str, tool_func: Callable, tool_type: str = "local"):
        """
        Register a tool in the toolkit.
        
        Args:
            tool_name: Tool name
            tool_func: Tool function
            tool_type: Tool type (local, mcp, agent)
        """
        self.toolkit.register_tool(tool_name, tool_func, tool_type)
    
    def register_agent(self, agent_type: str, agent: Callable):
        """
        Register an agent in the toolkit.
        
        Args:
            agent_type: Agent type
            agent: Agent function
        """
        self.toolkit.register_agent(agent_type, agent)
    
    def setup_route_framework(self):
        """
        Set up the route framework with registered agents.
        """
        if not self.toolkit.agents:
            logger.warning("No agents registered for route framework")
            return
        
        # Create route framework with registered agents
        route_framework = LCRouteAgentFramework(llm=self.llm, agents=self.toolkit.agents)
        self.register_framework("route", route_framework)
        logger.debug("Set up route framework with agents: %s", list(self.toolkit.agents.keys()))
    
    def setup_react_framework(self):
        """
        Set up the react framework with the unified toolkit.
        """
        if not self.toolkit.tools:
            logger.warning("No tools registered for react framework")
            return
        
        # Create react framework with all tools (including agent-wrapped tools)
        react_framework = LCReactAgentFramework(llm=self.llm, tools=self.toolkit.tools)
        self.register_framework("react", react_framework)
        logger.debug("Set up react framework with tools: %s", list(self.toolkit.tools.keys()))
    
    def setup_plan_framework(self):
        """
        Set up the plan framework with the unified toolkit.
        """
        if not self.toolkit.tools:
            logger.warning("No tools registered for plan framework")
            return
        
        # Create plan framework with all tools (including agent-wrapped tools)
        plan_framework = LCPlanAgentFramework(llm=self.llm, tools=self.toolkit.tools)
        self.register_framework("plan", plan_framework)
        logger.debug("Set up plan framework with tools: %s", list(self.toolkit.tools.keys()))
    
    async def process_query(self, query: str, session_id: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a user query using LangChain and LangGraph.
        
        Args:
            query: User query
            session_id: Optional session ID
            chat_history: Optional chat history
            
        Returns:
            Processing result
        """
        try:
            logger.debug("Processing query: %s", query)
            
            # Create initial state with session_id and chat_history
            state = AgentState(
                input=query,
                session_id=session_id,
                chat_history=chat_history or [],
            )
            
            # Process the query using the framework registry and LangGraph
            result = await self.framework_registry.process_query(query, session_id)
            
            return {
                "response": result.get("response", "I couldn't process your request."),
                "agent_outputs": result.get("agent_outputs", {}),
                "error": result.get("error")
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

    async def initialize(self):
        """Initialize the agent core."""
        try:
            # Initialize the toolkit
            self.toolkit.initialize()
            
            # Register MCP tools
            await self.register_mcp_tools()
            
            # First, register the framework selector with the orchestrator
            # This is the entry point for the graph
            self.framework_registry.orchestrator.register_framework(
                "framework_selector", 
                self.framework_selector
            )
            
            # For backward compatibility, register the LangChain frameworks with the old framework system
            # but avoid double registration in the orchestrator
            from ironbox.core.langchain_frameworks import (
                LCRouteAgentFramework,
                LCReactAgentFramework,
                LCPlanAgentFramework
            )
            
            # Create LangChain frameworks but don't register them with the orchestrator yet
            frameworks_config = self.config.get("agent_frameworks", [])
            
            # Log the current state of frameworks
            logger.debug("Current frameworks in registry: %s", list(self.framework_registry.frameworks.keys()))
            logger.debug("Current frameworks in orchestrator: %s", list(self.framework_registry.orchestrator.frameworks.keys()))
            
            # Only register these frameworks if they're not already loaded from config
            if not self.framework_registry.frameworks:
                logger.debug("No frameworks in registry, creating standard frameworks")
                
                # Create and register standard frameworks with consistent names
                # The key is to use consistent names that match what the framework_selector returns
                
                # Create and register route framework
                if len(frameworks_config) > 0:
                    lc_route_framework = LCRouteAgentFramework(
                        llm=self.llm,
                        agents=self.toolkit.agents,
                        config=frameworks_config[0].get("config", {})
                    )
                    # Use "route" as the framework name to match what framework_selector returns
                    self.framework_registry.frameworks["route"] = lc_route_framework
                    self.framework_registry.orchestrator.register_framework("route", lc_route_framework)
                    logger.debug("Registered route framework")
                
                # Create and register react framework
                if len(frameworks_config) > 1:
                    lc_react_framework = LCReactAgentFramework(
                        llm=self.llm,
                        tools=self.toolkit.tools,
                        config=frameworks_config[1].get("config", {})
                    )
                    # Use "react" as the framework name to match what framework_selector returns
                    self.framework_registry.frameworks["react"] = lc_react_framework
                    self.framework_registry.orchestrator.register_framework("react", lc_react_framework)
                    logger.debug("Registered react framework")
                
                # Create and register plan framework
                if len(frameworks_config) > 2:
                    lc_plan_framework = LCPlanAgentFramework(
                        llm=self.llm,
                        tools=self.toolkit.tools,
                        config=frameworks_config[2].get("config", {})
                    )
                    # Use "plan" as the framework name to match what framework_selector returns
                    self.framework_registry.frameworks["plan"] = lc_plan_framework
                    self.framework_registry.orchestrator.register_framework("plan", lc_plan_framework)
                    logger.debug("Registered plan framework")
                
                # Log the updated state of frameworks
                logger.debug("Updated frameworks in registry: %s", list(self.framework_registry.frameworks.keys()))
                logger.debug("Updated frameworks in orchestrator: %s", list(self.framework_registry.orchestrator.frameworks.keys()))
            
            # Now build the orchestrator graph with all the frameworks
            try:
                self.framework_registry.orchestrator.build_graph()
                logger.debug("Built orchestrator graph with frameworks: %s", 
                            list(self.framework_registry.orchestrator.frameworks.keys()))
            except Exception as e:
                logger.error(f"Error building graph: {e}")
                raise
            
            logger.debug("Agent core initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agent core: {e}")
            raise
    
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
                    self.register_tool(registered_name, tool_wrapper, "mcp")
            
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

# Initialize agent core asynchronously
async def initialize_default_agent_core():
    """Initialize the default agent core."""
    await default_agent_core.initialize()

# Run initialization in background if needed
# This can be called from the API server startup
