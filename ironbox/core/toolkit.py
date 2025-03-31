"""
Unified toolkit for managing tools and agents.

This module provides a central repository for all tools and agents in the system.
"""
import logging
import importlib
import inspect
import pkgutil
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from ironbox.core.graph import AgentState

# Configure logging
logger = logging.getLogger(__name__)


class Toolkit:
    """Unified toolkit for managing tools and agents."""
    
    def __init__(self, config=None):
        """
        Initialize the toolkit.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.tools = {}  # All tools (including agent-wrapped tools)
        self.agents = {}  # Original agent instances
        self.agent_tools = {}  # Agent instances wrapped as tools
        self.local_tools = {}  # Local tool functions
        self.mcp_tools = {}  # MCP tool functions
    
    def initialize(self):
        """Initialize the toolkit from configuration."""
        # Load tools and agents from configuration
        self._load_from_config()
        
        # Perform auto-discovery if enabled
        self._auto_discover()
        
        logger.debug(f"Toolkit initialized with {len(self.tools)} tools and {len(self.agents)} agents")
    
    def register_tool(self, tool_name: str, tool_func: Callable, tool_type: str = "local"):
        """
        Register a tool in the toolkit.
        
        Args:
            tool_name: Tool name
            tool_func: Tool function
            tool_type: Tool type (local, mcp, agent)
        """
        self.tools[tool_name] = tool_func
        
        # Also store in the appropriate category
        if tool_type == "local":
            self.local_tools[tool_name] = tool_func
        elif tool_type == "mcp":
            self.mcp_tools[tool_name] = tool_func
        elif tool_type == "agent":
            self.agent_tools[tool_name] = tool_func
        
        logger.debug(f"Registered {tool_type} tool: {tool_name}")
    
    def register_agent(self, agent_type: str, agent: Callable):
        """
        Register an agent in the toolkit.
        
        Args:
            agent_type: Agent type
            agent: Agent function
        """
        self.agents[agent_type] = agent
        
        # Also create and register an agent-as-tool wrapper
        agent_tool = self._create_agent_tool_wrapper(agent_type, agent)
        self.register_tool(f"agent_{agent_type}", agent_tool, tool_type="agent")
        
        logger.debug(f"Registered agent: {agent_type}")
    
    def _create_agent_tool_wrapper(self, agent_type: str, agent: Callable):
        """
        Create a wrapper that exposes an agent as a tool.
        
        Args:
            agent_type: Agent type
            agent: Agent function
            
        Returns:
            Wrapper function
        """
        async def agent_tool_wrapper(query: str, **kwargs):
            """Tool wrapper for agent."""
            try:
                # Create a minimal state for the agent
                state = AgentState(input=query)
                
                # Add any additional kwargs to the state
                for key, value in kwargs.items():
                    setattr(state, key, value)
                
                # Call the agent with the state
                agent_instance = agent()
                result_state = await agent_instance(state)
                
                # Extract and return the response
                if hasattr(result_state, 'agent_outputs') and agent_type in result_state.agent_outputs:
                    return result_state.agent_outputs[agent_type].get("response", "No response from agent")
                return "Agent execution completed but no response was generated"
            except Exception as e:
                logger.error(f"Error executing agent {agent_type} as tool: {e}")
                return f"Error executing agent {agent_type}: {str(e)}"
        
        # Set function name and docstring
        agent_tool_wrapper.__name__ = f"agent_{agent_type}"
        agent_tool_wrapper.__doc__ = f"Execute the {agent_type} agent with the given query"
        
        return agent_tool_wrapper
    
    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool function or None if not found
        """
        return self.tools.get(tool_name)
    
    def get_agent(self, agent_type: str) -> Optional[Callable]:
        """
        Get an agent by type.
        
        Args:
            agent_type: Agent type
            
        Returns:
            Agent function or None if not found
        """
        return self.agents.get(agent_type)
    
    def list_tools(self) -> Dict[str, Callable]:
        """
        Get all tools.
        
        Returns:
            Dictionary of tool names to tool functions
        """
        return self.tools
    
    def list_agents(self) -> Dict[str, Callable]:
        """
        Get all agents.
        
        Returns:
            Dictionary of agent types to agent functions
        """
        return self.agents
    
    def list_tool_descriptions(self) -> str:
        """
        Get descriptions of all tools.
        
        Returns:
            String containing tool descriptions
        """
        return "\n".join([
            f"- {name}: {tool.__doc__ or 'No description'}"
            for name, tool in self.tools.items()
        ])
    
    def _load_from_config(self):
        """Load tools and agents from configuration."""
        toolkit_config = self.config.get("toolkit", {})
        
        # Load tools from config
        for tool_config in toolkit_config.get("tools", []):
            if not tool_config.get("enabled", True):
                continue
                
            try:
                # Import the module and function
                module_name = tool_config.get("module")
                function_name = tool_config.get("function")
                
                if not module_name or not function_name:
                    logger.warning(f"Skipping tool with missing module or function: {tool_config}")
                    continue
                
                # Dynamically import the module and get the function
                module = importlib.import_module(module_name)
                tool_func = getattr(module, function_name)
                
                # Register the tool
                tool_name = tool_config.get("name", function_name)
                self.register_tool(tool_name, tool_func, "local")
                
                logger.debug(f"Loaded tool from config: {tool_name}")
            except Exception as e:
                logger.error(f"Error loading tool from config: {e}")
        
        # Load agents from config
        for agent_config in toolkit_config.get("agents", []):
            if not agent_config.get("enabled", True):
                continue
                
            try:
                # Import the class
                class_path = agent_config.get("class")
                
                if not class_path:
                    logger.warning(f"Skipping agent with missing class: {agent_config}")
                    continue
                
                # Split module and class name
                module_path, class_name = class_path.rsplit(".", 1)
                
                # Dynamically import the module and get the class
                module = importlib.import_module(module_path)
                agent_class = getattr(module, class_name)
                
                # Create a factory function for the agent
                agent_factory = lambda db_session=None, cls=agent_class: cls()
                
                # Register the agent
                agent_name = agent_config.get("name", class_name.lower().replace("agent", ""))
                self.register_agent(agent_name, agent_factory)
                
                logger.debug(f"Loaded agent from config: {agent_name}")
            except Exception as e:
                logger.error(f"Error loading agent from config: {e}")
    
    def _auto_discover(self):
        """Auto-discover tools and agents if enabled in config."""
        toolkit_config = self.config.get("toolkit", {})
        discovery_config = toolkit_config.get("discovery", {})
        
        # Auto-discover tools
        if discovery_config.get("tools", {}).get("enabled", False):
            tool_paths = discovery_config.get("tools", {}).get("paths", [])
            for path in tool_paths:
                self._discover_tools(path)
        
        # Auto-discover agents
        if discovery_config.get("agents", {}).get("enabled", False):
            agent_paths = discovery_config.get("agents", {}).get("paths", [])
            for path in agent_paths:
                self._discover_agents(path)
    
    def _discover_tools(self, package_path):
        """
        Discover tools in the specified package.
        
        Args:
            package_path: Package path
        """
        try:
            # Import the package
            package = importlib.import_module(package_path)
            
            # Get the package directory
            if hasattr(package, "__path__"):
                package_dir = package.__path__[0]
            else:
                logger.warning(f"Package {package_path} has no __path__ attribute")
                return
            
            # Iterate through all modules in the package
            for _, module_name, _ in pkgutil.iter_modules([package_dir]):
                try:
                    # Import the module
                    module = importlib.import_module(f"{package_path}.{module_name}")
                    
                    # Find all functions in the module
                    for name, obj in inspect.getmembers(module, inspect.isfunction):
                        # Check if it has a docstring (potential tool)
                        if obj.__doc__ and not name.startswith("_"):
                            # Register the function as a tool
                            self.register_tool(name, obj, "local")
                            logger.debug(f"Auto-discovered tool: {name}")
                except Exception as e:
                    logger.error(f"Error discovering tools in module {module_name}: {e}")
        except Exception as e:
            logger.error(f"Error discovering tools in package {package_path}: {e}")
    
    def _discover_agents(self, package_path):
        """
        Discover agents in the specified package.
        
        Args:
            package_path: Package path
        """
        try:
            # Import the package
            package = importlib.import_module(package_path)
            
            # Get the package directory
            if hasattr(package, "__path__"):
                package_dir = package.__path__[0]
            else:
                logger.warning(f"Package {package_path} has no __path__ attribute")
                return
            
            # Iterate through all modules in the package
            for _, module_name, _ in pkgutil.iter_modules([package_dir]):
                try:
                    # Import the module
                    module = importlib.import_module(f"{package_path}.{module_name}")
                    
                    # Find all classes in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Check if it's an agent class (has a __call__ method)
                        if hasattr(obj, "__call__") and name.endswith("Agent"):
                            # Register the agent
                            agent_type = name.lower().replace("agent", "")
                            self.register_agent(agent_type, lambda db_session=None, cls=obj: cls())
                            logger.debug(f"Auto-discovered agent: {agent_type}")
                except Exception as e:
                    logger.error(f"Error discovering agents in module {module_name}: {e}")
        except Exception as e:
            logger.error(f"Error discovering agents in package {package_path}: {e}")
