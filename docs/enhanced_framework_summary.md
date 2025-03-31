# Enhanced Core Framework Summary

This document provides a summary of the enhancements made to the IronBox core framework to enable React Agent, Plan Agent, and Route Agent to invoke specific agents and tools simultaneously.

## Overview

The enhanced architecture introduces a **Unified Toolkit** that serves as a central repository for all tools and agents in the system. This allows all frameworks to access both tools and agents through a consistent interface, providing greater flexibility and power.

## Key Components

### 1. Unified Toolkit (`ironbox/ironbox/core/toolkit.py`)

The Toolkit class serves as a central repository for all tools and agents:

```python
class Toolkit:
    """Unified toolkit for managing tools and agents."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.tools = {}  # All tools (including agent-wrapped tools)
        self.agents = {}  # Original agent instances
        self.agent_tools = {}  # Agent instances wrapped as tools
        self.local_tools = {}  # Local tool functions
        self.mcp_tools = {}  # MCP tool functions
```

Key features:
- Maintains separate collections for different types of tools and agents
- Provides methods for registering and retrieving tools and agents
- Supports configuration-based loading of tools and agents
- Implements auto-discovery of tools and agents

### 2. Enhanced Agent Core (`ironbox/ironbox/core/agent_core.py`)

The AgentCore class has been updated to use the Unified Toolkit:

```python
class AgentCore:
    def __init__(self, config=None, llm=None):
        self.config = config or load_config()
        self.llm = llm or default_llm
        self.framework_selector = FrameworkSelector(llm=self.llm)
        self.frameworks = {}
        self.toolkit = Toolkit(config=self.config)  # Unified toolkit instance
        self.mcp_tools_initialized = False
```

Key changes:
- Uses the Unified Toolkit for managing tools and agents
- Provides an `initialize()` method for setting up the toolkit and frameworks
- Sets up frameworks with the appropriate components from the toolkit
- Registers MCP tools with the toolkit

### 3. Agent-as-Tool Pattern

The Agent-as-Tool pattern allows specialized agents to be used as tools:

```python
def _create_agent_tool_wrapper(self, agent_type: str, agent: Callable):
    """Create a wrapper that exposes an agent as a tool."""
    async def agent_tool_wrapper(query: str, **kwargs):
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
```

This pattern allows React and Plan frameworks to use specialized agents just like regular tools.

### 4. Configuration-Based Setup

Tools and agents can be configured through YAML configuration files:

```yaml
# Toolkit settings
toolkit:
  # Tool definitions
  tools:
    - name: get_pod_count
      module: ironbox.tools.kubernetes
      function: get_pod_count
      description: Get the number of pods in a cluster
      enabled: true
  
  # Agent definitions
  agents:
    - name: cluster_register
      class: ironbox.agents.cluster_register.ClusterRegisterAgent
      enabled: true
  
  # Auto-discovery settings
  discovery:
    tools:
      enabled: true
      paths:
        - ironbox.tools
    agents:
      enabled: true
      paths:
        - ironbox.agents
```

This makes the system more flexible and extensible, as new tools and agents can be added without modifying the core code.

### 5. Auto-Discovery Mechanism

The toolkit can automatically discover tools and agents:

```python
def _discover_tools(self, package_path):
    """Discover tools in the specified package."""
    # Import the package
    package = importlib.import_module(package_path)
    
    # Iterate through all modules in the package
    for _, module_name, _ in pkgutil.iter_modules([package.__path__[0]]):
        # Import the module
        module = importlib.import_module(f"{package_path}.{module_name}")
        
        # Find all functions in the module
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            # Check if it has a docstring (potential tool)
            if obj.__doc__ and not name.startswith("_"):
                # Register the function as a tool
                self.register_tool(name, obj, "local")
```

This allows the system to be extended with new tools and agents without manual registration.

## Files Modified/Created

1. **New Files:**
   - `ironbox/ironbox/core/toolkit.py` - Unified toolkit implementation
   - `ironbox/config.yaml.example` - Example configuration file
   - `ironbox/docs/unified_toolkit_architecture.md` - Documentation of the unified toolkit architecture
   - `ironbox/docs/unified_toolkit.puml` - UML diagrams of the unified toolkit

2. **Modified Files:**
   - `ironbox/ironbox/core/agent_core.py` - Updated to use the unified toolkit
   - `ironbox/ironbox/config.py` - Added toolkit configuration section
   - `ironbox/docs/tool_repository.puml` - Updated architecture diagram
   - `ironbox/docs/agent_frameworks.puml` - Updated architecture diagram
   - `ironbox/notebooks/01_agent_core_and_frameworks.ipynb` - Updated with new examples
   - `ironbox/docs/enhanced_architecture_qa.md` - Added unified toolkit documentation
   - `ironbox/tests/test_agent_frameworks.py` - Updated to use configuration-based setup
   - `ironbox/tests/test_enhanced_architecture.py` - New test file for enhanced architecture

## Key Benefits

1. **Unified Access**: All frameworks can access both tools and agents through a consistent interface
2. **Agent-as-Tool Pattern**: Specialized agents can be used as tools by React and Plan frameworks
3. **Configuration-Driven**: System components can be configured through YAML files
4. **Auto-Discovery**: Tools and agents can be discovered automatically
5. **Flexible Orchestration**: Upper frameworks can use lower-level components
6. **Extensibility**: New specific agents can be easily added

## Usage Examples

### Using an Agent as a Tool

```python
# Process a query that would use an agent as a tool
query = "Use the cluster health agent to check the health of the production cluster"
result = await agent_core.process_query(query)
```

The React framework will use the agent-as-tool wrapper to invoke the cluster health agent.

### Configuration-Based Setup

```python
# Create agent core with configuration
config = load_config()
agent_core = AgentCore(config=config)

# Initialize the agent core
await agent_core.initialize()
```

The agent core will load tools and agents from the configuration file.

## Testing

You can run the test script to see the enhanced architecture in action:

```bash
cd ironbox
python -m tests.test_enhanced_architecture
```

This will demonstrate:
- The unified toolkit with tools and agents
- Using agents as tools in the React framework
- Using agents as tools in the Plan framework
- Configuration-based setup

## Conclusion

The enhanced core framework provides a more flexible and powerful system that can handle a wide range of query types efficiently, using the most appropriate approach for each situation. The unified toolkit allows all frameworks to access both tools and agents, enabling more complex and sophisticated AI actions.
