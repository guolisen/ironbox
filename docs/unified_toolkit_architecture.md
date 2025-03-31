# Unified Toolkit Architecture

## Overview

The Unified Toolkit is a central component of the IronBox system that provides a consistent interface for managing and accessing tools and agents. It serves as a repository that organizes different types of tools (local, MCP, and agent-wrapped) and makes them available to various agent frameworks.

## Core Components

### Toolkit Class

The `Toolkit` class (`ironbox/core/toolkit.py`) is the main implementation of the Unified Toolkit architecture. It provides:

- Registration and management of tools and agents
- Categorization of tools by type (local, MCP, agent)
- Configuration-based loading of tools and agents
- Auto-discovery of tools and agents based on conventions
- Agent-as-tool wrapping functionality

### Tool Types

The Unified Toolkit manages three main types of tools:

1. **Local Tools**: Python functions defined within the application that perform specific operations (e.g., `get_pod_count`, `restart_pod`).
2. **MCP Tools**: External tools accessed through the MCP (Model Context Protocol) client, which connects to external MCP servers.
3. **Agent Tools**: Specialized agents wrapped as tools, allowing them to be used by other frameworks.

### Integration with Agent Core

The `AgentCore` class initializes and uses the Toolkit, registering tools and agents and setting up different frameworks (Route, React, Plan) with tools from the toolkit.

## Configuration

The Unified Toolkit is configured through the YAML configuration file (`config.yaml`), which defines:

- Tool definitions (name, module, function, description)
- Agent definitions (name, class, enabled)
- Auto-discovery settings (enabled, paths)

## Tool Registration and Discovery

### Manual Registration

Tools and agents can be registered directly through the API:

```python
agent_core.register_tool("tool_name", tool_function)
agent_core.register_agent("agent_type", agent_factory)
```

### Configuration-based Loading

Tools and agents are loaded from the configuration file using the `_load_from_config` method, which dynamically imports and registers tools and agents.

### Auto-discovery

When enabled, the toolkit can automatically discover and register tools and agents:

- For tools: Looks for functions with docstrings in specified packages
- For agents: Looks for classes ending with "Agent" and having a `__call__` method

## Agent-as-Tool Wrapping

The toolkit creates wrapper functions for agents, allowing them to be used as tools. This is done through the `_create_agent_tool_wrapper` method, which:

1. Creates a wrapper function that accepts a query and additional parameters
2. Converts these parameters into an agent state
3. Calls the agent with this state
4. Extracts and returns the response

## MCP Tool Integration

The toolkit integrates with MCP servers to provide additional tools:

1. The `register_mcp_tools` method in `AgentCore` connects to MCP servers
2. For each tool on each server, it creates a wrapper function
3. These wrapper functions are registered with the toolkit as MCP tools

## Usage in Frameworks

The toolkit is used by different agent frameworks:

1. **Route Framework**: Uses agents directly for routing queries to specialized agents
2. **React Framework**: Uses all tools (including agent tools) for step-by-step reasoning and action
3. **Plan Framework**: Uses all tools for planning and executing multi-step tasks

## Tool Invocation Flow

1. A framework looks up a tool by name in the toolkit
2. The toolkit returns the function reference
3. The framework calls the function with arguments
4. For local tools, the function executes directly
5. For agent tools, the wrapper creates a state, calls the agent, and extracts the response
6. For MCP tools, the wrapper calls the MCP client, which sends a request to the MCP server

## Benefits of the Unified Toolkit

- **Centralized Management**: Single point of registration and access for all tools and agents
- **Consistent Interface**: Common interface for different types of tools
- **Flexible Configuration**: Tools and agents can be configured through YAML files
- **Auto-discovery**: Automatic discovery of tools and agents based on conventions
- **Agent-as-Tool Wrapping**: Agents can be used as tools by other frameworks
- **MCP Integration**: Seamless integration with external MCP servers
