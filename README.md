# IronBox

A multi-agent platform built with LangGraph.

## Overview

IronBox is a powerful multi-agent platform designed to integrate any kind of agent and functionality. It leverages LangGraph for agent orchestration, allowing for extensible functionality through a modular agent architecture. While the current implementation includes Kubernetes cluster management capabilities, the platform is designed to be a general-purpose framework that can be extended with new agents for various use cases.

## Features

- **Multi-Agent Architecture**: Specialized agents for different tasks
- **Multiple Agent Frameworks**: Different frameworks for different problem types
  - **Route Agent Framework**: For simple queries that fit into predefined categories
  - **React Agent Framework**: For problems that require reasoning and action
  - **Plan Agent Framework**: For complex problems that require planning
- **Intelligent Framework Selection**: Automatically selects the appropriate framework based on query type
- **Extensible Framework**: Add new agents and capabilities for any use case
- **Memory System**: Retain context and parameters across conversations
- **MCP Integration**: Use Model Context Protocol for tool integration
- **User Interfaces**: REST API and Streamlit UI

## Example Queries

IronBox can handle a wide range of query types, automatically selecting the appropriate framework:

### Route Framework (Simple Categorizable Queries)

```
Register a new Kubernetes cluster named production with API server https://k8s.example.com:6443
Show me all the registered clusters
Check the health of the production cluster
What was the last cluster I registered?
What's the weather like in London today?
```

### React Framework (Reasoning and Action Queries)

```
Check how many pods are in the production cluster and which ones are not running
Find any pods in the staging cluster that have been restarting frequently and restart them
Should we deploy to the outdoor edge clusters today based on the weather forecast?
```

### Plan Framework (Complex Multi-Step Problems)

```
I need to migrate workloads from the staging cluster to production. Help me plan and execute this.
Our production cluster is running out of resources. Analyze usage patterns and suggest optimization strategies.
We have 5 microservices that need to be deployed across 3 clusters with specific affinity rules. Help me determine the optimal placement.
```

### Direct LLM Response (Simple Informational Queries)

```
What is a Kubernetes pod?
What are the best practices for securing a Kubernetes cluster?
What's the difference between a Deployment and a StatefulSet?
```

For more detailed examples and explanations, see the following documentation:
- [Enhanced Architecture Q&A](docs/enhanced_architecture_qa.md)
- [Framework Selection Q&A](docs/framework_selection_qa.md) - Explains how the system decides which agent framework to use
- [Logging Configuration](docs/logging.md) - Details on logging configuration and viewing logs

### Current Implementations

- **K8s Cluster Management**: Register and manage multiple Kubernetes clusters
- **Health Analysis**: Monitor pod status, resource usage, and PVC/Volume health
- **Weather Information**: Demo MCP server for weather data

## Getting Started

### Prerequisites

- Python 3.9+
- Ollama server (default) or other LLM provider
- Access to Kubernetes clusters (if using K8s management features)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ironbox.git
cd ironbox

# Install dependencies
pip install -e .
```

### Database Initialization

IronBox uses SQLite for storing data. The database tables are automatically initialized when you start the API server for the first time. However, you can also initialize the database manually using one of the following methods:

#### Method 1: Using SQLAlchemy ORM (Default)

```python
# init_db.py
import asyncio
from ironbox.db.operations import init_db

async def main():
    await init_db()
    print("Database initialized successfully!")

if __name__ == "__main__":
    asyncio.run(main())
```

Save this script as `init_db.py` in the project root and run it:

```bash
python init_db.py
```

#### Method 2: Using SQL Script

IronBox also provides a SQL script and a Python utility to initialize the database directly using SQL commands. This can be useful for database administrators or for manual database setup.

The SQL script is located at `scripts/init_db.sql` and contains all the necessary SQL commands to create the database tables. You can run it directly using the SQLite command-line tool:

```bash
sqlite3 data/db/ironbox.db < scripts/init_db.sql
```

Alternatively, you can use the provided Python script:

```bash
# Run with default paths
python scripts/init_db.py

# Or specify custom paths
python scripts/init_db.py --db-path /path/to/your/database.db --sql-script /path/to/your/script.sql
```

Both methods will create the necessary database tables for storing cluster information, health checks, chat history, and function calls.

### Configuration

Create a `.env` file in the project root:

```
OLLAMA_BASE_URL=http://your-ollama-server:11434
OLLAMA_MODEL=qwen2.5
```

### Running the API Server

You can run the API server using one of the following methods:

```bash
# Using the entry point (if installed with pip)
ironbox

# Using the module directly
python -m ironbox.api.server

# Running the server file directly
python ironbox/api/server.py
```

The API server will start on the configured host and port (default: http://localhost:8000).

You can access the API documentation at http://localhost:8000/docs.

### Running the UI

You can run the Streamlit UI using one of the following methods:

```bash
# Using the entry point (if installed with pip)
ironbox-ui

# Using the module directly
python -m ironbox.ui.app

# Running the UI file directly
streamlit run ironbox/ui/app.py
```

The UI will be available at http://localhost:8501 by default.

## Architecture

IronBox uses a LangGraph-based agent orchestration system with multiple agent frameworks:

### Agent Core

The Agent Core is the main entry point for processing user queries. It:
- Analyzes the query to select the appropriate framework
- Initializes the selected framework with the necessary agents and tools
- Processes the query using the selected framework
- Returns the response to the user

### Agent Frameworks

IronBox supports multiple agent frameworks, each designed for different types of problems:

1. **Route Agent Framework** (Original Framework)
   - Directs requests to specialized agents based on query type
   - Good for simple queries that fit into predefined categories
   - Uses a router agent to determine which specialized agent should handle the request

2. **React Agent Framework**
   - Uses the React paradigm (Reason + Act)
   - Good for problems that require reasoning and action
   - Executes a loop of thinking, acting, and observing until the problem is solved

3. **Plan Agent Framework**
   - Creates a plan before execution
   - Good for complex problems that require planning
   - First creates a step-by-step plan, then executes each step in order

4. **Direct LLM Response**
   - For simple questions that don't require special handling
   - Bypasses frameworks and agents for efficiency

### Framework Selection

The system automatically selects the appropriate framework based on the query type:
- Simple categorizable queries → Route Framework
- Reasoning and action problems → React Framework
- Complex multi-step problems → Plan Framework
- Simple informational questions → Direct LLM Response

### Current Agent Types
- **Cluster Register Agent**: Handles Kubernetes cluster registration
- **Cluster Health Agent**: Analyzes Kubernetes cluster health
- **Cluster Info Agent**: Provides information about registered clusters
- **Memory Agent**: Retrieves information from conversation history
- **MCP Agent**: Interfaces with Model Context Protocol tools
- **LLM Agent**: Handles general queries that don't align with specialized agents

## Extending IronBox

IronBox is designed to be easily extended with new agents, frameworks, and functionality:

### Adding New Agents

1. **Create a new agent**: Implement a new agent class that follows the agent interface
2. **Register the agent**: Add the agent to the agent core
3. **Update the router**: Ensure the router can direct requests to your new agent

### Adding New Tools

1. **Create a new tool**: Implement a new tool function with appropriate documentation
2. **Register the tool**: Add the tool to the agent core
3. **Use in frameworks**: The tool will be available in React and Plan frameworks

### Creating Custom Frameworks

1. **Extend the base framework**: Implement a new framework class that extends AgentFramework
2. **Implement the process method**: Define how the framework processes queries
3. **Register the framework**: Add the framework to the agent core

### Adding MCP Servers

Integrate external tools through the MCP protocol:
1. **Create an MCP server**: Implement a new MCP server with tools and resources
2. **Register the server**: Add the server to the MCP client configuration
3. **Use in agents**: The MCP tools will be available to the MCP agent

See the documentation in `docs/` for detailed instructions on extending IronBox.

## Development

### Running Tests

IronBox includes a comprehensive test suite covering core functionality, API endpoints, and agent behavior.

```bash
# Run all tests
pytest

# Run specific test modules
pytest ironbox/tests/test_core.py
pytest ironbox/tests/test_api.py
pytest ironbox/tests/test_agents.py
pytest ironbox/tests/test_agent_frameworks.py
pytest ironbox/tests/test_import.py

# Run tests with coverage report
pytest --cov=ironbox

# Run tests with verbose output
pytest -v

# Run specific test functions
pytest ironbox/tests/test_core.py::test_router_agent
pytest ironbox/tests/test_agent_frameworks.py::test_react_framework
```

The test suite uses pytest fixtures to mock dependencies like the LLM, Kubernetes client, and MCP client, allowing for isolated testing of components without external dependencies.

For integration tests that require a real database, the tests use an in-memory SQLite database that is created and destroyed for each test session.

## License
