# IronBox

A multi-agent platform built with LangGraph.

## Overview

IronBox is a powerful multi-agent platform designed to integrate any kind of agent and functionality. It leverages LangGraph for agent orchestration, allowing for extensible functionality through a modular agent architecture. While the current implementation includes Kubernetes cluster management capabilities, the platform is designed to be a general-purpose framework that can be extended with new agents for various use cases.

## Features

- **Multi-Agent Architecture**: Specialized agents for different tasks
- **Extensible Framework**: Add new agents and capabilities for any use case
- **Memory System**: Retain context and parameters across conversations
- **MCP Integration**: Use Model Context Protocol for tool integration
- **User Interfaces**: REST API and Streamlit UI

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

IronBox uses a LangGraph-based agent orchestration system with the following components:

- **Router Agent**: Directs requests to specialized agents
- **Memory Agent**: Manages conversation context
- **MCP Agent**: Integrates with external tools
- **Specialized Agents**: Task-specific agents that can be added to the framework

### Current Agent Types
- **Cluster Register Agent**: Handles Kubernetes cluster registration
- **Cluster Health Agent**: Analyzes Kubernetes cluster health
- **Cluster Info Agent**: Provides information about registered clusters

## Extending IronBox

IronBox is designed to be easily extended with new agents and functionality:

1. **Create a new agent**: Implement a new agent class that follows the agent interface
2. **Register the agent**: Add the agent to the agent graph
3. **Update the router**: Ensure the router can direct requests to your new agent
4. **Add MCP servers**: Integrate external tools through the MCP protocol

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

# Run tests with coverage report
pytest --cov=ironbox

# Run tests with verbose output
pytest -v

# Run specific test functions
pytest ironbox/tests/test_core.py::test_router_agent
pytest ironbox/tests/test_api.py::test_chat
```

The test suite uses pytest fixtures to mock dependencies like the LLM, Kubernetes client, and MCP client, allowing for isolated testing of components without external dependencies.

For integration tests that require a real database, the tests use an in-memory SQLite database that is created and destroyed for each test session.

### Adding New Agents

See the documentation in `docs/` for details on extending IronBox with new agents.

## License
