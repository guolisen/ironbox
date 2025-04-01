"""
API server for IronBox.
"""
import logging
import asyncio
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from ironbox.config import config
from ironbox.db.operations import init_db
from ironbox.mcp.client import default_mcp_client
from ironbox.api.routes import router
# Import agent-related modules
from ironbox.core.graph import AgentType
from ironbox.agents.cluster_register import ClusterRegisterAgent
from ironbox.agents.cluster_info import ClusterInfoAgent
from ironbox.agents.cluster_health import ClusterHealthAgent
from ironbox.agents.memory_agent import MemoryAgent
from ironbox.agents.mcp_agent import MCPAgent
from ironbox.agents.llm_agent import LLMAgent

# Configure logging
logger = logging.getLogger(__name__)

# Import the agent core initialization function
from ironbox.api.routes import initialize_agent_core as routes_initialize_agent_core
from ironbox.core.agent_core import default_agent_core, initialize_default_agent_core

# Define the initialize_agent_core function here
async def initialize_agent_core():
    """Initialize agent core with agents and tools."""
    logger.info("Initializing agent core...")
    
    # Register agents
    default_agent_core.register_agent(AgentType.CLUSTER_REGISTER, create_cluster_register_agent)
    default_agent_core.register_agent(AgentType.CLUSTER_INFO, create_cluster_info_agent)
    default_agent_core.register_agent(AgentType.CLUSTER_HEALTH, create_cluster_health_agent)
    default_agent_core.register_agent(AgentType.MEMORY, create_memory_agent)
    default_agent_core.register_agent(AgentType.MCP, create_mcp_agent)
    default_agent_core.register_agent(AgentType.LLM, create_llm_agent)
    
    # Register local tools
    default_agent_core.register_tool("get_pod_count", get_pod_count)
    default_agent_core.register_tool("get_node_status", get_node_status)
    default_agent_core.register_tool("restart_pod", restart_pod)
    default_agent_core.register_tool("scale_deployment", scale_deployment)
    
    # Initialize the agent core with all registered agents and tools
    await initialize_default_agent_core()
    
    logger.info("Agent core initialized successfully")

# Helper functions for agent creation (placeholders)
def create_cluster_register_agent(db_session=None):
    """Create cluster register agent."""
    return ClusterRegisterAgent(db_session=db_session)

def create_cluster_info_agent(db_session=None):
    """Create cluster info agent."""
    return ClusterInfoAgent(db_session=db_session)

def create_cluster_health_agent(db_session=None):
    """Create cluster health agent."""
    return ClusterHealthAgent(db_session=db_session)

def create_memory_agent(db_session=None):
    """Create memory agent."""
    return MemoryAgent(db_session=db_session)

def create_mcp_agent(db_session=None):
    """Create MCP agent."""
    return MCPAgent(mcp_client=default_mcp_client)

def create_llm_agent(db_session=None):
    """Create LLM agent for general queries."""
    return LLMAgent()

# Sample Kubernetes tool implementations (will be moved to agent_core later)
async def get_pod_count(cluster_name: str) -> str:
    """Get the number of pods in a cluster."""
    # This is a mock implementation
    pod_counts = {
        "production": 42,
        "staging": 18,
        "development": 7,
    }
    return f"Cluster {cluster_name} has {pod_counts.get(cluster_name, 0)} pods."

async def get_node_status(cluster_name: str) -> str:
    """Get the status of nodes in a cluster."""
    # This is a mock implementation
    import json
    node_statuses = {
        "production": {"node1": "Ready", "node2": "Ready", "node3": "Ready"},
        "staging": {"node1": "Ready", "node2": "NotReady"},
        "development": {"node1": "Ready"},
    }
    status = node_statuses.get(cluster_name, {})
    return f"Cluster {cluster_name} node statuses: {json.dumps(status, indent=2)}"

async def restart_pod(cluster_name: str, pod_name: str) -> str:
    """Restart a pod in a cluster."""
    # This is a mock implementation
    return f"Pod {pod_name} in cluster {cluster_name} has been restarted."

async def scale_deployment(cluster_name: str, deployment_name: str, replicas: int) -> str:
    """Scale a deployment in a cluster."""
    # This is a mock implementation
    return f"Deployment {deployment_name} in cluster {cluster_name} has been scaled to {replicas} replicas."

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    Handles startup and shutdown events.
    """
    try:
        # Startup: Initialize database and MCP client
        await init_db()
        await default_mcp_client.initialize()
        
        # Initialize agent core
        logger.info("Initializing agent core from server lifespan...")
        # Call the initialize_agent_core function from routes.py
        await routes_initialize_agent_core()
        logger.info("Agent core initialized successfully")
        
        # Yield control to FastAPI
        yield
        
        # Shutdown: Clean up resources if needed
        # No cleanup needed for now
    except Exception as e:
        logger.error(f"Error in lifespan context manager: {e}")
        logger.error(traceback.format_exc())
        raise

# Create FastAPI app with lifespan
app = FastAPI(
    title="IronBox API",
    description="API for IronBox multi-agent K8s management platform",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "IronBox API",
        "version": "0.1.0",
        "description": "API for IronBox multi-agent K8s management platform",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    """Run the API server."""
    uvicorn.run(
        "ironbox.api.server:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=config["api"]["debug"],
        log_level="debug",  # Set log level to debug for detailed logging
    )


if __name__ == "__main__":
    main()
