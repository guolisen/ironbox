"""
API server for IronBox.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from ironbox.config import config
#from ironbox.core.graph import create_agent_graph, AgentType
#from ironbox.core.llm import default_llm
#from ironbox.core.memory import MemoryManager
from ironbox.db.operations import init_db
#from ironbox.agents.cluster_register import ClusterRegisterAgent
#from ironbox.agents.cluster_info import ClusterInfoAgent
#from ironbox.agents.cluster_health import ClusterHealthAgent
#from ironbox.agents.memory_agent import MemoryAgent
#from ironbox.agents.mcp_agent import MCPAgent
from ironbox.mcp.client import default_mcp_client

from ironbox.api.routes import router

# Configure logging
logger = logging.getLogger(__name__)

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    Handles startup and shutdown events.
    """
    # Startup: Initialize database and MCP client
    await init_db()
    await default_mcp_client.initialize()
    
    # Yield control to FastAPI
    yield
    
    # Shutdown: Clean up resources if needed
    # No cleanup needed for now

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
    )


if __name__ == "__main__":
    main()
