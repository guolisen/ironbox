"""
API routes for IronBox.
"""
import logging
import uuid
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ironbox.core.graph import create_agent_graph, AgentType
from ironbox.core.memory import MemoryManager
from ironbox.db.operations import get_db_session, ClusterOperations
from ironbox.agents.cluster_register import ClusterRegisterAgent
from ironbox.agents.cluster_info import ClusterInfoAgent
from ironbox.agents.cluster_health import ClusterHealthAgent
from ironbox.agents.memory_agent import MemoryAgent
from ironbox.agents.mcp_agent import MCPAgent
from ironbox.agents.llm_agent import LLMAgent
from ironbox.mcp.client import default_mcp_client

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Models
class ChatRequest(BaseModel):
    """Chat request model."""
    
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Chat session ID")


class ChatResponse(BaseModel):
    """Chat response model."""
    
    response: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Chat session ID")


class ClusterRequest(BaseModel):
    """Cluster request model."""
    
    name: str = Field(..., description="Cluster name")
    api_server: str = Field(..., description="API server URL")
    description: Optional[str] = Field(None, description="Cluster description")
    token: Optional[str] = Field(None, description="Authentication token")
    certificate: Optional[str] = Field(None, description="CA certificate")
    config_file: Optional[str] = Field(None, description="Path to kubeconfig file")
    config_context: Optional[str] = Field(None, description="Kubeconfig context")
    insecure_skip_tls_verify: Optional[bool] = Field(False, description="Skip TLS verification")


class ClusterResponse(BaseModel):
    """Cluster response model."""
    
    id: int = Field(..., description="Cluster ID")
    name: str = Field(..., description="Cluster name")
    api_server: str = Field(..., description="API server URL")
    description: Optional[str] = Field(None, description="Cluster description")
    health_status: Optional[str] = Field(None, description="Health status")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Update timestamp")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Health status")
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="Issues")
    pod_status: Dict[str, Any] = Field(..., description="Pod status")
    resource_usage: Dict[str, Any] = Field(..., description="Resource usage")
    storage_status: Dict[str, Any] = Field(..., description="Storage status")


# Chat endpoints
@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Chat with the assistant.
    
    Args:
        request: Chat request
        db: Database session
        
    Returns:
        Chat response
    """
    try:
        # Get or create session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Create memory manager
        memory_manager = MemoryManager(
            session_id=session_id,
            db_session=db
        )
        
        # Create agents
        agents = {
            AgentType.ROUTER: create_router_agent(),
            AgentType.CLUSTER_REGISTER: create_cluster_register_agent(db),
            AgentType.CLUSTER_INFO: create_cluster_info_agent(db),
            AgentType.CLUSTER_HEALTH: create_cluster_health_agent(db),
            AgentType.MEMORY: create_memory_agent(db),
            AgentType.MCP: create_mcp_agent(),
            AgentType.LLM: create_llm_agent(),
        }
        
        # Create agent graph
        logger.debug("Creating agent graph with agents: %s", list(agents.keys()))
        agent_graph = create_agent_graph(agents)
        
        # Invoke agent graph
        logger.debug("Invoking agent graph with message: %s", request.message)
        result = await agent_graph.invoke(request.message, session_id=session_id)
        logger.debug("Agent graph result: %s", result)
        
        # Save conversation to memory
        await memory_manager.conversation_memory.save_context(
            {"input": request.message},
            {"output": result["response"]}
        )
        
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# Cluster endpoints
@router.post("/clusters", response_model=ClusterResponse)
async def create_cluster(
    request: ClusterRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Create a new cluster.
    
    Args:
        request: Cluster request
        db: Database session
        
    Returns:
        Cluster response
    """
    try:
        # Check if cluster already exists
        existing_cluster = await ClusterOperations.get_cluster_by_name(db, request.name)
        if existing_cluster:
            raise HTTPException(status_code=400, detail=f"Cluster '{request.name}' already exists")
        
        # Create cluster
        cluster = await ClusterOperations.create_cluster(db, request.dict())
        
        return ClusterResponse(
            id=cluster.id,
            name=cluster.name,
            api_server=cluster.api_server,
            description=cluster.description,
            health_status=cluster.health_status,
            created_at=cluster.created_at.isoformat(),
            updated_at=cluster.updated_at.isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters", response_model=List[ClusterResponse])
async def list_clusters(
    db: AsyncSession = Depends(get_db_session),
):
    """
    List all clusters.
    
    Args:
        db: Database session
        
    Returns:
        List of clusters
    """
    try:
        clusters = await ClusterOperations.get_clusters(db)
        
        return [
            ClusterResponse(
                id=cluster.id,
                name=cluster.name,
                api_server=cluster.api_server,
                description=cluster.description,
                health_status=cluster.health_status,
                created_at=cluster.created_at.isoformat(),
                updated_at=cluster.updated_at.isoformat(),
            )
            for cluster in clusters
        ]
    except Exception as e:
        logger.error(f"Error listing clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters/{cluster_id}", response_model=ClusterResponse)
async def get_cluster(
    cluster_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get a cluster by ID.
    
    Args:
        cluster_id: Cluster ID
        db: Database session
        
    Returns:
        Cluster response
    """
    try:
        cluster = await ClusterOperations.get_cluster(db, cluster_id)
        
        if not cluster:
            raise HTTPException(status_code=404, detail=f"Cluster with ID {cluster_id} not found")
        
        return ClusterResponse(
            id=cluster.id,
            name=cluster.name,
            api_server=cluster.api_server,
            description=cluster.description,
            health_status=cluster.health_status,
            created_at=cluster.created_at.isoformat(),
            updated_at=cluster.updated_at.isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clusters/{cluster_id}")
async def delete_cluster(
    cluster_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Delete a cluster.
    
    Args:
        cluster_id: Cluster ID
        db: Database session
        
    Returns:
        Success message
    """
    try:
        # Check if cluster exists
        cluster = await ClusterOperations.get_cluster(db, cluster_id)
        if not cluster:
            raise HTTPException(status_code=404, detail=f"Cluster with ID {cluster_id} not found")
        
        # Delete cluster
        deleted = await ClusterOperations.delete_cluster(db, cluster_id)
        
        if not deleted:
            raise HTTPException(status_code=500, detail=f"Failed to delete cluster with ID {cluster_id}")
        
        return {"message": f"Cluster '{cluster.name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters/{cluster_id}/health", response_model=HealthCheckResponse)
async def check_cluster_health(
    cluster_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Check cluster health.
    
    Args:
        cluster_id: Cluster ID
        db: Database session
        
    Returns:
        Health check response
    """
    try:
        # Get cluster
        cluster = await ClusterOperations.get_cluster(db, cluster_id)
        if not cluster:
            raise HTTPException(status_code=404, detail=f"Cluster with ID {cluster_id} not found")
        
        # Create health agent
        health_agent = create_cluster_health_agent(db)
        
        # Create agent state
        from ironbox.core.graph import AgentState
        state = AgentState(input=f"Check health of cluster {cluster.name}")
        
        # Call health agent
        state = await health_agent(state)
        
        # Get health check from agent output
        agent_output = state.agent_outputs.get(AgentType.CLUSTER_HEALTH, {})
        if state.error or not agent_output:
            raise HTTPException(status_code=500, detail=state.error or "Failed to check cluster health")
        
        # Get health check from database
        health_checks = await ClusterOperations.get_health_checks(db, cluster_id, limit=1)
        if not health_checks:
            raise HTTPException(status_code=500, detail="No health check results found")
        
        health_check = health_checks[0]
        
        return HealthCheckResponse(
            status=health_check.status,
            issues=health_check.issues or [],
            pod_status=health_check.pod_status or {},
            resource_usage=health_check.resource_usage or {},
            storage_status=health_check.storage_status or {},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking cluster health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def create_router_agent():
    """Create router agent."""
    from ironbox.core.graph import RouterAgent
    return RouterAgent()


def create_cluster_register_agent(db: AsyncSession):
    """Create cluster register agent."""
    return ClusterRegisterAgent(db_session=db)


def create_cluster_info_agent(db: AsyncSession):
    """Create cluster info agent."""
    return ClusterInfoAgent(db_session=db)


def create_cluster_health_agent(db: AsyncSession):
    """Create cluster health agent."""
    return ClusterHealthAgent(db_session=db)


def create_memory_agent(db: AsyncSession):
    """Create memory agent."""
    return MemoryAgent(db_session=db)


def create_mcp_agent():
    """Create MCP agent."""
    return MCPAgent(mcp_client=default_mcp_client)


def create_llm_agent():
    """Create LLM agent for general queries."""
    return LLMAgent()
