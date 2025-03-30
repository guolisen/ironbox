"""
Tests for IronBox core functionality.
"""
import os
import pytest
import asyncio
from unittest.mock import MagicMock, patch

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ironbox.config import config
from ironbox.core.graph import create_agent_graph, AgentState, AgentType
from ironbox.core.llm import OllamaChat
from ironbox.core.memory import MemoryManager
from ironbox.db.models import Base, Cluster
from ironbox.db.operations import ClusterOperations
from ironbox.agents.cluster_register import ClusterRegisterAgent
from ironbox.agents.cluster_info import ClusterInfoAgent
from ironbox.agents.cluster_health import ClusterHealthAgent
from ironbox.agents.memory_agent import MemoryAgent
from ironbox.agents.mcp_agent import MCPAgent
from ironbox.mcp.client import MCPClient


# Test database URL
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def db_session():
    """Create a test database session."""
    engine = create_async_engine(TEST_DB_URL)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    mock = MagicMock(spec=OllamaChat)
    
    async def mock_ainvoke(messages):
        # Simple mock response
        mock_response = MagicMock()
        mock_response.generations = [MagicMock()]
        mock_response.generations[0].message = MagicMock()
        
        # Extract the last user message
        user_message = None
        for message in messages:
            if message.get("role") == "user":
                user_message = message.get("content", "")
        
        # Generate a response based on the user message
        if "register" in user_message.lower():
            mock_response.generations[0].message.content = '{"name": "test-cluster", "api_server": "https://test-cluster:6443"}'
        elif "health" in user_message.lower():
            mock_response.generations[0].message.content = "test-cluster"
        elif "info" in user_message.lower():
            mock_response.generations[0].message.content = '{"request_type": "list", "cluster_name": null}'
        elif "memory" in user_message.lower():
            mock_response.generations[0].message.content = '{"request_type": "summary", "query": null}'
        elif "mcp" in user_message.lower():
            mock_response.generations[0].message.content = '{"request_type": "list_servers", "server_name": null, "details": null}'
        else:
            mock_response.generations[0].message.content = "I'll help you with that."
        
        return mock_response
    
    mock.ainvoke = mock_ainvoke
    return mock


@pytest.fixture
def mock_k8s_client():
    """Create a mock Kubernetes client."""
    mock = MagicMock()
    mock.connect.return_value = True
    mock.get_cluster_info.return_value = {
        "version": {
            "major": "1",
            "minor": "26",
            "git_version": "v1.26.0",
            "platform": "linux/amd64",
        },
        "nodes": 3,
        "namespaces": 5,
    }
    mock.get_health_check.return_value = {
        "status": "healthy",
        "issues": [],
        "pod_status": {
            "status_count": {
                "Running": 10,
                "Pending": 0,
                "Succeeded": 2,
                "Failed": 0,
                "Unknown": 0,
            },
            "total": 12,
        },
        "resource_usage": {
            "cpu": {
                "requested": 4.5,
                "capacity": 12,
                "usage_percent": 37.5,
            },
            "memory": {
                "requested": 8589934592,  # 8 GB
                "capacity": 34359738368,  # 32 GB
                "usage_percent": 25.0,
            },
        },
        "storage_status": {
            "pvc_status": {
                "Bound": 5,
                "Pending": 0,
                "Lost": 0,
            },
            "pv_status": {
                "Bound": 5,
                "Available": 2,
                "Released": 0,
                "Failed": 0,
            },
        },
    }
    return mock


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client."""
    mock = MagicMock(spec=MCPClient)
    mock.enabled = True
    
    async def mock_list_servers():
        return [
            {
                "name": "weather",
                "version": "0.1.0",
                "description": "Weather information provider",
                "capabilities": {
                    "tools": True,
                    "resources": True,
                },
            }
        ]
    
    async def mock_list_tools(server_name):
        return [
            {
                "name": "get_current_weather",
                "description": "Get current weather for a city",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        },
                    },
                    "required": ["city"],
                },
            }
        ]
    
    async def mock_use_tool(server_name, tool_name, arguments):
        return {
            "city": arguments.get("city", "Unknown"),
            "temperature": 25.0,
            "condition": "Sunny",
        }
    
    mock.list_servers = mock_list_servers
    mock.list_tools = mock_list_tools
    mock.use_tool = mock_use_tool
    
    return mock


@pytest.mark.asyncio
async def test_router_agent(mock_llm):
    """Test the router agent."""
    from ironbox.core.graph import RouterAgent
    
    # Create router agent
    agent = RouterAgent(llm=mock_llm)
    
    # Create agent state
    state = AgentState(input="I want to register a new cluster")
    
    # Call agent
    result = await agent(state)
    
    # Check result
    assert AgentType.ROUTER in result.agent_outputs
    assert "next" in result.agent_outputs[AgentType.ROUTER]
    assert result.agent_outputs[AgentType.ROUTER]["next"] == AgentType.CLUSTER_REGISTER


@pytest.mark.asyncio
async def test_cluster_register_agent(mock_llm, mock_k8s_client, db_session):
    """Test the cluster register agent."""
    with patch("ironbox.agents.cluster_register.KubernetesClient", return_value=mock_k8s_client):
        # Create cluster register agent
        agent = ClusterRegisterAgent(llm=mock_llm, db_session=db_session)
        
        # Create agent state
        state = AgentState(input="Register a new cluster named test-cluster with API server https://test-cluster:6443")
        
        # Call agent
        result = await agent(state)
        
        # Check result
        assert AgentType.CLUSTER_REGISTER in result.agent_outputs
        assert "response" in result.agent_outputs[AgentType.CLUSTER_REGISTER]
        assert "test-cluster" in result.agent_outputs[AgentType.CLUSTER_REGISTER]["response"]
        
        # Check database
        clusters = await ClusterOperations.get_clusters(db_session)
        assert len(clusters) == 1
        assert clusters[0].name == "test-cluster"
        assert clusters[0].api_server == "https://test-cluster:6443"


@pytest.mark.asyncio
async def test_cluster_health_agent(mock_llm, mock_k8s_client, db_session):
    """Test the cluster health agent."""
    # Create a test cluster
    cluster = await ClusterOperations.create_cluster(
        db_session,
        {
            "name": "test-cluster",
            "api_server": "https://test-cluster:6443",
        }
    )
    
    with patch("ironbox.agents.cluster_health.KubernetesClient", return_value=mock_k8s_client):
        # Create cluster health agent
        agent = ClusterHealthAgent(llm=mock_llm, db_session=db_session)
        
        # Create agent state
        state = AgentState(input="Check the health of test-cluster")
        
        # Call agent
        result = await agent(state)
        
        # Check result
        assert AgentType.CLUSTER_HEALTH in result.agent_outputs
        assert "response" in result.agent_outputs[AgentType.CLUSTER_HEALTH]
        
        # Check database
        health_checks = await ClusterOperations.get_health_checks(db_session, cluster.id)
        assert len(health_checks) == 1
        assert health_checks[0].status == "healthy"


@pytest.mark.asyncio
async def test_agent_graph(mock_llm, mock_k8s_client, mock_mcp_client, db_session):
    """Test the agent graph."""
    with patch("ironbox.agents.cluster_register.KubernetesClient", return_value=mock_k8s_client), \
         patch("ironbox.agents.cluster_health.KubernetesClient", return_value=mock_k8s_client), \
         patch("ironbox.agents.cluster_info.KubernetesClient", return_value=mock_k8s_client):
        
        # Create agents
        agents = {
            AgentType.ROUTER: RouterAgent(llm=mock_llm),
            AgentType.CLUSTER_REGISTER: ClusterRegisterAgent(llm=mock_llm, db_session=db_session),
            AgentType.CLUSTER_INFO: ClusterInfoAgent(llm=mock_llm, db_session=db_session),
            AgentType.CLUSTER_HEALTH: ClusterHealthAgent(llm=mock_llm, db_session=db_session),
            AgentType.MEMORY: MemoryAgent(llm=mock_llm, db_session=db_session),
            AgentType.MCP: MCPAgent(llm=mock_llm, mcp_client=mock_mcp_client),
        }
        
        # Create agent graph
        agent_graph = create_agent_graph(agents)
        
        # Test cluster registration
        result = await agent_graph.invoke("Register a new cluster named test-cluster with API server https://test-cluster:6443")
        assert "response" in result
        assert "test-cluster" in result["response"]
        
        # Check database
        clusters = await ClusterOperations.get_clusters(db_session)
        assert len(clusters) == 1
        assert clusters[0].name == "test-cluster"
        
        # Test cluster health check
        result = await agent_graph.invoke("Check the health of test-cluster")
        assert "response" in result
        assert "test-cluster" in result["response"]
        
        # Check database
        health_checks = await ClusterOperations.get_health_checks(db_session, clusters[0].id)
        assert len(health_checks) == 1
        assert health_checks[0].status == "healthy"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
