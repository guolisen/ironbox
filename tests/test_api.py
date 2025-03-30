"""
Tests for IronBox API functionality.
"""
import os
import pytest
import asyncio
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ironbox.config import config
from ironbox.api.server import app
from ironbox.core.graph import create_agent_graph, AgentState, AgentType, RouterAgent
from ironbox.core.llm import OllamaChat
from ironbox.db.models import Base, Cluster
from ironbox.db.operations import get_db_session, ClusterOperations
from ironbox.agents.cluster_register import ClusterRegisterAgent
from ironbox.agents.cluster_info import ClusterInfoAgent
from ironbox.agents.cluster_health import ClusterHealthAgent
from ironbox.agents.memory_agent import MemoryAgent
from ironbox.agents.mcp_agent import MCPAgent
from ironbox.mcp.client import MCPClient


# Test database URL
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


# Override the get_db_session dependency
async def override_get_db_session():
    """Override the get_db_session dependency."""
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


# Override the dependencies
app.dependency_overrides[get_db_session] = override_get_db_session


# Create test client
client = TestClient(app)


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


@pytest.fixture
def mock_agent_graph():
    """Create a mock agent graph."""
    mock = MagicMock()
    
    async def mock_invoke(message, session_id=None):
        return {
            "response": f"I'll help you with: {message}",
            "agent_outputs": {},
            "error": None,
        }
    
    mock.invoke = mock_invoke
    return mock


def test_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "IronBox API"


def test_health():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("ironbox.api.routes.create_agent_graph")
@patch("ironbox.api.routes.create_router_agent")
@patch("ironbox.api.routes.create_cluster_register_agent")
@patch("ironbox.api.routes.create_cluster_info_agent")
@patch("ironbox.api.routes.create_cluster_health_agent")
@patch("ironbox.api.routes.create_memory_agent")
@patch("ironbox.api.routes.create_mcp_agent")
def test_chat(
    mock_create_mcp_agent,
    mock_create_memory_agent,
    mock_create_cluster_health_agent,
    mock_create_cluster_info_agent,
    mock_create_cluster_register_agent,
    mock_create_router_agent,
    mock_create_agent_graph,
    mock_agent_graph,
):
    """Test the chat endpoint."""
    # Set up mocks
    mock_create_agent_graph.return_value = mock_agent_graph
    
    # Test chat endpoint
    response = client.post(
        "/chat",
        json={
            "message": "Hello, I need help with my Kubernetes clusters",
            "session_id": "test-session",
        },
    )
    
    assert response.status_code == 200
    assert "response" in response.json()
    assert "session_id" in response.json()
    assert response.json()["session_id"] == "test-session"


@patch("ironbox.agents.cluster_register.KubernetesClient")
def test_create_cluster(mock_kubernetes_client, mock_k8s_client):
    """Test the create cluster endpoint."""
    # Set up mocks
    mock_kubernetes_client.return_value = mock_k8s_client
    
    # Test create cluster endpoint
    response = client.post(
        "/clusters",
        json={
            "name": "test-cluster",
            "api_server": "https://test-cluster:6443",
            "description": "Test cluster",
            "token": "test-token",
            "insecure_skip_tls_verify": True,
        },
    )
    
    assert response.status_code == 200
    assert response.json()["name"] == "test-cluster"
    assert response.json()["api_server"] == "https://test-cluster:6443"
    assert response.json()["description"] == "Test cluster"


@patch("ironbox.agents.cluster_register.KubernetesClient")
def test_list_clusters(mock_kubernetes_client, mock_k8s_client):
    """Test the list clusters endpoint."""
    # Set up mocks
    mock_kubernetes_client.return_value = mock_k8s_client
    
    # Create a test cluster
    client.post(
        "/clusters",
        json={
            "name": "test-cluster",
            "api_server": "https://test-cluster:6443",
        },
    )
    
    # Test list clusters endpoint
    response = client.get("/clusters")
    
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 1
    assert response.json()[0]["name"] == "test-cluster"


@patch("ironbox.agents.cluster_register.KubernetesClient")
def test_get_cluster(mock_kubernetes_client, mock_k8s_client):
    """Test the get cluster endpoint."""
    # Set up mocks
    mock_kubernetes_client.return_value = mock_k8s_client
    
    # Create a test cluster
    create_response = client.post(
        "/clusters",
        json={
            "name": "test-cluster",
            "api_server": "https://test-cluster:6443",
        },
    )
    
    cluster_id = create_response.json()["id"]
    
    # Test get cluster endpoint
    response = client.get(f"/clusters/{cluster_id}")
    
    assert response.status_code == 200
    assert response.json()["id"] == cluster_id
    assert response.json()["name"] == "test-cluster"
    assert response.json()["api_server"] == "https://test-cluster:6443"


@patch("ironbox.agents.cluster_register.KubernetesClient")
def test_delete_cluster(mock_kubernetes_client, mock_k8s_client):
    """Test the delete cluster endpoint."""
    # Set up mocks
    mock_kubernetes_client.return_value = mock_k8s_client
    
    # Create a test cluster
    create_response = client.post(
        "/clusters",
        json={
            "name": "test-cluster",
            "api_server": "https://test-cluster:6443",
        },
    )
    
    cluster_id = create_response.json()["id"]
    
    # Test delete cluster endpoint
    response = client.delete(f"/clusters/{cluster_id}")
    
    assert response.status_code == 200
    assert "message" in response.json()
    assert "test-cluster" in response.json()["message"]
    
    # Verify cluster is deleted
    get_response = client.get(f"/clusters/{cluster_id}")
    assert get_response.status_code == 404


@patch("ironbox.agents.cluster_health.KubernetesClient")
@patch("ironbox.agents.cluster_register.KubernetesClient")
def test_check_cluster_health(mock_register_client, mock_health_client, mock_k8s_client):
    """Test the check cluster health endpoint."""
    # Set up mocks
    mock_register_client.return_value = mock_k8s_client
    mock_health_client.return_value = mock_k8s_client
    
    # Create a test cluster
    create_response = client.post(
        "/clusters",
        json={
            "name": "test-cluster",
            "api_server": "https://test-cluster:6443",
        },
    )
    
    cluster_id = create_response.json()["id"]
    
    # Test check cluster health endpoint
    with patch("ironbox.api.routes.create_cluster_health_agent") as mock_create_agent:
        # Set up mock agent
        mock_agent = MagicMock()
        
        async def mock_call(state):
            state.agent_outputs[AgentType.CLUSTER_HEALTH] = {
                "response": "Cluster is healthy",
            }
            return state
        
        mock_agent.side_effect = mock_call
        mock_create_agent.return_value = mock_agent
        
        # Call health endpoint
        response = client.get(f"/clusters/{cluster_id}/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "pod_status" in response.json()
        assert "resource_usage" in response.json()
        assert "storage_status" in response.json()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
