"""
Tests for IronBox agent functionality.
"""
import os
import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ironbox.config import config
from ironbox.core.graph import AgentState, AgentType
from ironbox.core.llm import OllamaChat
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
        elif "weather" in user_message.lower():
            mock_response.generations[0].message.content = '{"request_type": "use_tool", "server_name": "weather", "details": {"tool_name": "get_current_weather", "arguments": {"city": "New York"}}}'
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
async def test_cluster_register_agent_extract_info(mock_llm):
    """Test the cluster register agent's ability to extract cluster information."""
    agent = ClusterRegisterAgent(llm=mock_llm)
    
    # Test with a simple message
    cluster_info = await agent._extract_cluster_info(
        "Register a cluster named test-cluster with API server https://test-cluster:6443"
    )
    
    assert cluster_info["name"] == "test-cluster"
    assert cluster_info["api_server"] == "https://test-cluster:6443"


@pytest.mark.asyncio
async def test_cluster_health_agent_extract_name(mock_llm):
    """Test the cluster health agent's ability to extract cluster name."""
    agent = ClusterHealthAgent(llm=mock_llm)
    
    # Test with a simple message
    cluster_name = await agent._extract_cluster_name(
        "Check the health of test-cluster"
    )
    
    assert cluster_name == "test-cluster"


@pytest.mark.asyncio
async def test_cluster_info_agent_analyze_request(mock_llm):
    """Test the cluster info agent's ability to analyze requests."""
    agent = ClusterInfoAgent(llm=mock_llm)
    
    # Test list request
    request_type, cluster_name = await agent._analyze_request(
        "List all clusters"
    )
    
    assert request_type == "list"
    assert cluster_name is None
    
    # Test detail request
    request_type, cluster_name = await agent._analyze_request(
        "Get details for cluster test-cluster"
    )
    
    assert request_type == "detail"
    assert cluster_name == "test-cluster"
    
    # Test health history request
    request_type, cluster_name = await agent._analyze_request(
        "Show health history for cluster test-cluster"
    )
    
    assert request_type == "health"
    assert cluster_name == "test-cluster"


@pytest.mark.asyncio
async def test_memory_agent_analyze_request(mock_llm):
    """Test the memory agent's ability to analyze requests."""
    agent = MemoryAgent(llm=mock_llm)
    
    # Test conversation request
    request_type, query = await agent._analyze_request(
        "What did we talk about earlier?"
    )
    
    assert request_type == "conversation"
    assert query is not None
    
    # Test function request
    request_type, query = await agent._analyze_request(
        "What parameters did I use for the register_cluster function?"
    )
    
    assert request_type == "function"
    assert query is not None
    
    # Test summary request
    request_type, query = await agent._analyze_request(
        "Summarize our conversation"
    )
    
    assert request_type == "summary"
    assert query is None


@pytest.mark.asyncio
async def test_mcp_agent_analyze_request(mock_llm):
    """Test the MCP agent's ability to analyze requests."""
    agent = MCPAgent(llm=mock_llm)
    
    # Test list servers request
    request_type, server_name, details = await agent._analyze_request(
        "List all MCP servers"
    )
    
    assert request_type == "list_servers"
    assert server_name is None
    assert details is None
    
    # Test list tools request
    request_type, server_name, details = await agent._analyze_request(
        "List tools for server weather"
    )
    
    assert request_type == "list_tools"
    assert server_name == "weather"
    assert details is None
    
    # Test use tool request
    request_type, server_name, details = await agent._analyze_request(
        "Use the get_current_weather tool on the weather server for New York"
    )
    
    assert request_type == "use_tool"
    assert server_name == "weather"
    assert details is not None
    assert details["tool_name"] == "get_current_weather"
    assert details["arguments"] == {"city": "New York"}


@pytest.mark.asyncio
async def test_cluster_register_agent_full(mock_llm, mock_k8s_client, db_session):
    """Test the full cluster register agent workflow."""
    with patch("ironbox.agents.cluster_register.KubernetesClient", return_value=mock_k8s_client):
        agent = ClusterRegisterAgent(llm=mock_llm, db_session=db_session)
        
        # Test registering a cluster
        state = AgentState(input="Register a cluster named test-cluster with API server https://test-cluster:6443")
        result = await agent(state)
        
        assert AgentType.CLUSTER_REGISTER in result.agent_outputs
        assert "response" in result.agent_outputs[AgentType.CLUSTER_REGISTER]
        assert "test-cluster" in result.agent_outputs[AgentType.CLUSTER_REGISTER]["response"]
        
        # Check database
        clusters = await ClusterOperations.get_clusters(db_session)
        assert len(clusters) == 1
        assert clusters[0].name == "test-cluster"
        assert clusters[0].api_server == "https://test-cluster:6443"


@pytest.mark.asyncio
async def test_cluster_health_agent_full(mock_llm, mock_k8s_client, db_session):
    """Test the full cluster health agent workflow."""
    # Create a test cluster
    cluster = await ClusterOperations.create_cluster(
        db_session,
        {
            "name": "test-cluster",
            "api_server": "https://test-cluster:6443",
        }
    )
    
    with patch("ironbox.agents.cluster_health.KubernetesClient", return_value=mock_k8s_client):
        agent = ClusterHealthAgent(llm=mock_llm, db_session=db_session)
        
        # Test checking cluster health
        state = AgentState(input="Check the health of test-cluster")
        result = await agent(state)
        
        assert AgentType.CLUSTER_HEALTH in result.agent_outputs
        assert "response" in result.agent_outputs[AgentType.CLUSTER_HEALTH]
        
        # Check database
        health_checks = await ClusterOperations.get_health_checks(db_session, cluster.id)
        assert len(health_checks) == 1
        assert health_checks[0].status == "healthy"


@pytest.mark.asyncio
async def test_cluster_info_agent_full(mock_llm, mock_k8s_client, db_session):
    """Test the full cluster info agent workflow."""
    # Create a test cluster
    cluster = await ClusterOperations.create_cluster(
        db_session,
        {
            "name": "test-cluster",
            "api_server": "https://test-cluster:6443",
        }
    )
    
    with patch("ironbox.agents.cluster_info.KubernetesClient", return_value=mock_k8s_client):
        agent = ClusterInfoAgent(llm=mock_llm, db_session=db_session)
        
        # Test listing clusters
        state = AgentState(input="List all clusters")
        result = await agent(state)
        
        assert AgentType.CLUSTER_INFO in result.agent_outputs
        assert "response" in result.agent_outputs[AgentType.CLUSTER_INFO]
        assert "test-cluster" in result.agent_outputs[AgentType.CLUSTER_INFO]["response"]


@pytest.mark.asyncio
async def test_mcp_agent_full(mock_llm, mock_mcp_client):
    """Test the full MCP agent workflow."""
    agent = MCPAgent(llm=mock_llm, mcp_client=mock_mcp_client)
    
    # Test listing servers
    state = AgentState(input="List all MCP servers")
    result = await agent(state)
    
    assert AgentType.MCP in result.agent_outputs
    assert "response" in result.agent_outputs[AgentType.MCP]
    assert "weather" in result.agent_outputs[AgentType.MCP]["response"]
    
    # Test using a tool
    state = AgentState(input="What's the weather in New York?")
    result = await agent(state)
    
    assert AgentType.MCP in result.agent_outputs
    assert "response" in result.agent_outputs[AgentType.MCP]
    # The response should contain weather information
    assert "temperature" in result.agent_outputs[AgentType.MCP]["response"] or "weather" in result.agent_outputs[AgentType.MCP]["response"]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
