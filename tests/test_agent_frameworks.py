"""
Test agent frameworks.
"""
import asyncio
import logging
import json
from typing import Dict, Any, List, Optional

from ironbox.core.agent_core import AgentCore
from ironbox.core.agent_framework import (
    RouteAgentFramework, 
    ReactAgentFramework, 
    PlanAgentFramework
)
from ironbox.core.llm import default_llm
from ironbox.core.graph import AgentType
from ironbox.agents.cluster_register import ClusterRegisterAgent
from ironbox.agents.cluster_info import ClusterInfoAgent
from ironbox.agents.cluster_health import ClusterHealthAgent
from ironbox.agents.memory_agent import MemoryAgent
from ironbox.agents.mcp_agent import MCPAgent
from ironbox.agents.llm_agent import LLMAgent
from ironbox.config import load_config

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Sample tools for React and Plan frameworks
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


# Create test configuration with tools and agents
def create_test_config() -> Dict[str, Any]:
    """Create a test configuration with tools and agents."""
    config = load_config()
    
    # Add test tools to config
    config["toolkit"]["tools"] = [
        {
            "name": "get_pod_count",
            "module": "ironbox.tests.test_agent_frameworks",
            "function": "get_pod_count",
            "description": "Get the number of pods in a cluster",
            "enabled": True
        },
        {
            "name": "get_node_status",
            "module": "ironbox.tests.test_agent_frameworks",
            "function": "get_node_status",
            "description": "Get the status of nodes in a cluster",
            "enabled": True
        },
        {
            "name": "restart_pod",
            "module": "ironbox.tests.test_agent_frameworks",
            "function": "restart_pod",
            "description": "Restart a pod in a cluster",
            "enabled": True
        },
        {
            "name": "scale_deployment",
            "module": "ironbox.tests.test_agent_frameworks",
            "function": "scale_deployment",
            "description": "Scale a deployment in a cluster",
            "enabled": True
        }
    ]
    
    # Add test agents to config
    config["toolkit"]["agents"] = [
        {
            "name": "cluster_register",
            "class": "ironbox.agents.cluster_register.ClusterRegisterAgent",
            "enabled": True
        },
        {
            "name": "cluster_info",
            "class": "ironbox.agents.cluster_info.ClusterInfoAgent",
            "enabled": True
        },
        {
            "name": "cluster_health",
            "class": "ironbox.agents.cluster_health.ClusterHealthAgent",
            "enabled": True
        },
        {
            "name": "memory",
            "class": "ironbox.agents.memory_agent.MemoryAgent",
            "enabled": True
        },
        {
            "name": "mcp",
            "class": "ironbox.agents.mcp_agent.MCPAgent",
            "enabled": True
        },
        {
            "name": "llm",
            "class": "ironbox.agents.llm_agent.LLMAgent",
            "enabled": True
        }
    ]
    
    # Disable auto-discovery for tests
    config["toolkit"]["discovery"]["tools"]["enabled"] = False
    config["toolkit"]["discovery"]["agents"]["enabled"] = False
    
    return config


# Create agent core
async def create_test_agent_core() -> AgentCore:
    """Create a test agent core with all frameworks."""
    # Create agent core with test config
    config = create_test_config()
    agent_core = AgentCore(config=config)
    
    # Initialize the agent core
    await agent_core.initialize()
    
    return agent_core


async def test_route_framework():
    """Test the route framework."""
    print("\n=== Testing Route Framework ===")
    
    agent_core = await create_test_agent_core()
    
    # Route framework is good for simple queries that fit into predefined categories
    query = "Register a new Kubernetes cluster named production with API server https://k8s.example.com:6443"
    
    result = await agent_core.process_query(query)
    print(f"Query: {query}")
    print(f"Selected Framework: {result['framework']}")
    print(f"Response: {result['response']}")


async def test_react_framework():
    """Test the react framework."""
    print("\n=== Testing React Framework ===")
    
    agent_core = await create_test_agent_core()
    
    # React framework is good for problems that require reasoning and action
    query = "Check the pod count in the production cluster and restart any pods that are not running"
    
    result = await agent_core.process_query(query)
    print(f"Query: {query}")
    print(f"Selected Framework: {result['framework']}")
    print(f"Response: {result['response']}")
    
    # Print steps if available
    if result['framework'] == 'react' and 'state' in result:
        state = result['state']
        if 'agent_outputs' in state and 'react' in state.agent_outputs:
            steps = state.agent_outputs['react'].get('steps', [])
            if steps:
                print("\nSteps:")
                for i, step in enumerate(steps):
                    print(f"Step {i+1}:")
                    print(f"  Thought: {step['thought']}")
                    print(f"  Action: {step['action']}")
                    print(f"  Action Input: {step['action_input']}")
                    print(f"  Observation: {step['observation']}")


async def test_plan_framework():
    """Test the plan framework."""
    print("\n=== Testing Plan Framework ===")
    
    agent_core = await create_test_agent_core()
    
    # Plan framework is good for complex problems that require planning
    query = "Scale the web deployment in the production cluster to 5 replicas, then check the node status to ensure all nodes are ready"
    
    result = await agent_core.process_query(query)
    print(f"Query: {query}")
    print(f"Selected Framework: {result['framework']}")
    print(f"Response: {result['response']}")
    
    # Print plan if available
    if result['framework'] == 'plan' and 'state' in result:
        state = result['state']
        if 'agent_outputs' in state and 'plan' in state.agent_outputs:
            plan = state.agent_outputs['plan'].get('plan', [])
            results = state.agent_outputs['plan'].get('results', [])
            if plan:
                print("\nPlan:")
                for i, step in enumerate(plan):
                    status = "✓" if step.completed else "✗"
                    print(f"{status} Step {i+1}: {step.description}")
                    if step.result:
                        print(f"   Result: {step.result}")


async def test_direct_response():
    """Test direct LLM response."""
    print("\n=== Testing Direct Response ===")
    
    agent_core = await create_test_agent_core()
    
    # Direct response is good for simple questions that don't require special handling
    query = "What is Kubernetes?"
    
    result = await agent_core.process_query(query)
    print(f"Query: {query}")
    print(f"Selected Framework: {result['framework']}")
    print(f"Response: {result['response']}")


async def test_framework_selection():
    """Test framework selection for different query types."""
    print("\n=== Testing Framework Selection ===")
    
    agent_core = await create_test_agent_core()
    
    # Test different query types
    queries = [
        "Register a new Kubernetes cluster named production",
        "How many pods are in the production cluster?",
        "Scale the web deployment in production to 5 replicas, then check node status",
        "What is a Kubernetes pod?",
        "I'm having an issue with my cluster where pods keep crashing",
    ]
    
    for query in queries:
        # Just test framework selection without running the full process
        framework_type = await agent_core.framework_selector.select_framework(query)
        print(f"Query: {query}")
        print(f"Selected Framework: {framework_type}\n")


async def test_agent_as_tool():
    """Test using an agent as a tool."""
    print("\n=== Testing Agent as Tool ===")
    
    agent_core = await create_test_agent_core()
    
    # Get the React framework
    react_framework = agent_core.frameworks.get("react")
    
    # Check if we have agent tools
    agent_tools = [name for name in agent_core.toolkit.tools.keys() if name.startswith("agent_")]
    print(f"Available agent tools: {agent_tools}")
    
    # Test using an agent as a tool
    if agent_tools and react_framework:
        # Create a test query that would use an agent tool
        query = "Use the cluster health agent to check the health of the production cluster"
        
        result = await agent_core.process_query(query)
        print(f"Query: {query}")
        print(f"Selected Framework: {result['framework']}")
        print(f"Response: {result['response']}")
        
        # Print steps if available
        if result['framework'] == 'react' and 'state' in result:
            state = result['state']
            if 'agent_outputs' in state and 'react' in state.agent_outputs:
                steps = state.agent_outputs['react'].get('steps', [])
                if steps:
                    print("\nSteps:")
                    for i, step in enumerate(steps):
                        print(f"Step {i+1}:")
                        print(f"  Thought: {step['thought']}")
                        print(f"  Action: {step['action']}")
                        print(f"  Action Input: {step['action_input']}")
                        print(f"  Observation: {step['observation']}")


async def main():
    """Run all tests."""
    await test_route_framework()
    await test_react_framework()
    await test_plan_framework()
    await test_direct_response()
    await test_framework_selection()
    await test_agent_as_tool()


if __name__ == "__main__":
    asyncio.run(main())
