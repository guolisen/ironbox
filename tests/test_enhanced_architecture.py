"""
Test the enhanced architecture with unified toolkit.
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
from ironbox.config import load_config

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Sample tools for testing
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
    config["toolkit"] = {
        "tools": [
            {
                "name": "get_pod_count",
                "module": "ironbox.tests.test_enhanced_architecture",
                "function": "get_pod_count",
                "description": "Get the number of pods in a cluster",
                "enabled": True
            },
            {
                "name": "get_node_status",
                "module": "ironbox.tests.test_enhanced_architecture",
                "function": "get_node_status",
                "description": "Get the status of nodes in a cluster",
                "enabled": True
            },
            {
                "name": "restart_pod",
                "module": "ironbox.tests.test_enhanced_architecture",
                "function": "restart_pod",
                "description": "Restart a pod in a cluster",
                "enabled": True
            },
            {
                "name": "scale_deployment",
                "module": "ironbox.tests.test_enhanced_architecture",
                "function": "scale_deployment",
                "description": "Scale a deployment in a cluster",
                "enabled": True
            }
        ],
        "agents": [
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
        ],
        "discovery": {
            "tools": {
                "enabled": False,
                "paths": ["ironbox.tools"]
            },
            "agents": {
                "enabled": False,
                "paths": ["ironbox.agents"]
            }
        }
    }
    
    return config


async def test_unified_toolkit():
    """Test the unified toolkit."""
    print("\n=== Testing Unified Toolkit ===")
    
    # Create agent core with test config
    config = create_test_config()
    agent_core = AgentCore(config=config)
    
    # Initialize the agent core
    await agent_core.initialize()
    
    # Print toolkit contents
    print("All Tools in Unified Toolkit:")
    for tool_name in agent_core.toolkit.tools.keys():
        print(f"- {tool_name}")
    
    print("\nLocal Tools:")
    for tool_name in agent_core.toolkit.local_tools.keys():
        print(f"- {tool_name}")
    
    print("\nMCP Tools:")
    for tool_name in agent_core.toolkit.mcp_tools.keys():
        print(f"- {tool_name}")
    
    print("\nAgent-as-Tools:")
    for tool_name in agent_core.toolkit.agent_tools.keys():
        print(f"- {tool_name}")
    
    print("\nOriginal Agents:")
    for agent_name in agent_core.toolkit.agents.keys():
        print(f"- {agent_name}")


async def test_react_with_agent_tools():
    """Test using agents as tools in the React framework."""
    print("\n=== Testing React Framework with Agent Tools ===")
    
    # Create agent core with test config
    config = create_test_config()
    agent_core = AgentCore(config=config)
    
    # Initialize the agent core
    await agent_core.initialize()
    
    # Process a query that would use an agent as a tool
    query = "Use the cluster health agent to check the health of the production cluster"
    print(f"Query: {query}")
    
    result = await agent_core.process_query(query)
    print(f"Selected Framework: {result['framework']}")
    print(f"Response: {result['response']}")
    
    # Print steps if available and using React framework
    if result['framework'] == 'react' and 'state' in result:
        state = result['state']
        if hasattr(state, 'agent_outputs') and 'react' in state.agent_outputs:
            steps = state.agent_outputs['react'].get('steps', [])
            if steps:
                print("\nSteps:")
                for i, step in enumerate(steps):
                    print(f"Step {i+1}:")
                    print(f"  Thought: {step.get('thought', 'No thought')}")
                    print(f"  Action: {step.get('action', 'No action')}")
                    print(f"  Action Input: {step.get('action_input', 'No input')}")
                    print(f"  Observation: {step.get('observation', 'No observation')}")


async def test_plan_with_agent_tools():
    """Test using agents as tools in the Plan framework."""
    print("\n=== Testing Plan Framework with Agent Tools ===")
    
    # Create agent core with test config
    config = create_test_config()
    agent_core = AgentCore(config=config)
    
    # Initialize the agent core
    await agent_core.initialize()
    
    # Process a query that would use an agent as a tool
    query = "Create a plan to check the health of all clusters and restart any pods that are not running"
    print(f"Query: {query}")
    
    result = await agent_core.process_query(query)
    print(f"Selected Framework: {result['framework']}")
    print(f"Response: {result['response']}")
    
    # Print plan if available and using Plan framework
    if result['framework'] == 'plan' and 'state' in result:
        state = result['state']
        if hasattr(state, 'agent_outputs') and 'plan' in state.agent_outputs:
            plan = state.agent_outputs['plan'].get('plan', [])
            if plan:
                print("\nPlan:")
                for i, step in enumerate(plan):
                    status = "✓" if step.completed else "✗"
                    print(f"{status} Step {i+1}: {step.description}")
                    if step.result:
                        print(f"   Result: {step.result}")


async def test_route_with_tools():
    """Test using tools in the Route framework."""
    print("\n=== Testing Route Framework with Tools ===")
    
    # Create agent core with test config
    config = create_test_config()
    agent_core = AgentCore(config=config)
    
    # Initialize the agent core
    await agent_core.initialize()
    
    # Process a query that would use the route framework
    query = "Register a new Kubernetes cluster named production with API server https://k8s.example.com:6443"
    print(f"Query: {query}")
    
    result = await agent_core.process_query(query)
    print(f"Selected Framework: {result['framework']}")
    print(f"Response: {result['response']}")
    
    # Print agent outputs if available
    if 'state' in result:
        state = result['state']
        if hasattr(state, 'agent_outputs') and 'router' in state.agent_outputs:
            router_output = state.agent_outputs['router']
            print(f"\nRouter Output:")
            print(f"  Response: {router_output.get('response', 'No response')}")
            print(f"  Next Agent: {router_output.get('next', 'None')}")


async def test_configuration_based_setup():
    """Test configuration-based setup."""
    print("\n=== Testing Configuration-Based Setup ===")
    
    # Create agent core with test config
    config = create_test_config()
    
    # Print toolkit configuration
    print("Toolkit Configuration:")
    print(json.dumps(config.get("toolkit", {}), indent=2))
    
    # Create agent core with this config
    agent_core = AgentCore(config=config)
    
    # Initialize the agent core
    await agent_core.initialize()
    
    # Print registered tools and agents
    print("\nRegistered Tools:")
    for tool_name in agent_core.toolkit.tools.keys():
        print(f"- {tool_name}")
    
    print("\nRegistered Agents:")
    for agent_name in agent_core.toolkit.agents.keys():
        print(f"- {agent_name}")


async def main():
    """Run all tests."""
    await test_unified_toolkit()
    await test_react_with_agent_tools()
    await test_plan_with_agent_tools()
    await test_route_with_tools()
    await test_configuration_based_setup()


if __name__ == "__main__":
    asyncio.run(main())
