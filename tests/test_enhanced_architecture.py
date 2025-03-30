#!/usr/bin/env python3
"""
Test script for the enhanced architecture with multiple agent frameworks.

This script demonstrates:
1. Framework selection based on query type
2. Local tool usage
3. MCP tool usage
4. Integration between different components
"""
import asyncio
import json
import logging
from typing import Dict, Any, List

from ironbox.core.agent_core import AgentCore
from ironbox.core.graph import AgentType
from ironbox.core.llm import default_llm
from ironbox.mcp.client import default_mcp_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample local tools
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


async def solve_logical_problem(problem_description: str) -> str:
    """Solve a logical problem step by step."""
    # This is a mock implementation that would actually use LLM
    return f"Problem solved: {problem_description}\n1. Analyzed problem\n2. Identified solution approach\n3. Applied solution\n4. Verified result"


class TestEnhancedArchitecture:
    """Test class for the enhanced architecture."""
    
    def __init__(self):
        """Initialize the test class."""
        self.agent_core = AgentCore(llm=default_llm)
    
    async def setup(self):
        """Set up the test environment."""
        # Register agents
        self.agent_core.register_agent(AgentType.CLUSTER_REGISTER, lambda db_session=None: None)
        self.agent_core.register_agent(AgentType.CLUSTER_INFO, lambda db_session=None: None)
        self.agent_core.register_agent(AgentType.CLUSTER_HEALTH, lambda db_session=None: None)
        self.agent_core.register_agent(AgentType.MEMORY, lambda db_session=None: None)
        self.agent_core.register_agent(AgentType.MCP, lambda db_session=None: None)
        self.agent_core.register_agent(AgentType.LLM, lambda db_session=None: None)
        
        # Register local tools
        self.agent_core.register_tool("get_pod_count", get_pod_count)
        self.agent_core.register_tool("get_node_status", get_node_status)
        self.agent_core.register_tool("restart_pod", restart_pod)
        self.agent_core.register_tool("scale_deployment", scale_deployment)
        self.agent_core.register_tool("solve_logical_problem", solve_logical_problem)
        
        # Initialize MCP client
        await default_mcp_client.initialize()
        
        # Register MCP tools
        await self._register_mcp_tools()
        
        # Set up frameworks
        self.agent_core.setup_route_framework()
        self.agent_core.setup_react_framework()
        self.agent_core.setup_plan_framework()
        
        logger.info("Test environment set up")
    
    async def _register_mcp_tools(self):
        """Register MCP tools."""
        try:
            # Get list of servers
            servers = await default_mcp_client.list_servers()
            
            for server in servers:
                server_name = server.get("name")
                if not server_name:
                    continue
                
                # Get tools for this server
                tools = await default_mcp_client.list_tools(server_name)
                
                for tool in tools:
                    tool_name = tool.get("name")
                    if not tool_name:
                        continue
                    
                    # Create a wrapper function for this tool
                    tool_wrapper = self._create_mcp_tool_wrapper(server_name, tool_name, tool.get("description"))
                    
                    # Register the wrapper
                    registered_name = f"mcp_{server_name}_{tool_name}"
                    self.agent_core.register_tool(registered_name, tool_wrapper)
            
            logger.info(f"Registered MCP tools from {len(servers)} servers")
        except Exception as e:
            logger.error(f"Error registering MCP tools: {e}")
    
    def _create_mcp_tool_wrapper(self, server_name: str, tool_name: str, description: str = None):
        """Create a wrapper function for an MCP tool."""
        async def mcp_tool_wrapper(**kwargs):
            """MCP tool wrapper."""
            try:
                return await default_mcp_client.use_tool(
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=kwargs
                )
            except Exception as e:
                return f"Error using MCP tool {tool_name} on server {server_name}: {str(e)}"
        
        # Set function name and docstring
        mcp_tool_wrapper.__name__ = f"mcp_{server_name}_{tool_name}"
        mcp_tool_wrapper.__doc__ = description or f"MCP tool {tool_name} from server {server_name}"
        
        return mcp_tool_wrapper
    
    async def test_framework_selection(self):
        """Test framework selection based on query type."""
        print("\n===== Testing Framework Selection =====")
        
        test_queries = [
            {
                "query": "Register a new Kubernetes cluster named production with API server https://k8s.example.com:6443",
                "expected_framework": "route",
                "description": "Simple categorizable query (cluster registration)"
            },
            {
                "query": "Check the pod count in the production cluster and restart any pods that are not running",
                "expected_framework": "react",
                "description": "Problem requiring reasoning and action"
            },
            {
                "query": "I need to scale the web deployment to 5 replicas, then check if all nodes are ready, and finally update the DNS if everything looks good",
                "expected_framework": "plan",
                "description": "Complex multi-step problem"
            },
            {
                "query": "What is a Kubernetes pod?",
                "expected_framework": "direct",
                "description": "Simple informational question"
            },
            {
                "query": "I have a logical problem where I need to figure out the optimal pod placement strategy for our cluster",
                "expected_framework": "plan",
                "description": "Logical problem requiring planning"
            }
        ]
        
        for test in test_queries:
            # Get framework selection
            framework_type = await self.agent_core.framework_selector.select_framework(test["query"])
            
            # Print result
            print(f"\nQuery: {test['query']}")
            print(f"Description: {test['description']}")
            print(f"Expected Framework: {test['expected_framework']}")
            print(f"Selected Framework: {framework_type}")
            print(f"Result: {'✓ PASS' if framework_type == test['expected_framework'] else '✗ FAIL'}")
    
    async def test_route_framework(self):
        """Test the route framework with a simple categorizable query."""
        print("\n===== Testing Route Framework =====")
        
        query = "Get information about the production cluster"
        
        print(f"\nQuery: {query}")
        print("Expected: Route to cluster_info agent")
        
        result = await self.agent_core.process_query(query)
        
        print(f"Selected Framework: {result['framework']}")
        print(f"Response: {result['response']}")
    
    async def test_react_framework(self):
        """Test the React framework with a reasoning and action problem."""
        print("\n===== Testing React Framework =====")
        
        query = "Check how many pods are in the production cluster"
        
        print(f"\nQuery: {query}")
        print("Expected: Use React framework to reason and take action")
        
        result = await self.agent_core.process_query(query)
        
        print(f"Selected Framework: {result['framework']}")
        print(f"Response: {result['response']}")
        
        # Print steps if available
        if result['framework'] == 'react' and 'state' in result:
            state = result['state']
            if hasattr(state, 'agent_outputs') and 'react' in state.agent_outputs:
                steps = state.agent_outputs['react'].get('steps', [])
                if steps:
                    print("\nSteps:")
                    for i, step in enumerate(steps):
                        print(f"Step {i+1}:")
                        print(f"  Thought: {step.get('thought', 'N/A')}")
                        print(f"  Action: {step.get('action', 'N/A')}")
                        print(f"  Action Input: {step.get('action_input', 'N/A')}")
                        print(f"  Observation: {step.get('observation', 'N/A')}")
    
    async def test_plan_framework(self):
        """Test the Plan framework with a complex multi-step problem."""
        print("\n===== Testing Plan Framework =====")
        
        query = "Scale the web deployment in production to 5 replicas, then check if all nodes are ready"
        
        print(f"\nQuery: {query}")
        print("Expected: Use Plan framework to create and execute a plan")
        
        result = await self.agent_core.process_query(query)
        
        print(f"Selected Framework: {result['framework']}")
        print(f"Response: {result['response']}")
        
        # Print plan if available
        if result['framework'] == 'plan' and 'state' in result:
            state = result['state']
            if hasattr(state, 'agent_outputs') and 'plan' in state.agent_outputs:
                plan = state.agent_outputs['plan'].get('plan', [])
                if plan:
                    print("\nPlan:")
                    for i, step in enumerate(plan):
                        status = "✓" if step.completed else "✗"
                        print(f"{status} Step {i+1}: {step.description}")
                        if hasattr(step, 'result') and step.result:
                            print(f"   Result: {step.result}")
    
    async def test_direct_response(self):
        """Test direct LLM response for a simple question."""
        print("\n===== Testing Direct Response =====")
        
        query = "What is Kubernetes?"
        
        print(f"\nQuery: {query}")
        print("Expected: Direct LLM response")
        
        result = await self.agent_core.process_query(query)
        
        print(f"Selected Framework: {result['framework']}")
        print(f"Response: {result['response']}")
    
    async def test_local_tools(self):
        """Test local tools usage."""
        print("\n===== Testing Local Tools =====")
        
        # Test get_pod_count
        result = await get_pod_count("production")
        print(f"get_pod_count('production'): {result}")
        
        # Test get_node_status
        result = await get_node_status("production")
        print(f"get_node_status('production'): {result}")
        
        # Test restart_pod
        result = await restart_pod("production", "web-1")
        print(f"restart_pod('production', 'web-1'): {result}")
        
        # Test scale_deployment
        result = await scale_deployment("production", "web", 5)
        print(f"scale_deployment('production', 'web', 5): {result}")
        
        # Test solve_logical_problem
        result = await solve_logical_problem("How to optimize pod placement")
        print(f"solve_logical_problem('How to optimize pod placement'): {result}")
    
    async def test_mcp_tools(self):
        """Test MCP tools usage."""
        print("\n===== Testing MCP Tools =====")
        
        try:
            # List available servers
            servers = await default_mcp_client.list_servers()
            print(f"Available MCP servers: {[s.get('name', 'Unknown') for s in servers]}")
            
            # Test weather MCP tool if available
            for server in servers:
                server_name = server.get("name")
                if server_name == "weather":
                    # List tools
                    tools = await default_mcp_client.list_tools(server_name)
                    print(f"Tools for {server_name}: {[t.get('name', 'Unknown') for t in tools]}")
                    
                    # Test get_current_weather
                    try:
                        result = await default_mcp_client.use_tool(
                            server_name="weather",
                            tool_name="get_current_weather",
                            arguments={"city": "London"}
                        )
                        print(f"get_current_weather('London'): {result}")
                    except Exception as e:
                        print(f"Error using get_current_weather: {e}")
                    
                    # Test get_forecast
                    try:
                        result = await default_mcp_client.use_tool(
                            server_name="weather",
                            tool_name="get_forecast",
                            arguments={"city": "London", "days": 3}
                        )
                        print(f"get_forecast('London', 3): {result}")
                    except Exception as e:
                        print(f"Error using get_forecast: {e}")
        except Exception as e:
            print(f"Error testing MCP tools: {e}")
    
    async def test_logical_problem_solving(self):
        """Test logical problem solving with the Plan framework."""
        print("\n===== Testing Logical Problem Solving =====")
        
        query = "I have a logical problem: I need to figure out the most efficient way to distribute workloads across our Kubernetes cluster"
        
        print(f"\nQuery: {query}")
        print("Expected: Use Plan framework to solve the logical problem")
        
        result = await self.agent_core.process_query(query)
        
        print(f"Selected Framework: {result['framework']}")
        print(f"Response: {result['response']}")
    
    async def run_all_tests(self):
        """Run all tests."""
        await self.setup()
        await self.test_framework_selection()
        await self.test_route_framework()
        await self.test_react_framework()
        await self.test_plan_framework()
        await self.test_direct_response()
        await self.test_local_tools()
        await self.test_mcp_tools()
        await self.test_logical_problem_solving()


async def main():
    """Main function."""
    test = TestEnhancedArchitecture()
    await test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
