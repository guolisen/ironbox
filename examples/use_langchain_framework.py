"""
Example of using the LangChain-based framework with LangGraph.

This script demonstrates how to use the enhanced agent framework
that leverages LangChain components and LangGraph for orchestration.
"""
import os
import sys
import asyncio
from typing import Dict, Any

from ironbox.core.agent_core import AgentCore, initialize_default_agent_core
from ironbox.config import load_config

# Sample queries to demonstrate the different frameworks
SAMPLE_QUERIES = [
    # Direct/Route framework query
    "What is Kubernetes and why is it useful?",
    
    # React framework query
    "I need to analyze the memory usage of pods in my cluster. Can you help me find the pods using the most memory?",
    
    # Plan framework query
    "I want to migrate my application from one Kubernetes cluster to another. Can you help me plan the migration?",
]


async def main():
    """Run the example."""
    print("Initializing agent core...")
    
    # Initialize the agent core
    # This loads the configuration, sets up the toolkit,
    # and creates the framework registry and orchestrator
    await initialize_default_agent_core()
    
    # Get the agent core instance
    from ironbox.core.agent_core import default_agent_core
    agent_core = default_agent_core
    
    # Process each sample query
    for i, query in enumerate(SAMPLE_QUERIES):
        print(f"\n\n{'='*80}")
        print(f"Query {i+1}: {query}")
        print(f"{'='*80}\n")
        
        # Process the query
        result = await agent_core.process_query(query)
        
        # Print the response
        print(f"\nResponse: {result['response']}")
        
        # Print debug information
        if 'agent_outputs' in result:
            # Get the framework type that was used
            framework_selector_output = result['agent_outputs'].get('framework_selector', {})
            if framework_selector_output:
                framework_type = framework_selector_output.get('framework_type')
                print(f"\nFramework used: {framework_type}")
        
        # Wait for user to press Enter before continuing
        if i < len(SAMPLE_QUERIES) - 1:
            input("\nPress Enter to continue to the next query...")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
