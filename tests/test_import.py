"""
Test script to verify imports are working correctly.
"""
try:
    from ironbox.core.agent_core import default_agent_core, initialize_default_agent_core
    print("Successfully imported default_agent_core and initialize_default_agent_core")
    
    from ironbox.core.graph import AgentType
    print("Successfully imported AgentType")
    
    from ironbox.agents.cluster_register import ClusterRegisterAgent
    print("Successfully imported ClusterRegisterAgent")
    
    print("All imports successful!")
except Exception as e:
    print(f"Error: {e}")
