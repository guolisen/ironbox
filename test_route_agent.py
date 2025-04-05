"""
Test script for the LCRouteAgentFramework with ClusterInfoAgent.
"""
import asyncio
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from ironbox.core.llm import default_llm
from ironbox.core.langchain_frameworks import LCRouteAgentFramework
from ironbox.agents.cluster_info import ClusterInfoAgent
from ironbox.core.graph import AgentState

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_route_agent():
    """Test the route agent framework with ClusterInfoAgent."""
    # Create a mock database session
    # In a real scenario, you would use your actual database connection
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    # Create agents
    cluster_info_agent = ClusterInfoAgent(llm=default_llm, db_session=None)
    
    # Create route agent framework
    route_framework = LCRouteAgentFramework(
        llm=default_llm,
        agents={"cluster_info": cluster_info_agent},
        config={"system_prompt": """
        You are a router agent for the IronBox system. Your job is to analyze the user's request 
        and determine which specialized agent should handle it.
        
        Available agents:
        {agent_descriptions}
        
        Respond with the name of the agent that should handle the request.
        """}
    )
    
    # Create initial state
    state = AgentState(input="Tell me about the clusters in my Kubernetes environment")
    
    # Process the state
    try:
        result_state = await route_framework.process(state)
        logger.info("Processing completed successfully")
        logger.info(f"Current agent: {result_state.current_agent}")
        logger.info(f"Agent outputs: {result_state.agent_outputs}")
        logger.info(f"Error: {result_state.error}")
        return True
    except Exception as e:
        logger.error(f"Error processing state: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_route_agent())
    print(f"Test {'passed' if success else 'failed'}")
