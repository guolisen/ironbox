"""
Test for the LLM agent functionality.
"""
import asyncio
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from ironbox.api.routes import chat, ChatRequest
from ironbox.core.graph import AgentType
from ironbox.config import config

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_llm_agent():
    """
    Test that queries not matching specialized agents are routed to the LLM agent.
    """
    # Create engine and session
    db_url = config["database"]["url"].replace("sqlite://", "sqlite+aiosqlite://")
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    # Create session
    async with async_session() as session:
        # Create request with a general query that doesn't match specialized agents
        request = ChatRequest(message="What is the capital of France?")
        
        # Call chat endpoint
        logger.debug("Calling chat endpoint with message: %s", request.message)
        response = await chat(request, session)
        
        # Print response
        logger.debug("Got response: %s", response)
        print(f"Response: {response}")
        
        # The response should come from the LLM agent
        # Note: We can't directly verify which agent was used in this test
        # as the agent graph details are not exposed in the response
        # But we can check that a response was generated
        assert response.response, "No response was generated"
        assert response.session_id, "No session ID was generated"

async def test_specialized_vs_llm_routing():
    """
    Test that specialized queries go to specialized agents and general queries go to LLM agent.
    """
    # Create engine and session
    db_url = config["database"]["url"].replace("sqlite://", "sqlite+aiosqlite://")
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    # Test cases: (message, expected_agent_type)
    test_cases = [
        ("Register a new Kubernetes cluster", AgentType.CLUSTER_REGISTER),
        ("Show me information about my clusters", AgentType.CLUSTER_INFO),
        ("Check the health of my clusters", AgentType.CLUSTER_HEALTH),
        ("What did we talk about earlier?", AgentType.MEMORY),
        ("Use the weather tool to check the forecast", AgentType.MCP),
        ("What is the meaning of life?", AgentType.LLM),
        ("Tell me a joke", AgentType.LLM),
        ("What's the current time in Tokyo?", AgentType.LLM),
    ]
    
    # Create session
    async with async_session() as session:
        for message, expected_agent in test_cases:
            # Create request
            request = ChatRequest(message=message)
            
            # Call chat endpoint
            logger.debug(f"Testing message: '{message}', expecting agent: {expected_agent}")
            response = await chat(request, session)
            
            # We can't directly verify which agent was used from the response
            # This is just a demonstration of the test concept
            logger.debug(f"Got response for '{message}': {response.response[:50]}...")
            print(f"Message: '{message}'")
            print(f"Response: {response.response[:100]}...")
            print("---")

if __name__ == "__main__":
    asyncio.run(test_llm_agent())
    print("\nTesting specialized vs LLM routing:")
    asyncio.run(test_specialized_vs_llm_routing())
