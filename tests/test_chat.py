import asyncio
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from ironbox.api.routes import chat, ChatRequest
from ironbox.config import config

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_chat():
    # Create engine and session
    db_url = config["database"]["url"].replace("sqlite://", "sqlite+aiosqlite://")
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    # Create session
    async with async_session() as session:
        # Create request
        request = ChatRequest(message="Tell me about the health of my clusters")
        
        # Call chat endpoint
        logger.debug("Calling chat endpoint with message: %s", request.message)
        response = await chat(request, session)
        
        # Print response
        logger.debug("Got response: %s", response)
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(test_chat())
