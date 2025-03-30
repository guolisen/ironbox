"""
LLM agent for handling general queries.
"""
import logging
from typing import Dict, Any, Optional

from langchain.schema import AIMessage
from ironbox.core.llm import default_llm
from ironbox.core.graph import AgentState

# Configure logging
logger = logging.getLogger(__name__)


class LLMAgent:
    """LLM agent for handling general queries that don't match specialized agents."""
    
    def __init__(self, llm=default_llm):
        """
        Initialize LLMAgent.
        
        Args:
            llm: LLM instance
        """
        self.llm = llm
        self.system_prompt = """
        You are an AI assistant for the IronBox Kubernetes management platform.
        
        You can help with general questions and tasks that don't require specialized Kubernetes knowledge.
        
        Provide helpful, accurate, and concise responses to user queries.
        """
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Process the user query and generate a response.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            logger.debug("LLMAgent called with input: %s", state.input)
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": state.input},
            ]
            
            # Add chat history for context
            for message in state.chat_history[-5:]:  # Last 5 messages
                messages.append(message)
            
            # Get response from LLM
            logger.debug("Calling LLM with messages: %s", messages)
            response = await self.llm.ainvoke(messages)
            logger.debug("LLM response: %s", response)
            
            # Extract response text
            if hasattr(response, 'generations'):
                # ChatResult object
                response_text = response.generations[0].message.content
            elif isinstance(response, AIMessage):
                # AIMessage object
                response_text = response.content
            else:
                # Fallback
                response_text = str(response)
            
            # Update state
            state.current_agent = "llm"
            state.agent_outputs["llm"] = {
                "response": response_text,
                "next": None,
            }
            
            return state
        except Exception as e:
            logger.error(f"Error in LLM agent: {e}")
            state.error = f"LLM agent error: {str(e)}"
            return state
