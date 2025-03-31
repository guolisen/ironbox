"""
Framework selector for IronBox.

This module provides a LangChain-based framework selector for determining which agent
framework should handle a given query.
"""
import logging
from typing import Dict, Any, List, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from ironbox.core.llm import default_llm
from ironbox.core.graph import AgentState

# Configure logging
logger = logging.getLogger(__name__)


class FrameworkSelection(BaseModel):
    """Framework selection result."""
    
    framework_type: Literal["route", "react", "plan", "direct"] = Field(
        ..., description="The selected framework type"
    )
    confidence: float = Field(
        ..., description="Confidence score between 0 and 1", ge=0, le=1
    )
    reasoning: str = Field(..., description="Reasoning behind the selection")


class FrameworkSelector:
    """
    Selector for choosing the appropriate agent framework based on the query type.
    This implementation uses LangChain components.
    """
    
    def __init__(self, llm=default_llm, config: Dict[str, Any] = None):
        """
        Initialize FrameworkSelector.
        
        Args:
            llm: LLM instance
            config: Configuration dictionary
        """
        self.llm = llm
        self.config = config or {}
        self.system_prompt = self.config.get("system_prompt", """
        You are a framework selector for the IronBox system. Your job is to analyze the user's request and determine which agent framework should handle it.
        
        Available frameworks:
        - route: The original framework that routes queries to specialized agents. Good for simple queries that fit into predefined categories.
        - react: A framework that uses the React paradigm (Reason + Act). Good for problems that require reasoning and action.
        - plan: A framework that creates a plan before execution. Good for complex problems that require planning.
        - direct: Direct LLM response without using any framework. Good for simple questions that don't require special handling.
        
        Analyze the following query and respond with the name of the framework that should handle it.
        
        Query: {query}
        
        Respond with just the framework name (route, react, plan, or direct).
        """)
    
    async def select_framework(self, query: str) -> str:
        """
        Select the appropriate framework for the query.
        
        Args:
            query: The user query
            
        Returns:
            Framework type
        """
        try:
            logger.debug("Selecting framework for query: %s", query)
            
            # Create prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt.format(query=query)),
            ])
            
            # Set up chain
            chain = prompt | self.llm | StrOutputParser()
            
            # Invoke chain
            response_text = await chain.ainvoke({})
            logger.debug("Framework selection response: %s", response_text)
            
            # Clean up response
            response_text = response_text.strip().lower()
            
            # Map response to framework type
            if "route" in response_text:
                return "route"
            elif "react" in response_text:
                return "react"
            elif "plan" in response_text:
                return "plan"
            elif "direct" in response_text:
                return "direct"
            else:
                # Default to route
                return "route"
            
        except Exception as e:
            logger.error(f"Error selecting framework: {e}")
            # Default to route on error
            return "route"
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Process the state and select a framework.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            logger.debug("FrameworkSelector called with input: %s", state.input)
            
            # Select framework
            framework_type = await self.select_framework(state.input)
            logger.debug("Selected framework: %s", framework_type)
            
            # Update state
            state.current_agent = "framework_selector"
            state.agent_outputs["framework_selector"] = {
                "response": f"Selected framework: {framework_type}",
                "next": framework_type,
                "framework_type": framework_type,
            }
            
            return state
        except Exception as e:
            logger.error(f"Error in framework selector: {e}")
            state.error = f"Framework selector error: {str(e)}"
            return state
