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
from ironbox.core.toolkit import Toolkit

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
    
    def __init__(self, llm=default_llm, config: Dict[str, Any] = None, toolkit: Toolkit = None):
        """
        Initialize FrameworkSelector.
        
        Args:
            llm: LLM instance
            config: Configuration dictionary
            toolkit: Toolkit instance for accessing available tools
        """
        self.llm = llm
        self.config = config or {}
        self.toolkit = toolkit
        
        # Get tool descriptions from toolkit if available
        tool_descriptions = ""
        if self.toolkit:
            tools_list = list(self.toolkit.list_tools().keys())
            tool_descriptions = ", ".join(tools_list)
        
        self.system_prompt = self.config.get("system_prompt", """
        You are a framework selector for the IronBox Kubernetes management system. Your job is to analyze the user's request and determine which agent framework should handle it.

        ## Available Frameworks:

        1. route: Routes queries to specialized agents. Best for simple, categorizable queries that match a specific agent's expertise.
        2. react: Uses the ReAct paradigm (Reason + Act). Best for multi-step problems requiring reasoning and tool usage.
        3. plan: Creates a detailed plan before execution. Best for complex problems requiring structured planning.
        4. direct: Direct LLM response without using frameworks or tools. Best for simple informational queries.

        ## Available Specialized Agents:

        - cluster_info: Provides information about registered Kubernetes clusters (list clusters, get details, health history)
        - cluster_register: Registers new Kubernetes clusters with the system
        - cluster_health: Performs health checks on Kubernetes clusters
        - memory: Retrieves information from conversation history and function calls
        - mcp: Interacts with Model Context Protocol servers (list servers, use tools, access resources)
        - llm: Handles general queries that don't match specialized agents

        ## Available Tools:

        IronBox supports the following tools: {tool_descriptions}
        
        If you need tools to solve the query problem, you need to choose a framework from one of: route, react, plan. Otherwise, choose direct.

        ## Framework Selection Guidelines:

        ### Use 'route' framework when:
        - The query clearly matches a specialized agent's domain
        - The request is straightforward and fits into predefined categories
        - The user is asking for specific information that a specialized agent can provide
        - Example queries:
          * "List all my Kubernetes clusters"
          * "Register a new cluster with API server at https://k8s.example.com"
          * "Check the health of my production cluster"
          * "What did we discuss yesterday about cluster scaling?"
          * "List available MCP servers"

        ### Use 'react' framework when:
        - The query requires multiple steps of reasoning and tool usage
        - The problem needs dynamic decision-making based on intermediate results
        - The request involves conditional logic or branching
        - Example queries:
          * "Find pods with high CPU usage and restart them"
          * "Compare resource usage across my dev and prod clusters"
          * "Analyze network traffic patterns in my cluster and suggest optimizations"
          * "Find and fix configuration issues in my deployment"

        ### Use 'plan' framework when:
        - The query involves a complex, multi-stage process
        - The task requires upfront planning before execution
        - The problem has dependencies between steps
        - Example queries:
          * "Migrate my application from cluster A to cluster B with zero downtime"
          * "Scale my deployment based on a schedule and resource availability"
          * "Implement a blue-green deployment strategy for my application"
          * "Create a disaster recovery plan for my Kubernetes infrastructure"

        ### Use 'direct' framework when:
        - The query is a simple informational question
        - The request doesn't require specialized tools or agents
        - The user is asking for explanations, concepts, or general advice
        - Example queries:
          * "What is a Kubernetes pod?"
          * "Explain the difference between a deployment and a statefulset"
          * "What are best practices for Kubernetes security?"
          * "How does Kubernetes handle load balancing?"

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
            
            # Ensure we're setting a proper framework_type that matches a registered framework
            # This is critical for the routing logic in LangGraphOrchestrator.build_graph
            state.agent_outputs["framework_selector"] = {
                "response": f"Selected framework: {framework_type}",
                "next": framework_type,
                "framework_type": framework_type,  # This must match a key in framework_routes
            }
            
            logger.debug("Updated state with framework_type: %s", framework_type)
            return state
        except Exception as e:
            logger.error(f"Error in framework selector: {e}")
            state.error = f"Framework selector error: {str(e)}"
            return state
    
    # Add process method to match BaseLCAgentFramework interface
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the state and select a framework.
        This method is used by the LangGraphOrchestrator.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        return await self.__call__(state)
