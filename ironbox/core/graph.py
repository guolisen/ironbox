"""
LangGraph orchestration for IronBox.
"""
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union, Annotated, TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langchain.schema import AIMessage

from ironbox.core.llm import default_llm

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Define state schema
class AgentState(BaseModel):
    """State for agent graph."""
    
    # Input from user
    input: str = Field(default="")
    
    # Chat history
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    
    # Current agent
    current_agent: Optional[str] = Field(default=None)
    
    # Agent outputs
    agent_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Final response
    response: Optional[str] = Field(default=None)
    
    # Error information
    error: Optional[str] = Field(default=None)
    
    # Session ID
    session_id: Optional[str] = Field(default=None)
    
    # Execution count to prevent infinite loops
    execution_count: Dict[str, int] = Field(default_factory=dict)


# Define agent types
class AgentType:
    """Agent types."""
    
    ROUTER = "router"
    CLUSTER_REGISTER = "cluster_register"
    CLUSTER_INFO = "cluster_info"
    CLUSTER_HEALTH = "cluster_health"
    MEMORY = "memory"
    MCP = "mcp"
    LLM = "llm"


class AgentOutput(TypedDict):
    """Output from an agent."""
    
    response: str
    next: Optional[str]
    error: Optional[str]


class AgentGraph:
    """LangGraph orchestration for agents."""
    
    def __init__(self, agents: Dict[str, Any]):
        """
        Initialize AgentGraph.
        
        Args:
            agents: Dictionary of agent instances
        """
        self.agents = agents
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the agent graph.
        
        Returns:
            StateGraph instance
        """
        # Create the graph
        graph = StateGraph(AgentState)
        
        # Add nodes for each agent
        for agent_name, agent in self.agents.items():
            graph.add_node(agent_name, agent)
        
        # Add router as the entry point
        graph.set_entry_point(AgentType.ROUTER)
        
        # Define conditional routing
        graph.add_conditional_edges(
            AgentType.ROUTER,
            self._route_from_router,
        )
        
        # Add edges from other agents to END
        for agent_name in self.agents.keys():
            if agent_name != AgentType.ROUTER:
                # All non-router agents go directly to END
                graph.add_edge(agent_name, END)
        
        # Compile the graph
        return graph.compile()
    
    def _route_from_router(self, state: AgentState) -> str:
        """
        Route from router agent.
        
        Args:
            state: Current state
            
        Returns:
            Next agent name or END
        """
        # Get the next agent from router output
        agent_outputs = state.agent_outputs
        if AgentType.ROUTER not in agent_outputs:
            return END
        
        router_output = agent_outputs[AgentType.ROUTER]
        next_agent = router_output.get("next")
        
        # Check if next agent is valid
        if next_agent and next_agent in self.agents:
            # Track execution count to prevent infinite loops
            if next_agent not in state.execution_count:
                state.execution_count[next_agent] = 0
            
            # Increment execution count
            state.execution_count[next_agent] += 1
            
            # If execution count exceeds limit, end the graph
            if state.execution_count[next_agent] > 1:
                logger.warning(f"Execution count for {next_agent} exceeded limit. Ending graph.")
                return END
            
            return next_agent
        
        # If no valid next agent, end the graph
        return END
    
    async def invoke(self, input_text: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke the agent graph.
        
        Args:
            input_text: Input text from user
            session_id: Optional session ID
            
        Returns:
            Result of graph execution
        """
        try:
            # Create initial state with session_id
            state = AgentState(input=input_text, session_id=session_id)
            logger.debug("Created initial state: %s", state)
            
            # Run the graph
            logger.debug("Running graph with input: %s", input_text)
            result = await self.graph.ainvoke(state)
            logger.debug("Graph execution result: %s", result)
            
            # Extract response and agent outputs directly from the result object
            response = None
            agent_outputs = {}
            error = None
            
            # Try to get the agent_outputs directly from the result
            try:
                # Try to access agent_outputs directly
                agent_outputs = result.agent_outputs
                logger.debug("Got agent_outputs directly: %s", agent_outputs)
            except (AttributeError, TypeError):
                # If that fails, try to convert the result to a dict
                try:
                    result_dict = dict(result)
                    agent_outputs = result_dict.get('agent_outputs', {})
                    logger.debug("Got agent_outputs from dict: %s", agent_outputs)
                except (TypeError, ValueError):
                    # If that fails too, try to access it as a property
                    try:
                        agent_outputs = getattr(result, 'agent_outputs', {})
                        logger.debug("Got agent_outputs from getattr: %s", agent_outputs)
                    except Exception as e:
                        logger.error(f"Failed to get agent_outputs: {e}")
            
            # Try to get the error directly from the result
            try:
                error = result.error
            except (AttributeError, TypeError):
                try:
                    result_dict = dict(result)
                    error = result_dict.get('error')
                except (TypeError, ValueError):
                    try:
                        error = getattr(result, 'error', None)
                    except Exception as e:
                        logger.error(f"Failed to get error: {e}")
            
            # Try to get the response directly from the result
            try:
                if hasattr(result, 'response') and result.response:
                    response = result.response
            except Exception:
                pass
            
            # Prioritize getting the response from the cluster_health agent
            if "cluster_health" in agent_outputs:
                response = agent_outputs["cluster_health"].get("response", "")
                logger.debug("Using cluster_health response: %s", response)
            # If no cluster_health response, try to get it from the last agent output
            elif not response and agent_outputs:
                # Try to get the current agent
                last_agent = None
                try:
                    last_agent = result.current_agent
                except (AttributeError, TypeError):
                    try:
                        result_dict = dict(result)
                        last_agent = result_dict.get('current_agent')
                    except (TypeError, ValueError):
                        try:
                            last_agent = getattr(result, 'current_agent', None)
                        except Exception as e:
                            logger.error(f"Failed to get current_agent: {e}")
                
                # If we have a last agent and it's in agent_outputs, get its response
                if last_agent and last_agent in agent_outputs:
                    response = agent_outputs[last_agent].get("response", "")
                else:
                    # Try to find any agent output with a response
                    for agent_name, agent_output in agent_outputs.items():
                        if isinstance(agent_output, dict) and agent_output.get("response"):
                            response = agent_output.get("response")
                            break
                
                # If we still don't have a response but we have a router response, use that
                if not response and "router" in agent_outputs:
                    response = agent_outputs["router"].get("response", "")
            
            logger.debug("Extracted response: %s", response)
            logger.debug("Extracted agent_outputs: %s", agent_outputs)
            
            return {
                "response": response or "I couldn't process your request.",
                "agent_outputs": agent_outputs,
                "error": error,
            }
        except Exception as e:
            logger.error(f"Error invoking agent graph: {e}")
            logger.error(traceback.format_exc())
            return {
                "response": "An error occurred while processing your request.",
                "error": str(e),
            }


class RouterAgent:
    """Router agent for directing requests to specialized agents."""
    
    def __init__(self, llm=default_llm):
        """
        Initialize RouterAgent.
        
        Args:
            llm: LLM instance
        """
        self.llm = llm
        self.system_prompt = """
        You are a router agent for the IronBox system. Your job is to analyze the user's request and determine which specialized agent should handle it.
        
        Available agents:
        - cluster_register: For registering new Kubernetes clusters
        - cluster_info: For getting information about registered clusters
        - cluster_health: For checking the health of clusters
        - memory: For retrieving information from memory
        - mcp: For using MCP tools
        - llm: For general queries that don't align with any of the specialized agents above
        
        Respond with the name of the agent that should handle the request. If the query doesn't align with any of the specialized areas (cluster registration, cluster information, health checks, memory retrieval, or MCP tools), route it to the llm agent.
        """
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Route the user request to the appropriate agent.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            logger.debug("RouterAgent called with input: %s", state.input)
            
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
            
            # Determine next agent
            next_agent = None
            if "cluster_register" in response_text.lower():
                next_agent = AgentType.CLUSTER_REGISTER
            elif "cluster_info" in response_text.lower():
                next_agent = AgentType.CLUSTER_INFO
            elif "cluster_health" in response_text.lower():
                next_agent = AgentType.CLUSTER_HEALTH
            elif "memory" in response_text.lower():
                next_agent = AgentType.MEMORY
            elif "mcp" in response_text.lower():
                next_agent = AgentType.MCP
            elif "llm" in response_text.lower():
                next_agent = AgentType.LLM
            
            # Update state
            state.current_agent = AgentType.ROUTER
            state.agent_outputs[AgentType.ROUTER] = {
                "response": response_text,
                "next": next_agent,
            }
            
            return state
        except Exception as e:
            logger.error(f"Error in router agent: {e}")
            state.error = f"Router agent error: {str(e)}"
            return state


def create_agent_graph(agents: Optional[Dict[str, Any]] = None) -> AgentGraph:
    """
    Create an agent graph.
    
    Args:
        agents: Optional dictionary of agent instances
        
    Returns:
        AgentGraph instance
    """
    # Use provided agents or create default ones
    if not agents:
        agents = {
            AgentType.ROUTER: RouterAgent(),
            # Other agents will be added as they are implemented
        }
    
    return AgentGraph(agents)
