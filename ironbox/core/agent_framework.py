"""
Agent framework for IronBox.

This module defines the base agent framework and different agent framework implementations.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from pydantic import BaseModel, Field
from langchain.schema import AIMessage

from ironbox.core.llm import default_llm
from ironbox.core.graph import AgentState

# Configure logging
logger = logging.getLogger(__name__)


class AgentFramework(ABC):
    """Base agent framework class."""
    
    def __init__(self, llm=default_llm):
        """
        Initialize AgentFramework.
        
        Args:
            llm: LLM instance
        """
        self.llm = llm
    
    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the user query using this framework.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        pass
    
    @classmethod
    def get_framework_type(cls) -> str:
        """
        Get the framework type.
        
        Returns:
            Framework type string
        """
        return cls.__name__


class RouteAgentFramework(AgentFramework):
    """
    Route agent framework that routes queries to specialized agents.
    This is the original framework used in IronBox.
    """
    
    def __init__(self, llm=default_llm, agents: Dict[str, Any] = None):
        """
        Initialize RouteAgentFramework.
        
        Args:
            llm: LLM instance
            agents: Dictionary of agent instances
        """
        super().__init__(llm)
        self.agents = agents or {}
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
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the user query by routing to specialized agents.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            logger.debug("RouteAgentFramework processing input: %s", state.input)
            
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
            from ironbox.core.graph import AgentType
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
            state.current_agent = "router"
            state.agent_outputs["router"] = {
                "response": response_text,
                "next": next_agent,
            }
            
            # If we have a next agent and it's in our agents dictionary, call it
            if next_agent and next_agent in self.agents:
                agent = self.agents[next_agent]
                state = await agent(state)
            
            return state
        except Exception as e:
            logger.error(f"Error in RouteAgentFramework: {e}")
            state.error = f"RouteAgentFramework error: {str(e)}"
            return state
    
    @classmethod
    def get_framework_type(cls) -> str:
        return "route"


class ReactStep(BaseModel):
    """A step in the React framework."""
    
    thought: str = Field(..., description="Reasoning about the current state")
    action: str = Field(..., description="Action to take")
    action_input: Dict[str, Any] = Field(..., description="Input for the action")
    observation: Optional[str] = Field(None, description="Observation from the action")


class ReactAgentFramework(AgentFramework):
    """
    React agent framework that uses the React paradigm (Reason + Act).
    This framework is suitable for problems that require reasoning and action.
    """
    
    def __init__(self, llm=default_llm, tools: Dict[str, Callable] = None):
        """
        Initialize ReactAgentFramework.
        
        Args:
            llm: LLM instance
            tools: Dictionary of tool functions
        """
        super().__init__(llm)
        self.tools = tools or {}
        self.system_prompt = """
        You are a React agent for the IronBox system. You solve problems by thinking step-by-step and taking actions.
        
        Follow this format:
        
        Thought: Reason about the current state and what to do next
        Action: The action to take (must be one of the available tools)
        Action Input: The input to the action as a JSON object
        Observation: The result of the action
        
        Available tools:
        {tool_descriptions}
        
        Continue this process until you have solved the problem, then respond with:
        
        Thought: I have solved the problem
        Final Answer: The final answer or solution
        """
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the user query using the React paradigm.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            logger.debug("ReactAgentFramework processing input: %s", state.input)
            
            # Create tool descriptions
            tool_descriptions = "\n".join([
                f"- {name}: {tool.__doc__ or 'No description'}"
                for name, tool in self.tools.items()
            ])
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": self.system_prompt.format(tool_descriptions=tool_descriptions)},
                {"role": "user", "content": state.input},
            ]
            
            # Add chat history for context
            for message in state.chat_history[-5:]:  # Last 5 messages
                messages.append(message)
            
            # Initialize steps
            steps = []
            max_steps = 10  # Prevent infinite loops
            
            # Execute React loop
            for _ in range(max_steps):
                # Get response from LLM
                response = await self.llm.ainvoke(messages)
                
                # Extract response text
                if hasattr(response, 'generations'):
                    response_text = response.generations[0].message.content
                elif isinstance(response, AIMessage):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Check if we have a final answer
                if "Final Answer:" in response_text:
                    # Extract final answer
                    final_answer = response_text.split("Final Answer:")[1].strip()
                    
                    # Update state
                    state.current_agent = "react"
                    state.agent_outputs["react"] = {
                        "response": final_answer,
                        "steps": steps,
                        "next": None,
                    }
                    
                    return state
                
                # Parse React step
                try:
                    thought = response_text.split("Thought:")[1].split("Action:")[0].strip()
                    action = response_text.split("Action:")[1].split("Action Input:")[0].strip()
                    action_input_text = response_text.split("Action Input:")[1].split("Observation:" if "Observation:" in response_text else "Final Answer:" if "Final Answer:" in response_text else "$$$")[0].strip()
                    
                    # Parse action input as JSON
                    import json
                    action_input = json.loads(action_input_text)
                    
                    # Create step
                    step = ReactStep(
                        thought=thought,
                        action=action,
                        action_input=action_input,
                    )
                    
                    # Execute action if it's a valid tool
                    observation = "Tool not found"
                    if action in self.tools:
                        tool = self.tools[action]
                        try:
                            observation = await tool(**action_input)
                        except Exception as e:
                            observation = f"Error executing tool: {str(e)}"
                    
                    # Update step with observation
                    step.observation = observation
                    steps.append(step.dict())
                    
                    # Add step to messages
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                except Exception as e:
                    logger.error(f"Error parsing React step: {e}")
                    observation = f"Error parsing step: {str(e)}"
                    messages.append({"role": "user", "content": f"Error: {observation}\nPlease follow the correct format."})
            
            # If we reach here, we've hit the max steps
            state.current_agent = "react"
            state.agent_outputs["react"] = {
                "response": "I couldn't solve the problem within the maximum number of steps.",
                "steps": steps,
                "next": None,
            }
            
            return state
        except Exception as e:
            logger.error(f"Error in ReactAgentFramework: {e}")
            state.error = f"ReactAgentFramework error: {str(e)}"
            return state
    
    @classmethod
    def get_framework_type(cls) -> str:
        return "react"


class PlanStep(BaseModel):
    """A step in a plan."""
    
    description: str = Field(..., description="Description of the step")
    completed: bool = Field(False, description="Whether the step is completed")
    result: Optional[str] = Field(None, description="Result of executing the step")


class PlanAgentFramework(AgentFramework):
    """
    Plan agent framework that creates a plan before execution.
    This framework is suitable for complex problems that require planning.
    """
    
    def __init__(self, llm=default_llm, tools: Dict[str, Callable] = None):
        """
        Initialize PlanAgentFramework.
        
        Args:
            llm: LLM instance
            tools: Dictionary of tool functions
        """
        super().__init__(llm)
        self.tools = tools or {}
        self.planning_prompt = """
        You are a planning agent for the IronBox system. Your job is to create a detailed plan to solve the user's problem.
        
        Available tools:
        {tool_descriptions}
        
        Create a step-by-step plan to solve the following problem:
        {problem}
        
        Your plan should be detailed and include all the necessary steps to solve the problem.
        Format your response as a numbered list of steps.
        """
        
        self.execution_prompt = """
        You are an execution agent for the IronBox system. Your job is to execute a plan to solve the user's problem.
        
        Original problem:
        {problem}
        
        Plan:
        {plan}
        
        Current step:
        {current_step}
        
        Previous steps and results:
        {previous_steps}
        
        Available tools:
        {tool_descriptions}
        
        Execute the current step by selecting a tool and providing the necessary input.
        
        Format your response as:
        
        Tool: The tool to use
        Tool Input: The input to the tool as a JSON object
        """
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the user query by planning and then executing.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            logger.debug("PlanAgentFramework processing input: %s", state.input)
            
            # Create tool descriptions
            tool_descriptions = "\n".join([
                f"- {name}: {tool.__doc__ or 'No description'}"
                for name, tool in self.tools.items()
            ])
            
            # Step 1: Create a plan
            plan = await self._create_plan(state.input, tool_descriptions)
            logger.debug("Created plan: %s", plan)
            
            # Step 2: Execute the plan
            results = await self._execute_plan(state.input, plan, tool_descriptions)
            logger.debug("Plan execution results: %s", results)
            
            # Step 3: Generate final response
            final_response = await self._generate_final_response(state.input, plan, results)
            logger.debug("Final response: %s", final_response)
            
            # Update state
            state.current_agent = "plan"
            state.agent_outputs["plan"] = {
                "response": final_response,
                "plan": plan,
                "results": results,
                "next": None,
            }
            
            return state
        except Exception as e:
            logger.error(f"Error in PlanAgentFramework: {e}")
            state.error = f"PlanAgentFramework error: {str(e)}"
            return state
    
    async def _create_plan(self, problem: str, tool_descriptions: str) -> List[PlanStep]:
        """
        Create a plan to solve the problem.
        
        Args:
            problem: The problem to solve
            tool_descriptions: Descriptions of available tools
            
        Returns:
            List of plan steps
        """
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": self.planning_prompt.format(
                tool_descriptions=tool_descriptions,
                problem=problem,
            )},
        ]
        
        # Get response from LLM
        response = await self.llm.ainvoke(messages)
        
        # Extract response text
        if hasattr(response, 'generations'):
            response_text = response.generations[0].message.content
        elif isinstance(response, AIMessage):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Parse plan steps
        steps = []
        for line in response_text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("- ")):
                # Remove numbering or bullet points
                step_text = line.split(".", 1)[1].strip() if "." in line and line[0].isdigit() else line[2:].strip()
                steps.append(PlanStep(description=step_text))
        
        return steps
    
    async def _execute_plan(self, problem: str, plan: List[PlanStep], tool_descriptions: str) -> List[PlanStep]:
        """
        Execute the plan.
        
        Args:
            problem: The original problem
            plan: The plan to execute
            tool_descriptions: Descriptions of available tools
            
        Returns:
            Updated plan with results
        """
        for i, step in enumerate(plan):
            # Prepare previous steps text
            previous_steps = ""
            for j, prev_step in enumerate(plan[:i]):
                if prev_step.completed:
                    previous_steps += f"{j+1}. {prev_step.description}\nResult: {prev_step.result}\n\n"
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": self.execution_prompt.format(
                    problem=problem,
                    plan="\n".join([f"{j+1}. {s.description}" for j, s in enumerate(plan)]),
                    current_step=f"{i+1}. {step.description}",
                    previous_steps=previous_steps,
                    tool_descriptions=tool_descriptions,
                )},
            ]
            
            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            
            # Extract response text
            if hasattr(response, 'generations'):
                response_text = response.generations[0].message.content
            elif isinstance(response, AIMessage):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Parse tool and input
            try:
                tool_name = response_text.split("Tool:")[1].split("Tool Input:")[0].strip()
                tool_input_text = response_text.split("Tool Input:")[1].strip()
                
                # Parse tool input as JSON
                import json
                tool_input = json.loads(tool_input_text)
                
                # Execute tool
                result = "Tool not found"
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    try:
                        result = await tool(**tool_input)
                    except Exception as e:
                        result = f"Error executing tool: {str(e)}"
                
                # Update step
                step.completed = True
                step.result = result
            except Exception as e:
                logger.error(f"Error executing step: {e}")
                step.completed = True
                step.result = f"Error: {str(e)}"
        
        return plan
    
    async def _generate_final_response(self, problem: str, plan: List[PlanStep], results: List[PlanStep]) -> str:
        """
        Generate a final response based on the plan execution.
        
        Args:
            problem: The original problem
            plan: The original plan
            results: The plan with results
            
        Returns:
            Final response
        """
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": f"""
            You are a response agent for the IronBox system. Your job is to generate a final response based on the execution of a plan.
            
            Original problem:
            {problem}
            
            Plan and results:
            {chr(10).join([f"{i+1}. {step.description}\nResult: {step.result}" for i, step in enumerate(results)])}
            
            Generate a comprehensive and helpful response that summarizes the results and provides a solution to the original problem.
            """}, 
        ]
        
        # Get response from LLM
        response = await self.llm.ainvoke(messages)
        
        # Extract response text
        if hasattr(response, 'generations'):
            response_text = response.generations[0].message.content
        elif isinstance(response, AIMessage):
            response_text = response.content
        else:
            response_text = str(response)
        
        return response_text
    
    @classmethod
    def get_framework_type(cls) -> str:
        return "plan"


class FrameworkSelector:
    """
    Selector for choosing the appropriate agent framework based on the query type.
    """
    
    def __init__(self, llm=default_llm):
        """
        Initialize FrameworkSelector.
        
        Args:
            llm: LLM instance
        """
        self.llm = llm
        self.system_prompt = """
        You are a framework selector for the IronBox system. Your job is to analyze the user's request and determine which agent framework should handle it.
        
        Available frameworks:
        - route: The original framework that routes queries to specialized agents. Good for simple queries that fit into predefined categories.
        - react: A framework that uses the React paradigm (Reason + Act). Good for problems that require reasoning and action.
        - plan: A framework that creates a plan before execution. Good for complex problems that require planning.
        - direct: Direct LLM response without using any framework. Good for simple questions that don't require special handling.
        
        Analyze the following query and respond with the name of the framework that should handle it.
        
        Query: {query}
        
        Respond with just the framework name (route, react, plan, or direct).
        """
    
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
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": self.system_prompt.format(query=query)},
            ]
            
            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            
            # Extract response text
            if hasattr(response, 'generations'):
                response_text = response.generations[0].message.content
            elif isinstance(response, AIMessage):
                response_text = response.content
            else:
                response_text = str(response)
            
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
