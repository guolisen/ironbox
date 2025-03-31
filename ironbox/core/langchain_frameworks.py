"""
LangChain-based agent frameworks for IronBox.

This module defines the base agent framework and different agent framework implementations
using LangChain and LangGraph components.
"""
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, TypedDict, Type

from pydantic import BaseModel, Field
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent, create_openai_functions_agent
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langgraph.graph import StateGraph, END

from ironbox.core.llm import default_llm
from ironbox.core.graph import AgentState

# Configure logging
logger = logging.getLogger(__name__)


class BaseLCAgentFramework(ABC):
    """Base LangChain agent framework class."""
    
    def __init__(self, llm=default_llm, tools: Dict[str, Callable] = None, config: Dict[str, Any] = None):
        """
        Initialize BaseLCAgentFramework.
        
        Args:
            llm: LLM instance
            tools: Dictionary of tool functions
            config: Configuration dictionary
        """
        self.llm = llm
        self.tools = tools or {}
        self.config = config or {}
        self.langchain_tools = self._create_langchain_tools()
    
    def _create_langchain_tools(self) -> List[BaseTool]:
        """
        Create LangChain tools from the tools dictionary.
        
        Returns:
            List of LangChain tools
        """
        langchain_tools = []
        
        for name, func in self.tools.items():
            # Create a LangChain tool from the function
            @tool(name=name, description=func.__doc__ or f"Call {name}")
            async def dynamic_tool(*args, **kwargs):
                return await func(*args, **kwargs)
            
            # Set the name and docstring
            dynamic_tool.__name__ = name
            dynamic_tool.__doc__ = func.__doc__ or f"Call {name}"
            
            langchain_tools.append(dynamic_tool)
        
        return langchain_tools
    
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


class LCRouteAgentFramework(BaseLCAgentFramework):
    """
    LangChain route agent framework that routes queries to specialized agents.
    This framework is based on LangChain components.
    """
    
    def __init__(self, llm=default_llm, agents: Dict[str, Any] = None, config: Dict[str, Any] = None):
        """
        Initialize LCRouteAgentFramework.
        
        Args:
            llm: LLM instance
            agents: Dictionary of agent instances
            config: Configuration dictionary
        """
        super().__init__(llm=llm, config=config)
        self.agents = agents or {}
        self.system_prompt = config.get("system_prompt", """
        You are a router agent for the IronBox system. Your job is to analyze the user's request and determine which specialized agent should handle it.
        
        Available agents:
        {agent_descriptions}
        
        Respond with the name of the agent that should handle the request.
        """)
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the user query by routing to specialized agents.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            logger.debug("LCRouteAgentFramework processing input: %s", state.input)
            
            # Create agent descriptions
            agent_descriptions = []
            for name, agent in self.agents.items():
                description = getattr(agent, "__doc__", None) or f"Agent for {name}"
                agent_descriptions.append(f"- {name}: {description}")
            
            agent_descriptions_text = "\n".join(agent_descriptions)
            
            # Create prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt.format(agent_descriptions=agent_descriptions_text)),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            # Set up chain
            chain = prompt | self.llm | StrOutputParser()
            
            # Prepare inputs
            inputs = {
                "input": state.input,
                "chat_history": [
                    HumanMessage(content=msg["content"]) if msg["role"] == "user" else 
                    AIMessage(content=msg["content"]) if msg["role"] == "assistant" else
                    SystemMessage(content=msg["content"])
                    for msg in state.chat_history[-5:]  # Last 5 messages
                ]
            }
            
            # Invoke chain
            response_text = await chain.ainvoke(inputs)
            logger.debug("Router response: %s", response_text)
            
            # Determine next agent
            next_agent = None
            for agent_name in self.agents.keys():
                if agent_name.lower() in response_text.lower():
                    next_agent = agent_name
                    break
            
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
            logger.error(f"Error in LCRouteAgentFramework: {e}")
            state.error = f"LCRouteAgentFramework error: {str(e)}"
            return state
    
    @classmethod
    def get_framework_type(cls) -> str:
        return "route"


class LCReactStep(BaseModel):
    """A step in the LangChain React framework."""
    
    thought: Optional[str] = Field(None, description="Reasoning about the current state")
    action: Optional[str] = Field(None, description="Action to take")
    action_input: Optional[Dict[str, Any]] = Field(None, description="Input for the action")
    observation: Optional[str] = Field(None, description="Observation from the action")


class LCReactAgentFramework(BaseLCAgentFramework):
    """
    LangChain React agent framework that uses the ReAct paradigm.
    This framework is based on LangChain components.
    """
    
    def __init__(self, llm=default_llm, tools: Dict[str, Callable] = None, config: Dict[str, Any] = None):
        """
        Initialize LCReactAgentFramework.
        
        Args:
            llm: LLM instance
            tools: Dictionary of tool functions
            config: Configuration dictionary
        """
        super().__init__(llm=llm, tools=tools, config=config)
        self.system_prompt = config.get("system_prompt", """
        You are a React agent for the IronBox system. You solve problems by thinking step-by-step and taking actions.
        """)
        self.max_iterations = config.get("max_iterations", 10)
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the user query using the React paradigm.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            logger.debug("LCReactAgentFramework processing input: %s", state.input)
            
            # Create prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create agent
            agent = create_react_agent(self.llm, self.langchain_tools, prompt)
            
            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.langchain_tools,
                verbose=True,
                max_iterations=self.max_iterations,
                return_intermediate_steps=True,
            )
            
            # Prepare inputs
            inputs = {
                "input": state.input,
                "chat_history": [
                    HumanMessage(content=msg["content"]) if msg["role"] == "user" else
                    AIMessage(content=msg["content"]) if msg["role"] == "assistant" else
                    SystemMessage(content=msg["content"])
                    for msg in state.chat_history[-5:]  # Last 5 messages
                ]
            }
            
            # Execute agent
            result = await agent_executor.ainvoke(inputs)
            logger.debug("React agent result: %s", result)
            
            # Extract steps and output
            steps = []
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    step = LCReactStep(
                        action=action.tool,
                        action_input=action.tool_input,
                        observation=observation
                    )
                    steps.append(step.dict())
            
            # Update state
            state.current_agent = "react"
            state.agent_outputs["react"] = {
                "response": result["output"],
                "steps": steps,
                "next": None,
            }
            
            return state
        except Exception as e:
            logger.error(f"Error in LCReactAgentFramework: {e}")
            state.error = f"LCReactAgentFramework error: {str(e)}"
            return state
    
    @classmethod
    def get_framework_type(cls) -> str:
        return "react"


class LCPlanStep(BaseModel):
    """A step in a LangChain plan."""
    
    description: str = Field(..., description="Description of the step")
    completed: bool = Field(False, description="Whether the step is completed")
    result: Optional[str] = Field(None, description="Result of executing the step")


class LCPlanAgentFramework(BaseLCAgentFramework):
    """
    LangChain Plan agent framework that creates a plan before execution.
    This framework is based on LangChain components.
    """
    
    def __init__(self, llm=default_llm, tools: Dict[str, Callable] = None, config: Dict[str, Any] = None):
        """
        Initialize LCPlanAgentFramework.
        
        Args:
            llm: LLM instance
            tools: Dictionary of tool functions
            config: Configuration dictionary
        """
        super().__init__(llm=llm, tools=tools, config=config)
        self.planning_prompt = config.get("planning_prompt", """
        You are a planning agent for the IronBox system. Your job is to create a detailed plan to solve the user's problem.
        
        Available tools:
        {tool_descriptions}
        
        Create a step-by-step plan to solve the following problem:
        {problem}
        
        Your plan should be detailed and include all the necessary steps to solve the problem.
        Format your response as a numbered list of steps.
        """)
        
        self.execution_prompt = config.get("execution_prompt", """
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
        """)
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the user query by planning and then executing.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            logger.debug("LCPlanAgentFramework processing input: %s", state.input)
            
            # Create tool descriptions
            tool_descriptions = "\n".join([
                f"- {tool.name}: {tool.description}"
                for tool in self.langchain_tools
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
                "plan": [step.dict() for step in plan],
                "results": [step.dict() for step in results],
                "next": None,
            }
            
            return state
        except Exception as e:
            logger.error(f"Error in LCPlanAgentFramework: {e}")
            state.error = f"LCPlanAgentFramework error: {str(e)}"
            return state
    
    async def _create_plan(self, problem: str, tool_descriptions: str) -> List[LCPlanStep]:
        """
        Create a plan to solve the problem.
        
        Args:
            problem: The problem to solve
            tool_descriptions: Descriptions of available tools
            
        Returns:
            List of plan steps
        """
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.planning_prompt.format(
                tool_descriptions=tool_descriptions,
                problem=problem,
            )),
        ])
        
        # Set up chain
        chain = prompt | self.llm | StrOutputParser()
        
        # Invoke chain
        response_text = await chain.ainvoke({})
        
        # Parse plan steps
        steps = []
        for line in response_text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("- ")):
                # Remove numbering or bullet points
                step_text = line.split(".", 1)[1].strip() if "." in line and line[0].isdigit() else line[2:].strip()
                steps.append(LCPlanStep(description=step_text))
        
        return steps
    
    async def _execute_plan(self, problem: str, plan: List[LCPlanStep], tool_descriptions: str) -> List[LCPlanStep]:
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
            
            # Create prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.execution_prompt.format(
                    problem=problem,
                    plan="\n".join([f"{j+1}. {s.description}" for j, s in enumerate(plan)]),
                    current_step=f"{i+1}. {step.description}",
                    previous_steps=previous_steps,
                    tool_descriptions=tool_descriptions,
                )),
            ])
            
            # Set up chain
            chain = prompt | self.llm | StrOutputParser()
            
            # Invoke chain
            response_text = await chain.ainvoke({})
            
            # Parse tool and input
            try:
                tool_name = response_text.split("Tool:")[1].split("Tool Input:")[0].strip()
                tool_input_text = response_text.split("Tool Input:")[1].strip()
                
                # Parse tool input as JSON
                tool_input = json.loads(tool_input_text)
                
                # Find the tool by name
                tool_func = None
                for name, func in self.tools.items():
                    if name.lower() == tool_name.lower():
                        tool_func = func
                        break
                
                # Execute tool
                result = "Tool not found"
                if tool_func:
                    try:
                        result = await tool_func(**tool_input)
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
    
    async def _generate_final_response(self, problem: str, plan: List[LCPlanStep], results: List[LCPlanStep]) -> str:
        """
        Generate a final response based on the plan execution.
        
        Args:
            problem: The original problem
            plan: The original plan
            results: The plan with results
            
        Returns:
            Final response
        """
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            You are a response agent for the IronBox system. Your job is to generate a final response based on the execution of a plan.
            
            Original problem:
            {problem}
            
            Plan and results:
            {chr(10).join([f"{i+1}. {step.description}\nResult: {step.result}" for i, step in enumerate(results)])}
            
            Generate a comprehensive and helpful response that summarizes the results and provides a solution to the original problem.
            """),
        ])
        
        # Set up chain
        chain = prompt | self.llm | StrOutputParser()
        
        # Invoke chain
        response_text = await chain.ainvoke({})
        
        return response_text
    
    @classmethod
    def get_framework_type(cls) -> str:
        return "plan"


class LangGraphOrchestrator:
    """LangGraph orchestration for agent frameworks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize LangGraphOrchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.frameworks = {}
        self.graph = None
    
    def register_framework(self, name: str, framework: BaseLCAgentFramework):
        """
        Register a framework.
        
        Args:
            name: Framework name
            framework: Framework instance
        """
        self.frameworks[name] = framework
        logger.debug(f"Registered framework: {name}")
    
    def build_graph(self) -> StateGraph:
        """
        Build the agent graph.
        
        Returns:
            StateGraph instance
        """
        # Create the graph
        graph = StateGraph(AgentState)
        
        # Add nodes for each framework
        for name, framework in self.frameworks.items():
            graph.add_node(name, framework.process)
        
        # Add framework selector as the entry point
        graph.set_entry_point("framework_selector")
        
        # Define conditional routing based on configuration
        graph_config = self.config.get("graph", {})
        edges = graph_config.get("edges", [])
        
        for edge in edges:
            from_node = edge.get("from")
            to_node = edge.get("to")
            condition = edge.get("condition")
            
            if from_node and to_node:
                if condition:
                    # Add conditional edge
                    graph.add_conditional_edges(
                        from_node,
                        lambda state, condition=condition: eval(condition),
                        {
                            True: to_node,
                            False: END
                        }
                    )
                else:
                    # Add direct edge
                    graph.add_edge(from_node, to_node)
        
        # Add edges from other frameworks to END
        for name in self.frameworks.keys():
            if name != "framework_selector" and not any(edge.get("from") == name for edge in edges):
                # All non-selector frameworks with no explicit edges go directly to END
                graph.add_edge(name, END)
        
        # Compile the graph
        self.graph = graph.compile()
        return self.graph
    
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
            
            # Make sure graph is built
            if not self.graph:
                self.build_graph()
            
            # Run the graph
            logger.debug("Running graph with input: %s", input_text)
            result = await self.graph.ainvoke(state)
            logger.debug("Graph execution result: %s", result)
            
            # Extract response
            response = None
            error = None
            
            # Try to get the agent_outputs
            agent_outputs = getattr(result, 'agent_outputs', {})
            
            # Try to get the error
            error = getattr(result, 'error', None)
            
            # Try to get the response directly from the result
            if hasattr(result, 'response') and result.response:
                response = result.response
            
            # Try to get the response from the last agent output
            if not response and agent_outputs:
                # Try to get the current agent
                last_agent = getattr(result, 'current_agent', None)
                
                # If we have a last agent and it's in agent_outputs, get its response
                if last_agent and last_agent in agent_outputs:
                    response = agent_outputs[last_agent].get("response", "")
                else:
                    # Try to find any agent output with a response
                    for agent_name, agent_output in agent_outputs.items():
                        if isinstance(agent_output, dict) and agent_output.get("response"):
                            response = agent_output.get("response")
                            break
            
            return {
                "response": response or "I couldn't process your request.",
                "agent_outputs": agent_outputs,
                "error": error,
            }
        except Exception as e:
            logger.error(f"Error invoking agent graph: {e}")
            return {
                "response": "An error occurred while processing your request.",
                "error": str(e),
            }


class FrameworkRegistry:
    """Registry for agent frameworks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize FrameworkRegistry.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.frameworks = {}
        self.framework_types = {
            "route": LCRouteAgentFramework,
            "react": LCReactAgentFramework,
            "plan": LCPlanAgentFramework,
        }
        self.orchestrator = LangGraphOrchestrator(config=config)
    
    def register_framework_type(self, type_name: str, framework_class: Type[BaseLCAgentFramework]):
        """
        Register a framework type.
        
        Args:
            type_name: Framework type name
            framework_class: Framework class
        """
        self.framework_types[type_name] = framework_class
        logger.debug(f"Registered framework type: {type_name}")
    
    def load_from_config(self, toolkit, llm=default_llm):
        """
        Load frameworks from configuration.
        
        Args:
            toolkit: Toolkit instance
            llm: LLM instance
        """
        # Get frameworks configuration
        frameworks_config = self.config.get("agent_frameworks", [])
        
        for framework_config in frameworks_config:
            name = framework_config.get("name")
            type_name = framework_config.get("type")
            
            if not name or not type_name:
                logger.warning(f"Skipping framework with missing name or type: {framework_config}")
                continue
            
            # Get framework class
            if type_name not in self.framework_types:
                logger.warning(f"Unknown framework type: {type_name}")
                continue
            
            framework_class = self.framework_types[type_name]
            
            # Instantiate framework
            try:
                if type_name == "route":
                    framework = framework_class(
                        llm=llm,
                        agents=toolkit.agents,
                        config=framework_config.get("config", {})
                    )
                else:
                    framework = framework_class(
                        llm=llm,
                        tools=toolkit.tools,
                        config=framework_config.get("config", {})
                    )
                
                # Register framework with the registry
                self.frameworks[name] = framework
                
                # Register framework with the orchestrator
                self.orchestrator.register_framework(name, framework)
                
                logger.debug(f"Loaded framework from config: {name} ({type_name})")
            except Exception as e:
                logger.error(f"Error loading framework from config: {e}")
        
        # Build the orchestrator graph
        self.orchestrator.build_graph()
    
    async def process_query(self, input_text: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query.
        
        Args:
            input_text: User query
            session_id: Optional session ID
            
        Returns:
            Processing result
        """
        return await self.orchestrator.invoke(input_text, session_id)
