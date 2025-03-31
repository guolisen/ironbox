# LangChain and LangGraph Framework Selection

## Question: How does the system decide which agent framework to use?

The IronBox system uses LangChain and LangGraph to implement a sophisticated framework selection mechanism that analyzes user queries and routes them to the appropriate framework. Here's how it works:

## Framework Selection Process

1. When a query is received, it's passed to the `process_query` method in the `AgentCore` class
2. The query is then passed to the `FrameworkRegistry`, which initializes the LangGraph state
3. The first node in the LangGraph is the `FrameworkSelector`, which determines which framework to use
4. Based on the LLM's analysis, the query is routed through the graph to the appropriate framework
5. The selected framework processes the query and returns the result

## The Framework Selector

The core of this decision-making is in the `FrameworkSelector` class (in `ironbox/core/framework_selector.py`), which:

1. Takes the user query as input
2. Uses LangChain to create a prompt for the LLM
3. Analyzes the LLM's response to determine which framework to use
4. Updates the state with the selected framework type
5. Returns the updated state which is used by LangGraph for routing

Here's the relevant system prompt used:

```
You are a framework selector for the IronBox system. Your job is to analyze the user's request and determine which agent framework should handle it.

Available frameworks:
- route: The original framework that routes queries to specialized agents. Good for simple queries that fit into predefined categories.
- react: A framework that uses the React paradigm (Reason + Act). Good for problems that require reasoning and action.
- plan: A framework that creates a plan before execution. Good for complex problems that require planning.
- direct: Direct LLM response without using any framework. Good for simple questions that don't require special handling.
```

## Framework Selection Criteria

Each framework is designed for specific types of queries:

1. **Route Framework**
   - For simple categorizable queries
   - Used when the query fits into predefined categories like:
     - Cluster registration
     - Cluster information retrieval
     - Health checks
     - Memory retrieval
     - MCP operations

2. **React Framework**
   - For problems requiring reasoning and action
   - Used for:
     - Multi-step tasks with tool usage
     - Conditional logic based on observations
     - Interactive problem solving

3. **Plan Framework**
   - For complex multi-step problems
   - Used for:
     - Tasks requiring upfront planning
     - Complex logical problems
     - Optimization problems
     - Multi-stage workflows

4. **Direct LLM**
   - For simple informational queries
   - Used for:
     - Kubernetes concepts
     - Best practices
     - General questions
     - Simple explanations

## LangGraph Integration

The framework selection is integrated with LangGraph for orchestration:

```python
# Define conditional routing in the graph configuration
graph_config = {
    "entry_point": "framework_selector",
    "edges": [
        {
            "from": "framework_selector",
            "to": "route_framework",
            "condition": "state.agent_outputs.get('framework_selector', {}).get('framework_type') == 'route'"
        },
        {
            "from": "framework_selector",
            "to": "react_framework",
            "condition": "state.agent_outputs.get('framework_selector', {}).get('framework_type') == 'react'"
        },
        {
            "from": "framework_selector",
            "to": "plan_framework",
            "condition": "state.agent_outputs.get('framework_selector', {}).get('framework_type') == 'plan'"
        }
    ]
}

# Build the graph
graph = StateGraph(AgentState)
        
# Add nodes for each framework
for name, framework in self.frameworks.items():
    graph.add_node(name, framework.process)

# Add framework selector as the entry point
graph.set_entry_point("framework_selector")

# Add conditional edges based on configuration
for edge in graph_config.get("edges", []):
    from_node = edge.get("from")
    to_node = edge.get("to")
    condition = edge.get("condition")
    
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
```

This graph-based architecture allows IronBox to intelligently select the most appropriate framework for each query and route it accordingly, providing a flexible and powerful system for handling a wide range of user requests.

## Related Files

- `ironbox/core/agent_core.py`: Contains the `AgentCore` class that initializes the system
- `ironbox/core/framework_selector.py`: Contains the `FrameworkSelector` class
- `ironbox/core/langchain_frameworks.py`: Contains LangChain-based framework implementations
- `ironbox/docs/langchain_langgraph_architecture.md`: Detailed architecture documentation
