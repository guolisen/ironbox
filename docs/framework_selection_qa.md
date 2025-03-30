# Framework Selection Q&A

## Question: How does the core code decide which agent framework to be used?

The IronBox system uses a sophisticated framework selection mechanism that analyzes each user query to determine the most appropriate agent framework. Here's how it works:

## Framework Selection Process

1. When a query is received, it's passed to the `process_query` method in the `AgentCore` class
2. This method calls `framework_selector.select_framework(query)` to determine which framework to use
3. The framework selector uses an LLM to analyze the query and select the appropriate framework
4. Based on the LLM's response, one of four frameworks is selected: Route, React, Plan, or Direct

## The FrameworkSelector Class

The core of this decision-making is in the `FrameworkSelector` class (in `ironbox/core/agent_framework.py`), which:

1. Takes the user query as input
2. Passes it to the LLM with a specific system prompt that explains the available frameworks
3. Analyzes the LLM's response to determine which framework to use
4. Returns the selected framework type as a string

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

## Implementation Details

After the framework is selected, the `process_query` method in `AgentCore` processes the query using the selected framework:

```python
# Select framework
framework_type = await self.framework_selector.select_framework(query)
logger.debug("Selected framework: %s", framework_type)

# Process query with selected framework
if framework_type == "direct":
    # Direct LLM response without using any framework
    state = await self._process_direct(state)
elif framework_type in self.frameworks:
    # Process with selected framework
    framework = self.frameworks[framework_type]
    state = await framework.process(state)
else:
    # Fallback to route framework
    logger.warning(f"Framework {framework_type} not found, falling back to route")
    if "route" in self.frameworks:
        framework = self.frameworks["route"]
        state = await framework.process(state)
    else:
        # Direct LLM response as last resort
        state = await self._process_direct(state)
```

This architecture allows IronBox to intelligently select the most appropriate framework for each query, providing a flexible and powerful system for handling a wide range of user requests.

## Related Files

- `ironbox/core/agent_core.py`: Contains the `AgentCore` class with the `process_query` method
- `ironbox/core/agent_framework.py`: Contains the `FrameworkSelector` class and framework implementations
- `ironbox/docs/framework_selection.puml`: UML diagram showing the framework selection logic
