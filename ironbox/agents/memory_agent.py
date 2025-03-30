"""
Memory agent for IronBox.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

from sqlalchemy.ext.asyncio import AsyncSession
from langchain.schema import AIMessage

from ironbox.core.graph import AgentState, AgentType
from ironbox.core.llm import default_llm
from ironbox.core.memory import MemoryManager
from ironbox.db.operations import MemoryOperations

# Configure logging
logger = logging.getLogger(__name__)


class MemoryAgent:
    """Agent for retrieving information from memory."""
    
    def __init__(self, llm=default_llm, db_session: Optional[AsyncSession] = None):
        """
        Initialize MemoryAgent.
        
        Args:
            llm: LLM instance
            db_session: Database session
        """
        self.llm = llm
        self.db_session = db_session
        self.system_prompt = """
        You are a memory agent for the IronBox system. Your job is to retrieve information from the conversation history and function call history.
        
        You can:
        - Recall previous conversations
        - Find parameters used in previous function calls
        - Summarize conversation history
        
        Extract the user's request and provide the requested information from memory.
        """
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Process a memory retrieval request.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            # Check if we have a database session
            if not self.db_session:
                state.error = "Database session not available"
                state.agent_outputs[AgentType.MEMORY] = {
                    "response": "I'm unable to access memory at the moment due to a database connection issue.",
                    "next": None,
                    "error": state.error,
                }
                return state
            
            # Create memory manager
            memory_manager = MemoryManager(
                session_id=state.session_id or "default",
                db_session=self.db_session
            )
            
            # Determine the type of memory request
            request_type, query = await self._analyze_request(state.input)
            
            if request_type == "conversation":
                # Retrieve conversation history
                response = await self._get_conversation_history(memory_manager, query)
            elif request_type == "function":
                # Retrieve function call history
                response = await self._get_function_history(memory_manager, query)
            elif request_type == "summary":
                # Generate a summary of the conversation
                response = await self._generate_summary(memory_manager)
            else:
                # Unknown request
                response = "I'm not sure what information you're looking for. You can ask about previous conversations, function calls, or request a summary of our conversation."
            
            state.agent_outputs[AgentType.MEMORY] = {
                "response": response,
                "next": None,
            }
            
            return state
        except Exception as e:
            logger.error(f"Error in memory agent: {e}")
            state.error = f"Memory agent error: {str(e)}"
            state.agent_outputs[AgentType.MEMORY] = {
                "response": "An error occurred while retrieving information from memory.",
                "next": None,
                "error": state.error,
            }
            return state
    
    async def _analyze_request(self, user_input: str) -> Tuple[str, Optional[str]]:
        """
        Analyze the user's request to determine the type and query.
        
        Args:
            user_input: User input text
            
        Returns:
            Tuple of (request_type, query)
        """
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": """
            Analyze the user's request for information from memory. Determine:
            1. The type of request: "conversation" (recall previous conversations), "function" (find function call parameters), or "summary" (summarize conversation history)
            2. The specific query or search term (if applicable)
            
            Return a JSON object with the following fields:
            - request_type: "conversation", "function", or "summary"
            - query: The search term or query (or null if not applicable)
            """},
            {"role": "user", "content": user_input},
        ]
        
        # Get response from LLM
        response = await self.llm.ainvoke(messages)
        
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
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            import json
            import re
            
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return data.get("request_type", "unknown"), data.get("query")
            
            # If no JSON block, try to parse the entire response
            data = json.loads(response_text)
            return data.get("request_type", "unknown"), data.get("query")
        except Exception as e:
            logger.error(f"Error analyzing request: {e}")
            
            # Fallback to simple text analysis
            if "function" in user_input.lower() or "parameter" in user_input.lower():
                # Extract function name if present
                function_match = re.search(r'function[:\s]+(["\']?)([^"\']+)\1', user_input, re.IGNORECASE)
                query = function_match.group(2) if function_match else None
                return "function", query
            elif "summary" in user_input.lower():
                return "summary", None
            else:
                return "conversation", user_input
    
    async def _get_conversation_history(self, memory_manager: MemoryManager, query: Optional[str]) -> str:
        """
        Retrieve conversation history.
        
        Args:
            memory_manager: Memory manager
            query: Optional search query
            
        Returns:
            Response text
        """
        # Get chat history
        chat_history = await memory_manager.get_chat_history()
        
        if not chat_history:
            return "I don't have any conversation history to recall."
        
        # If there's a query, use LLM to search for relevant messages
        if query:
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": f"""
                Search through the conversation history for information related to: "{query}"
                
                Conversation history:
                {chat_history}
                
                Return only the relevant parts of the conversation that match the query. If nothing is found, say so.
                """},
            ]
            
            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            
            # Extract response text
            if hasattr(response, 'generations'):
                # ChatResult object
                return response.generations[0].message.content
            elif isinstance(response, AIMessage):
                # AIMessage object
                return response.content
            else:
                # Fallback
                return str(response)
        
        # If no query, format the entire history
        response = "# Conversation History\n\n"
        
        for i, message in enumerate(chat_history):
            role_emoji = "👤" if message["role"] == "user" else "🤖" if message["role"] == "assistant" else "ℹ️"
            response += f"{role_emoji} **{message['role'].capitalize()}:** {message['content'][:100]}{'...' if len(message['content']) > 100 else ''}\n\n"
        
        response += "You can ask for specific information by including search terms in your query."
        
        return response
    
    async def _get_function_history(self, memory_manager: MemoryManager, function_name: Optional[str]) -> str:
        """
        Retrieve function call history.
        
        Args:
            memory_manager: Memory manager
            function_name: Optional function name
            
        Returns:
            Response text
        """
        # Get function calls
        function_calls = await memory_manager.function_memory.get_recent_calls(function_name)
        
        if not function_calls:
            if function_name:
                return f"I don't have any record of calls to the function '{function_name}'."
            else:
                return "I don't have any record of function calls."
        
        response = f"# Function Call History{' for ' + function_name if function_name else ''}\n\n"
        
        for i, call in enumerate(function_calls):
            response += f"## Call {i+1}: {call['timestamp']}\n\n"
            response += f"**Function:** {call['function_name']}\n\n"
            
            if call.get("parameters"):
                response += "**Parameters:**\n```json\n"
                import json
                response += json.dumps(call["parameters"], indent=2)
                response += "\n```\n\n"
            
            if call.get("result"):
                response += "**Result:**\n```json\n"
                import json
                response += json.dumps(call["result"], indent=2)
                response += "\n```\n\n"
            
            response += "---\n\n"
        
        return response
    
    async def _generate_summary(self, memory_manager: MemoryManager) -> str:
        """
        Generate a summary of the conversation.
        
        Args:
            memory_manager: Memory manager
            
        Returns:
            Response text
        """
        # Get chat history
        chat_history = await memory_manager.get_chat_history()
        
        if not chat_history:
            return "I don't have any conversation history to summarize."
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": f"""
            Generate a concise summary of the following conversation. Focus on:
            1. The main topics discussed
            2. Any decisions or conclusions reached
            3. Any pending questions or actions
            
            Conversation history:
            {chat_history}
            """},
        ]
        
        # Get response from LLM
        response = await self.llm.ainvoke(messages)
        
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
        
        return "# Conversation Summary\n\n" + response_text
