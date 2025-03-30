"""
Memory system for IronBox.
"""
import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from sqlalchemy.ext.asyncio import AsyncSession

from ironbox.db.operations import MemoryOperations

# Configure logging
logger = logging.getLogger(__name__)


class SQLiteMemory:
    """Memory implementation that stores conversations in SQLite."""
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        db_session: Optional[AsyncSession] = None,
        return_messages: bool = True,
    ):
        """
        Initialize SQLiteMemory.
        
        Args:
            session_id: Chat session ID
            db_session: Database session
            return_messages: Whether to return messages or a string
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.db_session = db_session
        self.return_messages = return_messages
    
    async def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables.
        
        Args:
            inputs: Input values
            
        Returns:
            Memory variables
        """
        if not self.db_session:
            return {"history": []}
        
        try:
            chat_history = await MemoryOperations.get_chat_history(self.db_session, self.session_id)
            messages = []
            
            for entry in chat_history:
                if entry.role == "user":
                    messages.append(HumanMessage(content=entry.content))
                elif entry.role == "assistant":
                    messages.append(AIMessage(content=entry.content))
                elif entry.role == "system":
                    messages.append(SystemMessage(content=entry.content))
            
            if self.return_messages:
                return {"history": messages}
            else:
                # Convert messages to string format
                string_messages = []
                for message in messages:
                    if isinstance(message, HumanMessage):
                        string_messages.append(f"Human: {message.content}")
                    elif isinstance(message, AIMessage):
                        string_messages.append(f"AI: {message.content}")
                    elif isinstance(message, SystemMessage):
                        string_messages.append(f"System: {message.content}")
                return {"history": "\n".join(string_messages)}
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            return {"history": [] if self.return_messages else ""}
    
    async def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Save context.
        
        Args:
            inputs: Input values
            outputs: Output values
        """
        if not self.db_session:
            return
        
        try:
            # Save user message
            input_str = inputs.get("input", "")
            if input_str:
                await MemoryOperations.add_chat_message(
                    self.db_session,
                    self.session_id,
                    "user",
                    input_str,
                    extra_data={}
                )
            
            # Save assistant message
            output_str = outputs.get("output", "")
            if output_str:
                await MemoryOperations.add_chat_message(
                    self.db_session,
                    self.session_id,
                    "assistant",
                    output_str,
                    extra_data={}
                )
        except Exception as e:
            logger.error(f"Error saving context: {e}")
    
    async def clear(self) -> None:
        """Clear memory."""
        # In a real implementation, you might want to delete all messages for this session
        # For now, we'll just log that this was called
        logger.info(f"Clear memory called for session {self.session_id}")


class FunctionMemory:
    """Memory for function calls and parameters."""
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        db_session: Optional[AsyncSession] = None
    ):
        """
        Initialize FunctionMemory.
        
        Args:
            session_id: Chat session ID
            db_session: Database session
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.db_session = db_session
    
    async def save_function_call(
        self,
        function_name: str,
        parameters: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a function call.
        
        Args:
            function_name: Function name
            parameters: Function parameters
            result: Function result
        """
        if not self.db_session:
            return
        
        try:
            await MemoryOperations.add_function_call(
                self.db_session,
                self.session_id,
                function_name,
                parameters,
                result
            )
        except Exception as e:
            logger.error(f"Error saving function call: {e}")
    
    async def get_recent_calls(
        self,
        function_name: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recent function calls.
        
        Args:
            function_name: Optional function name filter
            limit: Maximum number of calls to return
            
        Returns:
            List of function calls
        """
        if not self.db_session:
            return []
        
        try:
            calls = await MemoryOperations.get_function_calls(
                self.db_session,
                self.session_id,
                function_name,
                limit
            )
            return [call.to_dict() for call in calls]
        except Exception as e:
            logger.error(f"Error getting function calls: {e}")
            return []
    
    async def get_last_parameters(self, function_name: str) -> Optional[Dict[str, Any]]:
        """
        Get parameters from the last call to a function.
        
        Args:
            function_name: Function name
            
        Returns:
            Parameters or None if not found
        """
        calls = await self.get_recent_calls(function_name, 1)
        if calls:
            return calls[0].get("parameters", {})
        return None


class MemoryManager:
    """Manager for conversation and function memory."""
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        db_session: Optional[AsyncSession] = None
    ):
        """
        Initialize MemoryManager.
        
        Args:
            session_id: Chat session ID
            db_session: Database session
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.db_session = db_session
        self.conversation_memory = SQLiteMemory(
            session_id=self.session_id,
            db_session=self.db_session
        )
        self.function_memory = FunctionMemory(
            session_id=self.session_id,
            db_session=self.db_session
        )
    
    async def add_system_message(self, content: str) -> None:
        """
        Add a system message to the conversation.
        
        Args:
            content: Message content
        """
        if not self.db_session:
            return
        
        try:
            await MemoryOperations.add_chat_message(
                self.db_session,
                self.session_id,
                "system",
                content,
                extra_data={}
            )
        except Exception as e:
            logger.error(f"Error adding system message: {e}")
    
    async def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Get chat history.
        
        Returns:
            List of chat messages
        """
        if not self.db_session:
            return []
        
        try:
            history = await MemoryOperations.get_chat_history(self.db_session, self.session_id)
            return [entry.to_dict() for entry in history]
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    async def get_messages_for_llm(self) -> List[BaseMessage]:
        """
        Get messages formatted for LLM.
        
        Returns:
            List of messages
        """
        memory_vars = await self.conversation_memory.load_memory_variables({})
        return memory_vars.get("history", [])
