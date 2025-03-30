"""
LLM interface for IronBox.
"""
import logging
from typing import Dict, Any, List, Optional, Union, Callable

from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_ollama import OllamaLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ironbox.config import config

# Configure logging
logger = logging.getLogger(__name__)


class OllamaChat(BaseChatModel):
    """Chat model implementation for Ollama."""
    
    client: OllamaLLM
    model_name: str
    temperature: float
    max_tokens: int
    
    def __init__(
        self,
        base_url: str = config["llm"]["base_url"],
        model: str = config["llm"]["model"],
        temperature: float = config["llm"]["temperature"],
        max_tokens: int = config["llm"]["max_tokens"],
        **kwargs
    ):
        """
        Initialize OllamaChat.
        
        Args:
            base_url: Ollama API base URL
            model: Model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments
        """
        # Create the client and set required fields before calling super().__init__
        client = OllamaLLM(base_url=base_url, model=model)
        
        # Prepare the fields for Pydantic validation
        model_kwargs = {
            "client": client,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Initialize the parent class with all required fields
        super().__init__(**model_kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ollama-chat"
    
    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """
        Convert messages to a prompt string.
        
        Args:
            messages: List of messages
            
        Returns:
            Prompt string
        """
        prompt = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                prompt += f"Assistant: {message.content}\n"
            elif isinstance(message, SystemMessage):
                prompt += f"System: {message.content}\n"
            else:
                prompt += f"{message.type}: {message.content}\n"
        return prompt
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of messages
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments
            
        Returns:
            ChatResult with the generated response
        """
        prompt = self._convert_messages_to_prompt(messages)
        
        # Add any additional kwargs
        params = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        params.update(kwargs)
        
        # If stop sequences are provided, add them
        if stop:
            params["stop"] = stop
        
        try:
            # OllamaLLM doesn't accept temperature or max_tokens parameters
            response = self.client.invoke(
                prompt,
                stop=params.get("stop", None),
            )
            
            message = AIMessage(content=response)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            message = AIMessage(content="I encountered an error while processing your request.")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_llm(provider: str = config["llm"]["provider"], **kwargs) -> BaseChatModel:
        """
        Create an LLM instance.
        
        Args:
            provider: LLM provider
            **kwargs: Additional arguments
            
        Returns:
            LLM instance
        """
        if provider == "ollama":
            # Explicitly pass required parameters from config
            params = {
                "base_url": config["llm"]["base_url"],
                "model": config["llm"]["model"],
                "temperature": config["llm"]["temperature"],
                "max_tokens": config["llm"]["max_tokens"],
            }
            # Override with any provided kwargs
            params.update(kwargs)
            return OllamaChat(**params)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# Default LLM instance
default_llm = LLMFactory.create_llm()
