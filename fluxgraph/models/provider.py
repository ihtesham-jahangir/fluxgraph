# fluxgraph/models/provider.py
"""
Enhanced Model Provider System for FluxGraph
Supports multiple LLM providers with unified interface
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """Model capabilities"""
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    CODE_GENERATION = "code_generation"


@dataclass
class ModelConfig:
    """Configuration for model providers"""
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    retry_attempts: int = 3
    streaming: bool = False
    
    # Provider-specific settings
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


@dataclass
class ModelResponse:
    """Standardized response from model providers"""
    text: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "finish_reason": self.finish_reason,
            "usage": self.usage
        }


class ModelProvider(ABC):
    """
    Enhanced abstract base class for LLM model providers.
    Provides unified interface for all LLM operations.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize provider with configuration.
        
        Args:
            config: ModelConfig instance with provider settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._capabilities: List[ModelCapability] = []
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the provider (async setup, API checks, etc.)
        Must be called before using the provider.
        """
        pass
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a text response based on the prompt.
        
        Args:
            prompt: The input prompt for the model
            system_message: Optional system message for chat models
            **kwargs: Additional model-specific parameters
                - temperature: float (0.0-2.0)
                - max_tokens: int
                - top_p: float
                - stop: List[str] or str
                - stream: bool
        
        Returns:
            ModelResponse: Standardized response object
        
        Raises:
            ModelProviderError: If generation fails
        """
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ModelResponse:
        """
        Chat completion with message history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                Example: [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"}
                ]
            **kwargs: Additional parameters
        
        Returns:
            ModelResponse: Standardized response object
        """
        pass
    
    async def stream_generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream generation token by token.
        
        Args:
            prompt: Input prompt
            system_message: Optional system message
            **kwargs: Additional parameters
        
        Yields:
            str: Generated tokens
        
        Raises:
            NotImplementedError: If streaming not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming"
        )
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        
        Raises:
            NotImplementedError: If embeddings not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings"
        )
    
    def supports_capability(self, capability: ModelCapability) -> bool:
        """
        Check if provider supports a capability.
        
        Args:
            capability: ModelCapability to check
        
        Returns:
            bool: True if supported
        """
        return capability in self._capabilities
    
    @property
    def capabilities(self) -> List[ModelCapability]:
        """Get list of supported capabilities"""
        return self._capabilities.copy()
    
    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized"""
        return self._initialized
    
    def _merge_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Merge kwargs with config defaults.
        
        Args:
            **kwargs: Override parameters
        
        Returns:
            Dict with merged parameters
        """
        params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        params.update(kwargs)
        return params
    
    async def health_check(self) -> bool:
        """
        Check if provider is healthy and accessible.
        
        Returns:
            bool: True if healthy
        """
        try:
            response = await self.generate("test", max_tokens=5)
            return response.text is not None
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.config.model_name}, "
            f"capabilities={len(self._capabilities)})"
        )


class ModelProviderError(Exception):
    """Base exception for model provider errors"""
    pass


class ModelProviderTimeoutError(ModelProviderError):
    """Raised when model request times out"""
    pass


class ModelProviderAuthError(ModelProviderError):
    """Raised when authentication fails"""
    pass


class ModelProviderRateLimitError(ModelProviderError):
    """Raised when rate limit is exceeded"""
    pass
