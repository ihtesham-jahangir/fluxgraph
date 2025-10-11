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

# Assuming MultiModalInput is importable and defined somewhere
from ..multimodal.processor import MultiModalInput # <-- NEW IMPORT

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
        """
        pass
    
    @abstractmethod
    async def multimodal_generate(
        self,
        prompt: str,
        media_inputs: List['MultiModalInput'], # <-- NEW ABSTRACT METHOD
        system_message: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a text response based on a prompt and one or more media inputs (e.g., images).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support multimodal generation"
        )
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ModelResponse:
        """
        Chat completion with message history.
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
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming"
        )
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings"
        )
    
    def supports_capability(self, capability: ModelCapability) -> bool:
        """
        Check if provider supports a capability.
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