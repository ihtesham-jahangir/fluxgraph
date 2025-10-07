# fluxgraph/models/factory.py
"""
Model Provider Factory for creating provider instances
"""
from typing import Dict, Type, Optional
import os

from .provider import ModelProvider, ModelConfig
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .groq_provider import GroqProvider
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider
from .huggingface_provider import HuggingFaceProvider


class ModelProviderFactory:
    """Factory for creating model provider instances"""
    
    _providers: Dict[str, Type[ModelProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "groq": GroqProvider,
        "gemini": GeminiProvider,
        "google": GeminiProvider,  # Alias
        "ollama": OllamaProvider,
        "huggingface": HuggingFaceProvider,
        "hf": HuggingFaceProvider,  # Alias
    }
    
    @classmethod
    def create(
        cls,
        provider_name: str,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> ModelProvider:
        """
        Create a model provider instance.
        
        Args:
            provider_name: Name of provider (openai, anthropic, etc.)
            model_name: Model name to use
            api_key: API key (or will use environment variable)
            **kwargs: Additional ModelConfig parameters
        
        Returns:
            ModelProvider instance
        
        Raises:
            ValueError: If provider not found
        
        Example:
            provider = ModelProviderFactory.create(
                "openai",
                "gpt-4-turbo-preview",
                temperature=0.8
            )
        """
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {available}"
            )
        
        # Auto-detect API key from environment
        if api_key is None:
            api_key = cls._get_api_key(provider_name)
        
        config = ModelConfig(
            model_name=model_name,
            api_key=api_key,
            **kwargs
        )
        
        provider_class = cls._providers[provider_name]
        return provider_class(config)
    
    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: Type[ModelProvider]
    ) -> None:
        """
        Register a custom provider.
        
        Args:
            name: Provider name
            provider_class: Provider class
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def list_providers(cls) -> list:
        """Get list of available providers"""
        return list(cls._providers.keys())
    
    @staticmethod
    def _get_api_key(provider_name: str) -> Optional[str]:
        """Get API key from environment"""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "google": "GOOGLE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "hf": "HUGGINGFACE_API_KEY",
        }
        
        env_var = env_vars.get(provider_name)
        if env_var:
            return os.getenv(env_var)
        return None


# Convenience function
def create_provider(
    provider_name: str,
    model_name: str,
    **kwargs
) -> ModelProvider:
    """
    Convenience function to create provider.
    
    Example:
        from fluxgraph.models import create_provider
        
        provider = create_provider("openai", "gpt-4")
        response = await provider.generate("Hello world")
    """
    return ModelProviderFactory.create(provider_name, model_name, **kwargs)
