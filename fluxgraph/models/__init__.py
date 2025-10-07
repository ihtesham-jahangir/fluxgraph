# fluxgraph/models/__init__.py
"""
FluxGraph Model Provider System - Easy to Use!

Quick Start:
    from fluxgraph.models import ask
    
    answer = await ask("What is Python?")
    print(answer)
"""

# Core classes
from .provider import (
    ModelProvider,
    ModelConfig,
    ModelResponse,
    ModelCapability,
    ModelProviderError
)

# Factory
from .factory import ModelProviderFactory, create_provider

# Easy API (most users should use this)
from .easy import (
    ask,           # Simple question/answer
    chat,          # Chat with history
    stream,        # Streaming responses
    get_llm,       # Get provider instance
    ask_sync,      # Synchronous version
    chat_sync      # Synchronous chat
)

# Builder pattern
from .builder import LLM, LLMBuilder

# Presets
from .presets import get_preset, list_presets, PRESETS

# Simple shortcuts
from .simple import (
    ask_gpt4,
    ask_gpt3,
    ask_claude,
    ask_gemini,
    ask_local,
    gpt4,      # Sync
    gpt3,      # Sync
    claude,    # Sync
    compare_models
)

# All providers
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .groq_provider import GroqProvider
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider
from .huggingface_provider import HuggingFaceProvider

__version__ = "3.1.0"

__all__ = [
    # Easy API (recommended)
    "ask",
    "chat", 
    "stream",
    "get_llm",
    "ask_sync",
    "chat_sync",
    
    # Builder
    "LLM",
    "LLMBuilder",
    
    # Presets
    "get_preset",
    "list_presets",
    "PRESETS",
    
    # Simple shortcuts
    "ask_gpt4",
    "ask_gpt3",
    "ask_claude",
    "ask_gemini",
    "ask_local",
    "gpt4",
    "gpt3",
    "claude",
    "compare_models",
    
    # Advanced
    "create_provider",
    "ModelProviderFactory",
    "ModelProvider",
    "ModelConfig",
    "ModelResponse",
    "ModelCapability",
]
