# fluxgraph/models/easy.py
"""
Super easy-to-use API for FluxGraph model providers
One-line usage without configuration complexity
"""
from typing import Optional, List, Dict, Any, AsyncIterator
import os
import asyncio

from .factory import ModelProviderFactory
from .provider import ModelProvider, ModelResponse


# Global provider cache
_provider_cache: Dict[str, ModelProvider] = {}


async def ask(
    question: str,
    model: str = "gpt-3.5-turbo",
    provider: str = "openai",
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs
) -> str:
    """
    Ask a question and get an answer. Super simple!
    
    Args:
        question: Your question or prompt
        model: Model name (default: gpt-3.5-turbo)
        provider: Provider name (default: openai)
        system: Optional system message
        temperature: Temperature (0-2)
        max_tokens: Max response tokens
        **kwargs: Additional parameters
    
    Returns:
        str: The model's response text
    
    Example:
        answer = await ask("What is Python?")
        print(answer)
    """
    llm = await get_llm(provider, model, temperature=temperature, max_tokens=max_tokens, **kwargs)
    response = await llm.generate(question, system_message=system)
    return response.text


async def chat(
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo",
    provider: str = "openai",
    **kwargs
) -> str:
    """
    Chat with conversation history.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        provider: Provider name
        **kwargs: Additional parameters
    
    Returns:
        str: Response text
    
    Example:
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        response = await chat(messages)
    """
    llm = await get_llm(provider, model, **kwargs)
    response = await llm.chat(messages)
    return response.text


async def stream(
    question: str,
    model: str = "gpt-3.5-turbo",
    provider: str = "openai",
    system: Optional[str] = None,
    **kwargs
) -> AsyncIterator[str]:
    """
    Stream response token by token.
    
    Args:
        question: Your question
        model: Model name
        provider: Provider name
        system: Optional system message
        **kwargs: Additional parameters
    
    Yields:
        str: Response tokens
    
    Example:
        async for token in stream("Tell me a story"):
            print(token, end="", flush=True)
    """
    llm = await get_llm(provider, model, **kwargs)
    async for token in llm.stream_generate(question, system_message=system):
        yield token


async def get_llm(
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    **kwargs
) -> ModelProvider:
    """
    Get or create a provider instance (cached).
    
    Args:
        provider: Provider name
        model: Model name
        **kwargs: Configuration parameters
    
    Returns:
        ModelProvider instance
    
    Example:
        llm = await get_llm("openai", "gpt-4")
        response = await llm.generate("Hello")
    """
    cache_key = f"{provider}:{model}"
    
    if cache_key not in _provider_cache:
        llm = ModelProviderFactory.create(provider, model, **kwargs)
        await llm.initialize()
        _provider_cache[cache_key] = llm
    
    return _provider_cache[cache_key]


def clear_cache():
    """Clear the provider cache"""
    _provider_cache.clear()


# Synchronous wrappers for easy use
def ask_sync(question: str, **kwargs) -> str:
    """Synchronous version of ask()"""
    return asyncio.run(ask(question, **kwargs))


def chat_sync(messages: List[Dict[str, str]], **kwargs) -> str:
    """Synchronous version of chat()"""
    return asyncio.run(chat(messages, **kwargs))
