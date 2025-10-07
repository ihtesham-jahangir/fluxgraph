# fluxgraph/models/openai_provider.py
"""
OpenAI Model Provider with GPT-4, GPT-3.5, and more
"""
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
import openai
from openai import AsyncOpenAI

from .provider import (
    ModelProvider,
    ModelConfig,
    ModelResponse,
    ModelCapability,
    ModelProviderError,
    ModelProviderAuthError,
    ModelProviderRateLimitError
)


class OpenAIProvider(ModelProvider):
    """
    OpenAI provider supporting GPT-4, GPT-3.5-Turbo, etc.
    
    Supported models:
    - gpt-4-turbo-preview
    - gpt-4
    - gpt-3.5-turbo
    - gpt-3.5-turbo-16k
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client: Optional[AsyncOpenAI] = None
        
        # Set capabilities
        self._capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION  # GPT-4V
        ]
    
    async def initialize(self) -> None:
        """Initialize OpenAI client"""
        try:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            self._initialized = True
            self.logger.info(f"OpenAI provider initialized: {self.config.model_name}")
        except Exception as e:
            raise ModelProviderAuthError(f"Failed to initialize OpenAI: {e}")
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate text using OpenAI"""
        if not self._initialized:
            await self.initialize()
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        return await self.chat(messages, **kwargs)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ModelResponse:
        """Chat completion with OpenAI"""
        if not self._initialized:
            await self.initialize()
        
        params = self._merge_kwargs(**kwargs)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 1000),
                top_p=params.get("top_p", 1.0),
                frequency_penalty=params.get("frequency_penalty", 0.0),
                presence_penalty=params.get("presence_penalty", 0.0),
                stop=params.get("stop"),
                n=params.get("n", 1),
            )
            
            choice = response.choices[0]
            
            return ModelResponse(
                text=choice.message.content,
                model=response.model,
                provider="openai",
                finish_reason=choice.finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                raw_response=response
            )
            
        except openai.RateLimitError as e:
            raise ModelProviderRateLimitError(f"OpenAI rate limit: {e}")
        except openai.AuthenticationError as e:
            raise ModelProviderAuthError(f"OpenAI auth error: {e}")
        except Exception as e:
            raise ModelProviderError(f"OpenAI generation failed: {e}")
    
    async def stream_generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream generation from OpenAI"""
        if not self._initialized:
            await self.initialize()
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        params = self._merge_kwargs(**kwargs)
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 1000),
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise ModelProviderError(f"OpenAI streaming failed: {e}")
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        if not self._initialized:
            await self.initialize()
        
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            raise ModelProviderError(f"OpenAI embeddings failed: {e}")
