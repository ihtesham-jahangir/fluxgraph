# fluxgraph/models/groq_provider.py
"""
Enhanced Groq Provider with ultra-fast inference
"""
import os
from typing import Dict, Any, List, Optional, AsyncIterator
from groq import AsyncGroq
import asyncio

from .provider import (
    ModelProvider,
    ModelConfig,
    ModelResponse,
    ModelCapability,
    ModelProviderError,
    ModelProviderAuthError,
    ModelProviderRateLimitError
)


class GroqProvider(ModelProvider):
    """
    Groq provider with ultra-fast inference.
    
    Supported models:
    - mixtral-8x7b-32768
    - llama3-70b-8192
    - llama3-8b-8192
    - gemma-7b-it
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client: Optional[AsyncGroq] = None
        
        # Set capabilities
        self._capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.CODE_GENERATION
        ]
    
    async def initialize(self) -> None:
        """Initialize Groq client"""
        try:
            if not self.config.api_key:
                raise ModelProviderAuthError("GROQ_API_KEY not provided")
            
            self.client = AsyncGroq(api_key=self.config.api_key)
            self._initialized = True
            self.logger.info(f"Groq provider initialized: {self.config.model_name}")
            
        except Exception as e:
            raise ModelProviderAuthError(f"Failed to initialize Groq: {e}")
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate text using Groq"""
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
        """Chat completion with Groq"""
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
                stop=params.get("stop"),
            )
            
            choice = response.choices[0]
            
            return ModelResponse(
                text=choice.message.content.strip(),
                model=response.model,
                provider="groq",
                finish_reason=choice.finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                raw_response=response
            )
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise ModelProviderRateLimitError(f"Groq rate limit: {e}")
            raise ModelProviderError(f"Groq generation failed: {e}")
    
    async def stream_generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream generation from Groq"""
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
            raise ModelProviderError(f"Groq streaming failed: {e}")
