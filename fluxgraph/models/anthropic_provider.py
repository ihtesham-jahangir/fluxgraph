# fluxgraph/models/anthropic_provider.py
"""
Enhanced Anthropic Claude Provider
"""
import os
from typing import Dict, Any, List, Optional, AsyncIterator
from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT
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


class AnthropicProvider(ModelProvider):
    """
    Anthropic Claude provider with advanced features.
    
    Supported models:
    - claude-3-opus-20240229
    - claude-3-sonnet-20240229
    - claude-3-haiku-20240307
    - claude-2.1
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client: Optional[AsyncAnthropic] = None
        
        # Set capabilities
        self._capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.VISION,  # Claude 3 models
            ModelCapability.CODE_GENERATION
        ]
    
    async def initialize(self) -> None:
        """Initialize Anthropic client"""
        try:
            if not self.config.api_key:
                raise ModelProviderAuthError("ANTHROPIC_API_KEY not provided")
            
            self.client = AsyncAnthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            self._initialized = True
            self.logger.info(f"Anthropic provider initialized: {self.config.model_name}")
            
        except Exception as e:
            raise ModelProviderAuthError(f"Failed to initialize Anthropic: {e}")
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate text using Claude"""
        if not self._initialized:
            await self.initialize()
        
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, system_message=system_message, **kwargs)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Chat completion with Claude"""
        if not self._initialized:
            await self.initialize()
        
        params = self._merge_kwargs(**kwargs)
        
        # Extract system message from messages if present
        system = system_message
        filtered_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)
        
        try:
            response = await self.client.messages.create(
                model=self.config.model_name,
                messages=filtered_messages,
                system=system,
                max_tokens=params.get("max_tokens", 1000),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 1.0),
                stop_sequences=params.get("stop"),
            )
            
            text_content = ""
            if response.content and response.content[0].type == "text":
                text_content = response.content[0].text.strip()
            
            return ModelResponse(
                text=text_content,
                model=response.model,
                provider="anthropic",
                finish_reason=response.stop_reason,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                raw_response=response
            )
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise ModelProviderRateLimitError(f"Anthropic rate limit: {e}")
            raise ModelProviderError(f"Anthropic generation failed: {e}")
    
    async def stream_generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream generation from Claude"""
        if not self._initialized:
            await self.initialize()
        
        messages = [{"role": "user", "content": prompt}]
        params = self._merge_kwargs(**kwargs)
        
        try:
            async with self.client.messages.stream(
                model=self.config.model_name,
                messages=messages,
                system=system_message,
                max_tokens=params.get("max_tokens", 1000),
                temperature=params.get("temperature", 0.7),
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            raise ModelProviderError(f"Anthropic streaming failed: {e}")
