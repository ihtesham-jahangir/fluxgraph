# fluxgraph/models/openai_provider.py
"""
OpenAI Model Provider with GPT-4, GPT-3.5, and more
"""
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
import openai
from openai import AsyncOpenAI

# --- NEW IMPORTS ---
from ..multimodal.processor import MultiModalInput, MediaType
# --- END NEW IMPORTS ---

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

    async def multimodal_generate(
        self,
        prompt: str,
        media_inputs: List[MultiModalInput],
        system_message: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate a response based on a prompt and media using GPT-4V."""
        if not self._initialized:
            await self.initialize()

        if ModelCapability.VISION not in self.capabilities:
            raise NotImplementedError(f"Model {self.config.model_name} does not support vision capability.")

        params = self._merge_kwargs(**kwargs)
        
        # --- OpenAI Content Structure ---
        # OpenAI expects content as an array of objects: {type: "text", text: "..."}, {type: "image_url", image_url: {url: "..."}}
        content_parts = []
        
        # 1. Add media parts (image_url/b64 data)
        for media_input in media_inputs:
            if media_input.media_type == MediaType.IMAGE:
                # Use the full data URI (Base64 string with prefix)
                content_parts.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": media_input.encoded_content # This is the "data:mime_type;base64,..." string
                    }
                })
            # Note: Explicitly exclude non-image types as GPT-4V chat completion only supports images
        
        # 2. Add the text prompt
        content_parts.append({
            "type": "text", 
            "text": prompt
        })

        # 3. Construct messages (always a single user message for multimodal)
        messages = [{"role": "user", "content": content_parts}]
        
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})
            
        # --- End OpenAI Content Structure ---
        
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
            self.logger.error(f"OpenAI multimodal generation failed: {e}")
            raise ModelProviderError(f"OpenAI multimodal generation failed: {e}")

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