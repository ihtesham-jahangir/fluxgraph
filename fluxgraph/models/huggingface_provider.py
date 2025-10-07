# fluxgraph/models/huggingface_provider.py
"""
Enhanced Hugging Face Provider
"""
import os
from typing import Dict, Any, List, Optional, AsyncIterator
from huggingface_hub import InferenceClient
import asyncio

from .provider import (
    ModelProvider,
    ModelConfig,
    ModelResponse,
    ModelCapability,
    ModelProviderError,
    ModelProviderAuthError
)


class HuggingFaceProvider(ModelProvider):
    """
    Hugging Face Inference API provider.
    
    Supports thousands of models from the Hub.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client: Optional[InferenceClient] = None
        
        # Set capabilities
        self._capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.CODE_GENERATION,
            ModelCapability.EMBEDDINGS
        ]
    
    async def initialize(self) -> None:
        """Initialize HuggingFace client"""
        try:
            if not self.config.api_key:
                raise ModelProviderAuthError("HF_TOKEN not provided")
            
            self.client = InferenceClient(
                token=self.config.api_key,
                timeout=self.config.timeout
            )
            self._initialized = True
            self.logger.info(f"HuggingFace provider initialized: {self.config.model_name}")
            
        except Exception as e:
            raise ModelProviderAuthError(f"Failed to initialize HuggingFace: {e}")
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate text using HuggingFace"""
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
        """Chat completion with HuggingFace"""
        if not self._initialized:
            await self.initialize()
        
        params = self._merge_kwargs(**kwargs)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 1000),
                top_p=params.get("top_p", 0.95),
            )
            
            text_content = ""
            if completion.choices and completion.choices[0].message:
                text_content = completion.choices[0].message.content or ""
                text_content = text_content.strip()
            
            usage = {}
            if hasattr(completion, 'usage') and completion.usage:
                usage = {
                    "prompt_tokens": getattr(completion.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(completion.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(completion.usage, 'total_tokens', 0)
                }
            
            return ModelResponse(
                text=text_content,
                model=self.config.model_name,
                provider="huggingface",
                usage=usage,
                raw_response=completion
            )
            
        except Exception as e:
            raise ModelProviderError(f"HuggingFace generation failed: {e}")
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using HuggingFace"""
        if not self._initialized:
            await self.initialize()
        
        try:
            embeddings = []
            for text in texts:
                embedding = self.client.feature_extraction(text)
                embeddings.append(embedding)
            return embeddings
            
        except Exception as e:
            raise ModelProviderError(f"HuggingFace embeddings failed: {e}")
