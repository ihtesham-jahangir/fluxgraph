# fluxgraph/models/ollama_provider.py
"""
Enhanced Ollama Provider for local models
"""
import os
from typing import Dict, Any, List, Optional, AsyncIterator
import httpx
import asyncio

from .provider import (
    ModelProvider,
    ModelConfig,
    ModelResponse,
    ModelCapability,
    ModelProviderError,
    ModelProviderTimeoutError
)


class OllamaProvider(ModelProvider):
    """
    Ollama provider for local model inference.
    
    Supports all Ollama models:
    - llama2, llama3
    - mistral, mixtral
    - codellama
    - gemma, phi
    - And more!
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client: Optional[httpx.AsyncClient] = None
        self.base_url = config.base_url or "http://localhost:11434"
        
        # Set capabilities
        self._capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.CODE_GENERATION,
            ModelCapability.EMBEDDINGS
        ]
    
    async def initialize(self) -> None:
        """Initialize Ollama client"""
        try:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.config.timeout
            )
            
            # Check if Ollama is running
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            
            self._initialized = True
            self.logger.info(f"Ollama provider initialized: {self.config.model_name}")
            
        except httpx.ConnectError:
            raise ModelProviderError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running."
            )
        except Exception as e:
            raise ModelProviderError(f"Failed to initialize Ollama: {e}")
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate text using Ollama"""
        if not self._initialized:
            await self.initialize()
        
        params = self._merge_kwargs(**kwargs)
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params.get("temperature", 0.7),
                "num_predict": params.get("max_tokens", 1000),
                "top_p": params.get("top_p", 0.9),
            }
        }
        
        if system_message:
            payload["system"] = system_message
        
        try:
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            
            return ModelResponse(
                text=data.get("response", "").strip(),
                model=data.get("model", self.config.model_name),
                provider="ollama",
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                },
                raw_response=data
            )
            
        except httpx.TimeoutException:
            raise ModelProviderTimeoutError("Ollama request timed out")
        except Exception as e:
            raise ModelProviderError(f"Ollama generation failed: {e}")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ModelResponse:
        """Chat completion with Ollama"""
        if not self._initialized:
            await self.initialize()
        
        params = self._merge_kwargs(**kwargs)
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": params.get("temperature", 0.7),
                "num_predict": params.get("max_tokens", 1000),
            }
        }
        
        try:
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            
            message = data.get("message", {})
            text_content = message.get("content", "").strip()
            
            return ModelResponse(
                text=text_content,
                model=data.get("model", self.config.model_name),
                provider="ollama",
                raw_response=data
            )
            
        except Exception as e:
            raise ModelProviderError(f"Ollama chat failed: {e}")
    
    async def stream_generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream generation from Ollama"""
        if not self._initialized:
            await self.initialize()
        
        params = self._merge_kwargs(**kwargs)
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": params.get("temperature", 0.7),
            }
        }
        
        if system_message:
            payload["system"] = system_message
        
        try:
            async with self.client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                            
        except Exception as e:
            raise ModelProviderError(f"Ollama streaming failed: {e}")
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama"""
        if not self._initialized:
            await self.initialize()
        
        embeddings = []
        
        try:
            for text in texts:
                response = await self.client.post(
                    "/api/embeddings",
                    json={"model": self.config.model_name, "prompt": text}
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data.get("embedding", []))
            
            return embeddings
            
        except Exception as e:
            raise ModelProviderError(f"Ollama embeddings failed: {e}")
