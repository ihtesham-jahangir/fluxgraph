# fluxgraph/models/ollama_provider.py
"""
Enhanced Ollama Provider for local models
"""
import os
from typing import Dict, Any, List, Optional, AsyncIterator
import httpx
import asyncio
import tiktoken  # <--- NEW IMPORT for token counting

from .provider import (
    ModelProvider,
    ModelConfig,
    ModelResponse,
    ModelCapability,
    ModelProviderError,
    ModelProviderTimeoutError
)


# Global tokenizer cache for efficiency
_TKN_CACHE = {}

def _count_tokens(text: str, model_name: str) -> int:
    """Approximate token count for given text using tiktoken as a standard fallback."""
    # Use a generic encoding as a fallback for non-OpenAI models
    encoding_name = "cl100k_base"
    
    # Try to get model-specific encoding if it exists (though usually reserved for OpenAI)
    # The tiktoken library provides a stable, fast counting mechanism for most text data.
    if model_name not in _TKN_CACHE:
        try:
            _TKN_CACHE[model_name] = tiktoken.encoding_for_model(model_name)
        except KeyError:
            _TKN_CACHE[model_name] = tiktoken.get_encoding(encoding_name)
            
    return len(_TKN_CACHE[model_name].encode(text))


class OllamaProvider(ModelProvider):
    """
    Ollama provider for local model inference.
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
            
            # --- START: TOKEN COUNTING LOGIC FIX ---
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            
            # Fallback: if native token counts are missing (i.e., both are 0), calculate them.
            if prompt_tokens == 0 and completion_tokens == 0:
                text_content = data.get("response", "").strip()
                prompt_tokens = _count_tokens(prompt, self.config.model_name)
                completion_tokens = _count_tokens(text_content, self.config.model_name)

            usage_dict = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
            # --- END: TOKEN COUNTING LOGIC FIX ---
            
            return ModelResponse(
                text=data.get("response", "").strip(),
                model=data.get("model", self.config.model_name),
                provider="ollama",
                usage=usage_dict, # <-- Use the calculated/fallback usage
                raw_response=data
            )
            
        except httpx.TimeoutException:
            raise ModelProviderTimeoutError("Ollama request timed out")
        except Exception as e:
            raise ModelProviderError(f"Ollama generation failed: {e}")
    
    async def multimodal_generate(
        self,
        prompt: str,
        media_inputs: List['MultiModalInput'],
        system_message: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Ollama does not natively support multimodal generation in the /api/generate endpoint payload."""
        raise NotImplementedError(f"OllamaProvider does not natively support multimodal generation.")

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
            
            # --- START: TOKEN COUNTING LOGIC FIX ---
            # Ollama /api/chat may not return usage, so we estimate:
            full_prompt_text = "\n".join([msg['content'] for msg in messages if msg['content']])
            prompt_tokens = _count_tokens(full_prompt_text, self.config.model_name)
            completion_tokens = _count_tokens(text_content, self.config.model_name)

            usage_dict = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
            # --- END: TOKEN COUNTING LOGIC FIX ---

            return ModelResponse(
                text=text_content,
                model=data.get("model", self.config.model_name),
                provider="ollama",
                usage=usage_dict,
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