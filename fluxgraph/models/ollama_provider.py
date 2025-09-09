# fluxgraph/models/ollama_provider.py
import os
from typing import Dict, Any, Optional
import httpx
from .provider import ModelProvider

class OllamaProvider(ModelProvider):
    """
    Model provider for Ollama (local models).
    Assumes Ollama is running at http://localhost:11434 by default.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        """
        Initializes the Ollama provider.

        Args:
            model (str): The Ollama model name (e.g., 'llama3', 'mistral').
            base_url (str, optional): The base URL for the Ollama API.
                                      Defaults to "http://localhost:11434".
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient()

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the Ollama API.

        Args:
            prompt (str): The user's prompt.
            **kwargs: Additional arguments like temperature, max_tokens.

        Returns:
            Dict[str, Any]: The structured response.
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        payload.update(kwargs)

        try:
            response = await self.client.post(url, json=payload, timeout=120.0)
            response.raise_for_status()
            data = response.json()

            text_content = data.get("response", "").strip()

            return {
                "text": text_content,
                "model": data.get("model", self.model),
                "done": data.get("done", False),
                "raw_response": data
            }
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API call failed ({e.response.status_code}): {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}") from e
        finally:
             await self.client.aclose()
             self.client = httpx.AsyncClient()
