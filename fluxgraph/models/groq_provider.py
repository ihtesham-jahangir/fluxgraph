# fluxgraph/models/groq_provider.py
import os
from typing import Dict, Any, Optional
from groq import AsyncGroq  # requires groq>=0.4.0

from .provider import ModelProvider


class GroqProvider(ModelProvider):
    """
    Model provider for Groq's API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        """
        Initializes the Groq provider.

        Args:
            api_key (str, optional): Groq API key.
                                     Defaults to `os.getenv("GROQ_API_KEY")`.
            model (str, optional): The Groq model to use.
                                   Defaults to "llama-3.1-8b-instant".
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY environment "
                "variable or pass it directly."
            )
        self.client = AsyncGroq(api_key=self.api_key)
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the Groq API.

        Args:
            prompt (str): The user's prompt.
            **kwargs: Additional arguments like temperature, max_tokens, system_message.

        Returns:
            Dict[str, Any]: The structured response.
        """
        system_message = kwargs.pop("system_message", None)
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            text_content = response.choices[0].message.content.strip() if response.choices else ""

            return {
                "text": text_content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                "raw_response": response.model_dump()
            }
        except Exception as e:
            raise RuntimeError(f"Groq API call failed: {e}") from e