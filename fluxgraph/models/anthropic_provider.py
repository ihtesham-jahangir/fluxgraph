# fluxgraph/models/anthropic_provider.py
import os
from typing import Dict, Any, Optional
from anthropic import AsyncAnthropic
from .provider import ModelProvider

class AnthropicProvider(ModelProvider):
    """
    Model provider for Anthropic's API (Claude).
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        """
        Initializes the Anthropic provider.

        Args:
            api_key (str, optional): Anthropic API key.
                                     Defaults to `os.getenv("ANTHROPIC_API_KEY")`.
            model (str, optional): The Anthropic model to use.
                                   Defaults to "claude-3-haiku-20240307".
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment "
                "variable or pass it directly."
            )
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the Anthropic API.

        Args:
            prompt (str): The user's prompt.
            **kwargs: Additional arguments like temperature, max_tokens, system_message.

        Returns:
            Dict[str, Any]: The structured response.
        """
        system_message = kwargs.pop("system_message", None)
        max_tokens = kwargs.pop("max_tokens", 1024)

        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.client.messages.create(
                model=self.model,
                messages=messages,
                system=system_message,
                max_tokens=max_tokens,
                **kwargs
            )
            text_content = ""
            if response.content and response.content[0].type == "text":
                 text_content = response.content[0].text.strip()

            return {
                "text": text_content,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens if response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response.usage else 0,
                },
                "raw_response": response.model_dump()
            }
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}") from e
