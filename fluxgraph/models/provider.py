# fluxgraph/models/provider.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class ModelProvider(ABC):
    """
    Abstract base class for LLM model providers.
    Defines the interface for generating text.
    """

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a text response based on the prompt.

        Args:
            prompt (str): The input prompt for the model.
            **kwargs: Additional arguments for the model call
                      (e.g., temperature, max_tokens, system_message).

        Returns:
            Dict[str, Any]: A dictionary containing the model's response.
                           Expected to have a 'text' key with the generated text.
                           May include other metadata.
        """
        pass
