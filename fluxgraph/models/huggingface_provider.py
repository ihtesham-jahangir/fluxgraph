"""
Model provider for Hugging Face Inference API.
"""
import os
from typing import Dict, Any, Optional, List
from huggingface_hub import InferenceClient
from .provider import ModelProvider

class HuggingFaceProvider(ModelProvider):
    """
    Model provider for Hugging Face's Inference API.
    
    Supports chat completion models via InferenceClient.
    Requires a Hugging Face token with appropriate permissions.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "google/gemma-2-2b-it",
        base_url: Optional[str] = None,
        **client_kwargs
    ):
        """
        Initializes the Hugging Face provider.

        Args:
            api_key (str, optional): Hugging Face API token.
                                     Defaults to `os.getenv("HF_TOKEN")`.
            model (str, optional): The Hugging Face model ID to use.
                                   Defaults to "meta-llama/Meta-Llama-3-8B-Instruct".
            base_url (str, optional): Base URL for the inference endpoint.
                                      Useful for TGI deployments. Defaults to None.
            **client_kwargs: Additional keyword arguments passed to InferenceClient.
        """
        self.api_key = api_key or os.getenv("HF_TOKEN")
        if not self.api_key:
            raise ValueError(
                "Hugging Face API key is required. Set HF_TOKEN environment "
                "variable or pass it directly."
            )
        
        self.model = model
        self.client = InferenceClient(
            token=self.api_key,
            base_url=base_url,
            **client_kwargs
        )

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the Hugging Face Inference API.

        Args:
            prompt (str): The user's prompt. This will be treated as the user message.
            **kwargs: Additional arguments for the chat completion.
                      Supported keys:
                      - temperature (float)
                      - max_tokens (int)
                      - top_p (float)
                      - top_k (int)
                      - repetition_penalty (float)
                      - system_message (str): To set a system message.
                      - stop_sequences (List[str]): Sequences where generation stops.

        Returns:
            Dict[str, Any]: The structured response containing text, model info, usage, etc.
        """
        system_message = kwargs.pop("system_message", None)
        temperature = kwargs.pop("temperature", None)
        max_tokens = kwargs.pop("max_tokens", None)
        top_p = kwargs.pop("top_p", None)
        top_k = kwargs.pop("top_k", None)
        repetition_penalty = kwargs.pop("repetition_penalty", None)
        stop_sequences = kwargs.pop("stop_sequences", None)
        
        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            hf_kwargs = {}
            if temperature is not None:
                hf_kwargs["temperature"] = temperature
            if max_tokens is not None:
                hf_kwargs["max_tokens"] = max_tokens  # Use max_tokens instead of max_new_tokens
            if top_p is not None:
                hf_kwargs["top_p"] = top_p
            if top_k is not None:
                hf_kwargs["top_k"] = top_k
            if repetition_penalty is not None:
                hf_kwargs["repetition_penalty"] = repetition_penalty
            if stop_sequences is not None:
                hf_kwargs["stop_sequences"] = stop_sequences
            
            hf_kwargs.update(kwargs)

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **hf_kwargs
            )
            
            text_content = ""
            if completion.choices and completion.choices[0].message:
                text_content = completion.choices[0].message.content or ""
                text_content = text_content.strip()

            model_info = getattr(completion, 'model', self.model)

            usage_info = {}
            if hasattr(completion, 'usage'):
                usage = completion.usage
                if usage:
                    usage_info = {
                        "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(usage, 'completion_tokens', 0),
                        "total_tokens": getattr(usage, 'total_tokens', 0),
                    }

            return {
                "text": text_content,
                "model": model_info,
                "usage": usage_info,
                "raw_response": str(completion)
            }
        except Exception as e:
            raise RuntimeError(f"Hugging Face Inference API call failed: {e}") from e