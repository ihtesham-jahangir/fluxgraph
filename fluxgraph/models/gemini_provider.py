# fluxgraph/models/gemini_provider.py
"""
Model provider for Google Gemini's API.
"""
import os
from typing import Dict, Any, Optional
import google.generativeai as genai
# import google.ai.generativelanguage as glm # Might be needed for advanced safety handling
from .provider import ModelProvider

class GeminiProvider(ModelProvider):
    """
    Model provider for Google's Gemini API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        """
        Initializes the Gemini provider.

        Args:
            api_key (str, optional): Google API key.
                                     Defaults to `os.getenv("GOOGLE_API_KEY")`.
            model (str, optional): The Gemini model to use.
                                   Defaults to "gemini-1.5-flash".
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required for GeminiProvider. Set GOOGLE_API_KEY environment "
                "variable or pass it directly."
            )
        
        # Configure the library with the API key
        genai.configure(api_key=self.api_key)
        
        self.model_name = model
        # Initialize the model client instance
        self.model = genai.GenerativeModel(self.model_name)
        # Note: Setting system_instruction during model creation is possible:
        # self.model = genai.GenerativeModel(self.model_name, system_instruction="...")

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the Google Gemini API.

        Args:
            prompt (str): The user's prompt.
            **kwargs: Additional arguments like temperature, max_tokens.

        Returns:
            Dict[str, Any]: The structured response.
        """
        try:
            # Map common kwargs to Gemini's expected parameters for GenerationConfig
            generation_config_kwargs = {}
            if "temperature" in kwargs:
                generation_config_kwargs["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                 # Gemini uses `max_output_tokens`
                 generation_config_kwargs["max_output_tokens"] = kwargs["max_tokens"]
            
            # Handle system instruction
            # The cleanest way is often to set it on the model instance during initialization.
            # If passed per request, it needs to be handled carefully.
            # Older or current SDK versions might not support `system_instruction` directly
            # in `generate_content_async`. A common workaround is to prepend it to the prompt
            # or rely on the model's initial system instruction.
            system_instruction = kwargs.pop("system_message", None)
            
            # Prepare contents. Prepending system instruction to the prompt if needed
            # and not set globally on the model instance.
            # A more advanced approach would check the SDK version or use model._system_instruction
            # if set during init. For simplicity here, we'll just use the prompt.
            # If you need dynamic system instructions, consider setting it on the model
            # when you create the provider instance or handle it by modifying the prompt contextually.
            contents = prompt

            # Create GenerationConfig object if we have specific settings
            generation_config = genai.GenerationConfig(**generation_config_kwargs) if generation_config_kwargs else None

            # --- Call the async generate method ---
            # Pass contents, generation_config.
            # DO NOT pass system_instruction directly here if your SDK version doesn't support it.
            response = await self.model.generate_content_async(
                contents=contents,
                generation_config=generation_config # Pass the config object
                # system_instruction=system_instruction # REMOVED: Causes the error
                # **kwargs # REMOVED: Dangerous, might pass unsupported arguments
            )
            
            # --- Extract information from the response ---
            text_content = ""
            if response.candidates and response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                if hasattr(part, 'text'):
                    text_content = part.text.strip()
            
            model_info = getattr(response, 'model', self.model_name)

            usage_metadata = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_metadata = {
                    "prompt_token_count": response.usage_metadata.prompt_token_count,
                    "candidates_token_count": response.usage_metadata.candidates_token_count,
                    "total_token_count": response.usage_metadata.total_token_count,
                }
            
            # Optional: Extract safety ratings
            safety_ratings = []
            # ... (code for safety ratings if needed)

            return {
                "text": text_content,
                "model": model_info,
                "usage": usage_metadata,
                # "safety_ratings": safety_ratings, # Include if needed
                # Use .to_dict() if available, otherwise str() for raw response
                "raw_response": response.to_dict() if hasattr(response, 'to_dict') else str(response)
            }
        except Exception as e:
            # Provide more context in the error message
            raise RuntimeError(f"Gemini API call failed for model '{self.model_name}': {e}") from e
