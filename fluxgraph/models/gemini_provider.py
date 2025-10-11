# fluxgraph/models/gemini_provider.py
"""
Enhanced Google Gemini Provider with full features
"""
import os
from typing import Dict, Any, List, Optional, AsyncIterator
import google.generativeai as genai
from google.api_core import retry
import asyncio

# --- NEW IMPORTS ---
from ..multimodal.processor import MultiModalInput, MultiModalProcessor, MediaType
from google.generativeai.types import Part # Used for constructing multimodal requests
# --- END NEW IMPORTS ---

from .provider import (
    ModelProvider,
    ModelConfig,
    ModelResponse,
    ModelCapability,
    ModelProviderError,
    ModelProviderAuthError,
    ModelProviderRateLimitError,
    ModelProviderTimeoutError
)


class GeminiProvider(ModelProvider):
    """
    Enhanced Google Gemini provider with streaming and error handling.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_instance = None
        
        # Set capabilities
        self._capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.VISION,
            ModelCapability.CODE_GENERATION
        ]
    
    async def initialize(self) -> None:
        """Initialize Gemini client"""
        try:
            if not self.config.api_key:
                raise ModelProviderAuthError("GOOGLE_API_KEY not provided")
            
            genai.configure(api_key=self.config.api_key)
            
            # Configure generation settings
            generation_config = {
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }
            
            self.model_instance = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=generation_config
            )
            
            self._initialized = True
            self.logger.info(f"Gemini provider initialized: {self.config.model_name}")
            
        except Exception as e:
            raise ModelProviderAuthError(f"Failed to initialize Gemini: {e}")
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate text using Gemini"""
        if not self._initialized:
            await self.initialize()
        
        # Combine system message with prompt if provided
        full_prompt = prompt
        if system_message:
            full_prompt = f"System: {system_message}\n\nUser: {prompt}"
        
        params = self._merge_kwargs(**kwargs)
        
        # Build generation config
        generation_config = genai.GenerationConfig(
            temperature=params.get("temperature", self.config.temperature),
            max_output_tokens=params.get("max_tokens", self.config.max_tokens),
            top_p=params.get("top_p", self.config.top_p),
        )
        
        try:
            response = await self._retry_with_backoff(
                self.model_instance.generate_content_async,
                full_prompt,
                generation_config=generation_config
            )
            
            # Extract text
            text_content = ""
            if response.candidates and response.candidates[0].content.parts:
                text_content = response.candidates[0].content.parts[0].text.strip()
            
            # Extract usage
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
            
            return ModelResponse(
                text=text_content,
                model=self.config.model_name,
                provider="gemini",
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
                usage=usage,
                raw_response=response
            )
            
        except Exception as e:
            raise ModelProviderError(f"Gemini generation failed: {e}")
            
    async def multimodal_generate(
        self,
        prompt: str,
        media_inputs: List[MultiModalInput],
        system_message: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate a response based on a prompt and media using Gemini (GPT-4V compatible)."""
        if not self._initialized:
            await self.initialize()

        if ModelCapability.VISION not in self.capabilities:
            raise NotImplementedError(f"Model {self.config.model_name} does not support vision capability.")

        params = self._merge_kwargs(**kwargs)
        
        # Build multimodal parts list
        content_parts = []
        for media_input in media_inputs:
            if media_input.media_type == MediaType.TEXT:
                continue
            
            # Extract base64 and mime type
            # The MultiModalInput stores the Data URI string: "data:mime_type;base64,encoded_data"
            encoded_data = media_input.encoded_content.split("base64,")[1]
            mime_type = media_input.metadata["mime_type"]
                
            # Create the Part object required by the Google API
            content_parts.append(Part.from_bytes(
                data=base64.b64decode(encoded_data), # Decode the base64 string back to bytes
                mime_type=mime_type
            ))
        
        # Add the text prompt
        content_parts.append(prompt)
        
        # Apply system message if available
        system_instruction = system_message
        
        # Build generation config
        generation_config = genai.GenerationConfig(
            temperature=params.get("temperature", self.config.temperature),
            max_output_tokens=params.get("max_tokens", self.config.max_tokens),
            top_p=params.get("top_p", self.config.top_p),
            system_instruction=system_instruction
        )
        
        try:
            response = await self._retry_with_backoff(
                self.model_instance.generate_content_async,
                content_parts, # Pass the list of parts
                generation_config=generation_config
            )

            # Extract text
            text_content = ""
            if response.candidates and response.candidates[0].content.parts:
                text_content = response.candidates[0].content.parts[0].text.strip()
            
            # Extract usage
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
            
            return ModelResponse(
                text=text_content,
                model=self.config.model_name,
                provider="gemini",
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
                usage=usage,
                raw_response=response
            )

        except Exception as e:
            self.logger.error(f"Gemini multimodal generation failed: {e}")
            raise ModelProviderError(f"Gemini multimodal generation failed: {e}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ModelResponse:
        """Chat completion with Gemini"""
        if not self._initialized:
            await self.initialize()
        
        # Convert messages to Gemini format
        history = []
        current_message = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                # Prepend system message to first user message
                continue
            elif role == "user":
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})
        
        # Get the last user message
        if history and history[-1]["role"] == "user":
            current_message = history[-1]["parts"][0]
            history = history[:-1]
        
        try:
            chat = self.model_instance.start_chat(history=history)
            response = await chat.send_message_async(current_message)
            
            text_content = response.text.strip() if response.text else ""
            
            return ModelResponse(
                text=text_content,
                model=self.config.model_name,
                provider="gemini",
                raw_response=response
            )
            
        except Exception as e:
            raise ModelProviderError(f"Gemini chat failed: {e}")
    
    async def stream_generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream generation from Gemini"""
        if not self._initialized:
            await self.initialize()
        
        full_prompt = prompt
        if system_message:
            full_prompt = f"System: {system_message}\n\nUser: {prompt}"
        
        try:
            response = await self.model_instance.generate_content_async(
                full_prompt,
                stream=True
            )
            
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            raise ModelProviderError(f"Gemini streaming failed: {e}")
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry with exponential backoff"""
        max_retries = self.config.retry_attempts
        
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                wait_time = 2 ** attempt
                self.logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                await asyncio.sleep(wait_time)