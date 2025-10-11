# fluxgraph/multimodal/processor.py
"""
Multi-Modal Processing for FluxGraph.
Handles images, audio, and video inputs for agents.
"""

import logging
import base64
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class MediaType(Enum):
    """Supported media types."""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


class MultiModalInput:
    """Represents multi-modal input (text, image, audio, etc.)."""
    
    def __init__(
        self,
        content: Union[str, bytes],
        media_type: MediaType,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.media_type = media_type
        self.metadata = metadata or {}
        
        # Encode binary content
        if isinstance(content, bytes):
            # Store Base64 encoded string along with MIME type in metadata for LLM providers
            mime_type = self.metadata.get("mime_type", f"image/{Path(self.metadata.get('filename', 'default.jpg')).suffix[1:]}")
            # Create Data URI: data:mime_type;base64,content
            self.encoded_content = f"data:{mime_type};base64,{base64.b64encode(content).decode('utf-8')}"
            self.metadata["mime_type"] = mime_type
        else:
            self.encoded_content = content
    
    def to_dict(self) -> Dict[str, Any]:
        # Note: We return the full encoded content string for LLM provider consumption
        return {
            "media_type": self.media_type.value,
            "content": self.encoded_content,
            "metadata": self.metadata
        }


class MultiModalProcessor:
    """
    Processes multi-modal inputs for agents.
    Supports vision (GPT-4V, Claude 3), audio (Whisper), and video.
    """
    
    def __init__(self):
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        self.supported_audio_formats = ['.mp3', '.wav', '.m4a', '.ogg']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.webm']
        logger.info("MultiModalProcessor initialized")
    
    def process_image(
        self,
        image_path: str,
        description: Optional[str] = None
    ) -> MultiModalInput:
        """
        Process an image file for agent input.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if path.suffix.lower() not in self.supported_image_formats:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        
        # Read image
        with open(path, 'rb') as f:
            image_bytes = f.read()
        
        # Determine MIME type (simplified, a full implementation would use mimetypes)
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", 
            ".png": "image/png", ".gif": "image/gif", 
            ".webp": "image/webp"
        }
        mime_type = mime_map.get(path.suffix.lower(), "application/octet-stream")
        
        logger.info(f"[MultiModal] Processed image: {path.name} ({len(image_bytes)} bytes)")
        
        return MultiModalInput(
            content=image_bytes,
            media_type=MediaType.IMAGE,
            metadata={
                "filename": path.name,
                "format": path.suffix,
                "size_bytes": len(image_bytes),
                "description": description,
                "mime_type": mime_type # Explicitly set mime type for LLM
            }
        )
    
    def process_audio(
        self,
        audio_path: str,
        transcribe: bool = True
    ) -> MultiModalInput:
        """
        Process an audio file.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        
        if path.suffix.lower() not in self.supported_audio_formats:
            raise ValueError(f"Unsupported audio format: {path.suffix}")
        
        # Read audio
        with open(path, 'rb') as f:
            audio_bytes = f.read()
        
        metadata = {
            "filename": path.name,
            "format": path.suffix,
            "size_bytes": len(audio_bytes),
            "mime_type": f"audio/{path.suffix[1:]}"
        }
        
        # Transcribe if requested
        if transcribe:
            # Would use Whisper API here
            transcript = "[Transcription would be here]"
            metadata["transcript"] = transcript
        
        logger.info(f"[MultiModal] Processed audio: {path.name} ({len(audio_bytes)} bytes)")
        
        return MultiModalInput(
            content=audio_bytes,
            media_type=MediaType.AUDIO,
            metadata=metadata
        )
    
    def create_multimodal_prompt(
        self,
        text: str,
        media_inputs: List[MultiModalInput]
    ) -> Dict[str, Any]:
        """
        Create a multi-modal prompt content list suitable for a single message
        in the Gemini/OpenAI API structure.
        
        Returns:
            Formatted multi-modal prompt: List of content parts (Dicts/Strings)
        """
        # Gemini format uses a list of text/parts
        content_parts = []
        
        # Add media parts first
        for media_input in media_inputs:
            if media_input.media_type != MediaType.TEXT:
                
                # Extract the base64 content part (remove "data:mime_type;base64,")
                encoded_data = media_input.encoded_content.split("base64,")[1]
                mime_type = media_input.metadata["mime_type"]

                # Add a dict representing the media part - used internally by LLM providers
                content_parts.append({
                    "inline_data": {
                        "data": encoded_data,
                        "mime_type": mime_type
                    }
                })
        
        # Add the text part last
        content_parts.append({"text": text})
        
        logger.info(
            f"[MultiModal] Created prompt with text and {len(media_inputs)} media inputs"
        )
        
        # Return the structured content list
        return content_parts