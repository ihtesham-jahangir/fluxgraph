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
            self.encoded_content = base64.b64encode(content).decode('utf-8')
        else:
            self.encoded_content = content
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "media_type": self.media_type.value,
            "content": self.encoded_content if self.media_type != MediaType.TEXT else self.content,
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
        
        Args:
            image_path: Path to image file
            description: Optional image description
        
        Returns:
            MultiModalInput object
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if path.suffix.lower() not in self.supported_image_formats:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        
        # Read image
        with open(path, 'rb') as f:
            image_bytes = f.read()
        
        logger.info(f"[MultiModal] Processed image: {path.name} ({len(image_bytes)} bytes)")
        
        return MultiModalInput(
            content=image_bytes,
            media_type=MediaType.IMAGE,
            metadata={
                "filename": path.name,
                "format": path.suffix,
                "size_bytes": len(image_bytes),
                "description": description
            }
        )
    
    def process_audio(
        self,
        audio_path: str,
        transcribe: bool = True
    ) -> MultiModalInput:
        """
        Process an audio file.
        
        Args:
            audio_path: Path to audio file
            transcribe: Whether to transcribe audio to text
        
        Returns:
            MultiModalInput object
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
            "size_bytes": len(audio_bytes)
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
        Create a multi-modal prompt combining text and media.
        
        Args:
            text: Text prompt
            media_inputs: List of media inputs
        
        Returns:
            Formatted multi-modal prompt
        """
        prompt = {
            "type": "multimodal",
            "text": text,
            "media": [input.to_dict() for input in media_inputs]
        }
        
        logger.info(
            f"[MultiModal] Created prompt with text and {len(media_inputs)} media inputs"
        )
        
        return prompt
