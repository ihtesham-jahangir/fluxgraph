# fluxgraph/core/streaming.py
"""
Streaming Response System for FluxGraph.
Supports SSE (Server-Sent Events) and WebSocket streaming.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class StreamChunk:
    """Represents a single chunk in a streaming response."""
    
    def __init__(
        self,
        content: str,
        chunk_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.chunk_type = chunk_type
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_sse(self) -> str:
        """Convert chunk to SSE format."""
        data = {
            "content": self.content,
            "type": self.chunk_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
        return f"data: {json.dumps(data)}\n\n"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "type": self.chunk_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class StreamManager:
    """Manages streaming responses for agents."""
    
    def __init__(self):
        self.active_streams: Dict[str, bool] = {}
        logger.info("StreamManager initialized")
    
    async def stream_agent_response(
        self,
        agent_name: str,
        stream_func: AsyncGenerator[str, None],
        session_id: Optional[str] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream agent responses chunk by chunk.
        
        Args:
            agent_name: Name of the agent
            stream_func: Async generator producing text chunks
            session_id: Optional session identifier
        
        Yields:
            StreamChunk objects
        """
        stream_id = f"{agent_name}:{session_id or 'default'}"
        self.active_streams[stream_id] = True
        
        logger.info(f"[Stream:{stream_id}] Starting stream")
        
        try:
            chunk_count = 0
            async for chunk in stream_func:
                if not self.active_streams.get(stream_id, False):
                    logger.warning(f"[Stream:{stream_id}] Stream cancelled")
                    break
                
                chunk_count += 1
                yield StreamChunk(
                    content=chunk,
                    chunk_type="text",
                    metadata={
                        "agent": agent_name,
                        "chunk_index": chunk_count,
                        "session_id": session_id
                    }
                )
            
            # Send completion marker
            yield StreamChunk(
                content="",
                chunk_type="complete",
                metadata={
                    "agent": agent_name,
                    "total_chunks": chunk_count,
                    "session_id": session_id
                }
            )
            
            logger.info(f"[Stream:{stream_id}] Completed with {chunk_count} chunks")
            
        except Exception as e:
            logger.error(f"[Stream:{stream_id}] Error: {e}", exc_info=True)
            yield StreamChunk(
                content=str(e),
                chunk_type="error",
                metadata={"agent": agent_name, "session_id": session_id}
            )
        finally:
            self.active_streams.pop(stream_id, None)
    
    def cancel_stream(self, stream_id: str):
        """Cancel an active stream."""
        if stream_id in self.active_streams:
            self.active_streams[stream_id] = False
            logger.info(f"[Stream:{stream_id}] Cancellation requested")
