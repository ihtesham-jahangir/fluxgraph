"""
Advanced Memory System for FluxGraph
Provides hybrid memory with short-term, long-term, and episodic storage
"""

import asyncio
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"  # Session-based, temporary
    LONG_TERM = "long_term"    # Persistent, vector-based
    EPISODIC = "episodic"      # Specific past interactions
    SEMANTIC = "semantic"      # General knowledge


@dataclass
class MemoryEntry:
    """Single memory entry."""
    id: str
    content: str
    memory_type: MemoryType
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    importance: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance
        }


class AdvancedMemory:
    """
    Advanced memory system with multiple storage types.
    
    Features:
    - Short-term memory: Session-based, fast access
    - Long-term memory: Vector embeddings, semantic search
    - Episodic memory: Remember specific interactions
    - Semantic memory: General knowledge learned over time
    - Memory consolidation: Move important memories to long-term
    - Forgetting: Remove old, unimportant memories
    
    Example:
        memory = AdvancedMemory()
        
        # Store short-term memory
        memory.store("user asked about pricing", MemoryType.SHORT_TERM)
        
        # Recall similar memories
        results = memory.recall_similar("what are the prices?", k=5)
        
        # Consolidate important memories
        memory.consolidate()
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = "all-MiniLM-L6-v2",
        short_term_capacity: int = 100,
        long_term_capacity: int = 10000,
        consolidation_threshold: float = 0.7
    ):
        self.short_term: Dict[str, MemoryEntry] = {}
        self.long_term: Dict[str, MemoryEntry] = {}
        self.episodic: List[MemoryEntry] = []
        self.semantic: Dict[str, MemoryEntry] = {}
        
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        self.consolidation_threshold = consolidation_threshold
        
        # Initialize embedding model (lazy load)
        self.embedding_model_name = embedding_model
        self._embedding_model = None
        
        logger.info(f"AdvancedMemory initialized (ST:{short_term_capacity}, LT:{long_term_capacity})")
    
    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, embeddings disabled")
                self._embedding_model = None
        return self._embedding_model
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for memory entry."""
        return hashlib.md5(f"{content}{datetime.utcnow().timestamp()}".encode()).hexdigest()[:12]
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text."""
        model = self._get_embedding_model()
        if model is None:
            return None
        
        try:
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0
    ) -> str:
        """
        Store a memory entry.
        
        Args:
            content: Memory content
            memory_type: Type of memory storage
            metadata: Additional metadata
            importance: Importance score (0-1)
            
        Returns:
            Memory ID
        """
        memory_id = self._generate_id(content)
        
        # Generate embedding for long-term and semantic memory
        embedding = None
        if memory_type in [MemoryType.LONG_TERM, MemoryType.SEMANTIC]:
            embedding = self._get_embedding(content)
        
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            metadata=metadata or {},
            importance=importance
        )
        
        # Store in appropriate memory type
        if memory_type == MemoryType.SHORT_TERM:
            self.short_term[memory_id] = entry
            # Enforce capacity limit
            if len(self.short_term) > self.short_term_capacity:
                self._evict_short_term()
        
        elif memory_type == MemoryType.LONG_TERM:
            self.long_term[memory_id] = entry
            if len(self.long_term) > self.long_term_capacity:
                self._evict_long_term()
        
        elif memory_type == MemoryType.EPISODIC:
            self.episodic.append(entry)
        
        elif memory_type == MemoryType.SEMANTIC:
            self.semantic[memory_id] = entry
        
        logger.debug(f"Stored {memory_type.value} memory: {memory_id}")
        return memory_id
    
    def recall_similar(
        self,
        query: str,
        k: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        min_similarity: float = 0.0
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Recall similar memories using semantic search.
        
        Args:
            query: Query text
            k: Number of results
            memory_types: Types of memory to search (default: all)
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (MemoryEntry, similarity_score) tuples
        """
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            logger.warning("Cannot perform semantic search without embeddings")
            return []
        
        # Collect all memories to search
        memories_to_search = []
        
        if memory_types is None:
            memory_types = list(MemoryType)
        
        for mem_type in memory_types:
            if mem_type == MemoryType.SHORT_TERM:
                memories_to_search.extend(self.short_term.values())
            elif mem_type == MemoryType.LONG_TERM:
                memories_to_search.extend(self.long_term.values())
            elif mem_type == MemoryType.EPISODIC:
                memories_to_search.extend(self.episodic)
            elif mem_type == MemoryType.SEMANTIC:
                memories_to_search.extend(self.semantic.values())
        
        # Calculate similarities
        results = []
        for memory in memories_to_search:
            if memory.embedding is None:
                continue
            
            # Cosine similarity
            similarity = np.dot(query_embedding, memory.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding)
            )
            
            if similarity >= min_similarity:
                # Update access count
                memory.access_count += 1
                results.append((memory, float(similarity)))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def recall_recent(
        self,
        k: int = 10,
        memory_type: Optional[MemoryType] = None,
        time_window: Optional[timedelta] = None
    ) -> List[MemoryEntry]:
        """
        Recall recent memories.
        
        Args:
            k: Number of results
            memory_type: Specific memory type (default: all)
            time_window: Time window to search within
            
        Returns:
            List of recent memories
        """
        memories = []
        
        if memory_type is None or memory_type == MemoryType.SHORT_TERM:
            memories.extend(self.short_term.values())
        if memory_type is None or memory_type == MemoryType.LONG_TERM:
            memories.extend(self.long_term.values())
        if memory_type is None or memory_type == MemoryType.EPISODIC:
            memories.extend(self.episodic)
        if memory_type is None or memory_type == MemoryType.SEMANTIC:
            memories.extend(self.semantic.values())
        
        # Filter by time window
        if time_window:
            cutoff = datetime.utcnow() - time_window
            memories = [m for m in memories if m.timestamp >= cutoff]
        
        # Sort by timestamp
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        
        return memories[:k]
    
    def consolidate(self):
        """
        Consolidate important short-term memories to long-term.
        
        Moves frequently accessed or important short-term memories
        to long-term storage.
        """
        consolidated_count = 0
        
        for memory_id, memory in list(self.short_term.items()):
            # Calculate importance score
            importance_score = (
                memory.importance * 0.5 +
                min(memory.access_count / 10.0, 1.0) * 0.5
            )
            
            if importance_score >= self.consolidation_threshold:
                # Move to long-term
                memory.memory_type = MemoryType.LONG_TERM
                
                # Generate embedding if not exists
                if memory.embedding is None:
                    memory.embedding = self._get_embedding(memory.content)
                
                self.long_term[memory_id] = memory
                del self.short_term[memory_id]
                consolidated_count += 1
                
                logger.debug(f"Consolidated memory {memory_id} to long-term")
        
        logger.info(f"Consolidated {consolidated_count} memories to long-term")
    
    def forget(self, threshold_days: int = 30):
        """
        Remove old, unimportant memories.
        
        Args:
            threshold_days: Remove memories older than this many days
        """
        cutoff = datetime.utcnow() - timedelta(days=threshold_days)
        forgotten_count = 0
        
        # Forget old short-term memories
        for memory_id, memory in list(self.short_term.items()):
            if memory.timestamp < cutoff and memory.access_count < 2:
                del self.short_term[memory_id]
                forgotten_count += 1
        
        logger.info(f"Forgot {forgotten_count} old memories")
    
    def _evict_short_term(self):
        """Evict least important short-term memories."""
        # Sort by importance and access count
        memories = sorted(
            self.short_term.values(),
            key=lambda m: (m.importance, m.access_count)
        )
        
        # Remove least important 10%
        to_remove = max(1, len(memories) // 10)
        for memory in memories[:to_remove]:
            del self.short_term[memory.id]
            logger.debug(f"Evicted short-term memory: {memory.id}")
    
    def _evict_long_term(self):
        """Evict least important long-term memories."""
        memories = sorted(
            self.long_term.values(),
            key=lambda m: (m.importance, m.access_count, m.timestamp)
        )
        
        to_remove = max(1, len(memories) // 20)
        for memory in memories[:to_remove]:
            del self.long_term[memory.id]
            logger.debug(f"Evicted long-term memory: {memory.id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "episodic_count": len(self.episodic),
            "semantic_count": len(self.semantic),
            "total_memories": (
                len(self.short_term) +
                len(self.long_term) +
                len(self.episodic) +
                len(self.semantic)
            ),
            "embeddings_enabled": self._embedding_model is not None
        }
    
    def clear(self, memory_type: Optional[MemoryType] = None):
        """Clear memories."""
        if memory_type is None:
            self.short_term.clear()
            self.long_term.clear()
            self.episodic.clear()
            self.semantic.clear()
            logger.info("Cleared all memories")
        elif memory_type == MemoryType.SHORT_TERM:
            self.short_term.clear()
        elif memory_type == MemoryType.LONG_TERM:
            self.long_term.clear()
        elif memory_type == MemoryType.EPISODIC:
            self.episodic.clear()
        elif memory_type == MemoryType.SEMANTIC:
            self.semantic.clear()


# Example usage
if __name__ == "__main__":
    async def main():
        memory = AdvancedMemory()
        
        # Store some memories
        memory.store("User asked about pricing plans", MemoryType.SHORT_TERM, importance=0.8)
        memory.store("User prefers monthly billing", MemoryType.SHORT_TERM, importance=0.9)
        memory.store("Python is a programming language", MemoryType.SEMANTIC)
        
        # Recall similar
        results = memory.recall_similar("what are the prices?", k=3)
        print("\nSimilar memories:")
        for entry, score in results:
            print(f"  [{score:.2f}] {entry.content}")
        
        # Stats
        print(f"\nMemory stats: {json.dumps(memory.get_stats(), indent=2)}")
        
        # Consolidate
        memory.consolidate()
        print(f"After consolidation: {memory.get_stats()}")
    
    asyncio.run(main())
