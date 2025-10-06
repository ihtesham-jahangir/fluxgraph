"""
Agent Caching System for FluxGraph
Provides semantic and exact-match caching to reduce LLM calls
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache matching strategies."""
    EXACT = "exact"           # Exact string match
    SEMANTIC = "semantic"     # Embedding similarity
    HYBRID = "hybrid"         # Try exact first, then semantic


@dataclass
class CacheEntry:
    """Cached result entry."""
    key: str
    query: str
    result: Any
    embedding: Optional[np.ndarray] = None
    hits: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    ttl: Optional[int] = None  # Time to live in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "key": self.key,
            "query": self.query,
            "hits": self.hits,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "ttl": self.ttl,
            "metadata": self.metadata
        }


class AgentCache:
    """
    High-performance caching system for agent responses.
    
    Features:
    - Exact match caching (hash-based)
    - Semantic caching (embedding similarity)
    - TTL (time-to-live) support
    - LRU eviction
    - Cache statistics
    - Hit rate tracking
    
    Example:
        cache = AgentCache(strategy=CacheStrategy.SEMANTIC)
        
        # Check cache
        if cached := cache.get("what is Python?", threshold=0.9):
            return cached
        
        # Miss - call LLM
        result = await expensive_llm_call(query)
        
        # Store in cache
        cache.set("what is Python?", result, ttl=3600)
    """
    
    def __init__(
        self,
        strategy: CacheStrategy = CacheStrategy.HYBRID,
        max_size: int = 1000,
        default_ttl: Optional[int] = 3600,  # 1 hour
        embedding_model: Optional[str] = "all-MiniLM-L6-v2",
        semantic_threshold: float = 0.85
    ):
        self.strategy = strategy
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.semantic_threshold = semantic_threshold
        
        # Storage
        self.exact_cache: Dict[str, CacheEntry] = {}
        self.semantic_cache: Dict[str, CacheEntry] = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Embedding model (lazy load)
        self.embedding_model_name = embedding_model
        self._embedding_model = None
        
        logger.info(f"AgentCache initialized (strategy={strategy.value}, max_size={max_size})")
    
    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None and self.strategy in [CacheStrategy.SEMANTIC, CacheStrategy.HYBRID]:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, semantic caching disabled")
                self._embedding_model = None
        return self._embedding_model
    
    def _generate_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()
    
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
    
    def get(
        self,
        query: str,
        threshold: Optional[float] = None
    ) -> Optional[Any]:
        """
        Get cached result for query.
        
        Args:
            query: Query string
            threshold: Similarity threshold for semantic matching
            
        Returns:
            Cached result or None if not found
        """
        threshold = threshold or self.semantic_threshold
        
        # Try exact match first (always fast)
        key = self._generate_key(query)
        if key in self.exact_cache:
            entry = self.exact_cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.exact_cache[key]
                logger.debug(f"Expired cache entry removed: {key}")
            else:
                entry.hits += 1
                entry.last_accessed = datetime.utcnow()
                self.hits += 1
                logger.debug(f"Cache HIT (exact): {query[:50]}...")
                return entry.result
        
        # Try semantic match if enabled
        if self.strategy in [CacheStrategy.SEMANTIC, CacheStrategy.HYBRID]:
            query_embedding = self._get_embedding(query)
            if query_embedding is not None:
                best_match = None
                best_similarity = 0.0
                
                for entry in self.semantic_cache.values():
                    if entry.is_expired():
                        continue
                    
                    if entry.embedding is None:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, entry.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(entry.embedding)
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = entry
                
                # Check if match meets threshold
                if best_match and best_similarity >= threshold:
                    best_match.hits += 1
                    best_match.last_accessed = datetime.utcnow()
                    self.hits += 1
                    logger.debug(f"Cache HIT (semantic, {best_similarity:.3f}): {query[:50]}...")
                    return best_match.result
        
        # Cache miss
        self.misses += 1
        logger.debug(f"Cache MISS: {query[:50]}...")
        return None
    
    def set(
        self,
        query: str,
        result: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store result in cache.
        
        Args:
            query: Query string
            result: Result to cache
            ttl: Time to live in seconds (None = use default)
            metadata: Additional metadata
        """
        key = self._generate_key(query)
        ttl = ttl if ttl is not None else self.default_ttl
        
        # Generate embedding for semantic cache
        embedding = None
        if self.strategy in [CacheStrategy.SEMANTIC, CacheStrategy.HYBRID]:
            embedding = self._get_embedding(query)
        
        entry = CacheEntry(
            key=key,
            query=query,
            result=result,
            embedding=embedding,
            ttl=ttl,
            metadata=metadata or {}
        )
        
        # Store in exact cache
        self.exact_cache[key] = entry
        
        # Store in semantic cache if embedding available
        if embedding is not None:
            self.semantic_cache[key] = entry
        
        # Enforce size limits
        if len(self.exact_cache) > self.max_size:
            self._evict_lru()
        
        logger.debug(f"Cached result: {query[:50]}...")
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        # Sort by last access time
        sorted_entries = sorted(
            self.exact_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest 10%
        to_remove = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:to_remove]:
            del self.exact_cache[key]
            if key in self.semantic_cache:
                del self.semantic_cache[key]
            self.evictions += 1
        
        logger.debug(f"Evicted {to_remove} cache entries (LRU)")
    
    def clear(self):
        """Clear all cache entries."""
        count = len(self.exact_cache)
        self.exact_cache.clear()
        self.semantic_cache.clear()
        logger.info(f"Cleared {count} cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "strategy": self.strategy.value,
            "size": len(self.exact_cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "evictions": self.evictions,
            "total_requests": total_requests
        }
    
    def prune_expired(self):
        """Remove expired entries."""
        expired_count = 0
        
        for key in list(self.exact_cache.keys()):
            if self.exact_cache[key].is_expired():
                del self.exact_cache[key]
                if key in self.semantic_cache:
                    del self.semantic_cache[key]
                expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Pruned {expired_count} expired cache entries")


# Example usage
if __name__ == "__main__":
    async def main():
        cache = AgentCache(strategy=CacheStrategy.HYBRID)
        
        # Simulate expensive LLM call
        async def expensive_llm_call(query: str) -> str:
            await asyncio.sleep(1)  # Simulate delay
            return f"Response to: {query}"
        
        # First call - cache miss
        query1 = "What is Python?"
        start = time.time()
        if result := cache.get(query1):
            print(f"Cached: {result}")
        else:
            result = await expensive_llm_call(query1)
            cache.set(query1, result)
            print(f"Computed: {result} ({time.time()-start:.2f}s)")
        
        # Second call - cache hit (exact)
        start = time.time()
        if result := cache.get(query1):
            print(f"Cached: {result} ({time.time()-start:.2f}s)")
        
        # Similar query - cache hit (semantic)
        query2 = "What is the Python programming language?"
        start = time.time()
        if result := cache.get(query2, threshold=0.8):
            print(f"Cached (semantic): {result} ({time.time()-start:.2f}s)")
        
        # Stats
        print(f"\nCache stats: {json.dumps(cache.get_stats(), indent=2)}")
    
    asyncio.run(main())
