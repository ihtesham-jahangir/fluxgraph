# fluxgraph/core/rag.py
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class RAGConnector(ABC):
    """Abstract base class for RAG connectors."""

    @abstractmethod
    async def query(self, question: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query the RAG system."""
        pass

    # Optional: Add method for ingesting documents
    # @abstractmethod
    # async def ingest(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    #     """Ingest a document into the RAG system."""
    #     pass