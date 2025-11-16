# fluxgraph/core/rag.py
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field # <-- Added

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    A standard container for data to be ingested into the RAG system.
    This makes the RAG system data-source-agnostic.
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None


class RAGConnector(ABC):
    """Abstract base class for RAG connectors."""

    @abstractmethod
    async def ingest_documents(self, documents: List[Document]) -> List[str]:
        """
        Ingest a list of standard Document objects into the RAG system.
        Returns a list of document IDs that were added.
        """
        pass

    @abstractmethod
    async def query(self, question: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query the RAG system."""
        pass