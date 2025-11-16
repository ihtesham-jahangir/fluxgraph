# fluxgraph/core/universal_rag.py
"""
Universal Document RAG Connector for FluxGraph.
(Now with data-source-agnostic ingestion)
"""
import os
import logging
import uuid
from typing import List, Dict, Any, Optional, Union
import asyncio

# --- CRITICAL UPDATE: Correct LangChain v0.2.x+ imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    WebBaseLoader  # <-- Add WebBaseLoader for URL ingestion
)
from langchain_core.documents import Document as LCDocument

# Import FluxGraph RAG interface
try:
    # We now import Document as well
    from .rag import RAGConnector, Document
    RAG_INTERFACE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        from fluxgraph.core.rag import RAGConnector, Document
        RAG_INTERFACE_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        RAG_INTERFACE_AVAILABLE = False
        # Define dummy classes for type hints
        class RAGConnector: pass
        @dataclass
        class Document:
            content: str
            metadata: Dict[str, Any] = field(default_factory=dict)
            doc_id: Optional[str] = None
        logging.getLogger(__name__).debug("RAG interface not found. RAG features will be disabled.")

logger = logging.getLogger(__name__)
# --- END OF CRITICAL UPDATE ---

class UniversalRAG(RAGConnector):
    """
    A universal RAG connector supporting ingestion from various data sources
    and querying via LangChain + ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "fluxgraph_kb",
        embedding_model_name: str = "all-MiniLM-L6-v2", # Good default, small, fast
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initializes the UniversalRAG connector.
        (No changes to __init__)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info("Initializing UniversalRAG with ChromaDB at '%s' using model '%s'", persist_directory, embedding_model_name)

        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
            logger.debug("HuggingFace embedding model '%s' loaded.", embedding_model_name)
        except Exception as e:
            logger.error("Failed to load embedding model '%s': %s", embedding_model_name, e)
            raise RuntimeError(f"Failed to initialize embedding model: {e}") from e

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.debug("Text splitter configured (chunk_size=%d, chunk_overlap=%d)", chunk_size, chunk_overlap)

        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
            logger.info("ChromaDB vector store initialized/loaded at '%s'", persist_directory)
        except Exception as e:
            logger.error("Failed to initialize ChromaDB at '%s': %s", persist_directory, e)
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}") from e


    # --- NEW: Core Ingestion Method ---
    async def ingest_documents(self, documents: List[Document]) -> List[str]:
        """
        Ingests a list of standard Document objects into the RAG system.
        This is the new core ingestion method.

        Args:
            documents (List[Document]): A list of Document objects to ingest.

        Returns:
            List[str]: A list of the unique document IDs that were ingested.
        """
        logger.info(f"Starting ingestion of {len(documents)} documents.")
        
        # Convert our standard Document to LangChain's Document
        langchain_documents: List[LCDocument] = []
        doc_ids_processed = set()
        
        for doc in documents:
            # Ensure each document has a unique ID for metadata
            if not doc.doc_id:
                doc.doc_id = str(uuid.uuid4())
            doc.metadata["doc_id"] = doc.doc_id
            doc_ids_processed.add(doc.doc_id)
            
            langchain_documents.append(
                LCDocument(
                    page_content=doc.content,
                    metadata=doc.metadata
                )
            )

        try:
            # --- 1. Split Documents into Chunks ---
            split_documents: List[LCDocument] = self.text_splitter.split_documents(langchain_documents)
            logger.debug(f"Split {len(documents)} documents into {len(split_documents)} chunks.")

            if not split_documents:
                logger.warning("No text content was extracted from the documents. Skipping ingestion.")
                return []

            # --- 2. Add Chunks to Vector Store ---
            texts = [doc.page_content for doc in split_documents]
            metadatas = [doc.metadata for doc in split_documents]

            # Run in executor to prevent blocking the async event loop
            await asyncio.get_event_loop().run_in_executor(
                None, self.vector_store.add_texts, texts, metadatas
            )

            logger.info(f"Successfully ingested {len(documents)} documents ({len(split_documents)} chunks) into collection '{self.collection_name}'.")
            
            # Return the unique doc_ids
            return list(doc_ids_processed)

        except Exception as e:
            error_msg = f"Ingestion failed for batch of {len(documents)} documents: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    # --- MODIFIED: Old 'ingest' is now a wrapper ---
    async def ingest(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        (DEPRECATED - use ingest_from_file)
        Ingests a document from a *single file path*.
        This is now a wrapper for `ingest_from_file`.
        """
        doc_ids = await self.ingest_from_file(file_path, metadata)
        return bool(doc_ids)

    async def ingest_from_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Ingests a single document from a file path.
        This method now uses the core `ingest_documents`.
        """
        if not os.path.isfile(file_path):
            error_msg = f"Ingestion failed: File not found at path '{file_path}'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Loading file for ingestion: '{file_path}'")
        try:
            # Use UnstructuredFileLoader to get content
            loader = UnstructuredFileLoader(file_path=file_path, mode="single")
            
            # Run in executor to prevent blocking
            lc_docs: List[LCDocument] = await asyncio.get_event_loop().run_in_executor(None, loader.load)
            
            if not lc_docs:
                logger.warning(f"No content extracted from '{file_path}'. Skipping.")
                return []

            # Create our standard Document
            doc_metadata = metadata or {}
            doc_metadata.update(lc_docs[0].metadata) # Add metadata from loader
            doc_metadata["file_source"] = file_path # Ensure source is set
            
            document = Document(
                content=lc_docs[0].page_content,
                metadata=doc_metadata
            )
            
            # Call the new core ingestion method
            return await self.ingest_documents([document])

        except Exception as e:
            error_msg = f"Ingestion failed for file '{file_path}': {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    # --- NEW: URL Ingestion Method ---
    async def ingest_from_urls(self, urls: List[str], metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Ingests content from a list of web URLs.
        This demonstrates the new data-source-agnostic pipeline.
        """
        logger.info(f"Starting ingestion from {len(urls)} URLs.")
        
        # 1. Load content from URLs using WebBaseLoader
        try:
            loader = WebBaseLoader(urls)
            # We can also run this in an executor
            lc_docs: List[LCDocument] = await asyncio.get_event_loop().run_in_executor(None, loader.load)
        except Exception as e:
            logger.error(f"Failed to load content from URLs: {e}", exc_info=True)
            return [] # Return empty list if loading fails

        if not lc_docs:
            logger.warning(f"No content extracted from URLs: {urls}. Skipping.")
            return []

        # 2. Convert to standard Document objects
        documents: List[Document] = []
        for lc_doc in lc_docs:
            # Combine provided metadata with loader metadata
            doc_metadata = (metadata or {}).copy()
            doc_metadata.update(lc_doc.metadata) # Add metadata from loader (like 'source')
            documents.append(
                Document(
                    content=lc_doc.page_content,
                    metadata=doc_metadata
                )
            )

        # 3. Call the core ingestion method
        return await self.ingest_documents(documents)


    async def query(self, question: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Queries the RAG system to retrieve relevant document chunks.
        (This method remains unchanged)
        """
        logger.debug("RAG query initiated: '%s' (top_k=%d, filters=%s)", question, top_k, filters)
        try:
            # Perform similarity search using LangChain's Chroma wrapper
            search_kwargs = {"k": top_k}
            if filters:
                # Chroma supports filtering by metadata
                search_kwargs["filter"] = filters

            # Similarity search (usually sync, run in executor if needed to prevent blocking)
            results: List[LCDocument] = await asyncio.get_event_loop().run_in_executor(
                None, self.vector_store.similarity_search, question, **search_kwargs
            )

            # Convert LangChain Documents to standard Dict format
            formatted_results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]

            logger.debug("RAG query for '%s' returned %d results.", question, len(formatted_results))
            return formatted_results

        except Exception as e:
            error_msg = f"RAG query failed for question '{question}': {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Gets basic statistics about the ChromaDB collection.
        (This method remains unchanged)
        """
        try:
            # Access the underlying ChromaDB Collection object
            chroma_collection = self.vector_store._collection
            if chroma_collection:
                # Get the count of items in the collection
                count_result = chroma_collection.count()
                return {
                    "collection_name": self.collection_name,
                    "document_chunk_count": count_result
                }
            else:
                logger.warning("Could not access underlying ChromaDB collection for stats.")
                return {}
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}