# fluxgraph/core/universal_rag.py
"""
Universal Document RAG Connector for FluxGraph.

This module provides `UniversalRAG`, a concrete implementation of the
`RAGConnector` interface. It leverages `langchain` and `unstructured`
to load and process a wide variety of document types (PDF, DOCX, TXT, etc.)
and uses `ChromaDB` for vector storage and retrieval.

Features:
- Single `ingest(file_path, metadata)` method for all supported document types.
- Unified `query(question)` method returning relevant document chunks.
- Uses LangChain's document loaders, splitters, and embedding models.
- Integrates with ChromaDB for efficient similarity search.
"""
import os
import logging
import uuid
from typing import List, Dict, Any, Optional, Union
import asyncio

# --- CRITICAL UPDATE: Correct LangChain v0.2.x+ imports ---
# Import the correct embedding model class
# Requires: pip install langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings # <-- Corrected Import

# Import the correct Chroma vector store class
# Requires: pip install langchain-chroma
from langchain_chroma import Chroma # <-- Corrected Import

# Import other necessary LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    # Add more loaders as needed or rely on UnstructuredFileLoader's magic
)
from langchain_core.documents import Document as LCDocument

# Import FluxGraph RAG interface
try:
    from .rag import RAGConnector # Try relative import
    RAG_INTERFACE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        from fluxgraph.core.rag import RAGConnector # Try absolute import
        RAG_INTERFACE_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        RAG_INTERFACE_AVAILABLE = False
        # Define a dummy class for type hints if RAG interface is not available
        class RAGConnector: pass
        logging.getLogger(__name__).debug("RAG interface not found. RAG features will be disabled.")

logger = logging.getLogger(__name__)
# --- END OF CRITICAL UPDATE ---

class UniversalRAG(RAGConnector):
    """
    A universal RAG connector supporting ingestion of various document types
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

        Args:
            persist_directory (str): Directory to persist the ChromaDB database.
            collection_name (str): Name of the ChromaDB collection.
            embedding_model_name (str): Name of the sentence-transformers model.
            chunk_size (int): Size of text chunks for splitting documents.
            chunk_overlap (int): Overlap between chunks.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info("Initializing UniversalRAG with ChromaDB at '%s' using model '%s'", persist_directory, embedding_model_name)

        # 1. Initialize Embedding Model
        # Consider caching the model if used frequently
        try:
            # --- CRITICAL UPDATE: Use the correctly imported class ---
            # OLD/Wrong (leads to NameError if SentenceTransformerEmbeddings not imported):
            # self.embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
            
            # NEW/Correct (using langchain_huggingface):
            self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
            # --- END OF CRITICAL UPDATE ---
            
            logger.debug("HuggingFace embedding model '%s' loaded.", embedding_model_name)
        except Exception as e:
            logger.error("Failed to load embedding model '%s': %s", embedding_model_name, e)
            raise RuntimeError(f"Failed to initialize embedding model: {e}") from e

        # 2. Initialize Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.debug("Text splitter configured (chunk_size=%d, chunk_overlap=%d)", chunk_size, chunk_overlap)

        # 3. Initialize ChromaDB Vector Store
        try:
            # --- CRITICAL UPDATE: Use the correctly imported class ---
            # Ensure this line uses the `Chroma` imported from `langchain_chroma`
            self.vector_store = Chroma( # <-- This `Chroma` must be the one from `langchain_chroma`
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
            # --- END OF CRITICAL UPDATE ---
            logger.info("ChromaDB vector store initialized/loaded at '%s'", persist_directory)
        except Exception as e:
            logger.error("Failed to initialize ChromaDB at '%s': %s", persist_directory, e)
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}") from e

    async def ingest(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Ingests a document from a file path into the RAG system.

        Supports various file types thanks to `unstructured`.
        The document is loaded, split into chunks, and added to the vector store.

        Args:
            file_path (str): The path to the document file (e.g., 'docs/manual.pdf').
            metadata (Optional[Dict[str, Any]]): Optional metadata to associate
                                                with all chunks from this document.
                                                E.g., {'source': 'manual.pdf', 'author': '...'}.

        Returns:
            bool: True if ingestion was successful, False otherwise.

        Raises:
            ValueError: If the file path is invalid or the file type is unsupported.
            Exception: Propagates errors from loading, splitting, or storing.
        """
        if not os.path.isfile(file_path):
            error_msg = f"Ingestion failed: File not found at path '{file_path}'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        file_extension = os.path.splitext(file_path)[1].lower()
        logger.info("Starting ingestion of file: '%s' (type: %s)", file_path, file_extension)

        try:
            # --- 1. Load Document ---
            # LangChain's UnstructuredFileLoader often automatically chooses the right loader
            loader = UnstructuredFileLoader(file_path=file_path, mode="single") # 'single' loads as one doc
            # Alternative: Use specific loaders if needed for better control
            # if file_extension == '.pdf':
            #     loader = UnstructuredPDFLoader(file_path=file_path)
            # elif file_extension in ['.docx', '.doc']:
            #     loader = UnstructuredWordDocumentLoader(file_path=file_path)
            # elif file_extension == '.md':
            #     loader = UnstructuredMarkdownLoader(file_path=file_path)
            # else:
            #     loader = UnstructuredFileLoader(file_path=file_path)

            # Load documents (list of LangChain Documents)
            # run_in_executor is used to prevent blocking the async event loop
            langchain_documents: List[LCDocument] = await asyncio.get_event_loop().run_in_executor(None, loader.load)
            logger.debug("Loaded %d document(s) from '%s'", len(langchain_documents), file_path)

            if not langchain_documents:
                warning_msg = f"No content extracted from '{file_path}'. Skipping ingestion."
                logger.warning(warning_msg)
                return False # Not an error, just no content

            # --- 2. Assign/Update Metadata ---
            # Add common metadata
            doc_id = str(uuid.uuid4())
            common_metadata = {
                "file_source": file_path,
                "doc_id": doc_id,
                # Add file size, modification time etc. if needed
            }
            if metadata:
                common_metadata.update(metadata)

            # Update metadata for each loaded document part
            for lc_doc in langchain_documents:
                # Merge common metadata with existing doc metadata
                lc_doc.metadata.update(common_metadata)

            # --- 3. Split Documents into Chunks ---
            # Split the list of LangChain Documents
            split_documents: List[LCDocument] = self.text_splitter.split_documents(langchain_documents)
            logger.debug("Split into %d chunks.", len(split_documents))

            # --- 4. Add Chunks to Vector Store ---
            # Prepare lists of texts and metadatas for Chroma.add_texts
            texts = [doc.page_content for doc in split_documents]
            metadatas = [doc.metadata for doc in split_documents]

            # Add to ChromaDB (this is usually sync, but LangChain handles it)
            # Consider running in executor if it blocks for large amounts of data
            # run_in_executor is used to prevent blocking the async event loop
            await asyncio.get_event_loop().run_in_executor(None, self.vector_store.add_texts, texts, metadatas)
            # self.vector_store.add_texts(texts=texts, metadatas=metadatas) # If async is handled internally

            logger.info("Successfully ingested '%s' (%d chunks) into collection '%s'.", file_path, len(split_documents), self.collection_name)
            return True

        except ValueError as e:
            # Catch potential errors from unstructured loaders for unsupported types
            error_msg = f"Ingestion failed for '{file_path}': Unsupported file type or parsing error. {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Ingestion failed for '{file_path}': {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    async def query(self, question: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Queries the RAG system to retrieve relevant document chunks.

        Performs a similarity search in the ChromaDB vector store based on the
        embedded `question`.

        Args:
            question (str): The question or query string.
            top_k (int, optional): The maximum number of chunks to retrieve. Defaults to 5.
            filters (Optional[Dict[str, Any]], optional): Metadata filters for the search
                                                        (e.g., {"source": "manual.pdf"}).
                                                        Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                represents a retrieved chunk.
                                Example: [{'content': '...', 'metadata': {...}}, ...]
                                Returns an empty list if no relevant chunks are found.
        """
        logger.debug("RAG query initiated: '%s' (top_k=%d, filters=%s)", question, top_k, filters)
        try:
            # Perform similarity search using LangChain's Chroma wrapper
            # This returns LangChain Documents
            search_kwargs = {"k": top_k}
            if filters:
                # Chroma supports filtering by metadata
                search_kwargs["filter"] = filters

            # Similarity search (usually sync, run in executor if needed to prevent blocking)
            # run_in_executor is used to prevent blocking the async event loop
            results: List[LCDocument] = await asyncio.get_event_loop().run_in_executor(
                None, self.vector_store.similarity_search, question, search_kwargs
            )
            # results = self.vector_store.similarity_search(question, **search_kwargs)

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
            # Depending on requirements, you might want to return an empty list
            # or raise an error for the agent to handle.
            raise RuntimeError(error_msg) from e

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Gets basic statistics about the ChromaDB collection.

        Returns:
            Dict[str, Any]: A dictionary containing collection stats like name and count.
                            Returns {} if stats cannot be retrieved.
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
