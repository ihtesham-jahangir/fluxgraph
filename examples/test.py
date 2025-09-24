# examples/comprehensive_test.py
"""
Comprehensive Test Example for FluxGraph.

This script demonstrates various features of FluxGraph, including:
- Auto-initialized Universal RAG
- Memory Store (if configured)
- Tooling Layer
- Event Hooks
- Agent interactions via the /ask endpoint

Ensure you have installed the RAG dependencies:
pip install langchain langchain-community langchain-chroma unstructured[all-docs] chromadb sentence-transformers

To run:
1. Set environment variables (optional, for customization):
   export FLUXGRAPH_RAG_EMBEDDING_MODEL='BAAI/bge-small-en-v1.5' # Example
   export DATABASE_URL='...' # If using PostgreSQL memory

2. Run the server:
   flux run examples/comprehensive_test.py --reload
   # Or: uvicorn examples.comprehensive_test:app.api --reload

3. Interact with the agents using curl or an API tool.
"""
import os
import logging
from typing import List, Dict, Any

# Import FluxGraph core
from fluxgraph import FluxApp
# Import memory if needed (e.g., PostgresMemory)
# from fluxgraph.core.postgres_memory import PostgresMemory
# from fluxgraph.utils.db import DatabaseManager

# --- Configure logging for this example ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Initialize FluxGraph App ---
# FluxGraph will auto-initialize RAG if dependencies are met and rag_connector=None
app = FluxApp(title="FluxGraph Comprehensive Test")

# --- 2. Example: Define a Tool ---
@app.tool() # Name defaults to function name 'add_two_numbers'
def add_two_numbers(x: int, y: int) -> int:
    """
    A simple tool that adds two integers.
    """
    logger.info(f"[Tool Call] Adding {x} + {y}")
    return x + y

# --- 3. Example: Agent using Tool ---
@app.agent() # Agent name will be 'calculator_agent'
async def calculator_agent(a: int, b: int, tools):
    """
    An agent that uses the 'add_two_numbers' tool.
    """
    try:
        # Retrieve the tool function from the injected 'tools' registry
        add_tool_func = tools.get("add_two_numbers")
        if not add_tool_func:
            return {"error": "Tool 'add_two_numbers' not found."}

        # Execute the tool
        result = add_tool_func(a, b)
        
        return {
            "operation": "addition",
            "operand_a": a,
            "operand_b": b,
            "result": result
        }
    except Exception as e:
        return {"error": f"Calculation failed: {e}"}

# --- 4. Example: Agent using Memory (if available) ---
@app.agent() # Agent name will be 'memory_agent'
async def memory_agent(message: str, session_id: str, memory=None):
    """
    An agent that stores and recalls messages using the memory store.
    """
    if not memory:
        return {"response": "Memory is not configured for this agent.", "agent": "memory_agent"}

    try:
        # 1. Store the current user message
        await memory.add(session_id, {"role": "user", "content": message})
        logger.info(f"[Memory Agent] Stored message for session '{session_id}'.")

        # 2. Retrieve the last message
        history = await memory.get(session_id, limit=1)
        last_message_content = history[0]['content'] if history else "Nothing previously."

        # 3. Generate a response
        response_text = f"You said '{message}'. Before that, you said '{last_message_content}'."

        # 4. Store the agent's response
        await memory.add(session_id, {"role": "assistant", "content": response_text})
        logger.info(f"[Memory Agent] Stored agent response for session '{session_id}'.")

        return {
            "user_message": message,
            "agent_response": response_text,
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"[Memory Agent] Error: {e}")
        return {"error": f"Memory agent failed: {e}"}

# --- 5. Example: Agent using Auto-Initialized RAG ---
@app.agent() # Agent name will be 'rag_agent'
async def rag_agent(question: str, rag=None):
    """
    An agent that answers questions using the auto-initialized Universal RAG system.
    """
    if not rag:
        return {"error": "RAG connector is not configured.", "agent": "rag_agent"}

    try:
        logger.info(f"[RAG Agent] Processing question: {question}")

        # --- 1. Query the RAG system ---
        # You can pass filters here if your docs have metadata
        # e.g., filters = {"source": "policy_manual.pdf"}
        retrieved_chunks = await rag.query(question, top_k=3, filters=None)
        logger.debug(f"[RAG Agent] Retrieved {len(retrieved_chunks)} chunks.")

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                "chunks_used": 0
            }

        # --- 2. Format context (simplified for example) ---
        context_parts = [f"Document: {chunk.get('content', 'No content')[:100]}..." for chunk in retrieved_chunks]
        context_string = "\n\n".join(context_parts)

        # --- 3. (Simulate) Generate answer using context ---
        # In a real scenario, you would call an LLM provider here.
        # For this example, we'll create a simulated response.
        simulated_answer = (
            f"Based on the retrieved documents, the answer to '{question}' involves "
            f"information related to the content shown. "
            f"(This is a simulated response; a real LLM would generate a precise answer.)"
        )

        return {
            "question": question,
            "answer": simulated_answer,
            "context_preview": context_string,
            "chunks_used": len(retrieved_chunks)
        }
    except Exception as e:
        logger.error(f"[RAG Agent] Error processing question: {e}", exc_info=True)
        return {"error": f"RAG agent failed: {str(e)}", "agent": "rag_agent"}

# --- 6. Example: Event Hooks (Demonstration) ---
# These hooks will print messages to the console when triggered.
# They demonstrate the transparency aspect of FluxGraph.

@app.api.on_event("startup")
async def startup_event_print():
    """Print a message on startup."""
    print("\n🚀 FluxGraph Comprehensive Test App Started!")
    print("   - Auto-RAG is initialized if dependencies are met.")
    print("   - Check http://127.0.0.1:8000/rag/status for RAG details.")
    print("   - Available agents: calculator_agent, memory_agent, rag_agent")
    print("   - Available tools: add_two_numbers")
    print("-" * 60)

# --- CORRECT WAY TO REGISTER EVENT HOOKS ---
# Define async callback functions with correct syntax
async def on_request_received_hook( data:Dict[str, Any]):
    """Hook triggered when a request is received."""
    request_id = data.get("request_id", "N/A")
    agent_name = data.get("agent_name", "Unknown")
    print(f"[Hook] 📥 Request Received [ID: {request_id}] -> Agent: {agent_name}")

async def on_agent_completed_hook( data:Dict[str, Any]):
    """Hook triggered when an agent completes successfully."""
    request_id = data.get("request_id", "N/A")
    agent_name = data.get("agent_name", "Unknown")
    duration = data.get("duration", 0)
    print(f"[Hook] ✅ Agent Completed [ID: {request_id}] -> Agent: {agent_name} (Time: {duration:.4f}s)")

async def on_agent_error_hook( Dict[str, Any]):
    """Hook triggered when an agent encounters an error."""
    request_id = data.get("request_id", "N/A")
    agent_name = data.get("agent_name", "Unknown")
    error_msg = data.get("error", "Unknown error")
    duration = data.get("duration", 0)
    print(f"[Hook] ⚠️  Agent Error [ID: {request_id}] -> Agent: {agent_name} (Time: {duration:.4f}s) | Error: {error_msg}")

# Register the callback functions with specific events using app.hooks.on()
app.hooks.on("request_received", on_request_received_hook)
app.hooks.on("agent_completed", on_agent_completed_hook)
app.hooks.on("agent_error", on_agent_error_hook)
# --- END OF CORRECT WAY ---

# --- 7. Example: API Endpoint to Ingest Documents into RAG (WITH FILE UPLOAD) ---
# This demonstrates how you might add documents to the RAG system via file upload.
# Requires the rag_connector to be available.
if hasattr(app, 'rag_connector') and app.rag_connector:
    from fastapi import HTTPException, UploadFile, File # Import for file upload
    import tempfile
    # Import request context for logging (if available in your app.py)
    # from fluxgraph.core.app import request_id_context 
    
    @app.api.post("/rag/ingest", summary="Ingest Document (File Upload)", description="Ingest a document into the RAG system by uploading a file.")
    async def ingest_document(file: UploadFile = File(...)): # Use UploadFile
        """
        Endpoint to ingest a document into the RAG knowledge base via file upload.

        Args:
            file (UploadFile): The file to be ingested.
        """
        # request_id = request_id_context.get() # Get request ID for logging context
        request_id = "N/A (context import needed)" # Placeholder if context import is complex
        logger.info(f"[Request ID: {request_id}] 📥 Receiving file upload: '{file.filename}' (Content-Type: {file.content_type})")
        
        # --- Validate File ---
        if not file.filename:
             logger.warning(f"[Request ID: {request_id}] ⚠️  No filename provided in upload.")
             raise HTTPException(status_code=400, detail="No file uploaded or filename missing.")

        # --- Save File Temporarily ---
        temp_file_path = None
        try:
            # Create a temporary file to save the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file_path = temp_file.name
                logger.debug(f"[Request ID: {request_id}] Saving uploaded file to temporary path: {temp_file_path}")

                # Copy the contents of the uploaded file to the temporary file
                while chunk := await file.read(1024 * 1024): # Read 1MB chunks
                     temp_file.write(chunk)

            logger.info(f"[Request ID: {request_id}] ✅ File '{file.filename}' saved to temporary location.")

            # --- Ingest using UniversalRAG ---
            logger.info(f"[Request ID: {request_id}] 🔄 Starting ingestion process for '{file.filename}'...")
            # Pass the temporary file path and metadata to the RAG connector's ingest method
            success = await app.rag_connector.ingest(temp_file_path, metadata={"source_filename": file.filename})

            if success:
                logger.info(f"[Request ID: {request_id}] ✅ Document '{file.filename}' ingested successfully.")
                return {"message": f"Document '{file.filename}' ingested successfully."}
            else:
                 logger.warning(f"[Request ID: {request_id}] ⚠️  Document '{file.filename}' ingestion reported failure or no content.")
                 return {"message": f"Document '{file.filename}' ingestion might have partially failed or had no content."}

        except ValueError as e:
            # Catch potential errors from unstructured loaders for unsupported types
            logger.warning(f"[Request ID: {request_id}] ⚠️  Ingestion error for '{file.filename}' (ValueError): {e}")
            raise HTTPException(status_code=400, detail=f"Ingestion error: Unsupported file type or parsing error. {e}")
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] ❌ Ingestion failed for '{file.filename}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
        finally:
            # --- Cleanup Temporary File ---
            # Ensure the temporary file is deleted even if an error occurs
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"[Request ID: {request_id}] 🧹 Temporary file '{temp_file_path}' deleted.")
                except Exception as cleanup_error:
                    logger.warning(f"[Request ID: {request_id}] ⚠️  Failed to delete temporary file '{temp_file_path}': {cleanup_error}")

# --- Final Output ---
print(f"\n🚀 {app.title} is ready!")
print("Available agents:")
for agent_name in app.registry.list_agents():
    print(f"  - POST /ask/{agent_name}")
if hasattr(app, 'rag_connector') and app.rag_connector:
    print("  - POST /rag/ingest (for adding documents via file upload)") # Updated message
