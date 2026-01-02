import httpx
import time
import random
import asyncio
import os
from fastapi import FastAPI, HTTPException, Request, Body, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator

# ---------------------------------------------------------------------------
# 1. IMPORT YOUR FLUXGRAPH FRAMEWORK
#    (Assumes 'fluxgraph' folder is in the same directory or installed)
# ---------------------------------------------------------------------------
try:
    from fluxgraph.core.app import FluxApp  # Import your main class
except ImportError:
    print("Error: Could not import FluxApp. Make sure your 'fluxgraph' library is installed or in the PYTHONPATH.")
    print("Try running: pip install -e .  (if you are in the root of your GitHub repo)")
    exit(1)


# ---------------------------------------------------------------------------
# 2. DEFINE YOUR DEMO AGENTS & CHAINS (to show the UI works)
# ---------------------------------------------------------------------------

class SimpleAgent:
    """A simple demo agent that just echoes the input."""
    def __init__(self, name: str = "assistant"):
        self.name = name

    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        return {
            "response": f"FluxAgent '{self.name}' processed: '{query}'",
            "agent_name": self.name,
            "timestamp": time.time()
        }

async def streaming_chain_example(input: str) -> AsyncGenerator[Dict[str, Any], None]:
    """A demo for a streaming chain, as required by the UI."""
    yield {"status": f"Starting chain for input: '{input}'", "done": False}
    await asyncio.sleep(0.5)
    
    words = ["This", "is", "a", "real", "stream", "from", "a", 
             "FluxGraph", "chain.", "It's", "processing", 
             "your", "request", "live."]
    
    for word in words:
        yield {"chunk": f" {word}", "done": False}
        await asyncio.sleep(0.1)
        
    await asyncio.sleep(1)
    yield {
        "status": "Stream complete.",
        "final_message": " ".join(words),
        "done": True
    }

# ---------------------------------------------------------------------------
# 3. INITIALIZE FLUXAPP
#    This creates the main `flux_app` object
# ---------------------------------------------------------------------------

# Get database URL from environment (P0/P1/P2 features)
DB_URL = os.getenv("FLUXGRAPH_DB_URL") # e.g., "postgresql+asyncpg://user:pass@localhost/fluxdb"

if not DB_URL:
    print("Warning: FLUXGRAPH_DB_URL not set. Resumable workflows, logging, and audit logs will be disabled.")

flux_app = FluxApp(
    title="FluxGraph Live Demo",
    description="Serving a React UI with a real FluxGraph backend",
    database_url=DB_URL,
    enable_security=True,
    enable_workflows=True,
    enable_chains=True,
    cors_origins=["http://localhost:3000"] # Allow React app
)

# ---------------------------------------------------------------------------
# 4. REGISTER YOUR AGENTS & CHAINS
# ---------------------------------------------------------------------------

flux_app.register("assistant", SimpleAgent())
# TODO: Implement register_streaming_chain method in FluxApp
# flux_app.register_streaming_chain("streaming_chain_example", streaming_chain_example)


# ---------------------------------------------------------------------------
# 5. ADD THE PROXY ROUTES (for the LLM Chat UI)
#    We get the underlying FastAPI app from `flux_app.api`
# ---------------------------------------------------------------------------

app = flux_app.api  # Get the FastAPI app instance from FluxGraph
EXTERNAL_API_URL = "https://llm.themagnetismo.com"

# Pydantic models for the proxy
class ProxyLoginRequest(BaseModel):
    username: str
    password: str

class ProxyChatMessage(BaseModel):
    role: str
    content: str

class ProxyInferRequest(BaseModel):
    messages: List[ProxyChatMessage]

@app.post("/proxy/login", tags=["Proxy - LLM Chat"])
async def proxy_login(request: ProxyLoginRequest):
    """Proxies login to the external LLM server to avoid CORS."""
    async with httpx.AsyncClient() as client:
        try:
            data_to_send = {'grant_type': 'password', 'username': request.username, 'password': request.password}
            response = await client.post(
                f"{EXTERNAL_API_URL}/token",
                data=data_to_send,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status() 
            return response.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", "External login failed")
            raise HTTPException(status_code=e.response.status_code, detail=detail)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@app.post("/proxy/infer", tags=["Proxy - LLM Chat"])
async def proxy_infer(request_body: ProxyInferRequest, http_request: Request):
    """Proxies inference to the external LLM server to avoid CORS."""
    auth_header = http_request.headers.get('Authorization')
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    external_payload = {
        "messages": [msg.dict() for msg in request_body.messages],
        "model": "default-model", "temperature": 0.8,
        "max_tokens": 512, "stream": False
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{EXTERNAL_API_URL}/llm/infer",
                json=external_payload,
                headers={"Authorization": auth_header, "Content-Type": "application/json"},
                timeout=60.0 
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", "External infer failed")
            raise HTTPException(status_code=e.response.status_code, detail=detail)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

# ---------------------------------------------------------------------------
# 6. RUN THE SERVER
#    Use the built-in `flux run` command from your framework's CLI
#
#    IN YOUR TERMINAL, RUN:
#    flux run main.py --reload
#
#    (This `if __name__` block is a fallback for direct `python main.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("Starting server... Access docs at http://localhost:8000/docs")
    print("RECOMMENDED: Run this file using your framework's CLI: flux run main.py --reload")
    uvicorn.run(app, host="0.0.0.0", port=8000)