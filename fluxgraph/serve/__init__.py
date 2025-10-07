# fluxgraph/serve/__init__.py
"""
LangServe-style production deployment for FluxGraph chains
One-command deployment with full REST API, WebSocket streaming, and monitoring
"""
from typing import Any, Dict, List, Optional, Callable
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import asyncio
import json
import logging
from datetime import datetime
import uvicorn

from fluxgraph.chains import Runnable
from fluxgraph.tracing import get_tracer, RunType

logger = logging.getLogger(__name__)


class InvokeRequest(BaseModel):
    """Request model for chain invocation"""
    input: Any = Field(..., description="Input to the chain")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration options")
    stream: bool = Field(default=False, description="Enable streaming response")


class InvokeResponse(BaseModel):
    """Response model for chain invocation"""
    output: Any = Field(..., description="Output from the chain")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")


class BatchRequest(BaseModel):
    """Request model for batch processing"""
    inputs: List[Any] = Field(..., description="List of inputs")
    config: Optional[Dict[str, Any]] = None


class BatchResponse(BaseModel):
    """Response model for batch processing"""
    outputs: List[Any] = Field(..., description="List of outputs")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FluxServe:
    """
    Production-ready deployment server for FluxGraph chains.
    
    Features:
    - REST API for invoke/batch operations
    - WebSocket streaming
    - Automatic OpenAPI documentation
    - CORS support
    - Health checks
    - Metrics and monitoring
    - Request/response validation
    
    Example:
        from fluxgraph.serve import FluxServe
        
        # Create your chain
        chain = prompt | model | parser
        
        # Deploy
        server = FluxServe(chain, name="my-chain")
        server.run()
        
        # Or use decorator
        @serve(name="my-chain")
        def create_chain():
            return prompt | model | parser
    """
    
    def __init__(
        self,
        runnable: Runnable,
        name: str = "fluxgraph-chain",
        description: Optional[str] = None,
        version: str = "1.0.0",
        enable_tracing: bool = True,
        cors_origins: List[str] = ["*"],
        api_keys: Optional[List[str]] = None
    ):
        self.runnable = runnable
        self.name = name
        self.description = description or f"FluxGraph Chain: {name}"
        self.version = version
        self.enable_tracing = enable_tracing
        self.api_keys = api_keys
        
        # Create FastAPI app
        self.app = FastAPI(
            title=self.name,
            description=self.description,
            version=self.version,
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Setup routes
        self._setup_routes()
        
        # Metrics
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with service info"""
            return {
                "name": self.name,
                "description": self.description,
                "version": self.version,
                "endpoints": {
                    "invoke": "/invoke",
                    "batch": "/batch",
                    "stream": "/stream (WebSocket)",
                    "health": "/health",
                    "metrics": "/metrics",
                    "docs": "/docs"
                }
            }
        
        @self.app.post("/invoke", response_model=InvokeResponse)
        async def invoke(request: InvokeRequest):
            """
            Invoke the chain with a single input.
            
            Supports both regular and streaming responses.
            """
            start_time = datetime.now()
            
            try:
                if self.enable_tracing:
                    tracer = get_tracer()
                    async with tracer.span(
                        f"{self.name}_invoke",
                        run_type=RunType.CHAIN,
                        inputs={"input": request.input}
                    ):
                        if request.stream:
                            return await self._stream_response(request)
                        else:
                            output = await self.runnable.invoke(
                                request.input,
                                request.config
                            )
                else:
                    if request.stream:
                        return await self._stream_response(request)
                    output = await self.runnable.invoke(
                        request.input,
                        request.config
                    )
                
                # Update metrics
                self.request_count += 1
                latency = (datetime.now() - start_time).total_seconds()
                self.total_latency += latency
                
                return InvokeResponse(
                    output=output,
                    metadata={
                        "latency_ms": latency * 1000,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Invoke error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/batch", response_model=BatchResponse)
        async def batch(request: BatchRequest):
            """Process multiple inputs in batch"""
            start_time = datetime.now()
            
            try:
                if self.enable_tracing:
                    tracer = get_tracer()
                    async with tracer.span(
                        f"{self.name}_batch",
                        run_type=RunType.CHAIN,
                        inputs={"count": len(request.inputs)}
                    ):
                        outputs = await self.runnable.batch(
                            request.inputs,
                            request.config
                        )
                else:
                    outputs = await self.runnable.batch(
                        request.inputs,
                        request.config
                    )
                
                latency = (datetime.now() - start_time).total_seconds()
                
                return BatchResponse(
                    outputs=outputs,
                    metadata={
                        "count": len(outputs),
                        "latency_ms": latency * 1000,
                        "avg_latency_per_item_ms": (latency * 1000) / len(outputs)
                    }
                )
                
            except Exception as e:
                logger.error(f"Batch error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/stream")
        async def websocket_stream(websocket: WebSocket):
            """
            WebSocket endpoint for streaming responses.
            
            Client sends: {"input": <input>, "config": <config>}
            Server streams: {"chunk": <chunk>} or {"error": <error>}
            """
            await websocket.accept()
            
            try:
                # Receive input
                data = await websocket.receive_json()
                input_data = data.get("input")
                config = data.get("config")
                
                # Stream response
                async for chunk in self.runnable.stream(input_data, config):
                    await websocket.send_json({"chunk": chunk})
                
                # Send completion signal
                await websocket.send_json({"done": True})
                
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({"error": str(e)})
            finally:
                await websocket.close()
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "name": self.name,
                "version": self.version,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/metrics")
        async def metrics():
            """Metrics endpoint"""
            avg_latency = (
                self.total_latency / self.request_count
                if self.request_count > 0
                else 0
            )
            
            return {
                "requests_total": self.request_count,
                "errors_total": self.error_count,
                "success_rate": (
                    (self.request_count - self.error_count) / self.request_count
                    if self.request_count > 0
                    else 0
                ),
                "avg_latency_seconds": avg_latency,
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds()
            }
        
        @self.app.get("/config")
        async def config():
            """Get chain configuration"""
            return {
                "name": self.name,
                "runnable_type": type(self.runnable).__name__,
                "tracing_enabled": self.enable_tracing,
                "version": self.version
            }
    
    async def _stream_response(self, request: InvokeRequest):
        """Create streaming response"""
        async def generate():
            try:
                async for chunk in self.runnable.stream(request.input, request.config):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        reload: bool = False,
        **kwargs
    ):
        """
        Run the server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            workers: Number of worker processes
            reload: Enable auto-reload for development
            **kwargs: Additional uvicorn options
        """
        self._start_time = datetime.now()
        
        logger.info(f"Starting {self.name} on {host}:{port}")
        logger.info(f"Documentation available at http://{host}:{port}/docs")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            **kwargs
        )
    
    def mount(self, app: FastAPI, path: str = "/chain"):
        """
        Mount this chain as a sub-application.
        
        Example:
            main_app = FastAPI()
            chain_server = FluxServe(chain)
            chain_server.mount(main_app, path="/api/v1/chain")
        """
        app.mount(path, self.app)


def serve(
    name: str,
    description: Optional[str] = None,
    **kwargs
) -> Callable:
    """
    Decorator to quickly deploy a chain.
    
    Example:
        @serve(name="summarizer", description="Text summarization chain")
        def create_summarizer():
            prompt = PromptTemplate("Summarize: {text}")
            model = create_llm_runnable("openai", "gpt-4")
            return prompt | model
        
        # Chain is automatically deployed
    """
    def decorator(func: Callable) -> FluxServe:
        runnable = func()
        server = FluxServe(
            runnable,
            name=name,
            description=description,
            **kwargs
        )
        return server
    
    return decorator


def deploy_multiple(chains: Dict[str, Runnable], **kwargs) -> FastAPI:
    """
    Deploy multiple chains on a single server.
    
    Example:
        chains = {
            "summarize": summarize_chain,
            "translate": translate_chain,
            "analyze": analyze_chain
        }
        
        app = deploy_multiple(chains)
    """
    app = FastAPI(
        title="FluxGraph Multi-Chain Server",
        description="Multiple chains deployed together"
    )
    
    for name, runnable in chains.items():
        server = FluxServe(runnable, name=name, **kwargs)
        server.mount(app, path=f"/{name}")
    
    return app
