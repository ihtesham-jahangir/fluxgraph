"""
Main FastAPI application setup with enhanced features
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn
import logging
import sys
from typing import Optional
from datetime import datetime

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fluxgraph_api.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Import routers with better error handling
ROUTES_AVAILABLE = True
workflow_router = None
playground_router = None
websocket_router = None

try:
    from .workflow_routes import router as workflow_router
    logger.info("✅ Workflow router imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import workflow routes: {e}")
    ROUTES_AVAILABLE = False

try:
    from .playground_routes import router as playground_router
    logger.info("✅ Playground router imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import playground routes: {e}")
    ROUTES_AVAILABLE = False

try:
    from .websocket_routes import router as websocket_router
    logger.info("✅ WebSocket router imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import websocket routes: {e}")
    ROUTES_AVAILABLE = False


def create_app() -> FastAPI:
    """Create and configure the FastAPI application with enhanced middleware and error handling."""
    
    app = FastAPI(
        title="FluxGraph API",
        description="Production-grade AI agent orchestration with semantic caching",
        version="2.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        # Add response model validation
        response_model_exclude_unset=True,
        # Add OpenAPI tags for better organization
        openapi_tags=[
            {"name": "workflows", "description": "Workflow management operations"},
            {"name": "playground", "description": "Interactive testing endpoints"},
            {"name": "websockets", "description": "Real-time communication"},
            {"name": "health", "description": "Health and monitoring endpoints"}
        ]
    )

    # CORS middleware with enhanced configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            "https://fluxgraph.dev",
            "https://*.vercel.app",
            "https://*.netlify.app"
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Total-Count", "X-Request-ID"],
        max_age=3600,  # Cache preflight requests for 1 hour
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests with timing information."""
        start_time = datetime.now()
        
        # Generate request ID for tracking
        request_id = f"{start_time.timestamp()}-{id(request)}"
        
        logger.info(f"🔵 Request started: {request.method} {request.url.path} [ID: {request_id}]")
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = (datetime.now() - start_time).total_seconds()
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            logger.info(
                f"🟢 Request completed: {request.method} {request.url.path} "
                f"[Status: {response.status_code}] [Time: {process_time:.3f}s] [ID: {request_id}]"
            )
            
            return response
            
        except Exception as e:
            process_time = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"🔴 Request failed: {request.method} {request.url.path} "
                f"[Error: {str(e)}] [Time: {process_time:.3f}s] [ID: {request_id}]"
            )
            raise

    # Custom exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions with custom response."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path),
                "timestamp": datetime.now().isoformat()
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors with detailed messages."""
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "details": exc.errors(),
                "body": exc.body,
                "path": str(request.url.path),
                "timestamp": datetime.now().isoformat()
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(exc),
                "path": str(request.url.path),
                "timestamp": datetime.now().isoformat()
            }
        )

    # Include all routers with proper error handling
    if workflow_router:
        app.include_router(workflow_router, tags=["workflows"])
        logger.info("✅ Workflow routes loaded")
    
    if playground_router:
        app.include_router(playground_router, tags=["playground"])
        logger.info("✅ Playground routes loaded")
    
    if websocket_router:
        app.include_router(websocket_router, tags=["websockets"])
        logger.info("✅ WebSocket routes loaded")

    if not ROUTES_AVAILABLE:
        logger.warning("⚠️ Some or all API routes are not available")

    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Log startup information."""
        logger.info("=" * 80)
        logger.info("🚀 FluxGraph API Starting Up")
        logger.info(f"📦 Version: 2.1.0")
        logger.info(f"🐍 Python: {sys.version.split()[0]}")
        logger.info(f"🔧 Routes Available: {ROUTES_AVAILABLE}")
        logger.info("=" * 80)

    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Log shutdown information."""
        logger.info("=" * 80)
        logger.info("🛑 FluxGraph API Shutting Down")
        logger.info("=" * 80)

    # Enhanced health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check():
        """
        Comprehensive health check endpoint.
        Returns service status and feature availability.
        """
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.1.0",
            "features": {
                "semantic_caching": True,
                "hybrid_memory": True,
                "security_suite": True,
                "workflows": True,
                "websockets": True,
                "api_routes": ROUTES_AVAILABLE
            },
            "routes": {
                "workflow_router": workflow_router is not None,
                "playground_router": playground_router is not None,
                "websocket_router": websocket_router is not None
            }
        }

    # Readiness check (for Kubernetes/cloud deployments)
    @app.get("/ready", tags=["health"])
    async def readiness_check():
        """Check if the service is ready to accept traffic."""
        if ROUTES_AVAILABLE:
            return {"status": "ready", "timestamp": datetime.now().isoformat()}
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "Routes not fully initialized",
                "timestamp": datetime.now().isoformat()
            }
        )

    # Liveness check (for Kubernetes/cloud deployments)
    @app.get("/live", tags=["health"])
    async def liveness_check():
        """Check if the service is alive."""
        return {"status": "alive", "timestamp": datetime.now().isoformat()}

    # Root endpoint with enhanced HTML
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Root endpoint with FluxGraph information."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>FluxGraph API - Cut AI Agent Costs by 70%</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; 
                    background: linear-gradient(135deg, #1a1a1a 0%, #000000 100%);
                    color: #ffffff; 
                    min-height: 100vh;
                    display: flex; 
                    align-items: center; 
                    justify-content: center;
                    padding: 20px;
                }
                
                .container { 
                    background: rgba(255,255,255,0.03); 
                    padding: 60px 40px; 
                    border-radius: 20px; 
                    backdrop-filter: blur(10px); 
                    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
                    max-width: 1200px; 
                    width: 100%;
                    border: 1px solid rgba(255,255,255,0.1);
                }
                
                .logo { 
                    font-size: 4em; 
                    font-weight: bold; 
                    text-align: center; 
                    margin-bottom: 20px; 
                    background: linear-gradient(135deg, #ffffff 0%, #999999 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    text-shadow: 0 0 30px rgba(255,255,255,0.1);
                }
                
                .tagline { 
                    font-size: 1.8em; 
                    text-align: center; 
                    margin-bottom: 40px; 
                    opacity: 0.9;
                    color: #e0e0e0;
                }
                
                .stats { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 20px; 
                    margin: 40px 0; 
                }
                
                .stat { 
                    text-align: center; 
                    background: rgba(255,255,255,0.05);
                    padding: 30px 20px; 
                    border-radius: 15px;
                    border: 1px solid rgba(255,255,255,0.1);
                    transition: all 0.3s ease;
                }
                
                .stat:hover { 
                    transform: translateY(-5px);
                    background: rgba(255,255,255,0.08);
                    border-color: rgba(255,255,255,0.2);
                    box-shadow: 0 10px 40px rgba(255,255,255,0.1);
                }
                
                .stat-number { 
                    font-size: 3em; 
                    font-weight: bold; 
                    color: #ffffff; 
                    margin-bottom: 10px;
                    text-shadow: 0 0 20px rgba(255,255,255,0.3);
                }
                
                .stat-label { 
                    font-size: 1.1em; 
                    opacity: 0.7;
                    color: #cccccc;
                }
                
                .features { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
                    gap: 20px; 
                    margin: 40px 0; 
                }
                
                .feature { 
                    background: rgba(255,255,255,0.05); 
                    padding: 25px; 
                    border-radius: 15px;
                    border: 1px solid rgba(255,255,255,0.1);
                    transition: all 0.3s ease;
                }
                
                .feature:hover {
                    background: rgba(255,255,255,0.08);
                    border-color: rgba(255,255,255,0.2);
                    transform: translateY(-3px);
                }
                
                .feature h3 { 
                    margin-bottom: 15px; 
                    color: #ffffff; 
                    font-size: 1.3em;
                }
                
                .feature p { 
                    line-height: 1.6; 
                    opacity: 0.8;
                    color: #cccccc;
                }
                
                .api-section {
                    background: rgba(255,255,255,0.03);
                    padding: 30px;
                    border-radius: 15px;
                    margin: 40px 0;
                    border: 1px solid rgba(255,255,255,0.1);
                }
                
                .api-section h2 {
                    color: #ffffff;
                    font-size: 2em;
                    margin-bottom: 20px;
                    text-align: center;
                }
                
                .api-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                
                .api-card {
                    background: rgba(255,255,255,0.05);
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid rgba(255,255,255,0.1);
                    transition: all 0.3s ease;
                }
                
                .api-card:hover {
                    background: rgba(255,255,255,0.08);
                    transform: translateY(-2px);
                }
                
                .api-card h4 {
                    color: #ffffff;
                    margin-bottom: 10px;
                    font-size: 1.2em;
                }
                
                .api-card code {
                    background: rgba(0,0,0,0.3);
                    padding: 8px 12px;
                    border-radius: 5px;
                    display: block;
                    margin: 10px 0;
                    color: #00ff00;
                    font-family: 'Courier New', monospace;
                    font-size: 0.9em;
                    border: 1px solid rgba(255,255,255,0.1);
                }
                
                .api-card p {
                    color: #cccccc;
                    font-size: 0.95em;
                    line-height: 1.5;
                }
                
                .cta-section { 
                    text-align: center; 
                    margin: 50px 0 30px;
                }
                
                .btn { 
                    background: #ffffff; 
                    color: #000000; 
                    padding: 18px 40px; 
                    text-decoration: none; 
                    border-radius: 30px; 
                    font-weight: bold;
                    display: inline-block; 
                    margin: 10px;
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 15px rgba(255,255,255,0.2);
                    font-size: 1.1em;
                }
                
                .btn:hover { 
                    background: #e0e0e0; 
                    transform: translateY(-3px);
                    box-shadow: 0 6px 20px rgba(255,255,255,0.3);
                }
                
                .btn-secondary {
                    background: transparent;
                    color: #ffffff;
                    border: 2px solid #ffffff;
                }
                
                .btn-secondary:hover {
                    background: rgba(255,255,255,0.1);
                }
                
                .status-badge {
                    display: inline-block;
                    padding: 8px 20px;
                    background: rgba(255,255,255,0.1);
                    border: 2px solid #ffffff;
                    border-radius: 20px;
                    margin: 10px 5px;
                    font-size: 0.9em;
                    transition: all 0.3s ease;
                }
                
                .status-badge:hover {
                    background: rgba(255,255,255,0.15);
                }
                
                .footer {
                    text-align: center;
                    margin-top: 50px;
                    opacity: 0.7;
                    font-size: 0.95em;
                    color: #999999;
                }
                
                .footer a {
                    color: #ffffff;
                    text-decoration: none;
                    transition: opacity 0.3s ease;
                }
                
                .footer a:hover {
                    opacity: 0.7;
                }
                
                @media (max-width: 768px) {
                    .logo { font-size: 2.5em; }
                    .tagline { font-size: 1.3em; }
                    .stat-number { font-size: 2em; }
                    .container { padding: 40px 20px; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="logo">🔀 FluxGraph API</div>
                <div class="tagline">Cut AI Agent Costs by 70% with Semantic Caching</div>
                
                <div style="text-align: center;">
                    <span class="status-badge">✅ API Running</span>
                    <span class="status-badge">🚀 v2.1.0</span>
                    <span class="status-badge">🔓 MIT License</span>
                </div>
                
                <div class="stats">
                    <div class="stat">
                        <div class="stat-number">70%+</div>
                        <div class="stat-label">Cost Reduction</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">85-95%</div>
                        <div class="stat-label">Cache Hit Rate</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">146</div>
                        <div class="stat-label">Core Components</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">&lt;50ms</div>
                        <div class="stat-label">Memory Ops</div>
                    </div>
                </div>

                <div class="features">
                    <div class="feature">
                        <h3>⚡ Semantic Caching</h3>
                        <p>Reduce LLM costs by 70%+ with intelligent query similarity matching. No other framework has this built-in.</p>
                    </div>
                    <div class="feature">
                        <h3>🧠 4-Tier Memory</h3>
                        <p>Short-term, long-term, episodic, and semantic memory. Your agents truly learn and remember.</p>
                    </div>
                    <div class="feature">
                        <h3>🔒 Enterprise Security</h3>
                        <p>PII detection (9 types), prompt injection shield (7 vectors), RBAC, and blockchain audit logs.</p>
                    </div>
                    <div class="feature">
                        <h3>🔄 Graph Workflows</h3>
                        <p>Complex workflows with conditional routing, loops, and sophisticated state management.</p>
                    </div>
                </div>

                <div class="api-section">
                    <h2>🔌 Available APIs</h2>
                    
                    <div class="api-grid">
                        <div class="api-card">
                            <h4>🔄 Workflow API</h4>
                            <code>POST /api/workflows</code>
                            <p>Create, manage, and execute visual agent workflows with conditional routing and state management.</p>
                        </div>
                        
                        <div class="api-card">
                            <h4>⚡ WebSocket Live Updates</h4>
                            <code>WS /api/workflows/ws/{id}</code>
                            <p>Real-time workflow execution monitoring with node-by-node progress tracking.</p>
                        </div>
                        
                        <div class="api-card">
                            <h4>🎮 Playground</h4>
                            <code>POST /api/playground/execute</code>
                            <p>Interactive demo to test semantic caching with live cost savings visualization.</p>
                        </div>
                        
                        <div class="api-card">
                            <h4>📊 System Monitor</h4>
                            <code>WS /api/monitor</code>
                            <p>Live system metrics, active connections, and execution statistics.</p>
                        </div>
                        
                        <div class="api-card">
                            <h4>🔍 Workflow Stats</h4>
                            <code>GET /api/workflows/stats</code>
                            <p>Comprehensive analytics: workflow count, execution history, and success rates.</p>
                        </div>
                        
                        <div class="api-card">
                            <h4>🔐 Health Check</h4>
                            <code>GET /health</code>
                            <p>Service health status with feature availability flags.</p>
                        </div>
                    </div>
                </div>

                <div class="cta-section">
                    <a href="/api/docs" class="btn">📚 API Documentation</a>
                    <a href="/api/playground/demo-queries" class="btn">🎮 Try Playground</a>
                    <a href="https://github.com/ihtesham-jahangir/fluxgraph" class="btn btn-secondary">⭐ Star on GitHub</a>
                </div>

                <div class="footer">
                    <p><strong>Quick Start:</strong> pip install fluxgraph[full]</p>
                    <p style="margin-top: 10px;">FluxGraph v2.1.0 • MIT License • Made with ❤️ for the AI community</p>
                    <p style="margin-top: 10px;">Join Discord: <a href="https://discord.gg/Z9bAqjYvPc">discord.gg/Z9bAqjYvPc</a></p>
                </div>
            </div>
        </body>
        </html>
        """

    return app


# Create the app instance
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = True,
    log_level: str = "info",
    workers: Optional[int] = None
):
    """
    Run the FluxGraph API server with enhanced configuration.
    
    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)
        reload: Enable auto-reload for development (default: True)
        log_level: Logging level (default: info)
        workers: Number of worker processes for production (default: None)
    """
    logger.info("=" * 80)
    logger.info(f"🚀 Starting FluxGraph API server")
    logger.info(f"🌐 Host: {host}:{port}")
    logger.info(f"📖 API Documentation: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/api/docs")
    logger.info(f"🎮 Playground: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/api/playground/demo-queries")
    logger.info(f"🔌 WebSocket: ws://{host if host != '0.0.0.0' else 'localhost'}:{port}/api/workflows/ws/{{workflow_id}}")
    logger.info(f"💚 Health Check: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/health")
    logger.info(f"🔄 Auto-reload: {reload}")
    if workers:
        logger.info(f"👷 Workers: {workers}")
    logger.info("=" * 80)
    
    config = {
        "app": "fluxgraph.api.main:app",
        "host": host,
        "port": port,
        "log_level": log_level,
    }
    
    if reload:
        config["reload"] = True
    elif workers:
        config["workers"] = workers
    
    uvicorn.run(**config)


if __name__ == "__main__":
    run_server()