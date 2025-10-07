# examples/test.py
"""
FluxGraph Demo Example - Fixed version
"""

from fluxgraph.core.app import FluxApp

# Create app
app = FluxApp(
    title="FluxGraph Demo API",
    description="FluxGraph demo application",
    version="3.3.0",
    enable_analytics=True,
    enable_advanced_features=True,
    enable_workflows=True,
    enable_advanced_memory=True,
    enable_agent_cache=True,
    cache_strategy="hybrid",
    enable_visual_workflows=True,
    enable_chains=True,
    enable_tracing=True,
    enable_batch_optimization=True,
    enable_streaming_optimization=True,
    enable_langserve_api=True,
    log_level="INFO"
)


# Register agents
@app.agent("hello_agent")
async def hello_agent(name: str, **kwargs):
    """Simple hello world agent"""
    return {
        "message": f"Hello, {name}!",
        "agent": "hello_agent",
        "timestamp": str(__import__('datetime').datetime.now())
    }


@app.agent("calculator_agent")
async def calculator_agent(operation: str, a: float, b: float, tools, **kwargs):
    """Calculator agent"""
    result = 0
    
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b if b != 0 else 0
    
    return {
        "agent": "calculator_agent",
        "operation": operation,
        "input": {"a": a, "b": b},
        "result": result,
        "tools_available": tools.list_tools()
    }


# Register tools
@app.tool("add")
def add_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@app.tool("multiply")
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


@app.tool("search")
async def search(query: str) -> dict:
    """Mock search tool"""
    return {
        "query": query,
        "results": [
            f"Result 1 for: {query}",
            f"Result 2 for: {query}",
            f"Result 3 for: {query}"
        ]
    }


# Startup event - FIXED
@app.api.on_event("startup")
async def startup():
    print("\n" + "="*80)
    print("🚀 FluxGraph Demo Server Started!")
    print("="*80)
    
    # Get agents using correct method
    try:
        agents = []
        for agent_name in dir(app.registry):
            if not agent_name.startswith('_'):
                agent = getattr(app.registry, agent_name, None)
                if callable(agent):
                    agents.append(agent_name)
    except:
        agents = ["hello_agent", "calculator_agent"]  # Fallback
    
    print(f"📋 Registered Agents: {agents}")
    print(f"🛠️  Registered Tools: {app.tool_registry.list_tools()}")
    print("\n📚 API Documentation: http://127.0.0.1:8000/docs")
    print("🏠 Root Endpoint: http://127.0.0.1:8000/")
    print("="*80 + "\n")


# Custom endpoint
@app.api.get("/demo")
async def demo_info():
    """Demo endpoint"""
    # Safe way to get agent names
    try:
        agent_names = list(app.registry._agents.keys()) if hasattr(app.registry, '_agents') else []
    except:
        agent_names = ["hello_agent", "calculator_agent"]
    
    return {
        "message": "FluxGraph Demo",
        "version": app.version,
        "agents": agent_names,
        "tools": app.tool_registry.list_tools(),
        "features": {
            "workflows": app.workflows_enabled,
            "memory": app.memory_store is not None,
            "rag": app.rag_connector is not None,
            "cache": app.agent_cache_enabled,
            "chains": app.chains_enabled,
            "tracing": app.tracing_enabled
        },
        "endpoints": {
            "agents": "/ask/{agent_name}",
            "tools": "/tools",
            "health": "/health",
            "docs": "/docs"
        }
    }


# Run directly if needed
if __name__ == "__main__":
    app.run()
