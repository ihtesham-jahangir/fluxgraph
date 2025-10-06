# examples/app.py
"""
FluxGraph Enterprise Example Application

Usage:
    # Method 1: Run with FluxGraph CLI
    flux run examples/app.py
    
    # Method 2: Run directly with Python
    python examples/app.py
    
    # Method 3: Run with Uvicorn (FastAPI instance)
    uvicorn examples.app:app --reload
"""
from fluxgraph import FluxApp

# Initialize FluxApp with all enterprise features
# Keep this named 'flux' for now, we'll export 'app' at the end
flux_app = FluxApp(
    title="FluxGraph API",
    version="2.0.0",
    description="Enterprise AI Agent Orchestration Framework",
    enable_streaming=True,
    enable_sessions=True,
    enable_security=True,
    enable_orchestration=True,
    enable_mcp=True,
    enable_versioning=True,
    enable_templates=True,
    enable_analytics=True
)

# Register sample agents
@flux_app.agent()
async def echo_agent(message: str) -> dict:
    """Simple echo agent for testing."""
    return {
        "message": message,
        "echo": f"You said: {message}",
        "agent": "echo_agent"
    }

@flux_app.agent()
async def calculator_agent(expression: str) -> dict:
    """Calculator agent with basic arithmetic."""
    try:
        # In production, use ast.literal_eval for safety
        result = eval(expression)
        return {
            "expression": expression,
            "result": result,
            "agent": "calculator_agent"
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "agent": "calculator_agent"
        }

@flux_app.agent()
async def greeting_agent(name: str, language: str = "en") -> dict:
    """Multi-language greeting agent."""
    greetings = {
        "en": f"Hello, {name}!",
        "es": f"¡Hola, {name}!",
        "fr": f"Bonjour, {name}!",
        "de": f"Hallo, {name}!",
        "ur": f"السلام علیکم، {name}!"
    }
    
    return {
        "name": name,
        "language": language,
        "greeting": greetings.get(language, greetings["en"]),
        "agent": "greeting_agent"
    }

# Register sample tools
@flux_app.tool()
def uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()

@flux_app.tool()
def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()

@flux_app.tool()
def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())

# IMPORTANT: Export the correct instance based on usage
# For 'flux run' command: Export FluxApp instance as 'app'
# For 'uvicorn' command: Also support FastAPI instance
app = flux_app  # For flux run command - needs FluxApp instance

# Also export the FastAPI instance for uvicorn compatibility
fastapi_app = flux_app.api  # For uvicorn if needed

if __name__ == "__main__":
    # When running directly with python
    print("=" * 80)
    print("FluxGraph Enterprise API")
    print("=" * 80)
    print("\n📚 Available Agents:")
    print("  - echo_agent: Echo back messages")
    print("  - calculator_agent: Perform calculations")
    print("  - greeting_agent: Multi-language greetings")
    print("\n🛠️  Available Tools:")
    print("  - uppercase: Convert to uppercase")
    print("  - lowercase: Convert to lowercase")
    print("  - word_count: Count words")
    print("\n🌐 Starting server...")
    print("   Docs: http://localhost:8000/docs")
    print("   API: http://localhost:8000")
    print("=" * 80)
    
    flux_app.run()
