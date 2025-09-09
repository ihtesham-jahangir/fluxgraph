# examples/run_server.py
import os
import logging # Import logging
from fluxgraph import FluxApp
from fluxgraph.models import OpenAIProvider # Example
# Import our new custom API agent
from examples.optimized_llm_agent import create_optimized_llm_agent

# --- Configure basic logging ---
logging.basicConfig(level=logging.INFO) # Adjust level as needed (DEBUG, INFO, WARNING, ERROR)
logger = logging.getLogger(__name__) # Get logger for this module

# --- Simple Echo Agent ---
class EchoAgent:
    def run(self, query: str):
        return {"response": f"Echo: {query}"}

# --- Initialize FluxGraph ---
app = FluxApp(title="FluxGraph Demo - Multi-Model & Custom API")

# --- Register Simple Echo Agent ---
app.register("echo", EchoAgent())

# --- Register Agents using Standard Model Providers (Examples - simplified) ---
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    try:
        openai_model = OpenAIProvider(api_key=openai_key, model="gpt-3.5-turbo")
        class SmartAgent:
             def __init__(self, name, model): self.name, self.model = name, model
             async def run(self, query: str, **kwargs): res = await self.model.generate(query); return {"agent": self.name, "response": res.get("text", "")}
        app.register("openai_agent", SmartAgent("OpenAI", openai_model))
        logger.info("OpenAI agent registered.")
    except Exception as e: logger.error(f"OpenAI setup failed: {e}")

# --- Register the Custom Optimized LLM API Agent ---
OPTIMIZED_LLM_API_URL = os.getenv("OPTIMIZED_LLM_API_URL", "https://www.llm.alphanetwork.com.pk") # Default if not set
OPTIMIZED_LLM_IDENTIFIER = os.getenv("OPTIMIZED_LLM_IDENTIFIER", "ihteshamjahangir21@gmail.com")
OPTIMIZED_LLM_PASSWORD = os.getenv("OPTIMIZED_LLM_PASSWORD", "123456")

@app.api.on_event("startup") # Use app.api.on_event as per previous guidance
async def register_custom_agents():
    """Register agents that need async initialization."""
    global app # Ensure we are modifying the correct 'app' instance
    try:
        optimized_agent = await create_optimized_llm_agent(
            base_url=OPTIMIZED_LLM_API_URL,
            identifier=OPTIMIZED_LLM_IDENTIFIER,
            password=OPTIMIZED_LLM_PASSWORD
        )
        app.register("optimized_llm", optimized_agent) # Register with the FluxApp instance
        logger.info("Successfully registered 'optimized_llm' agent.")
    except Exception as e:
        # Log the error but don't prevent the app from starting
        logger.error(f"Failed to register 'optimized_llm' agent during startup: {e}. Agent will be unavailable.")
        # Optionally, you could register a placeholder/error agent
        # app.register("optimized_llm", ErrorAgent(f"Failed to initialize: {e}"))


# --- Run API ---
# Run with:
# export OPTIMIZED_LLM_API_URL='http://your-llm-api-host:port'
# export OPTIMIZED_LLM_IDENTIFIER='your_user_or_key'
# export OPTIMIZED_LLM_PASSWORD='your_password'
# uvicorn examples.run_server:app.api --reload # Remember to use :app.api

if __name__ == "__main__":
    print("Run with: uvicorn examples.run_server:app.api --reload")
