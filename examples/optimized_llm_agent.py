# examples/optimized_llm_agent.py
import httpx
from typing import Dict, Any, Optional
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)

# --- (OptimizedLLMAPIClient class remains the same) ---
class OptimizedLLMAPIClient:
    """
    Client to interact with the Optimized LLM API.
    Handles login and chat requests.
    """
    def __init__(self, base_url: str, identifier: str, password: str):
        self.base_url = base_url.rstrip('/')
        self.identifier = identifier
        self.password = password
        self.access_token: Optional[str] = None
        self.http_client = httpx.AsyncClient()

    async def login(self):
        """Authenticate and obtain an access token."""
        login_url = f"{self.base_url}/login"
        payload = {"identifier": self.identifier, "password": self.password}
        try:
            response = await self.http_client.post(login_url, json=payload)
            response.raise_for_status()
            data = response.json()
            self.access_token = data.get("access_token")
            if not self.access_token:
                raise ValueError("Login failed: No access token received.")
            logger.info(f"Successfully logged in. Token: {self.access_token[:10]}...")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Login failed with status {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Login failed: {e}") from e

    async def chat(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.8, depth_mode: bool = True) -> Dict[str, Any]:
        """Send a chat prompt to the API."""
        if not self.access_token:
            raise RuntimeError("Not authenticated. Please call login() first.")

        chat_url = f"{self.base_url}/chat"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "depth_mode": depth_mode
        }
        try:
            response = await self.http_client.post(chat_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                 self.access_token = None
            raise RuntimeError(f"Chat failed ({e.response.status_code}): {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Chat failed: {e}") from e

    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()

    async def __aenter__(self):
        await self.login()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# --- (OptimizedLLMAgent class - MODIFIED run method) ---
class OptimizedLLMAgent:
    """
    A FluxGraph agent that uses the Optimized LLM API.
    """
    def __init__(self, api_client: OptimizedLLMAPIClient):
        self.client = api_client

    async def run(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.8, depth_mode: bool = True, **kwargs): # <<<--- Accept **kwargs
        """
        Run the agent with a given prompt and parameters.
        Args (from FluxGraph API call):
            prompt (str): The user's prompt.
            max_tokens (int, optional): Max tokens for generation.
            temperature (float, optional): Sampling temperature.
            depth_mode (bool, optional): Enable depth mode.
            **kwargs: Additional arguments passed from the request payload.
                      They are ignored in this implementation.
        """
        # Log or handle unexpected arguments if needed for debugging
        if kwargs:
            logger.debug(f"OptimizedLLMAgent.run received unexpected arguments: {kwargs}. Ignoring.")

        try:
            response_data = await self.client.chat(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                depth_mode=depth_mode
            )
            return response_data
        except RuntimeError as e:
            logger.error(f"OptimizedLLMAgent execution error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in OptimizedLLMAgent: {e}")
            return {"error": f"Unexpected error in OptimizedLLMAgent: {e}"}

# --- Factory Function (remains the same) ---
async def create_optimized_llm_agent(base_url: str, identifier: str, password: str) -> OptimizedLLMAgent:
    """
    Factory function to create an agent instance, handling login.
    """
    client = OptimizedLLMAPIClient(base_url, identifier, password)
    await client.login()
    return OptimizedLLMAgent(client)
