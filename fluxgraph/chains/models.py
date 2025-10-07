# fluxgraph/chains/models.py
"""
Model wrappers for chain integration
"""
from typing import Any, Dict, List, Optional, AsyncIterator
from fluxgraph.chains import Runnable, RunnableConfig
from fluxgraph.models import ModelProvider, create_provider


class LLMRunnable(Runnable):
    """
    Wrap a model provider as a Runnable for use in chains.
    
    Example:
        llm = LLMRunnable("openai", "gpt-4")
        chain = prompt | llm | parser
    """
    
    def __init__(
        self,
        provider: str,
        model: str,
        **kwargs
    ):
        super().__init__(name=f"LLM_{provider}_{model}")
        self.provider_instance = create_provider(provider, model, **kwargs)
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Lazy initialization"""
        if not self._initialized:
            await self.provider_instance.initialize()
            self._initialized = True
    
    async def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        """Generate response"""
        await self._ensure_initialized()
        
        # Handle different input types
        if isinstance(input, str):
            response = await self.provider_instance.generate(input)
            return response.text
        elif isinstance(input, dict):
            if "messages" in input:
                response = await self.provider_instance.chat(input["messages"])
            else:
                prompt = input.get("prompt", str(input))
                system = input.get("system")
                response = await self.provider_instance.generate(prompt, system_message=system)
            return response.text
        elif isinstance(input, list):
            # Assume list of messages
            response = await self.provider_instance.chat(input)
            return response.text
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    
    async def stream(self, input: Any, config: Optional[RunnableConfig] = None) -> AsyncIterator[str]:
        """Stream response"""
        await self._ensure_initialized()
        
        if isinstance(input, str):
            async for chunk in self.provider_instance.stream_generate(input):
                yield chunk
        elif isinstance(input, dict):
            prompt = input.get("prompt", str(input))
            system = input.get("system")
            async for chunk in self.provider_instance.stream_generate(prompt, system_message=system):
                yield chunk
        else:
            # Fall back to non-streaming
            result = await self.invoke(input, config)
            yield result


def create_llm_runnable(provider: str, model: str, **kwargs) -> LLMRunnable:
    """
    Convenience function to create LLM runnable.
    
    Example:
        gpt4 = create_llm_runnable("openai", "gpt-4")
        chain = prompt | gpt4
    """
    return LLMRunnable(provider, model, **kwargs)
