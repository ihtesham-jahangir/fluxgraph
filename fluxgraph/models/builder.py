# fluxgraph/models/builder.py
"""
Fluent builder pattern for easy provider configuration
"""
from typing import Optional, List, Dict, Any
from .factory import ModelProviderFactory
from .provider import ModelProvider
from .presets import get_preset


class LLMBuilder:
    """
    Fluent builder for creating LLM providers.
    
    Example:
        llm = await (
            LLM()
            .use("openai")
            .model("gpt-4")
            .temperature(0.8)
            .max_tokens(2000)
            .build()
        )
    """
    
    def __init__(self):
        self._provider = "openai"
        self._model = "gpt-3.5-turbo"
        self._config = {}
    
    def use(self, provider: str) -> 'LLMBuilder':
        """Set provider"""
        self._provider = provider
        return self
    
    def model(self, model_name: str) -> 'LLMBuilder':
        """Set model name"""
        self._model = model_name
        return self
    
    def preset(self, preset_name: str) -> 'LLMBuilder':
        """Use a preset configuration"""
        preset = get_preset(preset_name)
        self._provider = preset.provider
        self._model = preset.model
        self._config.update({
            'temperature': preset.temperature,
            'max_tokens': preset.max_tokens,
        })
        if preset.extra_params:
            self._config.update(preset.extra_params)
        return self
    
    def temperature(self, temp: float) -> 'LLMBuilder':
        """Set temperature (0-2)"""
        self._config['temperature'] = temp
        return self
    
    def max_tokens(self, tokens: int) -> 'LLMBuilder':
        """Set max tokens"""
        self._config['max_tokens'] = tokens
        return self
    
    def api_key(self, key: str) -> 'LLMBuilder':
        """Set API key"""
        self._config['api_key'] = key
        return self
    
    def timeout(self, seconds: int) -> 'LLMBuilder':
        """Set timeout"""
        self._config['timeout'] = seconds
        return self
    
    def streaming(self, enabled: bool = True) -> 'LLMBuilder':
        """Enable streaming"""
        self._config['streaming'] = enabled
        return self
    
    def creative(self) -> 'LLMBuilder':
        """Preset for creative writing"""
        self._config['temperature'] = 1.2
        self._config['top_p'] = 0.95
        return self
    
    def precise(self) -> 'LLMBuilder':
        """Preset for precise/factual responses"""
        self._config['temperature'] = 0.2
        self._config['top_p'] = 0.9
        return self
    
    def balanced(self) -> 'LLMBuilder':
        """Preset for balanced responses"""
        self._config['temperature'] = 0.7
        self._config['top_p'] = 1.0
        return self
    
    async def build(self) -> ModelProvider:
        """Build and initialize the provider"""
        provider = ModelProviderFactory.create(
            self._provider,
            self._model,
            **self._config
        )
        await provider.initialize()
        return provider
    
    def __repr__(self) -> str:
        return f"LLMBuilder(provider={self._provider}, model={self._model})"


# Convenience alias
LLM = LLMBuilder
