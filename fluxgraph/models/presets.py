# fluxgraph/models/presets.py
"""
Pre-configured model presets for common use cases
"""
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ModelPreset:
    """Pre-configured model settings"""
    provider: str
    model: str
    temperature: float
    max_tokens: int
    description: str
    extra_params: Dict[str, Any] = None


# Common presets
PRESETS = {
    # OpenAI
    "gpt4": ModelPreset(
        provider="openai",
        model="gpt-4-turbo-preview",
        temperature=0.7,
        max_tokens=2000,
        description="GPT-4 Turbo - Most capable"
    ),
    
    "gpt3": ModelPreset(
        provider="openai",
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1000,
        description="GPT-3.5 Turbo - Fast and efficient"
    ),
    
    "gpt4-creative": ModelPreset(
        provider="openai",
        model="gpt-4-turbo-preview",
        temperature=1.2,
        max_tokens=2000,
        description="GPT-4 with high creativity"
    ),
    
    "gpt3-precise": ModelPreset(
        provider="openai",
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=500,
        description="GPT-3.5 with low temperature for accuracy"
    ),
    
    # Anthropic
    "claude": ModelPreset(
        provider="anthropic",
        model="claude-3-opus-20240229",
        temperature=0.7,
        max_tokens=2000,
        description="Claude 3 Opus - Most capable Claude"
    ),
    
    "claude-fast": ModelPreset(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        temperature=0.7,
        max_tokens=1000,
        description="Claude 3 Haiku - Fastest Claude"
    ),
    
    # Groq (Ultra fast)
    "groq-fast": ModelPreset(
        provider="groq",
        model="mixtral-8x7b-32768",
        temperature=0.7,
        max_tokens=1000,
        description="Groq Mixtral - Ultra fast inference"
    ),
    
    # Google
    "gemini": ModelPreset(
        provider="gemini",
        model="gemini-pro",
        temperature=0.7,
        max_tokens=2000,
        description="Google Gemini Pro"
    ),
    
    # Ollama (Local)
    "llama": ModelPreset(
        provider="ollama",
        model="llama2",
        temperature=0.7,
        max_tokens=1000,
        description="Llama 2 - Run locally"
    ),
    
    "codellama": ModelPreset(
        provider="ollama",
        model="codellama",
        temperature=0.3,
        max_tokens=2000,
        description="Code Llama - Optimized for code"
    ),
}


def get_preset(name: str) -> ModelPreset:
    """
    Get a preset by name.
    
    Args:
        name: Preset name
    
    Returns:
        ModelPreset
    
    Raises:
        ValueError: If preset not found
    
    Example:
        preset = get_preset("gpt4")
        llm = await get_llm(preset.provider, preset.model)
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    return PRESETS[name]


def list_presets() -> Dict[str, str]:
    """
    List all available presets.
    
    Returns:
        Dict of preset names and descriptions
    """
    return {name: preset.description for name, preset in PRESETS.items()}
