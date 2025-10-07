# fluxgraph/models/simple.py
"""
Ultra-simple interface - just import and use!
"""
from typing import Optional, List, Dict
from .easy import ask, chat, stream, ask_sync, chat_sync
from .builder import LLM
from .presets import list_presets, get_preset


# Quick functions with common models
async def ask_gpt4(question: str, system: Optional[str] = None) -> str:
    """Ask GPT-4"""
    return await ask(question, model="gpt-4-turbo-preview", system=system)


async def ask_gpt3(question: str, system: Optional[str] = None) -> str:
    """Ask GPT-3.5"""
    return await ask(question, model="gpt-3.5-turbo", system=system)


async def ask_claude(question: str, system: Optional[str] = None) -> str:
    """Ask Claude"""
    return await ask(question, provider="anthropic", model="claude-3-opus-20240229", system=system)


async def ask_gemini(question: str, system: Optional[str] = None) -> str:
    """Ask Gemini"""
    return await ask(question, provider="gemini", model="gemini-pro", system=system)


async def ask_local(question: str, model: str = "llama2") -> str:
    """Ask local Ollama model"""
    return await ask(question, provider="ollama", model=model)


# Synchronous versions
def gpt4(question: str) -> str:
    """Sync: Ask GPT-4"""
    return ask_sync(question, model="gpt-4-turbo-preview")


def gpt3(question: str) -> str:
    """Sync: Ask GPT-3.5"""
    return ask_sync(question, model="gpt-3.5-turbo")


def claude(question: str) -> str:
    """Sync: Ask Claude"""
    return ask_sync(question, provider="anthropic", model="claude-3-opus-20240229")


# Model comparison
async def compare_models(
    question: str,
    models: List[str] = None
) -> Dict[str, str]:
    """
    Compare responses from multiple models.
    
    Args:
        question: Question to ask
        models: List of preset names (default: ["gpt3", "gpt4", "claude"])
    
    Returns:
        Dict of model name to response
    
    Example:
        results = await compare_models("What is AI?")
        for model, answer in results.items():
            print(f"{model}: {answer}")
    """
    if models is None:
        models = ["gpt3", "gpt4", "claude"]
    
    results = {}
    for preset_name in models:
        try:
            preset = get_preset(preset_name)
            response = await ask(
                question,
                provider=preset.provider,
                model=preset.model
            )
            results[preset_name] = response
        except Exception as e:
            results[preset_name] = f"Error: {e}"
    
    return results
