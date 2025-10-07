# fluxgraph/chains/prompts.py
"""
Prompt templates for FluxGraph chains
"""
from typing import Any, Dict, List, Optional
from fluxgraph.chains import Runnable, RunnableConfig


class PromptTemplate(Runnable):
    """
    Template for formatting prompts with variables.
    
    Example:
        prompt = PromptTemplate(
            "Tell me about {topic} in {style} style"
        )
        
        result = await prompt.invoke({
            "topic": "AI",
            "style": "simple"
        })
    """
    
    def __init__(
        self,
        template: str,
        input_variables: Optional[List[str]] = None,
        partial_variables: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="PromptTemplate")
        self.template = template
        self.input_variables = input_variables or self._extract_variables(template)
        self.partial_variables = partial_variables or {}
    
    async def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> str:
        """Format the template with input variables"""
        variables = {**self.partial_variables, **input}
        
        # Validate all required variables are present
        missing = set(self.input_variables) - set(variables.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        return self.template.format(**variables)
    
    def partial(self, **kwargs) -> 'PromptTemplate':
        """
        Create a new template with some variables pre-filled.
        
        Example:
            base = PromptTemplate("Translate {text} to {language}")
            spanish = base.partial(language="Spanish")
        """
        return PromptTemplate(
            self.template,
            self.input_variables,
            {**self.partial_variables, **kwargs}
        )
    
    @staticmethod
    def _extract_variables(template: str) -> List[str]:
        """Extract variable names from template"""
        import re
        return re.findall(r'\{(\w+)\}', template)


class ChatPromptTemplate(Runnable):
    """
    Chat-style prompt with multiple messages.
    
    Example:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("user", "Tell me about {topic}")
        ])
    """
    
    def __init__(self, messages: List[tuple]):
        super().__init__(name="ChatPromptTemplate")
        self.messages = messages
    
    async def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> List[Dict[str, str]]:
        """Format messages"""
        formatted = []
        for role, content in self.messages:
            if isinstance(content, str):
                formatted_content = content.format(**input)
            else:
                formatted_content = content
            
            formatted.append({
                "role": role,
                "content": formatted_content
            })
        
        return formatted
    
    @classmethod
    def from_messages(cls, messages: List[tuple]) -> 'ChatPromptTemplate':
        """Create from list of (role, content) tuples"""
        return cls(messages)
