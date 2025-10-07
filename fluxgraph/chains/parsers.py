# fluxgraph/chains/parsers.py
"""
Output parsers for structured data extraction
"""
from typing import Any, Dict, List, Optional, Type
from fluxgraph.chains import Runnable, RunnableConfig
import json
import re
from pydantic import BaseModel


class OutputParser(Runnable):
    """Base class for output parsers"""
    
    def __init__(self, name: str = "OutputParser"):
        super().__init__(name=name)
    
    async def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> Any:
        """Parse the input"""
        return self.parse(input)
    
    def parse(self, text: str) -> Any:
        """Override this to implement parsing logic"""
        return text
    
    def get_format_instructions(self) -> str:
        """Get instructions for the model on how to format output"""
        return ""


class JsonOutputParser(OutputParser):
    """
    Parse JSON output from model.
    
    Example:
        parser = JsonOutputParser()
        chain = prompt | model | parser
        result = await chain.invoke(input)  # Returns dict
    """
    
    def __init__(self):
        super().__init__(name="JsonOutputParser")
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON"""
        # Extract JSON from markdown code blocks
        if "```" in text and "```json" in text:
            json_str = text.split("```json").split("```")[0]
        elif "```" in text:
            json_str = text.split("``````")[0]
        else:
            json_str = text
        
        return json.loads(json_str.strip())
    
    def get_format_instructions(self) -> str:
        return "Return your response as valid JSON."


class PydanticOutputParser(OutputParser):
    """
    Parse output into Pydantic model.
    
    Example:
        class Person(BaseModel):
            name: str
            age: int
        
        parser = PydanticOutputParser(pydantic_object=Person)
        chain = prompt | model | parser
        person = await chain.invoke(input)  # Returns Person instance
    """
    
    def __init__(self, pydantic_object: Type[BaseModel]):
        super().__init__(name="PydanticOutputParser")
        self.pydantic_object = pydantic_object
    
    def parse(self, text: str) -> BaseModel:
        """Parse into Pydantic model"""
        # First extract JSON
        json_parser = JsonOutputParser()
        data = json_parser.parse(text)
        
        # Then validate with Pydantic
        return self.pydantic_object(**data)
    
    def get_format_instructions(self) -> str:
        schema = self.pydantic_object.schema()
        return f"Return JSON matching this schema:\n{json.dumps(schema, indent=2)}"


class ListOutputParser(OutputParser):
    """
    Parse comma or newline-separated list.
    
    Example:
        parser = ListOutputParser()
        result = await parser.invoke("apple, banana, orange")
        # Returns: ["apple", "banana", "orange"]
    """
    
    def parse(self, text: str) -> List[str]:
        """Parse list from text"""
        # Try comma-separated first
        if ',' in text:
            items = [item.strip() for item in text.split(',')]
        else:
            # Try newline-separated
            items = [item.strip() for item in text.split('\n') if item.strip()]
        
        # Remove numbers/bullets
        cleaned = []
        for item in items:
            cleaned_item = re.sub(r'^\d+[\.)]\s*', '', item)
            cleaned_item = re.sub(r'^[-*]\s*', '', cleaned_item)
            if cleaned_item:
                cleaned.append(cleaned_item)
        
        return cleaned
    
    def get_format_instructions(self) -> str:
        return "Return a comma-separated list."


class StructuredOutputParser(OutputParser):
    """
    Parse structured output with named fields.
    
    Example:
        parser = StructuredOutputParser.from_response_schemas([
            ResponseSchema(name="answer", description="The answer"),
            ResponseSchema(name="source", description="The source")
        ])
    """
    
    def __init__(self, response_schemas: List[Dict[str, str]]):
        super().__init__(name="StructuredOutputParser")
        self.response_schemas = response_schemas
    
    def parse(self, text: str) -> Dict[str, str]:
        """Parse structured output"""
        result = {}
        
        for schema in self.response_schemas:
            name = schema["name"]
            # Try to extract field
            pattern = f"{name}:\\s*(.+?)(?:\\n|$)"
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            
            if match:
                result[name] = match.group(1).strip()
            else:
                result[name] = ""
        
        return result
    
    @classmethod
    def from_response_schemas(cls, schemas: List[Dict[str, str]]) -> 'StructuredOutputParser':
        return cls(schemas)
    
    def get_format_instructions(self) -> str:
        lines = ["Return the response in this format:"]
        for schema in self.response_schemas:
            lines.append(f"{schema['name']}: {schema['description']}")
        return "\n".join(lines)
