# fluxgraph/marketplace/templates.py
"""
Agent Template Marketplace for FluxGraph.
Pre-built agent templates for common use cases.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TemplateCategory(Enum):
    """Agent template categories."""
    CUSTOMER_SERVICE = "customer_service"
    DATA_ANALYSIS = "data_analysis"
    CONTENT_GENERATION = "content_generation"
    CODE_ASSISTANCE = "code_assistance"
    RESEARCH = "research"
    AUTOMATION = "automation"
    TRANSLATION = "translation"
    MODERATION = "moderation"


class AgentTemplate:
    """Represents a reusable agent template."""
    
    def __init__(
        self,
        template_id: str,
        name: str,
        description: str,
        category: TemplateCategory,
        code_template: str,
        config: Dict[str, Any],
        required_tools: List[str],
        example_usage: str,
        author: str,
        version: str = "1.0.0",
        tags: Optional[List[str]] = None
    ):
        self.template_id = template_id
        self.name = name
        self.description = description
        self.category = category
        self.code_template = code_template
        self.config = config
        self.required_tools = required_tools
        self.example_usage = example_usage
        self.author = author
        self.version = version
        self.tags = tags or []
        self.created_at = datetime.utcnow()
        self.downloads = 0
        self.rating = 0.0
        self.reviews = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "config": self.config,
            "required_tools": self.required_tools,
            "example_usage": self.example_usage,
            "author": self.author,
            "version": self.version,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "downloads": self.downloads,
            "rating": self.rating,
            "reviews": len(self.reviews)
        }


class TemplateMarketplace:
    """
    Agent template marketplace with discovery and installation.
    """
    
    def __init__(self):
        self.templates: Dict[str, AgentTemplate] = {}
        self._register_builtin_templates()
        logger.info("TemplateMarketplace initialized")
    
    def _register_builtin_templates(self):
        """Register built-in templates."""
        
        # Customer Service Agent
        self.register_template(AgentTemplate(
            template_id="customer_service_basic",
            name="Customer Service Agent",
            description="Handles customer inquiries with empathy and efficiency",
            category=TemplateCategory.CUSTOMER_SERVICE,
            code_template="""
@app.agent()
async def customer_service_agent(query: str, customer_id: str, tools, memory):
    # Retrieve customer history
    history = await memory.get(customer_id, limit=5) if memory else []
    
    # Analyze sentiment
    sentiment = "positive"  # Would use sentiment analysis
    
    # Generate response
    response = f"Thank you for contacting us. Regarding: {query}"
    
    # Log interaction
    if memory:
        await memory.add(customer_id, {
            "query": query,
            "response": response,
            "sentiment": sentiment
        })
    
    return {
        "response": response,
        "sentiment": sentiment,
        "requires_escalation": False
    }
""",
            config={
                "max_response_length": 500,
                "tone": "professional_friendly",
                "escalation_threshold": 0.7
            },
            required_tools=["sentiment_analysis", "ticket_system"],
            example_usage="customer_service_agent(query='Order status', customer_id='C12345')",
            author="FluxGraph Team",
            tags=["customer-support", "chatbot", "service"]
        ))
        
        # Data Analysis Agent
        self.register_template(AgentTemplate(
            template_id="data_analyst",
            name="Data Analysis Agent",
            description="Analyzes datasets and generates insights",
            category=TemplateCategory.DATA_ANALYSIS,
            code_template="""
@app.agent()
async def data_analyst_agent(dataset_path: str, analysis_type: str, tools):
    import pandas as pd
    
    # Load data
    df = pd.read_csv(dataset_path)
    
    # Perform analysis
    if analysis_type == "summary":
        result = df.describe().to_dict()
    elif analysis_type == "correlation":
        result = df.corr().to_dict()
    else:
        result = {"error": "Unknown analysis type"}
    
    return {
        "analysis_type": analysis_type,
        "results": result,
        "row_count": len(df),
        "columns": list(df.columns)
    }
""",
            config={
                "max_file_size_mb": 100,
                "supported_formats": ["csv", "json", "xlsx"]
            },
            required_tools=["file_reader", "data_visualizer"],
            example_usage="data_analyst_agent(dataset_path='data.csv', analysis_type='summary')",
            author="FluxGraph Team",
            tags=["data", "analytics", "pandas"]
        ))
        
        # Code Assistant Agent
        self.register_template(AgentTemplate(
            template_id="code_assistant",
            name="Code Assistant Agent",
            description="Helps with code generation and debugging",
            category=TemplateCategory.CODE_ASSISTANCE,
            code_template="""
@app.agent()
async def code_assistant_agent(task: str, language: str, context: str, tools):
    # Generate code based on task
    prompt = f"Generate {language} code for: {task}\\nContext: {context}"
    
    # Would call LLM here
    generated_code = "# Generated code\\nprint('Hello, World!')"
    
    # Validate syntax
    is_valid = True  # Would use syntax checker
    
    return {
        "generated_code": generated_code,
        "language": language,
        "valid_syntax": is_valid,
        "explanation": "Code explanation here"
    }
""",
            config={
                "supported_languages": ["python", "javascript", "typescript", "java"],
                "max_code_length": 1000
            },
            required_tools=["syntax_validator", "code_formatter"],
            example_usage="code_assistant_agent(task='Sort array', language='python', context='')",
            author="FluxGraph Team",
            tags=["coding", "development", "llm"]
        ))
        
        logger.info(f"Registered {len(self.templates)} built-in templates")
    
    def register_template(self, template: AgentTemplate):
        """Register a new template in the marketplace."""
        self.templates[template.template_id] = template
        logger.info(f"[Marketplace] Registered template: {template.name}")
    
    def search_templates(
        self,
        query: Optional[str] = None,
        category: Optional[TemplateCategory] = None,
        tags: Optional[List[str]] = None
    ) -> List[AgentTemplate]:
        """
        Search templates by query, category, or tags.
        
        Args:
            query: Search query
            category: Filter by category
            tags: Filter by tags
        
        Returns:
            List of matching templates
        """
        results = list(self.templates.values())
        
        if query:
            query_lower = query.lower()
            results = [
                t for t in results
                if query_lower in t.name.lower() or query_lower in t.description.lower()
            ]
        
        if category:
            results = [t for t in results if t.category == category]
        
        if tags:
            results = [
                t for t in results
                if any(tag in t.tags for tag in tags)
            ]
        
        return results
    
    def get_template(self, template_id: str) -> Optional[AgentTemplate]:
        """Get template by ID."""
        return self.templates.get(template_id)
    
    def install_template(
        self,
        template_id: str,
        app,
        agent_name: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Install template as an agent in FluxApp.
        
        Args:
            template_id: Template to install
            app: FluxApp instance
            agent_name: Custom agent name (uses template name if None)
            config_override: Override default config
        
        Returns:
            Name of installed agent
        """
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        agent_name = agent_name or template.name.lower().replace(" ", "_")
        
        # Merge config
        config = {**template.config, **(config_override or {})}
        
        # Increment download counter
        template.downloads += 1
        
        logger.info(
            f"[Marketplace] Installing template '{template.name}' as agent '{agent_name}'"
        )
        
        # In practice, would execute template.code_template
        # This is simplified for demonstration
        
        return agent_name
    
    def get_popular_templates(self, limit: int = 10) -> List[AgentTemplate]:
        """Get most popular templates by downloads."""
        sorted_templates = sorted(
            self.templates.values(),
            key=lambda t: t.downloads,
            reverse=True
        )
        return sorted_templates[:limit]
