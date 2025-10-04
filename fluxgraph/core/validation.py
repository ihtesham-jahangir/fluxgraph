# fluxgraph/core/validation.py
"""
Output Validation System for FluxGraph.
Ensures agent outputs conform to expected schemas using Pydantic.
"""

import logging
from typing import Any, Dict, Type, Optional, Union
from pydantic import BaseModel, ValidationError, Field
import json

logger = logging.getLogger(__name__)


class AgentOutput(BaseModel):
    """Base model for validated agent outputs."""
    success: bool = Field(default=True, description="Whether the operation succeeded")
    data: Dict[str, Any] = Field(default_factory=dict, description="Output data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ValidationResult:
    """Result of output validation."""
    
    def __init__(
        self,
        is_valid: bool,
        validated_data: Optional[Any] = None,
        errors: Optional[list] = None
    ):
        self.is_valid = is_valid
        self.validated_data = validated_data
        self.errors = errors or []


class OutputValidator:
    """Validates agent outputs against Pydantic schemas."""
    
    def __init__(self):
        self.registered_schemas: Dict[str, Type[BaseModel]] = {}
        logger.info("OutputValidator initialized")
    
    def register_schema(self, agent_name: str, schema: Type[BaseModel]):
        """Register a Pydantic schema for an agent."""
        self.registered_schemas[agent_name] = schema
        logger.info(f"Registered validation schema for agent: {agent_name}")
    
    def validate(
        self,
        agent_name: str,
        output: Any,
        schema: Optional[Type[BaseModel]] = None
    ) -> ValidationResult:
        """
        Validate agent output against a schema.
        
        Args:
            agent_name: Name of the agent
            output: Output to validate
            schema: Pydantic model (uses registered if None)
        
        Returns:
            ValidationResult with validation status and errors
        """
        schema = schema or self.registered_schemas.get(agent_name)
        
        if not schema:
            logger.warning(f"No schema registered for agent: {agent_name}")
            return ValidationResult(is_valid=True, validated_data=output)
        
        try:
            # Handle dict outputs
            if isinstance(output, dict):
                validated = schema(**output)
            # Handle string outputs (try JSON parse)
            elif isinstance(output, str):
                try:
                    parsed = json.loads(output)
                    validated = schema(**parsed)
                except json.JSONDecodeError:
                    return ValidationResult(
                        is_valid=False,
                        errors=["Output is not valid JSON"]
                    )
            # Handle Pydantic models
            elif isinstance(output, BaseModel):
                validated = output
            else:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Unsupported output type: {type(output)}"]
                )
            
            logger.info(f"[Validator:{agent_name}] Output validation passed")
            return ValidationResult(is_valid=True, validated_data=validated)
            
        except ValidationError as e:
            logger.warning(f"[Validator:{agent_name}] Validation failed: {e}")
            errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
            return ValidationResult(is_valid=False, errors=errors)
    
    def validate_or_raise(
        self,
        agent_name: str,
        output: Any,
        schema: Optional[Type[BaseModel]] = None
    ) -> Any:
        """Validate and raise exception if invalid."""
        result = self.validate(agent_name, output, schema)
        if not result.is_valid:
            error_msg = f"Validation failed for agent '{agent_name}': {', '.join(result.errors)}"
            raise ValidationError(error_msg)
        return result.validated_data
