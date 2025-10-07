# fluxgraph/crew/human.py
"""
Enhanced human-in-the-loop with multiple input types and validation.
"""
from typing import Optional, Any, Callable, Dict, List
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class InputType(Enum):
    """Types of human input."""
    TEXT = "text"
    CHOICE = "choice"
    CONFIRMATION = "confirmation"
    REVIEW = "review"
    RATING = "rating"
    FILE_UPLOAD = "file_upload"


@dataclass
class HumanInput:
    """Request for human input."""
    prompt: str
    input_type: InputType
    context: Optional[Dict[str, Any]] = None
    choices: Optional[List[str]] = None
    default: Optional[Any] = None
    timeout: Optional[int] = None
    validation: Optional[Callable] = None


class HumanInputHandler:
    """
    Handle human input with validation and timeout.
    
    Example:
        handler = HumanInputHandler()
        
        approved = await handler.request_confirmation(
            "Approve this content for publication?",
            context={"content": article}
        )
        
        rating = await handler.request_rating(
            "Rate the quality (1-5)",
            min_rating=1,
            max_rating=5
        )
    """
    
    def __init__(self, input_func: Optional[Callable] = None):
        """
        Initialize handler.
        
        Args:
            input_func: Custom function for getting input (default: console)
        """
        self.input_func = input_func or self._console_input
        self.history: List[Dict] = []
    
    async def request_input(
        self,
        prompt: str,
        input_type: InputType = InputType.TEXT,
        **kwargs
    ) -> Any:
        """
        Request input from human.
        
        Args:
            prompt: Question/prompt for human
            input_type: Type of input expected
            **kwargs: Additional parameters
            
        Returns:
            Human's response
        """
        request = HumanInput(
            prompt=prompt,
            input_type=input_type,
            **kwargs
        )
        
        logger.info(f"👤 Requesting human input: {prompt}")
        
        # Call input function
        response = await self.input_func(request)
        
        # Validate if validator provided
        if request.validation:
            while not request.validation(response):
                print("❌ Invalid input. Please try again.")
                response = await self.input_func(request)
        
        # Store in history
        self.history.append({
            "request": request,
            "response": response,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        return response
    
    async def request_text(
        self,
        prompt: str,
        default: Optional[str] = None,
        validation: Optional[Callable] = None
    ) -> str:
        """Request text input."""
        return await self.request_input(
            prompt,
            InputType.TEXT,
            default=default,
            validation=validation
        )
    
    async def request_choice(
        self,
        prompt: str,
        choices: List[str],
        **kwargs
    ) -> str:
        """Request choice from list."""
        return await self.request_input(
            prompt,
            InputType.CHOICE,
            choices=choices,
            **kwargs
        )
    
    async def request_confirmation(
        self,
        prompt: str,
        **kwargs
    ) -> bool:
        """Request yes/no confirmation."""
        response = await self.request_input(
            prompt,
            InputType.CONFIRMATION,
            **kwargs
        )
        return response.lower() in ['y', 'yes', 'true', '1']
    
    async def request_review(
        self,
        content: str,
        prompt: str = "Please review this content",
        **kwargs
    ) -> Dict[str, Any]:
        """Request content review."""
        return await self.request_input(
            prompt,
            InputType.REVIEW,
            context={"content": content},
            **kwargs
        )
    
    async def request_rating(
        self,
        prompt: str,
        min_rating: int = 1,
        max_rating: int = 5,
        **kwargs
    ) -> int:
        """Request numeric rating."""
        def validate_rating(value):
            try:
                val = int(value)
                return min_rating <= val <= max_rating
            except:
                return False
        
        response = await self.request_input(
            prompt,
            InputType.RATING,
            validation=validate_rating,
            **kwargs
        )
        return int(response)
    
    async def _console_input(self, request: HumanInput) -> Any:
        """Default console input handler."""
        print("\n" + "="*60)
        print("🙋 HUMAN INPUT REQUIRED")
        print("="*60)
        print(f"\n{request.prompt}\n")
        
        if request.context:
            print("Context:")
            for key, value in request.context.items():
                val_str = str(value)[:200]
                print(f"  {key}: {val_str}...")
            print()
        
        if request.input_type == InputType.CHOICE:
            print("Choices:")
            for i, choice in enumerate(request.choices, 1):
                print(f"  {i}. {choice}")
            print()
            
            while True:
                try:
                    choice = input("Enter choice number: ")
                    idx = int(choice) - 1
                    if 0 <= idx < len(request.choices):
                        return request.choices[idx]
                    print("Invalid choice. Try again.")
                except ValueError:
                    print("Invalid input. Enter a number.")
        
        elif request.input_type == InputType.CONFIRMATION:
            response = input("(y/n): ")
            return response.lower() in ['y', 'yes']
        
        elif request.input_type == InputType.REVIEW:
            print("Enter feedback (or press Enter to skip):")
            feedback = input("> ")
            approved = input("Approve? (y/n): ")
            return {
                "approved": approved.lower() in ['y', 'yes'],
                "feedback": feedback
            }
        
        elif request.input_type == InputType.RATING:
            rating = input("> ")
            return rating
        
        else:  # TEXT
            if request.default:
                response = input(f"[{request.default}]: ")
                return response or request.default
            return input("> ")


# Add to agent
def enable_human_input(agent_func):
    """
    Decorator to add human input handler to agent.
    
    Example:
        @enable_human_input
        @app.agent("reviewer")
        async def review_agent(content, human):
            approved = await human.request_confirmation(
                "Approve this content?"
            )
            return {"approved": approved}
    """
    async def wrapper(*args, **kwargs):
        handler = HumanInputHandler()
        kwargs['human'] = handler
        return await agent_func(*args, **kwargs)
    return wrapper
