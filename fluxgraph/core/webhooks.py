"""
FluxGraph Webhook System

Provides event-driven webhook notifications for:
- Agent execution events
- Workflow state changes
- Error notifications
- Custom events
"""

import asyncio
import httpx
import logging
import json
import hmac
import hashlib
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Standard webhook event types"""
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"

    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_STEP_COMPLETED = "workflow.step.completed"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"

    COST_THRESHOLD_EXCEEDED = "cost.threshold.exceeded"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"

    CUSTOM = "custom"


@dataclass
class WebhookPayload:
    """Standard webhook payload structure"""
    event: str
    timestamp: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class WebhookClient:
    """HTTP client for sending webhooks"""

    def __init__(self, timeout: int = 10, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(timeout=timeout)

    async def send(
        self,
        url: str,
        payload: Dict[str, Any],
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Send webhook with retries and signature

        Args:
            url: Webhook endpoint URL
            payload: Data to send
            secret: Optional secret for HMAC signature
            headers: Additional headers

        Returns:
            True if successful, False otherwise
        """
        headers = headers or {}
        headers["Content-Type"] = "application/json"
        headers["User-Agent"] = "FluxGraph-Webhook/1.0"

        # Add HMAC signature if secret provided
        if secret:
            payload_str = json.dumps(payload, sort_keys=True)
            signature = hmac.new(
                secret.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-FluxGraph-Signature"] = f"sha256={signature}"

        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    url,
                    json=payload,
                    headers=headers
                )

                if response.status_code < 300:
                    logger.debug(f"✅ Webhook sent to {url}")
                    return True
                else:
                    logger.warning(
                        f"⚠️ Webhook failed: {url} (status {response.status_code})"
                    )

            except Exception as e:
                logger.error(f"❌ Webhook error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        return False

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class WebhookManager:
    """
    Manages webhook subscriptions and delivery

    Example:
        webhook_mgr = WebhookManager()

        # Register webhook endpoint
        webhook_mgr.register(
            event=WebhookEvent.AGENT_COMPLETED,
            url="https://myapp.com/webhooks/agent-complete",
            secret="my_secret_key"
        )

        # Emit event
        await webhook_mgr.emit(
            event=WebhookEvent.AGENT_COMPLETED,
            data={"agent_name": "research_agent", "result": "..."}
        )
    """

    def __init__(self):
        self.subscriptions: Dict[str, List[Dict]] = {}
        self.client = WebhookClient()
        self._event_handlers: Dict[str, List[Callable]] = {}

    def register(
        self,
        event: str,
        url: str,
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        filter_fn: Optional[Callable] = None
    ):
        """
        Register a webhook endpoint for an event

        Args:
            event: Event type (use WebhookEvent enum)
            url: Webhook endpoint URL
            secret: Optional secret for HMAC signatures
            headers: Additional headers to send
            filter_fn: Optional filter function to conditionally send
        """
        if event not in self.subscriptions:
            self.subscriptions[event] = []

        self.subscriptions[event].append({
            "url": url,
            "secret": secret,
            "headers": headers,
            "filter_fn": filter_fn
        })

        logger.info(f"📡 Registered webhook: {event} → {url}")

    def on(self, event: str, handler: Callable):
        """
        Register in-process event handler (alternative to HTTP webhooks)

        Example:
            @webhook_mgr.on(WebhookEvent.AGENT_COMPLETED)
            async def handle_completion(data):
                print(f"Agent completed: {data}")
        """
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
        logger.info(f"📡 Registered event handler for: {event}")

    async def emit(
        self,
        event: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Emit an event to all subscribed webhooks and handlers

        Args:
            event: Event type
            data: Event data
            metadata: Optional metadata (e.g., user_id, session_id)
        """
        payload = WebhookPayload(
            event=event,
            timestamp=datetime.utcnow().isoformat(),
            data=data,
            metadata=metadata
        )

        # Send to HTTP webhooks
        if event in self.subscriptions:
            tasks = []
            for sub in self.subscriptions[event]:
                # Apply filter if present
                if sub["filter_fn"] and not sub["filter_fn"](data):
                    continue

                task = self.client.send(
                    url=sub["url"],
                    payload=payload.to_dict(),
                    secret=sub["secret"],
                    headers=sub["headers"]
                )
                tasks.append(task)

            if tasks:
                # Fire and forget (non-blocking)
                asyncio.create_task(self._send_all(tasks))

        # Call in-process handlers
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                asyncio.create_task(self._call_handler(handler, data))

    async def _send_all(self, tasks: List):
        """Send all webhooks concurrently"""
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _call_handler(self, handler: Callable, data: Dict):
        """Call event handler safely"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(data)
            else:
                handler(data)
        except Exception as e:
            logger.error(f"❌ Event handler error: {e}")

    def unregister(self, event: str, url: Optional[str] = None):
        """
        Unregister webhook(s)

        Args:
            event: Event type
            url: Specific URL to remove (if None, removes all for event)
        """
        if event in self.subscriptions:
            if url:
                self.subscriptions[event] = [
                    sub for sub in self.subscriptions[event]
                    if sub["url"] != url
                ]
            else:
                del self.subscriptions[event]

    def list_subscriptions(self, event: Optional[str] = None) -> Dict:
        """
        List all webhook subscriptions

        Args:
            event: Filter by event type (None for all)

        Returns:
            Dictionary of subscriptions
        """
        if event:
            return {event: self.subscriptions.get(event, [])}
        return self.subscriptions

    async def close(self):
        """Cleanup resources"""
        await self.client.close()


class WebhookDecorator:
    """
    Decorator for easy webhook emission

    Example:
        @webhook("agent.completed")
        async def my_agent(query: str):
            result = process(query)
            return result  # Automatically emits webhook on completion
    """

    def __init__(self, manager: WebhookManager):
        self.manager = manager

    def emit(self, event: str, extract_data: Optional[Callable] = None):
        """
        Decorator to emit webhook after function execution

        Args:
            event: Event type to emit
            extract_data: Function to extract data from result
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

                    # Extract data for webhook
                    data = extract_data(result) if extract_data else {"result": result}

                    # Emit success webhook
                    await self.manager.emit(event, data)

                    return result

                except Exception as e:
                    # Emit error webhook
                    await self.manager.emit(
                        f"{event}.failed",
                        {"error": str(e), "type": type(e).__name__}
                    )
                    raise

            return wrapper
        return decorator


# Global instance (optional)
_default_webhook_manager = None


def get_webhook_manager() -> WebhookManager:
    """Get or create default webhook manager"""
    global _default_webhook_manager
    if _default_webhook_manager is None:
        _default_webhook_manager = WebhookManager()
    return _default_webhook_manager


# Convenience decorators
def webhook(event: str):
    """Decorator using default webhook manager"""
    return WebhookDecorator(get_webhook_manager()).emit(event)
