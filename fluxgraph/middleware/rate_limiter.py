"""
FluxGraph Rate Limiting Middleware

Provides flexible rate limiting with:
- Per-user limits
- Per-IP limits
- Per-endpoint limits
- Redis and in-memory backends
- Customizable strategies
"""

import time
import asyncio
import hashlib
from typing import Optional, Dict, Callable
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import logging


logger = logging.getLogger(__name__)


class RateLimitBackend:
    """Base class for rate limit storage backends"""

    async def increment(self, key: str, window: int) -> int:
        """Increment counter and return current count"""
        raise NotImplementedError

    async def reset(self, key: str):
        """Reset counter for key"""
        raise NotImplementedError

    async def get(self, key: str) -> int:
        """Get current count for key"""
        raise NotImplementedError


class InMemoryBackend(RateLimitBackend):
    """In-memory rate limit backend (single instance)"""

    def __init__(self):
        self._counters: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "reset_at": 0})
        self._lock = asyncio.Lock()

    async def increment(self, key: str, window: int) -> int:
        async with self._lock:
            now = time.time()
            data = self._counters[key]

            # Reset if window expired
            if now >= data["reset_at"]:
                data["count"] = 0
                data["reset_at"] = now + window

            data["count"] += 1
            return data["count"]

    async def reset(self, key: str):
        async with self._lock:
            if key in self._counters:
                del self._counters[key]

    async def get(self, key: str) -> int:
        return self._counters[key]["count"]

    def get_reset_time(self, key: str) -> float:
        """Get reset timestamp"""
        return self._counters[key]["reset_at"]


class RedisBackend(RateLimitBackend):
    """Redis-backed rate limiting (distributed)"""

    def __init__(self, redis_client):
        """
        Args:
            redis_client: Redis client instance (from redis-py or aioredis)
        """
        self.redis = redis_client

    async def increment(self, key: str, window: int) -> int:
        """Increment with automatic expiry"""
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        results = await pipe.execute()
        return results[0]

    async def reset(self, key: str):
        await self.redis.delete(key)

    async def get(self, key: str) -> int:
        count = await self.redis.get(key)
        return int(count) if count else 0


class RateLimitStrategy:
    """Rate limiting strategy"""

    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        backend: Optional[RateLimitBackend] = None
    ):
        """
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            backend: Storage backend (defaults to in-memory)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.backend = backend or InMemoryBackend()

    async def is_allowed(self, key: str) -> tuple[bool, Dict]:
        """
        Check if request is allowed

        Returns:
            (allowed: bool, info: dict)
        """
        count = await self.backend.increment(key, self.window_seconds)

        allowed = count <= self.max_requests
        remaining = max(0, self.max_requests - count)

        info = {
            "limit": self.max_requests,
            "remaining": remaining,
            "reset": time.time() + self.window_seconds
        }

        return allowed, info


class RateLimiter:
    """
    FluxGraph Rate Limiter

    Usage:
        # Create limiter
        limiter = RateLimiter(
            default_limit=100,
            window_seconds=60
        )

        # Apply to FastAPI app
        app.middleware("http")(limiter.middleware)

        # Or use as dependency
        @app.get("/api/endpoint")
        async def endpoint(request: Request, _: None = Depends(limiter.check)):
            return {"message": "success"}
    """

    def __init__(
        self,
        default_limit: int = 100,
        window_seconds: int = 60,
        per_user: bool = True,
        per_ip: bool = False,
        backend: Optional[RateLimitBackend] = None,
        key_func: Optional[Callable] = None,
        exempt_ips: Optional[list] = None
    ):
        """
        Args:
            default_limit: Default requests per window
            window_seconds: Time window in seconds
            per_user: Rate limit per user ID (from auth)
            per_ip: Rate limit per IP address
            backend: Storage backend
            key_func: Custom function to generate rate limit key
            exempt_ips: List of IPs to exempt from rate limiting
        """
        self.strategy = RateLimitStrategy(default_limit, window_seconds, backend)
        self.per_user = per_user
        self.per_ip = per_ip
        self.key_func = key_func
        self.exempt_ips = set(exempt_ips or [])

        # Endpoint-specific limits
        self._endpoint_limits: Dict[str, RateLimitStrategy] = {}

    def set_endpoint_limit(
        self,
        path: str,
        max_requests: int,
        window_seconds: Optional[int] = None
    ):
        """
        Set custom rate limit for specific endpoint

        Args:
            path: Endpoint path (e.g., "/api/chat")
            max_requests: Max requests for this endpoint
            window_seconds: Optional custom window
        """
        window = window_seconds or self.strategy.window_seconds
        self._endpoint_limits[path] = RateLimitStrategy(
            max_requests,
            window,
            self.strategy.backend
        )

    async def get_key(self, request: Request) -> str:
        """Generate rate limit key from request"""
        if self.key_func:
            return await self.key_func(request)

        parts = []

        # Add user ID if available
        if self.per_user and hasattr(request.state, "user_id"):
            parts.append(f"user:{request.state.user_id}")

        # Add IP address
        if self.per_ip or not parts:
            client_ip = request.client.host if request.client else "unknown"
            parts.append(f"ip:{client_ip}")

        # Add endpoint path
        parts.append(f"path:{request.url.path}")

        return ":".join(parts)

    async def check(self, request: Request):
        """
        Check rate limit for request

        Raises:
            HTTPException: If rate limit exceeded
        """
        # Check if IP is exempt
        if request.client and request.client.host in self.exempt_ips:
            return

        # Get appropriate strategy
        strategy = self._endpoint_limits.get(
            request.url.path,
            self.strategy
        )

        # Generate key
        key = await self.get_key(request)

        # Check limit
        allowed, info = await strategy.is_allowed(key)

        # Add headers
        request.state.rate_limit_info = info

        if not allowed:
            logger.warning(f"⚠️ Rate limit exceeded: {key}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": info["limit"],
                    "reset_at": info["reset"]
                },
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(info["reset"])),
                    "Retry-After": str(int(info["reset"] - time.time()))
                }
            )

    async def middleware(self, request: Request, call_next):
        """
        FastAPI middleware for automatic rate limiting

        Usage:
            app.middleware("http")(limiter.middleware)
        """
        try:
            await self.check(request)
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content=e.detail,
                headers=e.headers
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        if hasattr(request.state, "rate_limit_info"):
            info = request.state.rate_limit_info
            response.headers["X-RateLimit-Limit"] = str(info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(int(info["reset"]))

        return response


# Convenience function
def create_rate_limiter(
    requests_per_minute: int = 60,
    per_user: bool = True,
    redis_url: Optional[str] = None
) -> RateLimiter:
    """
    Create a rate limiter with common settings

    Args:
        requests_per_minute: Requests allowed per minute
        per_user: Rate limit per user (True) or globally (False)
        redis_url: Optional Redis URL for distributed limiting

    Example:
        limiter = create_rate_limiter(requests_per_minute=100)
        app.middleware("http")(limiter.middleware)
    """
    backend = None
    if redis_url:
        try:
            import redis.asyncio as aioredis
            backend = RedisBackend(aioredis.from_url(redis_url))
        except ImportError:
            logger.warning("Redis not available, using in-memory backend")

    return RateLimiter(
        default_limit=requests_per_minute,
        window_seconds=60,
        per_user=per_user,
        backend=backend
    )
