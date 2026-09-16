"""Authentication and bounded HTTP resource use for private services."""

import asyncio
import hmac
import os
import time
from pathlib import Path

import anyio
from fastapi import HTTPException, Request
from starlette.responses import JSONResponse


def secret(name: str) -> str:
    filename = os.getenv(f"{name}_FILE")
    value = Path(filename).read_text().strip() if filename else os.getenv(name, "")
    if len(value) < 32:
        raise RuntimeError(f"{name} must contain at least 32 characters")
    return value


def authorize(request: Request, name: str) -> None:
    expected = request.app.state.secrets.get(name)
    supplied = request.headers.get("authorization", "")
    if not expected or not hmac.compare_digest(supplied, f"Bearer {expected}"):
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})


class ResourceLimits:
    """Per-process global limits; deliberately independent of spoofable forwarded IPs."""

    def __init__(self, app, max_body=16384, concurrency=32, requests_per_minute=1200):
        self.app = app
        self.max_body = max_body
        self.semaphore = asyncio.Semaphore(concurrency)
        self.rate = requests_per_minute
        self.tokens = float(requests_per_minute)
        self.updated = time.monotonic()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"] in ("/health", "/ready", "/metrics", "/metrics/"):
            return await self.app(scope, receive, send)
        now = time.monotonic()
        self.tokens = min(self.rate, self.tokens + (now - self.updated) * self.rate / 60)
        self.updated = now
        if self.tokens < 1 or self.semaphore.locked():
            return await JSONResponse(
                {"detail": "Over capacity"}, 429, headers={"Retry-After": "1"}
            )(scope, receive, send)
        self.tokens -= 1
        async with self.semaphore:
            chunks = []
            size = 0
            try:
                with anyio.fail_after(10):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        body = message.get("body", b"")
                        size += len(body)
                        if size > self.max_body:
                            return await JSONResponse({"detail": "Body too large"}, 413)(
                                scope, receive, send
                            )
                        chunks.append(body)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                return await JSONResponse({"detail": "Request timeout"}, 408)(scope, receive, send)
            delivered = False

            async def buffered_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
                return await receive()

            await self.app(scope, buffered_receive, send)
