"""Request guard for managed mode.

In managed mode the runtime sits behind the platform gateway, which adds the
runtime's internal token as ``X-Runtime-Auth`` to every request it forwards
(the token never reaches the browser). This ASGI middleware wraps the whole
app, Socket.IO included, and:

- rejects requests without the token (``/api/health`` stays open for the
  control plane's readiness probe), and
- blocks endpoints that must not be reachable on a hosted runtime: connecting
  or pairing arbitrary devices and probing arbitrary URLs (these would let a
  user reach other hosts from inside the platform network), changing the
  platform-managed model configuration, and the host shell terminal.
"""

from __future__ import annotations

import hmac
import json
import re

from starlette.types import ASGIApp, Receive, Scope, Send

RUNTIME_AUTH_HEADER = b"x-runtime-auth"

_OPEN_PATHS = {"/api/health"}

# (HTTP method or None for any, path regex)
_BLOCKED: tuple[tuple[str | None, re.Pattern[str]], ...] = (
    (None, re.compile(r"^/api/terminal(/|$)")),
    (None, re.compile(r"^/api/devices/qr_pair(/|$)")),
    ("POST", re.compile(r"^/api/devices/connect_wifi(_manual)?$")),
    ("POST", re.compile(r"^/api/devices/disconnect_wifi$")),
    ("POST", re.compile(r"^/api/devices/pair_wifi$")),
    ("GET", re.compile(r"^/api/devices/discover_mdns$")),
    ("POST", re.compile(r"^/api/devices/discover_remote$")),
    ("POST", re.compile(r"^/api/devices/(add|remove)_remote$")),
    ("POST", re.compile(r"^/api/config$")),
    ("DELETE", re.compile(r"^/api/config$")),
    ("POST", re.compile(r"^/api/config/model-connection-check$")),
)


def is_blocked(method: str, path: str) -> bool:
    return any(
        (blocked_method is None or blocked_method == method) and pattern.match(path)
        for blocked_method, pattern in _BLOCKED
    )


class ManagedGuard:
    def __init__(self, app: ASGIApp, token: str) -> None:
        if not token:
            raise ValueError("ManagedGuard needs a non-empty token")
        self._app = app
        self._token = token.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        method: str = scope.get("method", "GET")

        if not (scope["type"] == "http" and path in _OPEN_PATHS):
            supplied = dict(scope.get("headers") or []).get(RUNTIME_AUTH_HEADER, b"")
            if not hmac.compare_digest(supplied, self._token):
                await _deny(
                    scope, receive, send, 401, "Missing or invalid runtime auth"
                )
                return

        if is_blocked(method, path):
            await _deny(scope, receive, send, 403, "Not available on a managed runtime")
            return

        await self._app(scope, receive, send)


async def _deny(
    scope: Scope, receive: Receive, send: Send, status: int, detail: str
) -> None:
    if scope["type"] == "websocket":
        # Closing before accept makes the server answer the upgrade with 403.
        await receive()
        await send({"type": "websocket.close", "code": 4000 + status})
        return
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
