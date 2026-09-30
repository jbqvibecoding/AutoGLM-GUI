"""AutoGLM-GUI Backend API Server (FastAPI + Socket.IO)."""

from socketio import ASGIApp

from AutoGLM_GUI.api import app as fastapi_app
from AutoGLM_GUI.managed import load_managed_settings
from AutoGLM_GUI.managed_guard import ManagedGuard
from AutoGLM_GUI.socketio_server import sio

_asgi_app = ASGIApp(
    other_asgi_app=fastapi_app, socketio_server=sio, socketio_path="/socket.io"
)

# Managed mode: only answer requests the platform gateway forwarded.
_managed_settings = load_managed_settings()
app = (
    ManagedGuard(_asgi_app, token=_managed_settings.internal_token or "")
    if _managed_settings.enabled
    else _asgi_app
)

__all__ = ["app"]
