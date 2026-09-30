"""Tests for the managed-mode request guard."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from socketio import ASGIApp
from starlette.websockets import WebSocketDisconnect

from AutoGLM_GUI.managed_guard import ManagedGuard, is_blocked

pytestmark = pytest.mark.unit

TOKEN = "gateway-token-" + "x" * 32
AUTH = {"X-Runtime-Auth": TOKEN}


@pytest.fixture
def managed_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AUTOGLM_MANAGED_MODE", "1")
    monkeypatch.setenv("AUTOGLM_DEVICE_REMOTE_URL", "http://agent:8001")
    monkeypatch.setenv("AUTOGLM_INTERNAL_TOKEN", TOKEN)
    monkeypatch.setenv("AUTOGLM_BASE_URL", "http://gateway:4000/v1")
    monkeypatch.setenv("AUTOGLM_MODEL_NAME", "phone-executor")
    monkeypatch.setenv("AUTOGLM_API_KEY", "sk-user-secret-key")


@pytest.fixture
def client(managed_env: None) -> Iterator[TestClient]:
    from AutoGLM_GUI.api import create_app
    from AutoGLM_GUI.socketio_server import sio

    app = ManagedGuard(
        ASGIApp(other_asgi_app=create_app(), socketio_server=sio), token=TOKEN
    )
    # No lifespan: the guard answers before the app is reached.
    yield TestClient(app)


def test_health_is_open_for_readiness_probe(client: TestClient) -> None:
    assert client.get("/api/health").status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-Runtime-Auth": "wrong"}])
def test_requests_without_valid_token_are_rejected(
    client: TestClient, headers: dict[str, str]
) -> None:
    resp = client.get("/api/devices", headers=headers)
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Missing or invalid runtime auth"}


def test_request_with_token_reaches_app(client: TestClient) -> None:
    resp = client.get("/api/devices", headers=AUTH)
    assert resp.status_code == 200
    assert "devices" in resp.json()


def test_socketio_requires_token(client: TestClient) -> None:
    path = "/socket.io/?EIO=4&transport=polling"
    assert client.get(path).status_code == 401
    assert client.get(path, headers=AUTH).status_code == 200


@pytest.mark.parametrize(
    "method, path",
    [
        ("POST", "/api/devices/connect_wifi"),
        ("POST", "/api/devices/connect_wifi_manual"),
        ("POST", "/api/devices/disconnect_wifi"),
        ("POST", "/api/devices/pair_wifi"),
        ("GET", "/api/devices/discover_mdns"),
        ("POST", "/api/devices/discover_remote"),
        ("POST", "/api/devices/add_remote"),
        ("POST", "/api/devices/remove_remote"),
        ("POST", "/api/devices/qr_pair/generate"),
        ("GET", "/api/devices/qr_pair/abc"),
        ("POST", "/api/config"),
        ("DELETE", "/api/config"),
        ("POST", "/api/config/model-connection-check"),
        ("POST", "/api/terminal/sessions"),
        ("GET", "/api/terminal/sessions/abc"),
    ],
)
def test_dangerous_endpoints_blocked_even_with_token(
    client: TestClient, method: str, path: str
) -> None:
    resp = client.request(method, path, headers=AUTH, json={})
    assert resp.status_code == 403
    assert resp.json() == {"detail": "Not available on a managed runtime"}


def test_blocked_list_leaves_normal_endpoints_alone() -> None:
    for method, path in [
        ("GET", "/api/devices"),
        ("GET", "/api/config"),
        ("POST", "/api/chat"),
        ("POST", "/api/control/tap"),
        ("GET", "/api/devices/abc/name"),
    ]:
        assert not is_blocked(method, path), (method, path)


def test_config_hides_platform_keys(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from AutoGLM_GUI.config_manager import config_manager

    # The CLI loads env config at startup; do the same here, restoring it afterwards.
    monkeypatch.setattr(config_manager, "_env_layer", config_manager._env_layer)
    monkeypatch.setattr(config_manager, "_effective_config", None)
    config_manager.load_env_config()

    resp = client.get("/api/config", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["api_key"] == "********"
    assert "sk-user-secret-key" not in resp.text
    assert body["base_url"] == "http://gateway:4000/v1"


def test_websocket_without_token_is_rejected(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/terminal/sessions/x/stream"):
            pass


def test_terminal_feature_is_off_in_managed_mode(
    managed_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from AutoGLM_GUI.api import terminal

    monkeypatch.setenv("AUTOGLM_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("AUTOGLM_ENABLE_WEB_TERMINAL", "1")
    assert terminal._is_terminal_feature_enabled() is False


def test_guard_requires_token() -> None:
    with pytest.raises(ValueError):
        ManagedGuard(lambda scope, receive, send: None, token="")  # type: ignore[arg-type,return-value]


def test_server_app_is_guarded_only_in_managed_mode(
    monkeypatch: pytest.MonkeyPatch, managed_env: None
) -> None:
    import AutoGLM_GUI.server as server

    try:
        assert isinstance(importlib.reload(server).app, ManagedGuard)
        monkeypatch.delenv("AUTOGLM_MANAGED_MODE")
        assert not isinstance(importlib.reload(server).app, ManagedGuard)
    finally:
        monkeypatch.delenv("AUTOGLM_MANAGED_MODE", raising=False)
        importlib.reload(server)
