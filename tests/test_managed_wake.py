"""Tests for managed-mode wake-on-demand and activity heartbeats."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from AutoGLM_GUI import managed
from AutoGLM_GUI.managed import (
    ControlPlaneClient,
    ManagedRuntime,
    ManagedSettings,
    ManagedWakeError,
    ensure_device_awake,
    get_managed_runtime,
    set_managed_runtime,
    start_managed_runtime,
)

pytestmark = pytest.mark.unit

TOKEN = "runtime-token-" + "x" * 32
SETTINGS = ManagedSettings(
    enabled=True,
    device_remote_url="http://agent:8001",
    device_remote_id="phone-1",
    control_plane_url="http://control-plane:8000",
    internal_token=TOKEN,
)


class _ControlPlane:
    """Records requests; answers wake and activity."""

    def __init__(self, wake_status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.wake_status = wake_status

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/internal/runtime/wake":
            return httpx.Response(self.wake_status, json={"ok": self.wake_status < 400})
        return httpx.Response(204)

    def client(self) -> ControlPlaneClient:
        return ControlPlaneClient(
            "http://control-plane:8000/",
            TOKEN,
            transport=httpx.MockTransport(self.handler),
        )

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


class _DeviceManager:
    def __init__(self, bind_ok: bool = True) -> None:
        self.bind_ok = bind_ok
        self.binds = 0

    def add_remote_device(self, base_url: str, device_id: str) -> tuple[bool, str, str]:
        self.binds += 1
        if not self.bind_ok:
            return False, "Connection failed", ""
        return False, f"Remote device {device_id} already exists", ""

    def force_refresh(self) -> None:
        pass


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _runtime(
    cp: _ControlPlane,
    devices: _DeviceManager | None = None,
    clock: _Clock | None = None,
) -> ManagedRuntime:
    return ManagedRuntime(
        SETTINGS,
        devices or _DeviceManager(),  # type: ignore[arg-type]
        cp.client(),
        rebind_attempts=2,
        rebind_delay=0,
        clock=clock or _Clock(),
    )


@pytest.fixture(autouse=True)
def _reset_runtime() -> Any:
    yield
    set_managed_runtime(None)


# ------------------------------------------------------------------- client


def test_wake_posts_with_runtime_token() -> None:
    cp = _ControlPlane()
    asyncio.run(cp.client().wake())
    request = cp.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://control-plane:8000/internal/runtime/wake"
    assert request.headers["authorization"] == f"Bearer {TOKEN}"


def test_wake_raises_on_error_status() -> None:
    cp = _ControlPlane(wake_status=503)
    with pytest.raises(ManagedWakeError, match="HTTP 503"):
        asyncio.run(cp.client().wake())


def test_wake_raises_when_unreachable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = ControlPlaneClient(
        "http://control-plane:8000", TOKEN, transport=httpx.MockTransport(refuse)
    )
    with pytest.raises(ManagedWakeError, match="unreachable"):
        asyncio.run(client.wake())


def test_activity_report_payload_and_errors_are_swallowed() -> None:
    cp = _ControlPlane()
    asyncio.run(cp.client().report_activity(busy=True, viewers=2))
    assert cp.paths() == ["/internal/runtime/activity"]
    assert json.loads(cp.requests[0].content) == {"busy": True, "viewers": 2}

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    client = ControlPlaneClient(
        "http://control-plane:8000", TOKEN, transport=httpx.MockTransport(boom)
    )
    asyncio.run(client.report_activity(busy=False, viewers=0))


# ------------------------------------------------------------------ runtime


def test_ensure_awake_wakes_then_rebinds() -> None:
    cp = _ControlPlane()
    devices = _DeviceManager()
    asyncio.run(_runtime(cp, devices).ensure_device_awake())
    assert cp.paths() == ["/internal/runtime/wake"]
    assert devices.binds == 1


def test_ensure_awake_is_debounced() -> None:
    cp = _ControlPlane()
    clock = _Clock()
    runtime = _runtime(cp, clock=clock)

    async def scenario() -> None:
        await runtime.ensure_device_awake()
        clock.now += 10
        await runtime.ensure_device_awake()
        clock.now += 30
        await runtime.ensure_device_awake()

    asyncio.run(scenario())
    assert cp.paths().count("/internal/runtime/wake") == 2


def test_concurrent_callers_share_one_wake() -> None:
    cp = _ControlPlane()
    runtime = _runtime(cp)

    async def scenario() -> None:
        await asyncio.gather(*(runtime.ensure_device_awake() for _ in range(5)))

    asyncio.run(scenario())
    assert cp.paths().count("/internal/runtime/wake") == 1


def test_rebind_failure_raises_and_is_not_debounced() -> None:
    cp = _ControlPlane()
    devices = _DeviceManager(bind_ok=False)
    runtime = _runtime(cp, devices)

    with pytest.raises(ManagedWakeError, match="reconnect"):
        asyncio.run(runtime.ensure_device_awake())
    assert not runtime.woke_within(1000)


def test_heartbeat_reports_busy_and_viewers() -> None:
    cp = _ControlPlane()
    clock = _Clock()
    runtime = _runtime(cp, clock=clock)

    async def idle() -> bool:
        return False

    async def busy() -> bool:
        return True

    async def scenario() -> None:
        await runtime.heartbeat_once(idle, lambda: 0)
        await runtime.heartbeat_once(busy, lambda: 0)
        await runtime.heartbeat_once(idle, lambda: 3)
        await runtime.ensure_device_awake()  # a recent wake counts as busy
        await runtime.heartbeat_once(idle, lambda: 0)
        clock.now += 60
        await runtime.heartbeat_once(idle, lambda: 0)

    asyncio.run(scenario())
    reports = [
        json.loads(r.content)
        for r in cp.requests
        if r.url.path == "/internal/runtime/activity"
    ]
    assert reports == [
        {"busy": False, "viewers": 0},
        {"busy": True, "viewers": 0},
        {"busy": False, "viewers": 3},
        {"busy": True, "viewers": 0},
        {"busy": False, "viewers": 0},
    ]


# ------------------------------------------------------------- module hooks


def test_hook_is_noop_without_managed_runtime() -> None:
    set_managed_runtime(None)
    asyncio.run(ensure_device_awake())


def test_hook_logs_instead_of_raising() -> None:
    cp = _ControlPlane(wake_status=500)
    set_managed_runtime(_runtime(cp))
    asyncio.run(ensure_device_awake())
    assert cp.paths() == ["/internal/runtime/wake"]


def test_start_managed_runtime_needs_control_plane_url() -> None:
    no_url = ManagedSettings(
        enabled=True, device_remote_url="http://agent:8001", internal_token=TOKEN
    )
    assert start_managed_runtime(no_url, _DeviceManager(), "adb") is None  # type: ignore[arg-type]
    assert get_managed_runtime() is None

    runtime = start_managed_runtime(SETTINGS, _DeviceManager(), "adb")  # type: ignore[arg-type]
    assert runtime is not None
    assert get_managed_runtime() is runtime
    asyncio.run(runtime.client.aclose())


def test_acquire_device_wakes_before_acquiring(monkeypatch: pytest.MonkeyPatch) -> None:
    from AutoGLM_GUI import phone_agent_manager as pam
    from AutoGLM_GUI.exceptions import AgentNotInitializedError

    calls: list[str] = []

    async def fake_wake() -> None:
        calls.append("wake")

    monkeypatch.setattr(pam, "ensure_device_awake", fake_wake)
    manager = pam.PhoneAgentManager()

    with pytest.raises(AgentNotInitializedError):
        asyncio.run(manager.acquire_device_async("never-initialized"))
    assert calls == ["wake"]


def test_scheduler_wakes_before_checking_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from AutoGLM_GUI.scheduler_manager import SchedulerManager

    calls: list[str] = []

    async def fake_wake() -> None:
        calls.append("wake")

    monkeypatch.setattr(managed, "ensure_device_awake", fake_wake)

    class NoDevices:
        def get_devices(self) -> list[Any]:
            calls.append("get_devices")
            return []

    result = asyncio.run(
        SchedulerManager()._execute_single_device(
            "serial-1", {}, "task", object(), NoDevices(), object()
        )
    )
    assert result.message == "Device offline"
    assert calls == ["wake", "get_devices"]


def test_stream_connect_wakes_before_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    from AutoGLM_GUI import socketio_server

    calls: list[str] = []

    async def fake_wake() -> None:
        calls.append("wake")

    class _Stop(Exception):
        pass

    async def stop_here(sid: str) -> None:
        calls.append("stop_existing")
        raise _Stop

    monkeypatch.setattr(socketio_server, "ensure_device_awake", fake_wake)
    monkeypatch.setattr(socketio_server, "_stop_stream_for_sid", stop_here)

    with pytest.raises(_Stop):
        asyncio.run(socketio_server.connect_device("sid-1", {"device_id": "d1"}))
    assert calls == ["wake", "stop_existing"]


def test_active_stream_count(monkeypatch: pytest.MonkeyPatch) -> None:
    from AutoGLM_GUI import socketio_server

    monkeypatch.setattr(
        socketio_server, "_socket_streamers", {"a": object(), "b": object()}
    )
    assert socketio_server.active_stream_count() == 2
