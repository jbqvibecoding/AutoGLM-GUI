"""Tests for managed-mode approvals and takeover notices."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import AutoGLM_GUI.task_store as task_store_module
from AutoGLM_GUI.actions.async_handler import AsyncActionHandler
from AutoGLM_GUI.managed import (
    ApprovalClient,
    ManagedRuntime,
    ManagedSettings,
    managed_confirmation,
    managed_takeover,
    set_managed_runtime,
)
from AutoGLM_GUI.trace import trace_context

pytestmark = pytest.mark.unit

TOKEN = "runtime-token-" + "x" * 32
SETTINGS = ManagedSettings(
    enabled=True,
    device_remote_url="http://agent:8001",
    control_plane_url="http://control-plane:8000",
    internal_token=TOKEN,
)


class _ControlPlane:
    """Fake approval API. ``decisions`` are handed out one per poll."""

    def __init__(
        self,
        decisions: list[str] | None = None,
        *,
        create_status: int = 201,
        expires_in: float = 600,
        poll_errors: int = 0,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.decisions = list(decisions or [])
        self.create_status = create_status
        self.expires_in = expires_in
        self.poll_errors = poll_errors

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/internal/runtime/approvals":
            if self.create_status != 201:
                return httpx.Response(self.create_status)
            return httpx.Response(
                201,
                json={"id": "a1", "status": "pending", "expires_in": self.expires_in},
            )
        if request.method == "GET" and path == "/internal/runtime/approvals/a1":
            if self.poll_errors:
                self.poll_errors -= 1
                raise httpx.ConnectError("control plane restarting", request=request)
            status = self.decisions.pop(0) if self.decisions else "pending"
            return httpx.Response(200, json={"id": "a1", "status": status})
        if request.method == "POST" and path == "/internal/runtime/events":
            return httpx.Response(204)
        return httpx.Response(404)

    def client(self, **kwargs: Any) -> ApprovalClient:
        return ApprovalClient(
            "http://control-plane:8000/",
            TOKEN,
            transport=httpx.MockTransport(self.handler),
            sleep=lambda _: None,
            **kwargs,
        )

    def paths(self) -> list[str]:
        return [f"{r.method} {r.url.path}" for r in self.requests]


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 10.0
        return self.now


def _install(cp: _ControlPlane) -> None:
    set_managed_runtime(
        ManagedRuntime(
            SETTINGS,
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            approvals=cp.client(),
        )
    )


@pytest.fixture(autouse=True)
def _reset_runtime() -> Any:
    yield
    set_managed_runtime(None)


@pytest.fixture
def store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> task_store_module.TaskStore:
    isolated = task_store_module.TaskStore(tmp_path / "tasks.db")
    monkeypatch.setattr(task_store_module, "task_store", isolated)
    return isolated


# ------------------------------------------------------------------- client


def test_request_sends_context_and_waits_for_decision() -> None:
    cp = _ControlPlane(["pending", "approved"])
    created: list[str] = []

    status = cp.client().request(
        "确认支付", device_id="phone-1", context="chat:s1", on_created=created.append
    )

    assert status == "approved"
    assert created == ["a1"]
    create = cp.requests[0]
    assert create.headers["authorization"] == f"Bearer {TOKEN}"
    assert json.loads(create.content) == {
        "message": "确认支付",
        "device_id": "phone-1",
        "context": "chat:s1",
    }
    polls = [r for r in cp.requests if r.method == "GET"]
    assert len(polls) == 2
    assert polls[0].url.params["wait"] == "25.0"


@pytest.mark.parametrize("decision", ["denied", "expired"])
def test_request_returns_refusals(decision: str) -> None:
    assert cp_request(_ControlPlane([decision])) == decision


def cp_request(cp: _ControlPlane, **kwargs: Any) -> str:
    return cp.client(**kwargs).request("确认支付", device_id="d", context="c")


def test_rejected_or_unreachable_create_is_an_error() -> None:
    assert cp_request(_ControlPlane(create_status=401)) == "error"

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = ApprovalClient(
        "http://control-plane:8000", TOKEN, transport=httpx.MockTransport(refuse)
    )
    assert client.request("x", device_id="d", context="c") == "error"


def test_poll_retries_transient_errors() -> None:
    cp = _ControlPlane(["approved"], poll_errors=2)
    assert cp_request(cp) == "approved"
    assert cp.paths().count("GET /internal/runtime/approvals/a1") == 3


def test_runtime_stops_asking_after_expiry() -> None:
    # Never decided: the runtime gives up once expiry plus grace has passed.
    cp = _ControlPlane(expires_in=15)
    assert cp_request(cp, clock=_Clock()) == "expired"


def test_report_event_is_best_effort() -> None:
    cp = _ControlPlane()
    cp.client().report_event("takeover", "请登录", device_id="d", context="scheduled")
    assert json.loads(cp.requests[0].content) == {
        "kind": "takeover",
        "message": "请登录",
        "device_id": "d",
        "context": "scheduled",
    }

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    ApprovalClient(
        "http://control-plane:8000", TOKEN, transport=httpx.MockTransport(boom)
    ).report_event("takeover", "x", device_id="d", context="c")


# ---------------------------------------------------------------- callbacks


def test_confirmation_approves_only_on_approval() -> None:
    _install(_ControlPlane(["approved"]))
    assert managed_confirmation("d", "chat:s1")("确认支付") is True

    _install(_ControlPlane(["denied"]))
    assert managed_confirmation("d", "chat:s1")("确认支付") is False

    _install(_ControlPlane(create_status=500))
    assert managed_confirmation("d", "chat:s1")("确认支付") is False


def test_confirmation_without_control_plane_denies() -> None:
    set_managed_runtime(None)
    assert managed_confirmation("d", "chat:s1")("确认支付") is False


def test_takeover_reports_event() -> None:
    cp = _ControlPlane()
    _install(cp)
    managed_takeover("d", "scheduled")("请完成验证码")
    assert cp.paths() == ["POST /internal/runtime/events"]

    set_managed_runtime(None)
    managed_takeover("d", "scheduled")("no control plane: only logged")


def test_confirmation_records_events_on_the_running_task(
    store: task_store_module.TaskStore,
) -> None:
    task = store.create_task_run(
        source="chat",
        executor_key="classic_chat",
        device_id="d",
        device_serial="d",
        input_text="买一杯咖啡",
        trace_id="trace-1",
    )
    _install(_ControlPlane(["approved"]))

    with trace_context("trace-1"):
        assert managed_confirmation("d", "chat:s1")("确认支付")

    events = [
        (e["event_type"], e["payload"])
        for e in store.list_task_events(str(task["id"]))
        if e["event_type"].startswith("approval_")
    ]
    assert events == [
        ("approval_required", {"approval_id": "a1", "message": "确认支付"}),
        ("approval_resolved", {"approval_id": "a1", "status": "approved"}),
    ]


def test_confirmation_without_task_still_works(
    store: task_store_module.TaskStore,
) -> None:
    _install(_ControlPlane(["approved"]))
    with trace_context("unknown-trace"):
        assert managed_confirmation("d", "mcp")("确认支付")


def test_sensitive_tap_is_blocked_when_denied() -> None:
    _install(_ControlPlane(["denied"]))

    class Device:
        def __init__(self) -> None:
            self.taps: list[tuple[int, int]] = []

        async def tap(self, x: int, y: int, delay: float | None = None) -> None:
            self.taps.append((x, y))

    device = Device()
    handler = AsyncActionHandler(
        device,  # type: ignore[arg-type]
        confirmation_callback=managed_confirmation("d", "chat:s1"),
        takeover_callback=managed_takeover("d", "chat:s1"),
    )
    result = asyncio.run(
        handler.execute(
            {
                "_metadata": "do",
                "action": "Tap",
                "element": [500, 500],
                "message": "确认支付",
            },
            1080,
            2400,
        )
    )
    assert result.should_finish is True
    assert device.taps == []


@pytest.fixture
def captured_init(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub config and agent creation; returns the kwargs agents are created with."""
    from AutoGLM_GUI import phone_agent_manager as pam
    from AutoGLM_GUI.config_manager import config_manager

    captured: dict[str, Any] = {}

    async def fake_init(self: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    config = SimpleNamespace(
        base_url="http://llm:4000/v1",
        api_key="k",
        model_name="m",
        agent_config_params=None,
        agent_type="glm-async",
    )
    monkeypatch.setattr(config_manager, "load_file_config", lambda: None)
    monkeypatch.setattr(config_manager, "sync_to_env", lambda: None)
    monkeypatch.setattr(config_manager, "get_effective_config", lambda: config)
    monkeypatch.setattr(
        pam.PhoneAgentManager, "_initialize_agent_with_factory_unsafe", fake_init
    )
    return captured


def test_agent_manager_uses_managed_callbacks(
    monkeypatch: pytest.MonkeyPatch, captured_init: dict[str, Any]
) -> None:
    from AutoGLM_GUI import phone_agent_manager as pam

    monkeypatch.setenv("AUTOGLM_MANAGED_MODE", "1")
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        pam,
        "managed_confirmation",
        lambda device_id, context: calls.append((device_id, context))
        or (lambda m: False),
    )

    manager = pam.PhoneAgentManager()
    asyncio.run(
        manager._auto_initialize_agent_unsafe("10.0.0.2:5555:chat:s1", "10.0.0.2:5555")
    )
    assert calls == [("10.0.0.2:5555", "chat:s1")]
    assert captured_init["confirmation_callback"]("确认支付") is False


def test_agent_manager_keeps_noop_callbacks_outside_managed_mode(
    monkeypatch: pytest.MonkeyPatch, captured_init: dict[str, Any]
) -> None:
    from AutoGLM_GUI import phone_agent_manager as pam

    monkeypatch.delenv("AUTOGLM_MANAGED_MODE", raising=False)
    asyncio.run(pam.PhoneAgentManager()._auto_initialize_agent_unsafe("d", "d"))
    assert captured_init["confirmation_callback"]("确认支付") is True


def test_malformed_reply_is_denied() -> None:
    def garbage(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, text="not json")

    set_managed_runtime(
        ManagedRuntime(
            SETTINGS,
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            approvals=ApprovalClient(
                "http://control-plane:8000",
                TOKEN,
                transport=httpx.MockTransport(garbage),
            ),
        )
    )
    assert managed_confirmation("d", "chat:s1")("确认支付") is False
