"""Managed mode: finished tasks are reported so the control plane can notify the user."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import AutoGLM_GUI.task_store as task_store_module
from AutoGLM_GUI import task_manager as task_manager_module
from AutoGLM_GUI.managed import (
    ApprovalClient,
    ManagedRuntime,
    ManagedSettings,
    report_task_finished,
    set_managed_runtime,
)

pytestmark = pytest.mark.unit

TOKEN = "runtime-token-" + "x" * 32
SETTINGS = ManagedSettings(
    enabled=True,
    device_remote_url="http://agent:8001",
    control_plane_url="http://control-plane:8000",
    internal_token=TOKEN,
)


class _ControlPlane:
    def __init__(self, fail: bool = False) -> None:
        self.events: list[dict[str, Any]] = []
        self.fail = fail

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        if self.fail:
            raise httpx.ConnectError("control plane down", request=request)
        if request.method == "POST" and request.url.path == "/internal/runtime/events":
            self.events.append(json.loads(request.content))
            return httpx.Response(204)
        return httpx.Response(404)


def _install(cp: _ControlPlane) -> None:
    client = ApprovalClient(
        "http://control-plane:8000/", TOKEN, transport=httpx.MockTransport(cp.handler)
    )
    set_managed_runtime(
        ManagedRuntime(
            SETTINGS,
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            approvals=client,
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


def _task(store: task_store_module.TaskStore, source: str = "scheduled") -> str:
    task = store.create_task_run(
        source=source,
        executor_key="scheduled_workflow",
        device_id="phone-1",
        device_serial="phone-1",
        input_text="  每天早上\n查一下快递  " + "很长" * 50,
    )
    return str(task["id"])


def test_reports_the_finished_task(store: task_store_module.TaskStore) -> None:
    cp = _ControlPlane()
    _install(cp)
    task_id = _task(store)

    report_task_finished(
        task_id,
        status="SUCCEEDED",
        message="已查到 3 个快递",
        stop_reason="completed",
        duration_ms=12_345,
    )

    [event] = cp.events
    assert event["kind"] == "task_finished"
    assert event["message"] == "已查到 3 个快递"
    assert event["status"] == "SUCCEEDED"
    assert event["source"] == "scheduled"
    assert event["duration_seconds"] == 12.3
    assert event["device_id"] == "phone-1"
    assert event["context"] == f"task:{task_id}"
    assert event["title"].startswith("每天早上 查一下快递 很长")
    assert len(event["title"]) == 60


@pytest.mark.parametrize("stop_reason", ["takeover", "user_stopped"])
def test_known_stops_are_not_reported(
    store: task_store_module.TaskStore, stop_reason: str
) -> None:
    cp = _ControlPlane()
    _install(cp)
    report_task_finished(
        _task(store),
        status="CANCELLED",
        message="",
        stop_reason=stop_reason,
        duration_ms=1,
    )
    assert cp.events == []


def test_long_messages_are_cut(store: task_store_module.TaskStore) -> None:
    cp = _ControlPlane()
    _install(cp)
    report_task_finished(
        _task(store),
        status="FAILED",
        message="x" * 5000,
        stop_reason="error",
        duration_ms=1,
    )
    assert len(cp.events[0]["message"]) == 2000


def test_nothing_happens_outside_managed_mode_or_for_unknown_tasks(
    store: task_store_module.TaskStore,
) -> None:
    # No managed runtime installed: no request at all.
    report_task_finished(
        _task(store), status="SUCCEEDED", message="", stop_reason=None, duration_ms=1
    )
    cp = _ControlPlane()
    _install(cp)
    report_task_finished(
        "missing", status="SUCCEEDED", message="", stop_reason=None, duration_ms=1
    )
    assert cp.events == []


def test_control_plane_errors_are_swallowed(store: task_store_module.TaskStore) -> None:
    _install(_ControlPlane(fail=True))
    report_task_finished(
        _task(store), status="SUCCEEDED", message="ok", stop_reason=None, duration_ms=1
    )


def test_task_manager_reports_in_the_background(
    store: task_store_module.TaskStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_report(task_id: str, **kwargs: Any) -> None:
        calls.append({"task_id": task_id, **kwargs})
        raise RuntimeError("must not escape")

    monkeypatch.setattr(task_manager_module, "is_managed_mode", lambda: True)
    monkeypatch.setattr(task_manager_module, "report_task_finished", fake_report)

    async def run() -> None:
        manager = task_manager_module.TaskManager.__new__(
            task_manager_module.TaskManager
        )
        manager._reports = set()
        manager._report_finished("t1", "SUCCEEDED", "done", "completed", 1500)
        await asyncio.gather(*manager._reports)

    asyncio.run(run())
    assert calls == [
        {
            "task_id": "t1",
            "status": "SUCCEEDED",
            "message": "done",
            "stop_reason": "completed",
            "duration_ms": 1500,
        }
    ]
