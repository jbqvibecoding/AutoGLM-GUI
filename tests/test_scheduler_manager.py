"""Unit tests for scheduler task execution semantics."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import AutoGLM_GUI.device_manager as device_manager_module
import AutoGLM_GUI.task_manager as task_manager_module
import AutoGLM_GUI.task_store as task_store_module
import AutoGLM_GUI.workflow_manager as workflow_manager_module
from AutoGLM_GUI.models.scheduled_task import ScheduledTask
from AutoGLM_GUI.scheduler_manager import SchedulerManager
from AutoGLM_GUI.task_store import TaskStatus, TaskStore


def test_scheduler_execution_counts_offline_devices_in_latest_summary(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeWorkflowManager:
        @staticmethod
        def get_workflow(workflow_uuid: str) -> dict[str, str] | None:
            if workflow_uuid != "wf-1":
                return None
            return {"uuid": "wf-1", "name": "Morning", "text": "执行签到"}

    class FakeDeviceManager:
        @staticmethod
        def get_devices() -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    serial="online-1",
                    primary_device_id="device-online-1",
                    state=SimpleNamespace(value="online"),
                )
            ]

    class FakeTaskManager:
        def __init__(self, store: TaskStore) -> None:
            self.store = store

        async def enqueue_scheduled_task(
            self,
            *,
            scheduled_task_id: str,
            workflow_uuid: str,
            device_id: str,
            device_serial: str,
            input_text: str,
            schedule_fire_id: str,
            executor_key: str = "scheduled_workflow",
        ) -> dict[str, object]:
            task = self.store.create_task_run(
                source="scheduled",
                executor_key=executor_key,
                scheduled_task_id=scheduled_task_id,
                workflow_uuid=workflow_uuid,
                schedule_fire_id=schedule_fire_id,
                device_id=device_id,
                device_serial=device_serial,
                input_text=input_text,
            )
            self.store.update_task_terminal(
                task_id=task["id"],
                status=TaskStatus.SUCCEEDED.value,
                final_message="完成",
                error_message=None,
                step_count=1,
            )
            return task

    store = TaskStore(tmp_path / "tasks.db")
    fake_task_manager = FakeTaskManager(store)

    SchedulerManager._instance = None
    manager = SchedulerManager()
    manager._tasks = {
        "scheduled-1": ScheduledTask(
            id="scheduled-1",
            name="Morning",
            workflow_uuid="wf-1",
            device_serialnos=["online-1", "offline-1"],
            cron_expression="0 8 * * *",
            enabled=True,
        )
    }

    monkeypatch.setattr(
        workflow_manager_module, "workflow_manager", FakeWorkflowManager()
    )
    monkeypatch.setattr(
        device_manager_module.DeviceManager,
        "get_instance",
        classmethod(lambda cls: FakeDeviceManager()),
    )
    monkeypatch.setattr(task_manager_module, "task_manager", fake_task_manager)
    monkeypatch.setattr(task_store_module, "task_store", store)

    try:
        asyncio.run(manager._execute_task("scheduled-1"))

        summary = store.get_latest_schedule_summary("scheduled-1")
        tasks, total = store.list_tasks(
            source="scheduled", limit=10, offset=0, device_serial=None
        )
    finally:
        store.close()
        SchedulerManager._instance = None

    assert total == 2
    assert summary is not None
    assert summary["last_run_status"] == "partial"
    assert summary["last_run_success_count"] == 1
    assert summary["last_run_total_count"] == 2
    assert {task["device_serial"] for task in tasks} == {"online-1", "offline-1"}
    offline_task = next(task for task in tasks if task["device_serial"] == "offline-1")
    assert offline_task["status"] == TaskStatus.FAILED.value
    assert offline_task["error_message"] == "Device offline"


def test_scheduler_uses_layered_executor_when_task_mode_is_layered(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeWorkflowManager:
        @staticmethod
        def get_workflow(workflow_uuid: str) -> dict[str, str] | None:
            if workflow_uuid != "wf-1":
                return None
            return {"uuid": "wf-1", "name": "Planner", "text": "执行复杂任务"}

    class FakeDeviceManager:
        @staticmethod
        def get_devices() -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    serial="online-1",
                    primary_device_id="device-online-1",
                    state=SimpleNamespace(value="online"),
                )
            ]

    class FakeTaskManager:
        def __init__(self) -> None:
            self.enqueued: list[dict[str, object]] = []

        async def enqueue_scheduled_task(self, **kwargs) -> dict[str, object]:
            self.enqueued.append(kwargs)
            return {"id": "task-1"}

    store = TaskStore(tmp_path / "tasks.db")
    fake_task_manager = FakeTaskManager()

    SchedulerManager._instance = None
    manager = SchedulerManager()
    manager._tasks = {
        "scheduled-1": ScheduledTask(
            id="scheduled-1",
            name="Planner",
            workflow_uuid="wf-1",
            device_serialnos=["online-1"],
            cron_expression="0 8 * * *",
            enabled=True,
            execution_mode="layered",
        )
    }

    monkeypatch.setattr(
        workflow_manager_module, "workflow_manager", FakeWorkflowManager()
    )
    monkeypatch.setattr(
        device_manager_module.DeviceManager,
        "get_instance",
        classmethod(lambda cls: FakeDeviceManager()),
    )
    monkeypatch.setattr(task_manager_module, "task_manager", fake_task_manager)
    monkeypatch.setattr(task_store_module, "task_store", store)

    try:
        asyncio.run(manager._execute_task("scheduled-1"))
    finally:
        store.close()
        SchedulerManager._instance = None

    assert len(fake_task_manager.enqueued) == 1
    assert fake_task_manager.enqueued[0]["executor_key"] == "scheduled_layered_workflow"


# ---------------------------------------------------------------- warm-up


def _fresh_manager(tmp_path: Path) -> SchedulerManager:
    SchedulerManager._instance = None
    manager = SchedulerManager()
    manager._tasks_path = tmp_path / "scheduled_tasks.json"
    return manager


def _cron_in(minutes: int) -> str:
    """A daily cron for ``minutes`` from now (local time, like the scheduler)."""
    from datetime import timedelta

    at = datetime.now() + timedelta(minutes=minutes)
    return f"{at.minute} {at.hour} * * *"


class _Waker:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def __call__(self) -> None:
        self.calls += 1
        if self.fail:
            raise RuntimeError("control plane down")


def _with_waker(monkeypatch, waker: _Waker) -> None:
    import AutoGLM_GUI.managed as managed

    monkeypatch.setattr(managed, "ensure_device_awake", waker)


def test_tasks_due_within_the_lead(tmp_path: Path) -> None:
    async def run() -> None:
        manager = _fresh_manager(tmp_path)
        await manager.start(prewarm_seconds=300)
        try:
            soon = manager.create_task("soon", "wf", ["d"], _cron_in(3))
            assert manager.due_within(300)
            assert manager.due_soon()
            manager.set_enabled(soon.id, False)
            assert not manager.due_within(300)
            manager.create_task("later", "wf", ["d"], _cron_in(20))
            assert not manager.due_soon()
            assert manager.due_within(30 * 60)
        finally:
            await manager.shutdown()

    asyncio.run(run())


def test_each_run_is_warmed_up_once(tmp_path: Path, monkeypatch) -> None:
    waker = _Waker()
    _with_waker(monkeypatch, waker)

    async def run() -> None:
        manager = _fresh_manager(tmp_path)
        await manager.start(prewarm_seconds=300)
        try:
            # Nothing due: no wake.
            manager.create_task("later", "wf", ["d"], _cron_in(20))
            assert await manager.prewarm_once() is False
            task = manager.create_task("soon", "wf", ["d"], _cron_in(3))
            assert await manager.prewarm_once() is True
            assert await manager.prewarm_once() is False  # the next sweep
            assert waker.calls == 1
            # The following run (a changed schedule here) is warmed up again.
            manager.update_task(task.id, cron_expression=_cron_in(4))
            assert await manager.prewarm_once() is True
            assert waker.calls == 2
        finally:
            await manager.shutdown()

    asyncio.run(run())


def test_a_task_created_shortly_before_it_is_due_wakes_the_phone_now(
    tmp_path: Path, monkeypatch
) -> None:
    waker = _Waker()
    _with_waker(monkeypatch, waker)

    async def run() -> None:
        manager = _fresh_manager(tmp_path)
        await manager.start(prewarm_seconds=300)
        await asyncio.sleep(0.1)  # the check at start finds nothing due
        assert waker.calls == 0
        try:
            manager.create_task("soon", "wf", ["d"], _cron_in(2))
            for _ in range(100):
                if waker.calls:
                    break
                await asyncio.sleep(0.02)
        finally:
            await manager.shutdown()

    asyncio.run(run())
    assert waker.calls == 1


def test_a_failed_warm_up_is_only_logged(tmp_path: Path, monkeypatch) -> None:
    _with_waker(monkeypatch, _Waker(fail=True))

    async def run() -> bool:
        manager = _fresh_manager(tmp_path)
        await manager.start(prewarm_seconds=300)
        try:
            manager.create_task("soon", "wf", ["d"], _cron_in(3))
            return await manager.prewarm_once()
        finally:
            await manager.shutdown()

    assert asyncio.run(run()) is True


def test_no_warm_up_outside_managed_mode(tmp_path: Path, monkeypatch) -> None:
    from AutoGLM_GUI.scheduler_manager import PREWARM_JOB_ID

    waker = _Waker()
    _with_waker(monkeypatch, waker)
    monkeypatch.delenv("AUTOGLM_MANAGED_MODE", raising=False)

    async def run() -> None:
        manager = _fresh_manager(tmp_path)
        await manager.start()  # reads the (non-managed) settings
        try:
            manager.create_task("soon", "wf", ["d"], _cron_in(2))
            assert manager._scheduler.get_job(PREWARM_JOB_ID) is None
            assert not manager.due_soon()
            assert await manager.prewarm_once() is False
        finally:
            await manager.shutdown()

    asyncio.run(run())
    assert waker.calls == 0


def test_the_warm_up_sweep_runs_in_managed_mode(tmp_path: Path) -> None:
    from AutoGLM_GUI.scheduler_manager import PREWARM_CHECK_SECONDS, PREWARM_JOB_ID

    async def run() -> None:
        manager = _fresh_manager(tmp_path)
        await manager.start(prewarm_seconds=180)
        try:
            job = manager._scheduler.get_job(PREWARM_JOB_ID)
            assert job is not None
            assert job.trigger.interval.total_seconds() == PREWARM_CHECK_SECONDS
        finally:
            await manager.shutdown()

    asyncio.run(run())


def test_prewarm_settings() -> None:
    from AutoGLM_GUI.managed import load_managed_settings

    env = {
        "AUTOGLM_MANAGED_MODE": "1",
        "AUTOGLM_DEVICE_SERIAL": "10.0.0.2:5555",
        "AUTOGLM_INTERNAL_TOKEN": "t" * 40,
    }
    assert load_managed_settings(env).schedule_prewarm_seconds == 180
    off = {**env, "AUTOGLM_SCHEDULE_PREWARM_SECONDS": "0"}
    assert load_managed_settings(off).schedule_prewarm_seconds == 0
    for bad in ("-5", "soon", "nan"):
        with pytest.raises(ValueError, match="AUTOGLM_SCHEDULE_PREWARM_SECONDS"):
            load_managed_settings({**env, "AUTOGLM_SCHEDULE_PREWARM_SECONDS": bad})


def test_a_task_due_soon_keeps_the_managed_phone_awake(monkeypatch) -> None:
    """The heartbeat reports the runtime busy while a task is about to run."""
    from AutoGLM_GUI.api import managed_busy
    from AutoGLM_GUI.phone_agent_manager import PhoneAgentManager
    from AutoGLM_GUI.scheduler_manager import scheduler_manager

    agents = SimpleNamespace(busy=False)

    async def has_busy_agent_async() -> bool:
        return agents.busy

    monkeypatch.setattr(
        PhoneAgentManager,
        "get_instance",
        classmethod(
            lambda cls: SimpleNamespace(has_busy_agent_async=has_busy_agent_async)
        ),
    )
    due = SimpleNamespace(soon=False)
    monkeypatch.setattr(scheduler_manager, "due_soon", lambda: due.soon)

    assert asyncio.run(managed_busy()) is False
    due.soon = True
    assert asyncio.run(managed_busy()) is True
    due.soon, agents.busy = False, True
    assert asyncio.run(managed_busy()) is True
