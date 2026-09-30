"""Tests for managed (hosted) runtime mode."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import AutoGLM_GUI.adb as adb_module
from AutoGLM_GUI import managed
from AutoGLM_GUI.managed import (
    DEFAULT_REMOTE_DEVICE_ID,
    ManagedSettings,
    bind_managed_device,
    load_managed_settings,
)

pytestmark = pytest.mark.unit

TOKEN = "t" * 32


class TestLoadManagedSettings:
    def test_disabled_by_default(self) -> None:
        assert load_managed_settings({}) == ManagedSettings()

    def test_disabled_ignores_device_vars(self) -> None:
        settings = load_managed_settings(
            {"AUTOGLM_MANAGED_MODE": "0", "AUTOGLM_DEVICE_SERIAL": "10.0.0.2:5555"}
        )
        assert settings.enabled is False
        assert settings.device_serial is None

    def test_serial_binding(self) -> None:
        settings = load_managed_settings(
            {
                "AUTOGLM_MANAGED_MODE": "true",
                "AUTOGLM_DEVICE_SERIAL": " 10.0.0.2:5555 ",
                "AUTOGLM_CONTROL_PLANE_URL": "http://control-plane:8000",
                "AUTOGLM_INTERNAL_TOKEN": TOKEN,
            }
        )
        assert settings.enabled is True
        assert settings.device_serial == "10.0.0.2:5555"
        assert settings.device_remote_url is None
        assert settings.control_plane_url == "http://control-plane:8000"
        assert settings.internal_token == TOKEN

    def test_remote_url_binding(self) -> None:
        settings = load_managed_settings(
            {
                "AUTOGLM_MANAGED_MODE": "1",
                "AUTOGLM_DEVICE_REMOTE_URL": "http://device-agent:8001/",
                "AUTOGLM_INTERNAL_TOKEN": TOKEN,
            }
        )
        assert settings.device_remote_url == "http://device-agent:8001"
        assert settings.device_remote_id == DEFAULT_REMOTE_DEVICE_ID

    def test_remote_device_id_override(self) -> None:
        settings = load_managed_settings(
            {
                "AUTOGLM_MANAGED_MODE": "1",
                "AUTOGLM_DEVICE_REMOTE_URL": "http://device-agent:8001",
                "AUTOGLM_DEVICE_REMOTE_ID": "phone-42",
                "AUTOGLM_INTERNAL_TOKEN": TOKEN,
            }
        )
        assert settings.device_remote_id == "phone-42"

    @pytest.mark.parametrize(
        "env, match",
        [
            ({}, "requires"),
            (
                {
                    "AUTOGLM_DEVICE_SERIAL": "10.0.0.2:5555",
                    "AUTOGLM_DEVICE_REMOTE_URL": "http://device-agent:8001",
                },
                "only one",
            ),
            ({"AUTOGLM_DEVICE_SERIAL": "emulator-5554"}, "host:port"),
            ({"AUTOGLM_DEVICE_REMOTE_URL": "device-agent:8001"}, "http://"),
            (
                {
                    "AUTOGLM_DEVICE_REMOTE_URL": "http://device-agent:8001",
                    "AUTOGLM_INTERNAL_TOKEN": "",
                },
                "AUTOGLM_INTERNAL_TOKEN",
            ),
            (
                {
                    "AUTOGLM_DEVICE_REMOTE_URL": "http://device-agent:8001",
                    "AUTOGLM_INTERNAL_TOKEN": "short",
                },
                "at least 16",
            ),
        ],
    )
    def test_invalid_binding_rejected(self, env: dict[str, str], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            load_managed_settings(
                {"AUTOGLM_MANAGED_MODE": "1", "AUTOGLM_INTERNAL_TOKEN": TOKEN, **env}
            )


class _FakeDeviceManager:
    def __init__(self, results: list[tuple[bool, str, str]]) -> None:
        self._results = list(results)
        self.add_calls: list[tuple[str, str]] = []
        self.refresh_calls = 0

    def add_remote_device(self, base_url: str, device_id: str) -> tuple[bool, str, str]:
        self.add_calls.append((base_url, device_id))
        return self._results.pop(0)

    def force_refresh(self) -> None:
        self.refresh_calls += 1

    def start_polling(self) -> None:
        pass


def _bind(
    device_manager: Any, settings: ManagedSettings, attempts: int = 3
) -> str | None:
    return asyncio.run(
        bind_managed_device(
            device_manager, settings, adb_path="adb", attempts=attempts, retry_delay=0
        )
    )


class TestBindManagedDevice:
    def test_noop_when_disabled(self) -> None:
        manager = _FakeDeviceManager([])
        assert _bind(manager, ManagedSettings()) is None
        assert manager.add_calls == []

    def test_serial_retries_until_connected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcomes = [(False, "connection refused"), (True, "Connected to 10.0.0.2:5555")]
        connect_calls: list[tuple[str, str]] = []

        class FakeConnection:
            def __init__(self, adb_path: str = "adb") -> None:
                self.adb_path = adb_path

            async def connect_async(self, address: str) -> tuple[bool, str]:
                connect_calls.append((self.adb_path, address))
                return outcomes.pop(0)

        monkeypatch.setattr(adb_module, "ADBConnection", FakeConnection)
        manager = _FakeDeviceManager([])
        settings = ManagedSettings(enabled=True, device_serial="10.0.0.2:5555")

        assert _bind(manager, settings) == "10.0.0.2:5555"
        assert connect_calls == [("adb", "10.0.0.2:5555")] * 2
        assert manager.refresh_calls == 1

    def test_serial_gives_up_after_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FailingConnection:
            def __init__(self, adb_path: str = "adb") -> None:
                pass

            async def connect_async(self, address: str) -> tuple[bool, str]:
                return False, "connection refused"

        monkeypatch.setattr(adb_module, "ADBConnection", FailingConnection)
        manager = _FakeDeviceManager([])
        settings = ManagedSettings(enabled=True, device_serial="10.0.0.2:5555")

        assert _bind(manager, settings, attempts=2) is None
        assert manager.refresh_calls == 0

    def test_remote_device_retries_until_added(self) -> None:
        serial = "remote:http://agent:8001:phone-42"
        manager = _FakeDeviceManager(
            [(False, "Connection failed: refused", ""), (True, "added", serial)]
        )
        settings = ManagedSettings(
            enabled=True,
            device_remote_url="http://agent:8001",
            device_remote_id="phone-42",
        )

        assert _bind(manager, settings) == serial
        assert manager.add_calls == [("http://agent:8001", "phone-42")] * 2

    def test_remote_device_already_registered_counts_as_bound(self) -> None:
        manager = _FakeDeviceManager(
            [(False, "Remote device phone-42 already exists", "")]
        )
        settings = ManagedSettings(
            enabled=True,
            device_remote_url="http://agent:8001",
            device_remote_id="phone-42",
        )

        assert _bind(manager, settings) == "remote:http://agent:8001:phone-42"


def test_lifespan_binds_managed_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app lifespan starts binding when managed mode is enabled."""
    from fastapi.testclient import TestClient

    from AutoGLM_GUI import api as api_module
    from AutoGLM_GUI.device_manager import DeviceManager
    from AutoGLM_GUI.scheduler_manager import scheduler_manager
    from AutoGLM_GUI.task_manager import task_manager

    monkeypatch.setenv("AUTOGLM_MANAGED_MODE", "1")
    monkeypatch.setenv("AUTOGLM_INTERNAL_TOKEN", TOKEN)
    monkeypatch.setenv("AUTOGLM_DEVICE_REMOTE_URL", "http://agent:8001")
    monkeypatch.setenv("AUTOGLM_DEVICE_REMOTE_ID", "phone-42")

    fake_manager = _FakeDeviceManager([])
    monkeypatch.setattr(
        DeviceManager, "get_instance", lambda adb_path="adb": fake_manager
    )

    async def _noop() -> None:
        return None

    monkeypatch.setattr(task_manager, "start", _noop)
    monkeypatch.setattr(task_manager, "shutdown", _noop)
    monkeypatch.setattr(scheduler_manager, "start", _noop)
    monkeypatch.setattr(scheduler_manager, "shutdown", _noop)

    bound: list[ManagedSettings] = []

    async def fake_bind(
        device_manager: Any, settings: ManagedSettings, **_: Any
    ) -> str:
        assert device_manager is fake_manager
        bound.append(settings)
        return "remote:http://agent:8001:phone-42"

    monkeypatch.setattr(api_module, "bind_managed_device", fake_bind)

    with TestClient(api_module.create_app()) as client:
        assert client.get("/api/health").status_code == 200

    assert [s.device_remote_id for s in bound] == ["phone-42"]


def test_lifespan_skips_binding_when_not_managed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    from AutoGLM_GUI import api as api_module
    from AutoGLM_GUI.device_manager import DeviceManager
    from AutoGLM_GUI.scheduler_manager import scheduler_manager
    from AutoGLM_GUI.task_manager import task_manager

    monkeypatch.delenv("AUTOGLM_MANAGED_MODE", raising=False)

    fake_manager = _FakeDeviceManager([])
    monkeypatch.setattr(
        DeviceManager, "get_instance", lambda adb_path="adb": fake_manager
    )

    async def _noop() -> None:
        return None

    monkeypatch.setattr(task_manager, "start", _noop)
    monkeypatch.setattr(task_manager, "shutdown", _noop)
    monkeypatch.setattr(scheduler_manager, "start", _noop)
    monkeypatch.setattr(scheduler_manager, "shutdown", _noop)

    async def fail_bind(*_: Any, **__: Any) -> None:
        raise AssertionError("bind_managed_device must not run outside managed mode")

    monkeypatch.setattr(api_module, "bind_managed_device", fail_bind)

    with TestClient(api_module.create_app()) as client:
        assert client.get("/api/health").status_code == 200


def test_module_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOGLM_MANAGED_MODE", "yes")
    monkeypatch.setenv("AUTOGLM_INTERNAL_TOKEN", TOKEN)
    monkeypatch.setenv("AUTOGLM_DEVICE_SERIAL", "10.0.0.2:5555")
    assert managed.load_managed_settings().device_serial == "10.0.0.2:5555"
