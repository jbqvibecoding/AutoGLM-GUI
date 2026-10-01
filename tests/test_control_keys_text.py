"""Contract tests for the key and text control endpoints (takeover controls)."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import AutoGLM_GUI.api.control as control_api

pytestmark = [pytest.mark.contract]

SECRET = "密码 hunter2"


class _Device:
    def __init__(self, ime: str = "com.sohu.inputmethod.sogou/.SogouIME") -> None:
        self.calls: list[tuple[str, Any]] = []
        self.ime = ime
        self.fail_typing = False
        self.device_id = "10.0.0.2:5555"

    def back(self, delay: float | None = None) -> None:
        self.calls.append(("back", delay))

    def home(self, delay: float | None = None) -> None:
        self.calls.append(("home", delay))

    def detect_and_set_adb_keyboard(self) -> str:
        self.calls.append(("set_ime", None))
        return self.ime

    def type_text(self, text: str) -> None:
        if self.fail_typing:
            raise RuntimeError(f"adb shell am broadcast --es msg {text}")
        self.calls.append(("type_text", text))

    def restore_keyboard(self, ime: str) -> None:
        self.calls.append(("restore_ime", ime))


class _AdbDevice(_Device):
    pass


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    devices: dict[str, _Device] = {"adb": _AdbDevice(), "remote": _Device()}
    keyevents: list[tuple[str, int]] = []

    def device_for(device_id: str) -> _Device:
        if device_id not in devices:
            raise ValueError(f"Device {device_id} not found in DeviceManager")
        return devices[device_id]

    monkeypatch.setattr(control_api, "_device_for", device_for)
    monkeypatch.setattr(control_api, "ADBDevice", _AdbDevice)
    monkeypatch.setattr(
        control_api.adb,
        "keyevent",
        lambda device_id, code: keyevents.append((device_id, code)),
    )
    app = FastAPI()
    app.include_router(control_api.router)
    return {"client": TestClient(app), "devices": devices, "keyevents": keyevents}


def _key(env: dict[str, Any], device_id: str, key: str) -> Any:
    return env["client"].post(
        "/api/control/key", json={"device_id": device_id, "key": key}
    )


def test_back_and_home_work_on_any_device(env: dict[str, Any]) -> None:
    for device_id in ("adb", "remote"):
        assert _key(env, device_id, "back").json() == {"success": True, "error": None}
        assert _key(env, device_id, "home").json()["success"] is True
        assert env["devices"][device_id].calls == [("back", 0.0), ("home", 0.0)]


def test_other_keys_need_adb(env: dict[str, Any]) -> None:
    assert _key(env, "adb", "enter").json()["success"] is True
    assert _key(env, "adb", "delete").json()["success"] is True
    assert _key(env, "adb", "app_switch").json()["success"] is True
    assert env["keyevents"] == [
        ("10.0.0.2:5555", 66),
        ("10.0.0.2:5555", 67),
        ("10.0.0.2:5555", 187),
    ]

    resp = _key(env, "remote", "enter").json()
    assert resp["success"] is False
    assert "not supported" in resp["error"]


def test_bad_key_and_unknown_device(env: dict[str, Any]) -> None:
    assert _key(env, "adb", "power").status_code == 422
    resp = _key(env, "nope", "back").json()
    assert resp["success"] is False
    assert "not found" in resp["error"]


def test_text_is_typed_through_the_adb_keyboard(env: dict[str, Any]) -> None:
    resp = env["client"].post(
        "/api/control/text", json={"device_id": "adb", "text": "你好 world"}
    )
    assert resp.json() == {"success": True, "error": None}
    assert env["devices"]["adb"].calls == [
        ("set_ime", None),
        ("type_text", "你好 world"),
        ("restore_ime", "com.sohu.inputmethod.sogou/.SogouIME"),
    ]


def test_keyboard_left_alone_when_it_already_was_the_adb_keyboard(
    env: dict[str, Any],
) -> None:
    device = env["devices"]["remote"]
    device.ime = "com.android.adbkeyboard/.AdbIME"
    env["client"].post("/api/control/text", json={"device_id": "remote", "text": "x"})
    assert [c[0] for c in device.calls] == ["set_ime", "type_text"]


def test_failed_typing_restores_keyboard_and_never_leaks_the_text(
    env: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    device = env["devices"]["adb"]
    device.fail_typing = True
    with caplog.at_level(logging.DEBUG):
        resp = env["client"].post(
            "/api/control/text", json={"device_id": "adb", "text": SECRET}
        )
    assert resp.json() == {"success": False, "error": "Typing failed"}
    assert ("restore_ime", "com.sohu.inputmethod.sogou/.SogouIME") in device.calls
    assert SECRET not in caplog.text
    assert "hunter2" not in resp.text


def test_text_validation(env: dict[str, Any]) -> None:
    client = env["client"]
    assert (
        client.post(
            "/api/control/text", json={"device_id": "adb", "text": ""}
        ).status_code
        == 422
    )
    too_long = {"device_id": "adb", "text": "x" * 2001}
    assert client.post("/api/control/text", json=too_long).status_code == 422
