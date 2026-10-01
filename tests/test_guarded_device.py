"""Tests for the managed-mode app guard and approval screenshots."""

from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from typing import Any

import httpx
import pytest
from PIL import Image

from AutoGLM_GUI.actions.async_handler import AsyncActionHandler
from AutoGLM_GUI.adb import parse_focused_package
from AutoGLM_GUI.device_protocol import Screenshot
from AutoGLM_GUI.devices import guarded_device
from AutoGLM_GUI.devices.async_adapter import AsyncDeviceAdapter
from AutoGLM_GUI.devices.guarded_device import (
    GuardedDevice,
    agent_context,
    guard_device,
    is_guarded,
)
from AutoGLM_GUI.exceptions import ActionDeniedError
from AutoGLM_GUI.managed import (
    ApprovalClient,
    ManagedRuntime,
    ManagedSettings,
    jpeg_thumbnail,
    last_screenshot,
    managed_confirmation,
    parse_guarded_apps,
    set_managed_runtime,
)
from AutoGLM_GUI.trace import trace_context

pytestmark = pytest.mark.unit

TOKEN = "runtime-token-" + "x" * 32
ALIPAY = "com.eg.android.AlipayGphone"


def _png(width: int = 1080, height: int = 2400) -> str:
    buf = BytesIO()
    Image.new("RGB", (width, height), (40, 120, 200)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class _ControlPlane:
    def __init__(self, decisions: list[str]) -> None:
        self.decisions = decisions
        self.created: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            self.created.append(json.loads(request.content))
            return httpx.Response(
                201,
                json={
                    "id": f"a{len(self.created)}",
                    "status": "pending",
                    "expires_in": 600,
                },
            )
        status = self.decisions.pop(0) if self.decisions else "denied"
        return httpx.Response(200, json={"id": "a", "status": status})


def _install(cp: _ControlPlane) -> None:
    set_managed_runtime(
        ManagedRuntime(
            ManagedSettings(enabled=True, internal_token=TOKEN),
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            approvals=ApprovalClient(
                "http://cp:8000",
                TOKEN,
                transport=httpx.MockTransport(cp.handler),
                sleep=lambda _: None,
            ),
        )
    )


@pytest.fixture(autouse=True)
def _reset() -> Any:
    yield
    set_managed_runtime(None)


class _Device:
    """Async device whose foreground package can be changed."""

    def __init__(self, package: str | None = ALIPAY) -> None:
        self.package = package
        self.calls: list[str] = []

    @property
    def device_id(self) -> str:
        return "phone-1"

    async def get_current_package(self) -> str | None:
        return self.package

    async def get_current_app(self) -> str:
        return self.package or "System Home"

    async def get_screenshot(self, timeout: int = 10) -> Screenshot:
        return Screenshot(_png(), 1080, 2400)

    async def tap(self, x: int, y: int, delay: float | None = None) -> None:
        self.calls.append("tap")

    async def swipe(self, *args: Any) -> None:
        self.calls.append("swipe")

    async def detect_and_set_adb_keyboard(self) -> str:
        self.calls.append("set_ime")
        return "com.other/.Ime"

    async def clear_text(self) -> None:
        self.calls.append("clear_text")

    async def type_text(self, text: str) -> None:
        self.calls.append("type_text")

    async def restore_keyboard(self, ime: str) -> None:
        self.calls.append("restore_ime")

    async def back(self, delay: float | None = None) -> None:
        self.calls.append("back")


def _guarded(device: _Device, guarded: tuple[str, ...] = (ALIPAY,)) -> GuardedDevice:
    return GuardedDevice(device, guarded=guarded, context="chat:s1")  # type: ignore[arg-type]


# ------------------------------------------------------------------ basics


def test_parse_focused_package() -> None:
    dumpsys = (
        "  mCurrentFocus=Window{1a u0 NotificationShade}\n"
        f"  mFocusedApp=ActivityRecord{{2b u0 {ALIPAY}/com.alipay.Launcher t9}}\n"
    )
    assert parse_focused_package(dumpsys) == ALIPAY
    dumpsys = (
        "  mCurrentFocus=Window{1a u0 com.tencent.mm/com.tencent.mm.ui.LauncherUI}\n"
        f"  mFocusedApp=ActivityRecord{{2b u0 {ALIPAY}/.X t9}}\n"
    )
    assert parse_focused_package(dumpsys) == "com.tencent.mm"
    assert parse_focused_package("no focus here") is None


def test_guarded_matching_and_settings() -> None:
    assert is_guarded(ALIPAY, (ALIPAY,))
    assert is_guarded("com.icbc.mobile", ("com.icbc",))
    assert not is_guarded("com.icbcx", ("com.icbc",))
    assert not is_guarded("com.tencent.mm", (ALIPAY,))
    assert parse_guarded_apps(" com.a, ,com.b ") == ("com.a", "com.b")
    assert parse_guarded_apps(None) == ()
    assert agent_context("10.0.0.2:5555:chat:s1", "10.0.0.2:5555") == "chat:s1"
    assert agent_context("10.0.0.2:5555", "10.0.0.2:5555") == "default"


def test_thumbnail_is_a_small_jpeg() -> None:
    thumb = jpeg_thumbnail(Screenshot(_png(), 1080, 2400))
    assert thumb is not None
    raw = base64.b64decode(thumb)
    img = Image.open(BytesIO(raw))
    assert img.format == "JPEG"
    assert img.size == (540, 1200)
    assert len(raw) < 100_000
    assert jpeg_thumbnail(None) is None
    assert jpeg_thumbnail(Screenshot("", 1, 1)) is None


# ------------------------------------------------------------------- guard


def test_unguarded_app_is_not_asked() -> None:
    cp = _ControlPlane([])
    _install(cp)
    device = _Device(package="com.tencent.mm")
    asyncio.run(_guarded(device).tap(1, 2))
    assert device.calls == ["tap"]
    assert cp.created == []


def test_guarded_app_asks_once_per_task_with_a_screenshot() -> None:
    cp = _ControlPlane(["approved", "approved"])
    _install(cp)
    device = _Device()
    guarded = _guarded(device)

    async def task() -> None:
        await guarded.get_screenshot()
        await guarded.tap(1, 2)
        await guarded.swipe(1, 2, 3, 4, None, None)

    with trace_context("trace-1"):
        asyncio.run(task())
    assert device.calls == ["tap", "swipe"]
    assert len(cp.created) == 1
    request = cp.created[0]
    assert request["kind"] == "app_access"
    assert request["package"] == ALIPAY
    assert "允许 agent" in request["message"]
    assert request["screenshot"]  # the screen the agent saw
    assert last_screenshot("phone-1") is not None

    # A new task asks again.
    with trace_context("trace-2"):
        asyncio.run(guarded.tap(1, 2))
    assert len(cp.created) == 2


def test_denied_app_raises_and_ends_the_task() -> None:
    _install(_ControlPlane(["denied"]))
    device = _Device()
    handler = AsyncActionHandler(AsyncDeviceAdapter(_guarded(device)))  # type: ignore[arg-type]

    with trace_context("trace-1"):
        result = asyncio.run(
            handler.execute(
                {"_metadata": "do", "action": "Tap", "element": [500, 500]}, 1080, 2400
            )
        )
    assert result.should_finish is True
    assert result.success is False
    assert "未允许" in (result.message or "")
    assert device.calls == []


def test_denial_on_type_happens_before_the_keyboard_switch() -> None:
    _install(_ControlPlane(["denied"]))
    device = _Device()
    handler = AsyncActionHandler(AsyncDeviceAdapter(_guarded(device)))  # type: ignore[arg-type]
    with trace_context("trace-1"):
        result = asyncio.run(
            handler.execute(
                {"_metadata": "do", "action": "Type", "text": "123456"}, 1080, 2400
            )
        )
    assert result.should_finish is True
    assert device.calls == []


def test_no_control_plane_denies_guarded_apps() -> None:
    set_managed_runtime(None)
    with pytest.raises(ActionDeniedError):
        asyncio.run(_guarded(_Device()).tap(1, 2))


def test_unknown_foreground_app_is_denied_not_guessed() -> None:
    class Broken(_Device):
        async def get_current_package(self) -> str | None:
            raise RuntimeError("adb offline")

    with pytest.raises(ActionDeniedError, match="无法确认"):
        asyncio.run(_guarded(Broken()).tap(1, 2))


def test_remote_devices_fall_back_to_the_reported_app() -> None:
    cp = _ControlPlane(["approved"])
    _install(cp)

    class Remote(_Device):
        # Remote device agents have no package lookup; they report the package
        # as the current app.
        get_current_package = None  # type: ignore[assignment]

    with trace_context("trace-1"):
        asyncio.run(_guarded(Remote()).tap(1, 2))
    assert cp.created and cp.created[0]["package"] == ALIPAY


def test_confirmation_attaches_the_last_screenshot() -> None:
    cp = _ControlPlane(["approved"])
    _install(cp)
    asyncio.run(_guarded(_Device(package="com.tencent.mm")).get_screenshot())
    assert managed_confirmation("phone-1", "chat:s1")("确认付款") is True
    assert cp.created[0]["kind"] == "action"
    assert cp.created[0]["screenshot"]


def test_standing_grant_answers_without_waiting() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST", "must not poll a decided approval"
        return httpx.Response(
            201, json={"id": "", "status": "approved", "expires_in": 0}
        )

    set_managed_runtime(
        ManagedRuntime(
            ManagedSettings(enabled=True, internal_token=TOKEN),
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            approvals=ApprovalClient(
                "http://cp:8000", TOKEN, transport=httpx.MockTransport(handler)
            ),
        )
    )
    device = _Device()
    with trace_context("trace-1"):
        asyncio.run(_guarded(device).tap(1, 2))
    assert device.calls == ["tap"]


def test_devices_are_wrapped_only_in_managed_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _Device()
    monkeypatch.delenv("AUTOGLM_MANAGED_MODE", raising=False)
    assert guard_device(device, agent_key="d", device_id="d") is device  # type: ignore[arg-type]

    monkeypatch.setenv("AUTOGLM_MANAGED_MODE", "1")
    monkeypatch.setenv("AUTOGLM_DEVICE_REMOTE_URL", "http://agent:8001")
    monkeypatch.setenv("AUTOGLM_INTERNAL_TOKEN", TOKEN)
    monkeypatch.setenv("AUTOGLM_GUARDED_APPS", f"{ALIPAY}, com.icbc")
    wrapped = guard_device(device, agent_key="d:chat:s1", device_id="d")  # type: ignore[arg-type]
    assert isinstance(wrapped, guarded_device.GuardedDevice)
    assert wrapped._guarded == (ALIPAY, "com.icbc")
    assert wrapped._context == "chat:s1"
