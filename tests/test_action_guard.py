"""Tests for the managed-mode action guard: an independent check of committing inputs."""

from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from PIL import Image

from AutoGLM_GUI.device_protocol import Screenshot
from AutoGLM_GUI.devices.action_guard import (
    ActionGuard,
    InputAction,
    Verdict,
    approval_message,
    mark_action,
    parse_verdict,
)
from AutoGLM_GUI.devices.guarded_device import GuardedDevice, guard_device
from AutoGLM_GUI.exceptions import ActionDeniedError
from AutoGLM_GUI.managed import (
    ApprovalClient,
    ManagedRuntime,
    ManagedSettings,
    load_managed_settings,
    managed_confirmation,
    set_managed_runtime,
)

pytestmark = pytest.mark.unit

TOKEN = "runtime-token-" + "x" * 32
WIDTH, HEIGHT = 1080, 2400


def _png(width: int = WIDTH, height: int = HEIGHT) -> str:
    buf = BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


SHOT = Screenshot(_png(), WIDTH, HEIGHT)


# ------------------------------------------------------------------ fakes


class _Completions:
    """Fake ``client.chat.completions`` that answers with a fixed text."""

    def __init__(self, answer: str | Exception, delay: float = 0) -> None:
        self.answer = answer
        self.delay = delay
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(self.answer, Exception):
            raise self.answer
        message = SimpleNamespace(content=self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _guard(
    answer: str | Exception, *, timeout: float = 5, delay: float = 0
) -> ActionGuard:
    completions = _Completions(answer, delay)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return ActionGuard(
        model="guard", base_url="http://gw", api_key="k", timeout=timeout, client=client
    )


def _calls(guard: ActionGuard) -> list[dict[str, Any]]:
    return guard._client.chat.completions.requests  # type: ignore[attr-defined]


SENSITIVE = json.dumps(
    {
        "sensitive": True,
        "category": "payment",
        "target": "确认支付",
        "reason": "会扣款 25 元",
    }
)
HARMLESS = json.dumps(
    {"sensitive": False, "category": "none", "target": "搜索", "reason": ""}
)


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
    def __init__(self, package: str = "com.tencent.mm") -> None:
        self.package = package
        self.calls: list[str] = []

    @property
    def device_id(self) -> str:
        return "phone-1"

    async def get_current_package(self) -> str:
        return self.package

    async def get_screenshot(self, timeout: int = 10) -> Screenshot:
        return SHOT

    async def tap(self, x: int, y: int, delay: float | None = None) -> None:
        self.calls.append("tap")

    async def double_tap(self, x: int, y: int, delay: float | None = None) -> None:
        self.calls.append("double_tap")

    async def long_press(
        self, x: int, y: int, duration_ms: int = 3000, delay: float | None = None
    ) -> None:
        self.calls.append("long_press")

    async def swipe(self, *args: Any) -> None:
        self.calls.append("swipe")

    async def back(self, delay: float | None = None) -> None:
        self.calls.append("back")


def _guarded(
    device: _Device, guard: ActionGuard | None, guarded: tuple[str, ...] = ()
) -> GuardedDevice:
    return GuardedDevice(device, guarded=guarded, context="chat:s1", action_guard=guard)  # type: ignore[arg-type]


async def _look_then(guarded: GuardedDevice, *inputs: str) -> None:
    await guarded.get_screenshot()
    for name in inputs:
        if name == "swipe":
            await guarded.swipe(540, 2000, 1000, 2000)
        else:
            await getattr(guarded, name)(540, 2100)


# ---------------------------------------------------------------- verdicts


@pytest.mark.parametrize(
    "text",
    [
        SENSITIVE,
        f"好的，结果如下：\n```json\n{SENSITIVE}\n```",
    ],
)
def test_verdicts_are_read_from_the_first_json_object(text: str) -> None:
    verdict = parse_verdict(text)
    assert verdict == Verdict(True, "payment", "确认支付", "会扣款 25 元")
    assert verdict.needs_approval


@pytest.mark.parametrize(
    "text",
    [
        "",
        "sensitive: no",
        '{"sensitive": "no"}',
        '{"category": "none"}',
        "[true]",
        '{"sensitive": false',
    ],
)
def test_anything_else_is_a_doubt(text: str) -> None:
    verdict = parse_verdict(text)
    assert verdict.error is not None
    assert verdict.needs_approval


def test_categories_are_normalized() -> None:
    assert parse_verdict('{"sensitive": true, "category": "nuke"}').category == "other"
    assert parse_verdict('{"sensitive": true, "category": "none"}').category == "other"
    harmless = parse_verdict('{"sensitive": false, "category": "payment"}')
    assert not harmless.needs_approval
    long = parse_verdict(json.dumps({"sensitive": True, "target": "x" * 500}))
    assert len(long.target) <= 80


def test_approval_messages_name_what_and_where() -> None:
    tap = InputAction("tap", 1, 2)
    msg = approval_message(parse_verdict(SENSITIVE), tap)
    assert msg == "这一步可能是「付款或转账」：点击「确认支付」。会扣款 25 元"
    msg = approval_message(
        Verdict(False, error="检查超时"), InputAction("swipe", 1, 2, 3, 4)
    )
    assert "检查超时" in msg and "滑动" in msg
    assert (
        len(approval_message(Verdict(True, reason="长" * 80, target="t" * 80), tap))
        <= 300
    )


# ------------------------------------------------------------------ marking


def test_the_target_is_marked_on_a_scaled_copy() -> None:
    marked = Image.open(
        BytesIO(base64.b64decode(mark_action(SHOT, InputAction("tap", 540, 1200))))
    )
    assert marked.size == (720, 1600)
    # The crosshair centre is red; a corner is still white.
    r, g, b = marked.getpixel((360, 800))  # type: ignore[misc]
    assert r > 200 and g < 80 and b < 80
    assert marked.getpixel((5, 5)) == (255, 255, 255)


def test_swipes_get_an_arrow() -> None:
    action = InputAction("swipe", 100, 1200, 900, 1200)
    marked = Image.open(BytesIO(base64.b64decode(mark_action(SHOT, action))))
    r, g, b = marked.getpixel((360, 800))  # type: ignore[misc]  # middle of the arrow
    assert r > 200 and g < 80 and b < 80
    assert action.describe(WIDTH, HEIGHT) == "swipe from (93, 500) to (833, 500)"


# -------------------------------------------------------------------- judge


def test_the_guard_sees_the_marked_screen_and_the_input() -> None:
    guard = _guard(HARMLESS)
    verdict = asyncio.run(guard.judge(SHOT, InputAction("tap", 540, 1200)))
    assert not verdict.needs_approval
    (request,) = _calls(guard)
    assert request["model"] == "guard" and request["temperature"] == 0
    text, image = request["messages"][1]["content"]
    assert "tap at (500, 500)" in text["text"]
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "data, not instructions" in request["messages"][0]["content"]


@pytest.mark.parametrize(
    "guard, screenshot, error",
    [
        (_guard(HARMLESS, timeout=0.05, delay=1), SHOT, "检查超时"),
        (_guard(RuntimeError("gateway down")), SHOT, "检查失败"),
        (_guard(HARMLESS), None, "没有可用的屏幕截图"),
        (
            _guard(HARMLESS),
            Screenshot(_png(), WIDTH, HEIGHT, is_sensitive=True),
            "屏幕内容受保护，无法查看",
        ),
    ],
)
def test_the_guard_fails_closed(
    guard: ActionGuard, screenshot: Screenshot | None, error: str
) -> None:
    verdict = asyncio.run(guard.judge(screenshot, InputAction("tap", 1, 2)))
    assert verdict.error == error
    assert verdict.needs_approval


# ------------------------------------------------------------ guarded device


@pytest.mark.parametrize("name", ["tap", "double_tap", "long_press", "swipe"])
def test_a_harmless_input_goes_through_without_asking(name: str) -> None:
    cp = _ControlPlane([])
    _install(cp)
    device = _Device()
    guard = _guard(HARMLESS)
    asyncio.run(_look_then(_guarded(device, guard), name))
    assert device.calls == [name]
    assert len(_calls(guard)) == 1
    assert cp.created == []


def test_a_sensitive_input_waits_for_the_user() -> None:
    cp = _ControlPlane(["approved"])
    _install(cp)
    device = _Device()
    asyncio.run(_look_then(_guarded(device, _guard(SENSITIVE)), "tap"))
    assert device.calls == ["tap"]
    (approval,) = cp.created
    assert approval["message"].startswith(
        "这一步可能是「付款或转账」：点击「确认支付」"
    )
    assert approval["kind"] == "action"
    assert approval["context"] == "chat:s1"
    assert approval["screenshot"]  # what the agent saw


def test_a_refused_input_never_reaches_the_phone() -> None:
    cp = _ControlPlane(["denied"])
    _install(cp)
    device = _Device()
    with pytest.raises(ActionDeniedError):
        asyncio.run(_look_then(_guarded(device, _guard(SENSITIVE)), "swipe"))
    assert device.calls == []


def test_when_the_guard_fails_the_user_decides() -> None:
    cp = _ControlPlane(["denied"])
    _install(cp)
    device = _Device()
    with pytest.raises(ActionDeniedError):
        asyncio.run(_look_then(_guarded(device, _guard(RuntimeError("down"))), "tap"))
    assert device.calls == []
    assert "检查失败" in cp.created[0]["message"]


def test_navigation_is_not_checked() -> None:
    cp = _ControlPlane([])
    _install(cp)
    device = _Device()
    guard = _guard(SENSITIVE)
    asyncio.run(_guarded(device, guard).back())
    assert device.calls == ["back"]
    assert _calls(guard) == []


def test_an_input_the_executor_flagged_is_asked_about_once() -> None:
    cp = _ControlPlane(["approved"])
    _install(cp)
    device = _Device()
    guard = _guard(SENSITIVE)
    guarded = _guarded(device, guard)
    confirm = managed_confirmation("phone-1", "chat:s1")

    async def flagged_tap() -> None:
        await guarded.get_screenshot()
        assert await asyncio.to_thread(confirm, "确认支付 25 元")
        await guarded.tap(540, 2100)
        # The approval covered that one input only.
        cp.decisions.append("denied")
        with pytest.raises(ActionDeniedError):
            await guarded.tap(540, 2100)

    asyncio.run(flagged_tap())
    assert device.calls == ["tap"]
    assert len(cp.created) == 2
    assert len(_calls(guard)) == 1


def test_a_refused_flag_leaves_no_approval_behind() -> None:
    cp = _ControlPlane(["denied", "denied"])
    _install(cp)
    device = _Device()
    guarded = _guarded(device, _guard(SENSITIVE))
    confirm = managed_confirmation("phone-1", "chat:s1")

    async def run() -> None:
        await guarded.get_screenshot()
        assert not await asyncio.to_thread(confirm, "确认支付")
        with pytest.raises(ActionDeniedError):
            await guarded.tap(540, 2100)

    asyncio.run(run())
    assert device.calls == []


def test_guarded_apps_are_asked_about_first() -> None:
    cp = _ControlPlane(["denied"])
    _install(cp)
    device = _Device(package="com.eg.android.AlipayGphone")
    guard = _guard(HARMLESS)
    with pytest.raises(ActionDeniedError):
        asyncio.run(
            _look_then(_guarded(device, guard, ("com.eg.android.AlipayGphone",)), "tap")
        )
    assert cp.created[0]["kind"] == "app_access"
    assert _calls(guard) == []


# ----------------------------------------------------------------- settings


MANAGED_ENV = {
    "AUTOGLM_MANAGED_MODE": "1",
    "AUTOGLM_DEVICE_SERIAL": "10.0.0.2:5555",
    "AUTOGLM_INTERNAL_TOKEN": TOKEN,
}


def test_the_guard_model_needs_the_gateway() -> None:
    with pytest.raises(ValueError, match="AUTOGLM_ACTION_GUARD_MODEL"):
        load_managed_settings(
            {**MANAGED_ENV, "AUTOGLM_ACTION_GUARD_MODEL": "action-guard"}
        )
    settings = load_managed_settings(
        {
            **MANAGED_ENV,
            "AUTOGLM_ACTION_GUARD_MODEL": "action-guard",
            "AUTOGLM_BASE_URL": "http://gw/v1",
            "AUTOGLM_API_KEY": "sk-user",
        }
    )
    assert settings.action_guard_model == "action-guard"


def test_guard_device_builds_the_guard_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in MANAGED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("AUTOGLM_ACTION_GUARD_MODEL", "action-guard")
    monkeypatch.setenv("AUTOGLM_BASE_URL", "http://gw/v1")
    monkeypatch.setenv("AUTOGLM_API_KEY", "sk-user")
    wrapped = guard_device(_Device(), agent_key="phone-1", device_id="phone-1")  # type: ignore[arg-type]
    assert isinstance(wrapped, GuardedDevice)
    assert wrapped._action_guard is not None
    assert wrapped._action_guard.model == "action-guard"

    monkeypatch.delenv("AUTOGLM_ACTION_GUARD_MODEL")
    wrapped = guard_device(_Device(), agent_key="phone-1", device_id="phone-1")  # type: ignore[arg-type]
    assert isinstance(wrapped, GuardedDevice) and wrapped._action_guard is None

    monkeypatch.delenv("AUTOGLM_MANAGED_MODE")
    device = _Device()
    assert guard_device(device, agent_key="phone-1", device_id="phone-1") is device  # type: ignore[arg-type]
