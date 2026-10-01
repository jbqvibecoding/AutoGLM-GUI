"""Managed mode: ask the user before the agent acts inside guarded apps.

The model only sometimes labels a sensitive action itself (``message=`` on a
Tap), and a prompt-injected model may not label it at all. This wrapper does
not ask the model: before every input action (tap, swipe, typing...) it reads
the foreground app's package, and if it is on the guarded list (payment and
bank apps) the user must allow the agent in that app first. A grant lasts for
the current task; the control plane may also answer from a grant the user made
permanent ("always allow"). A refusal raises :class:`ActionDeniedError`, which
ends the task.

Outside guarded apps, every input that can commit something (tap, double tap,
long press, swipe) is first shown to an independent guard model, which only
says whether it pays, buys, sends, posts, deletes, changes account security or
authorizes something (``devices/action_guard.py``). If it does, or if the
guard cannot tell, the user is asked; a refusal also ends the task. An input
the user just approved because the executor flagged it is not asked about
twice.

It also remembers the last screenshot so approvals can show the user what the
agent is looking at, and so the guard sees the screen the agent acted on.
"""

from __future__ import annotations

import asyncio

from AutoGLM_GUI.adb.apps import get_app_name
from AutoGLM_GUI.device_protocol import AsyncDeviceProtocol, Screenshot
from AutoGLM_GUI.devices.action_guard import (
    ActionGuard,
    InputAction,
    approval_message,
)
from AutoGLM_GUI.exceptions import ActionDeniedError
from AutoGLM_GUI.logger import logger
from AutoGLM_GUI.managed import (
    ask_user,
    is_managed_mode,
    last_screenshot,
    load_managed_settings,
    remember_screenshot,
    take_input_approval,
)
from AutoGLM_GUI.trace import current_trace_id


def is_guarded(package: str, guarded: tuple[str, ...]) -> bool:
    """Exact package, or a package under a guarded prefix (``com.icbc`` covers ``com.icbc.x``)."""
    return any(package == g or package.startswith(g + ".") for g in guarded)


class GuardedDevice:
    """Wraps an async device; see the module docstring."""

    def __init__(
        self,
        inner: AsyncDeviceProtocol,
        *,
        guarded: tuple[str, ...],
        context: str,
        action_guard: ActionGuard | None = None,
        managed_id: str | None = None,
    ) -> None:
        self._inner = inner
        # The id the agent manager knows this phone by. Approvals, remembered
        # screenshots and the executor's own confirmations use it; a remote
        # device may call itself something else.
        self._managed_id = managed_id or inner.device_id
        self._guarded = guarded
        self._context = context
        self._action_guard = action_guard
        self._grants: set[tuple[str, str]] = set()

    @property
    def device_id(self) -> str:
        return self._inner.device_id

    # ------------------------------------------------------------ the guard

    async def _foreground_package(self) -> str | None:
        get_package = getattr(self._inner, "get_current_package", None)
        if get_package is not None:
            return await get_package()
        # Remote device agents report the package as the current app.
        return await self._inner.get_current_app()

    async def _check(self) -> None:
        if not self._guarded:
            return
        try:
            package = await self._foreground_package()
        except Exception as exc:
            # Cannot tell which app is in front: do not act blind.
            raise ActionDeniedError(f"无法确认当前 App，已停止：{exc}") from exc
        if not package or not is_guarded(package, self._guarded):
            return

        grant = (current_trace_id() or self._context, package)
        if grant in self._grants:
            return
        name = get_app_name(package) or package
        allowed = await asyncio.to_thread(
            ask_user,
            f"允许 agent 在「{name}」中操作？",
            device_id=self._managed_id,
            context=self._context,
            kind="app_access",
            package=package,
            app_name=name,
        )
        if not allowed:
            raise ActionDeniedError(f"用户未允许 agent 在「{name}」中操作")
        logger.info(f"[Managed] Agent allowed in {package} for this task")
        self._grants.add(grant)

    async def _judge(self, action: InputAction) -> None:
        """Ask the user first if the guard model finds the input sensitive (or can't tell)."""
        if self._action_guard is None:
            return
        if take_input_approval(self._managed_id, self._context):
            return  # the user just approved this very input
        screenshot = last_screenshot(self._managed_id)
        verdict = await self._action_guard.judge(screenshot, action)
        if not verdict.needs_approval:
            return
        allowed = await asyncio.to_thread(
            ask_user,
            approval_message(verdict, action),
            device_id=self._managed_id,
            context=self._context,
        )
        if not allowed:
            raise ActionDeniedError("用户未允许这一步操作")

    # ---------------------------------------------------------- read access

    async def get_screenshot(self, timeout: int = 10) -> Screenshot:
        screenshot = await self._inner.get_screenshot(timeout)
        remember_screenshot(self._managed_id, screenshot)
        return screenshot

    async def get_current_app(self) -> str:
        return await self._inner.get_current_app()

    # ------------------------------------------------------- guarded inputs

    async def tap(self, x: int, y: int, delay: float | None = None) -> None:
        await self._check()
        await self._judge(InputAction("tap", x, y))
        await self._inner.tap(x, y, delay)

    async def double_tap(self, x: int, y: int, delay: float | None = None) -> None:
        await self._check()
        await self._judge(InputAction("double_tap", x, y))
        await self._inner.double_tap(x, y, delay)

    async def long_press(
        self, x: int, y: int, duration_ms: int = 3000, delay: float | None = None
    ) -> None:
        await self._check()
        await self._judge(InputAction("long_press", x, y))
        await self._inner.long_press(x, y, duration_ms, delay)

    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int | None = None,
        delay: float | None = None,
    ) -> None:
        await self._check()
        await self._judge(InputAction("swipe", start_x, start_y, end_x, end_y))
        await self._inner.swipe(start_x, start_y, end_x, end_y, duration_ms, delay)

    async def type_text(self, text: str) -> None:
        await self._check()
        await self._inner.type_text(text)

    async def clear_text(self) -> None:
        await self._check()
        await self._inner.clear_text()

    async def detect_and_set_adb_keyboard(self) -> str:
        # First call of a Type action: refuse before the keyboard is switched.
        await self._check()
        return await self._inner.detect_and_set_adb_keyboard()

    # ------------------------------------------------- not guarded (harmless)

    async def restore_keyboard(self, ime: str) -> None:
        await self._inner.restore_keyboard(ime)

    async def back(self, delay: float | None = None) -> None:
        await self._inner.back(delay)

    async def home(self, delay: float | None = None) -> None:
        await self._inner.home(delay)

    async def launch_app(self, app_name: str, delay: float | None = None) -> bool:
        return await self._inner.launch_app(app_name, delay)


def agent_context(agent_key: str, device_id: str) -> str:
    """The context part of an agent key (``<device_id>:<context>``); device ids
    may themselves contain colons."""
    if agent_key == device_id:
        return "default"
    return agent_key.removeprefix(f"{device_id}:")


def guard_device(
    device: AsyncDeviceProtocol, *, agent_key: str, device_id: str
) -> AsyncDeviceProtocol | GuardedDevice:
    """Wrap ``device`` in managed mode; return it unchanged otherwise."""
    if not is_managed_mode():
        return device
    settings = load_managed_settings()
    action_guard = None
    if (
        settings.action_guard_model
        and settings.model_base_url
        and settings.model_api_key
    ):
        action_guard = ActionGuard(
            model=settings.action_guard_model,
            base_url=settings.model_base_url,
            api_key=settings.model_api_key,
        )
    else:
        logger.warning(
            "[Managed] No AUTOGLM_ACTION_GUARD_MODEL: only the executor model "
            "decides which inputs outside guarded apps need approval"
        )
    return GuardedDevice(
        device,
        guarded=settings.guarded_apps,
        context=agent_context(agent_key, device_id),
        action_guard=action_guard,
        managed_id=device_id,
    )
