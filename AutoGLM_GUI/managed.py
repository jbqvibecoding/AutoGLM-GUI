"""Managed (hosted) runtime mode.

When ``AUTOGLM_MANAGED_MODE=1``, AutoGLM-GUI runs as a single-user runtime
behind a platform control plane and is bound to exactly one device that the
control plane provisioned. The device is supplied through the environment:

- ``AUTOGLM_DEVICE_SERIAL``: ADB TCP address of the phone (``host:port``).
- ``AUTOGLM_DEVICE_REMOTE_URL``: base URL of an HTTP Device Agent (see
  :mod:`AutoGLM_GUI.devices.remote_device`), used for mock devices in CI.
  ``AUTOGLM_DEVICE_REMOTE_ID`` selects the device on that agent.

Exactly one of the two must be set.

``AUTOGLM_INTERNAL_TOKEN`` is required: the gateway sends it on every request
(see :mod:`AutoGLM_GUI.managed_guard`), so the runtime only answers traffic
that the control plane authorized. ``AUTOGLM_CONTROL_PLANE_URL`` is where the
runtime reaches the control plane's internal API.

The control plane may put the phone to sleep while the runtime is idle. Before
the runtime uses the phone (tasks, scheduled tasks, the live stream) it calls
:func:`ensure_device_awake`, which asks the control plane to wake the phone and
reconnects to it; a heartbeat tells the control plane when the runtime is busy.

Managed mode is off by default; nothing here changes the behaviour of a normal
local installation.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from AutoGLM_GUI.logger import logger

if TYPE_CHECKING:
    from AutoGLM_GUI.device_manager import DeviceManager

_TRUE_VALUES = {"1", "true", "yes", "on"}
MIN_INTERNAL_TOKEN_LENGTH = 16

DEFAULT_REMOTE_DEVICE_ID = "managed-device"
DEFAULT_BIND_ATTEMPTS = 15
DEFAULT_BIND_RETRY_DELAY = 2.0

# Waking a phone can include a cold boot.
DEFAULT_WAKE_TIMEOUT = 240.0
# The control plane sleeps phones only after minutes of inactivity, and every
# wake counts as activity, so a wake is still valid for a short while.
WAKE_DEBOUNCE_SECONDS = 30.0
HEARTBEAT_INTERVAL_SECONDS = 30.0


@dataclass(frozen=True)
class ManagedSettings:
    """Managed-mode settings loaded from the environment."""

    enabled: bool = False
    device_serial: str | None = None
    device_remote_url: str | None = None
    device_remote_id: str = DEFAULT_REMOTE_DEVICE_ID
    control_plane_url: str | None = None
    internal_token: str | None = None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def load_managed_settings(env: Mapping[str, str] | None = None) -> ManagedSettings:
    """Read managed-mode settings.

    Raises:
        ValueError: If managed mode is enabled but the device binding is
            missing, ambiguous or malformed.
    """
    env = os.environ if env is None else env

    if not is_managed_mode(env):
        return ManagedSettings()

    device_serial = _clean(env.get("AUTOGLM_DEVICE_SERIAL"))
    device_remote_url = _clean(env.get("AUTOGLM_DEVICE_REMOTE_URL"))

    if device_serial and device_remote_url:
        raise ValueError(
            "Managed mode: set only one of AUTOGLM_DEVICE_SERIAL and "
            "AUTOGLM_DEVICE_REMOTE_URL"
        )
    if not device_serial and not device_remote_url:
        raise ValueError(
            "Managed mode requires AUTOGLM_DEVICE_SERIAL or AUTOGLM_DEVICE_REMOTE_URL"
        )
    if device_serial and ":" not in device_serial:
        raise ValueError(
            "Managed mode: AUTOGLM_DEVICE_SERIAL must be an ADB TCP address "
            f"in host:port form, got {device_serial!r}"
        )
    if device_remote_url and not device_remote_url.startswith(("http://", "https://")):
        raise ValueError(
            "Managed mode: AUTOGLM_DEVICE_REMOTE_URL must start with http:// or https://"
        )
    internal_token = _clean(env.get("AUTOGLM_INTERNAL_TOKEN"))
    if not internal_token or len(internal_token) < MIN_INTERNAL_TOKEN_LENGTH:
        raise ValueError(
            "Managed mode requires AUTOGLM_INTERNAL_TOKEN "
            f"(at least {MIN_INTERNAL_TOKEN_LENGTH} characters)"
        )

    return ManagedSettings(
        enabled=True,
        device_serial=device_serial,
        device_remote_url=device_remote_url.rstrip("/") if device_remote_url else None,
        device_remote_id=_clean(env.get("AUTOGLM_DEVICE_REMOTE_ID"))
        or DEFAULT_REMOTE_DEVICE_ID,
        control_plane_url=_clean(env.get("AUTOGLM_CONTROL_PLANE_URL")),
        internal_token=internal_token,
    )


def is_managed_mode(env: Mapping[str, str] | None = None) -> bool:
    """Whether managed mode is switched on (without validating its settings)."""
    env = os.environ if env is None else env
    return (env.get("AUTOGLM_MANAGED_MODE") or "").strip().lower() in _TRUE_VALUES


async def bind_managed_device(
    device_manager: DeviceManager,
    settings: ManagedSettings,
    *,
    adb_path: str = "adb",
    attempts: int = DEFAULT_BIND_ATTEMPTS,
    retry_delay: float = DEFAULT_BIND_RETRY_DELAY,
) -> str | None:
    """Connect the runtime to its single managed device.

    Retries because the phone may still be booting when the runtime starts.

    Returns:
        The device serial registered with the device manager, or ``None`` if
        every attempt failed.
    """
    if not settings.enabled:
        return None

    last_message = ""
    for attempt in range(1, attempts + 1):
        ok, last_message, serial = await _try_bind(device_manager, settings, adb_path)
        if ok:
            logger.info(f"[Managed] Bound device {serial} (attempt {attempt})")
            if settings.device_serial:
                await asyncio.to_thread(device_manager.force_refresh)
            return serial

        logger.warning(
            f"[Managed] Device bind attempt {attempt}/{attempts} failed: {last_message}"
        )
        if attempt < attempts:
            await asyncio.sleep(retry_delay)

    logger.error(
        f"[Managed] Could not bind managed device after {attempts} attempts: "
        f"{last_message}"
    )
    return None


async def _try_bind(
    device_manager: DeviceManager, settings: ManagedSettings, adb_path: str
) -> tuple[bool, str, str | None]:
    if settings.device_serial:
        from AutoGLM_GUI.adb import ADBConnection

        conn = ADBConnection(adb_path=adb_path)
        ok, message = await conn.connect_async(settings.device_serial)
        return ok, message, settings.device_serial if ok else None

    assert settings.device_remote_url is not None
    ok, message, serial = await asyncio.to_thread(
        device_manager.add_remote_device,
        settings.device_remote_url,
        settings.device_remote_id,
    )
    if not ok and "already exists" in message:
        existing = f"remote:{settings.device_remote_url}:{settings.device_remote_id}"
        return True, message, existing
    return ok, message, serial if ok else None


# --------------------------------------------------------------------------- wake


class ManagedWakeError(RuntimeError):
    """The phone could not be woken or reconnected."""


class ControlPlaneClient:
    """Calls the control plane's internal runtime API with the runtime token."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        wake_timeout: float = DEFAULT_WAKE_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
            transport=transport,
        )
        self._wake_timeout = wake_timeout

    async def wake(self) -> None:
        """Block until the control plane reports the phone running."""
        try:
            resp = await self._client.post(
                "/internal/runtime/wake", timeout=self._wake_timeout
            )
        except httpx.HTTPError as exc:
            raise ManagedWakeError(f"control plane unreachable: {exc}") from exc
        if resp.status_code >= 400:
            raise ManagedWakeError(
                f"wake failed: HTTP {resp.status_code}: {resp.text[:200]}"
            )

    async def report_activity(self, *, busy: bool, viewers: int) -> None:
        """Best effort: a missed heartbeat only makes an idle sleep more likely."""
        try:
            resp = await self._client.post(
                "/internal/runtime/activity",
                json={"busy": busy, "viewers": viewers},
                timeout=5.0,
            )
            if resp.status_code >= 400:
                logger.warning(
                    f"[Managed] Activity report rejected: {resp.status_code}"
                )
        except httpx.HTTPError as exc:
            logger.warning(f"[Managed] Activity report failed: {exc}")

    async def aclose(self) -> None:
        await self._client.aclose()


class ManagedRuntime:
    """Keeps the managed phone awake while the runtime needs it."""

    def __init__(
        self,
        settings: ManagedSettings,
        device_manager: DeviceManager,
        client: ControlPlaneClient,
        *,
        adb_path: str = "adb",
        debounce_seconds: float = WAKE_DEBOUNCE_SECONDS,
        rebind_attempts: int = 10,
        rebind_delay: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._device_manager = device_manager
        self.client = client
        self._adb_path = adb_path
        self._debounce = debounce_seconds
        self._rebind_attempts = rebind_attempts
        self._rebind_delay = rebind_delay
        self._clock = clock
        self._lock = asyncio.Lock()
        self._last_wake: float | None = None

    def woke_within(self, seconds: float) -> bool:
        return self._last_wake is not None and self._clock() - self._last_wake < seconds

    async def ensure_device_awake(self) -> None:
        async with self._lock:
            if self.woke_within(self._debounce):
                return
            await self.client.wake()
            # The ADB TCP connection drops while the phone sleeps; reconnect.
            serial = await bind_managed_device(
                self._device_manager,
                self._settings,
                adb_path=self._adb_path,
                attempts=self._rebind_attempts,
                retry_delay=self._rebind_delay,
            )
            if serial is None:
                raise ManagedWakeError(
                    "phone is awake but the runtime could not reconnect"
                )
            self._last_wake = self._clock()

    async def heartbeat_once(
        self,
        busy_probe: Callable[[], Awaitable[bool]],
        viewers_probe: Callable[[], int],
    ) -> None:
        busy = await busy_probe() or self.woke_within(HEARTBEAT_INTERVAL_SECONDS)
        await self.client.report_activity(busy=busy, viewers=viewers_probe())

    async def heartbeat_loop(
        self,
        busy_probe: Callable[[], Awaitable[bool]],
        viewers_probe: Callable[[], int],
        interval: float = HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        while True:
            try:
                await self.heartbeat_once(busy_probe, viewers_probe)
            except Exception:
                logger.exception("[Managed] Heartbeat failed")
            await asyncio.sleep(interval)


_managed_runtime: ManagedRuntime | None = None


def set_managed_runtime(runtime: ManagedRuntime | None) -> None:
    global _managed_runtime
    _managed_runtime = runtime


def get_managed_runtime() -> ManagedRuntime | None:
    return _managed_runtime


def start_managed_runtime(
    settings: ManagedSettings, device_manager: DeviceManager, adb_path: str
) -> ManagedRuntime | None:
    """Create and register the process-wide ManagedRuntime.

    Without ``AUTOGLM_CONTROL_PLANE_URL`` the phone is never put to sleep by a
    control plane we can reach, so wake-on-demand stays off.
    """
    if not settings.enabled or not settings.control_plane_url:
        if settings.enabled:
            logger.warning(
                "[Managed] AUTOGLM_CONTROL_PLANE_URL not set; wake-on-demand is off"
            )
        return None
    runtime = ManagedRuntime(
        settings,
        device_manager,
        ControlPlaneClient(settings.control_plane_url, settings.internal_token or ""),
        adb_path=adb_path,
    )
    set_managed_runtime(runtime)
    return runtime


async def ensure_device_awake() -> None:
    """Wake the managed phone before using it. No-op outside managed mode.

    Failures are logged, not raised: the phone may well be awake already, and a
    phone that is really unreachable fails the caller with its usual device error.
    """
    runtime = _managed_runtime
    if runtime is None:
        return
    try:
        await runtime.ensure_device_awake()
    except ManagedWakeError as exc:
        logger.error(f"[Managed] {exc}")
