"""Managed (hosted) runtime mode.

When ``AUTOGLM_MANAGED_MODE=1``, AutoGLM-GUI runs as a single-user runtime
behind a platform control plane and is bound to exactly one device that the
control plane provisioned. The device is supplied through the environment:

- ``AUTOGLM_DEVICE_SERIAL``: ADB TCP address of the phone (``host:port``).
- ``AUTOGLM_DEVICE_REMOTE_URL``: base URL of an HTTP Device Agent (see
  :mod:`AutoGLM_GUI.devices.remote_device`), used for mock devices in CI.
  ``AUTOGLM_DEVICE_REMOTE_ID`` selects the device on that agent.

Exactly one of the two must be set. ``AUTOGLM_CONTROL_PLANE_URL`` and
``AUTOGLM_INTERNAL_TOKEN`` are read here so later managed-mode features share
one settings object.

Managed mode is off by default; nothing here changes the behaviour of a normal
local installation.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from AutoGLM_GUI.logger import logger

if TYPE_CHECKING:
    from AutoGLM_GUI.device_manager import DeviceManager

_TRUE_VALUES = {"1", "true", "yes", "on"}

DEFAULT_REMOTE_DEVICE_ID = "managed-device"
DEFAULT_BIND_ATTEMPTS = 15
DEFAULT_BIND_RETRY_DELAY = 2.0


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

    enabled = (env.get("AUTOGLM_MANAGED_MODE") or "").strip().lower() in _TRUE_VALUES
    if not enabled:
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

    return ManagedSettings(
        enabled=True,
        device_serial=device_serial,
        device_remote_url=device_remote_url.rstrip("/") if device_remote_url else None,
        device_remote_id=_clean(env.get("AUTOGLM_DEVICE_REMOTE_ID"))
        or DEFAULT_REMOTE_DEVICE_ID,
        control_plane_url=_clean(env.get("AUTOGLM_CONTROL_PLANE_URL")),
        internal_token=_clean(env.get("AUTOGLM_INTERNAL_TOKEN")),
    )


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
