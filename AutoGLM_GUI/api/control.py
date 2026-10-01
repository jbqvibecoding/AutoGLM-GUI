"""Device control routes (tap/swipe/touch, keys and text).

These are the user's own hands on the phone (e.g. during a takeover), not the
agent's, so they bypass the agent's device guard. Typed text is never logged.
"""

import asyncio
from typing import Any

from fastapi import APIRouter

from AutoGLM_GUI import adb
from AutoGLM_GUI.devices.adb_device import ADBDevice
from AutoGLM_GUI.logger import logger
from AutoGLM_GUI.schemas import (
    KeyRequest,
    KeyResponse,
    SwipeRequest,
    SwipeResponse,
    TapRequest,
    TapResponse,
    TextRequest,
    TextResponse,
    TouchDownRequest,
    TouchDownResponse,
    TouchMoveRequest,
    TouchMoveResponse,
    TouchUpRequest,
    TouchUpResponse,
)

router = APIRouter()


@router.post("/api/control/tap", response_model=TapResponse)
async def control_tap(request: TapRequest) -> TapResponse:
    """Execute tap at specified device coordinates."""
    try:
        if not request.device_id:
            return TapResponse(success=False, error="device_id is required")

        device = ADBDevice(request.device_id)
        await asyncio.to_thread(
            device.tap,
            x=request.x,
            y=request.y,
            delay=request.delay,
        )

        return TapResponse(success=True)
    except Exception as e:
        return TapResponse(success=False, error=str(e))


@router.post("/api/control/swipe", response_model=SwipeResponse)
async def control_swipe(request: SwipeRequest) -> SwipeResponse:
    """Execute swipe from start to end coordinates."""
    try:
        if not request.device_id:
            return SwipeResponse(success=False, error="device_id is required")

        device = ADBDevice(request.device_id)
        await asyncio.to_thread(
            device.swipe,
            start_x=request.start_x,
            start_y=request.start_y,
            end_x=request.end_x,
            end_y=request.end_y,
            duration_ms=request.duration_ms,
            delay=request.delay,
        )

        return SwipeResponse(success=True)
    except Exception as e:
        return SwipeResponse(success=False, error=str(e))


@router.post("/api/control/touch/down", response_model=TouchDownResponse)
async def control_touch_down(request: TouchDownRequest) -> TouchDownResponse:
    """Send touch DOWN event at specified device coordinates."""
    try:
        from AutoGLM_GUI.adb_plus import touch_down_async

        await touch_down_async(
            x=request.x,
            y=request.y,
            device_id=request.device_id,
            delay=request.delay,
        )

        return TouchDownResponse(success=True)
    except Exception as e:
        return TouchDownResponse(success=False, error=str(e))


@router.post("/api/control/touch/move", response_model=TouchMoveResponse)
async def control_touch_move(request: TouchMoveRequest) -> TouchMoveResponse:
    """Send touch MOVE event at specified device coordinates."""
    try:
        from AutoGLM_GUI.adb_plus import touch_move_async

        await touch_move_async(
            x=request.x,
            y=request.y,
            device_id=request.device_id,
            delay=request.delay,
        )

        return TouchMoveResponse(success=True)
    except Exception as e:
        return TouchMoveResponse(success=False, error=str(e))


@router.post("/api/control/touch/up", response_model=TouchUpResponse)
async def control_touch_up(request: TouchUpRequest) -> TouchUpResponse:
    """Send touch UP event at specified device coordinates."""
    try:
        from AutoGLM_GUI.adb_plus import touch_up_async

        await touch_up_async(
            x=request.x,
            y=request.y,
            device_id=request.device_id,
            delay=request.delay,
        )

        return TouchUpResponse(success=True)
    except Exception as e:
        return TouchUpResponse(success=False, error=str(e))


# Android key codes for keys without a DeviceProtocol method.
_KEYCODES = {"enter": 66, "delete": 67, "app_switch": 187}
_ADB_KEYBOARD_IME = "com.android.adbkeyboard/.AdbIME"


def _device_for(device_id: str) -> Any:
    """ADB or remote device for a UI device id (blocking; call in a thread)."""
    from AutoGLM_GUI.device_manager import DeviceManager

    return DeviceManager.get_instance().get_device_protocol(device_id)


def _press(device_id: str, key: str) -> str | None:
    """Press ``key``; returns an error message, or None on success."""
    device = _device_for(device_id)
    if key == "back":
        device.back(delay=0.0)
    elif key == "home":
        device.home(delay=0.0)
    elif isinstance(device, ADBDevice):
        adb.keyevent(device.device_id, _KEYCODES[key])
    else:
        return f"Key {key!r} is not supported on this device"
    return None


def _type(device_id: str, text: str) -> None:
    """Type through the ADB Keyboard (handles Chinese), then restore the IME."""
    device = _device_for(device_id)
    original_ime = device.detect_and_set_adb_keyboard()
    try:
        device.type_text(text)
    finally:
        if original_ime and original_ime != _ADB_KEYBOARD_IME:
            device.restore_keyboard(original_ime)


@router.post("/api/control/key", response_model=KeyResponse)
async def control_key(request: KeyRequest) -> KeyResponse:
    """Press Back, Home, Enter, Delete or the app switcher."""
    try:
        error = await asyncio.to_thread(_press, request.device_id, request.key)
        return KeyResponse(success=error is None, error=error)
    except Exception as e:
        return KeyResponse(success=False, error=str(e))


@router.post("/api/control/text", response_model=TextResponse)
async def control_text(request: TextRequest) -> TextResponse:
    """Type text into the focused field. The text is never logged or traced."""
    try:
        await asyncio.to_thread(_type, request.device_id, request.text)
        return TextResponse(success=True)
    except Exception as e:
        # The error (e.g. a failed adb command line) may contain the text.
        logger.warning(f"Typing on {request.device_id} failed: {type(e).__name__}")
        return TextResponse(success=False, error="Typing failed")
