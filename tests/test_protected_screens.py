"""Tests for pausing on protected (FLAG_SECURE) screens in managed mode."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncGenerator
from io import BytesIO
from typing import Any

import pytest
from PIL import Image, ImageDraw

from AutoGLM_GUI.adb_plus import screenshot as screenshot_module
from AutoGLM_GUI.adb_plus.screenshot import is_blank_frame
from AutoGLM_GUI.agents.base import AsyncAgentBase
from AutoGLM_GUI.agents.base.async_agent_base import PROTECTED_SCREEN_MESSAGE
from AutoGLM_GUI.agents.gemini.async_agent import AsyncGeminiAgent
from AutoGLM_GUI.agents.glm.async_agent import AsyncGLMAgent
from AutoGLM_GUI.agents.mai.async_agent import AsyncMAIAgent
from AutoGLM_GUI.agents.qwen.async_agent import AsyncQwenAgent
from AutoGLM_GUI.config import AgentConfig, ModelConfig
from AutoGLM_GUI.device_protocol import Screenshot

pytestmark = pytest.mark.unit

W, H = 108, 240


def _secure_frame(status_bar: bool = True) -> Image.Image:
    """Black app window; the status and navigation bars may still be drawn."""
    img = Image.new("RGB", (W, H), "black")
    if status_bar:
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, W, int(H * 0.04)), fill="white")
        draw.rectangle((0, int(H * 0.94), W, H), fill=(200, 200, 200))
    return img


def _dark_mode_frame() -> Image.Image:
    img = Image.new("RGB", (W, H), "black")
    ImageDraw.Draw(img).text((10, 100), "Hello", fill="white")
    return img


def _png(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- detection


def test_blank_frame_detection() -> None:
    assert is_blank_frame(_secure_frame(status_bar=False))
    assert is_blank_frame(_secure_frame(status_bar=True))
    assert not is_blank_frame(_dark_mode_frame())
    assert not is_blank_frame(Image.new("RGB", (W, H), (30, 30, 30)))
    assert not is_blank_frame(Image.new("RGB", (W, 0)))


def test_decode_flags_protected_screens_only_in_managed_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secure = _png(_secure_frame())
    visible = _png(_dark_mode_frame())

    monkeypatch.delenv("AUTOGLM_MANAGED_MODE", raising=False)
    decoded = screenshot_module._decode_screenshot(secure, "d")
    assert decoded is not None and decoded.is_sensitive is False
    assert screenshot_module._fallback_screenshot().is_sensitive is False

    monkeypatch.setenv("AUTOGLM_MANAGED_MODE", "1")
    decoded = screenshot_module._decode_screenshot(secure, "d")
    assert decoded is not None and decoded.is_sensitive is True
    assert (decoded.width, decoded.height) == (W, H)
    decoded = screenshot_module._decode_screenshot(visible, "d")
    assert decoded is not None and decoded.is_sensitive is False
    # Capture failed entirely: the agent cannot see the screen either.
    assert screenshot_module._fallback_screenshot().is_sensitive is True

    decoded = asyncio.run(screenshot_module._decode_screenshot_async(secure, "d"))
    assert decoded is not None and decoded.is_sensitive is True


# -------------------------------------------------------------------- agents


class _Device:
    def __init__(self, sensitive: bool) -> None:
        self.sensitive = sensitive
        self.taps: list[tuple[int, int]] = []

    @property
    def device_id(self) -> str:
        return "phone-1"

    def get_screenshot(self, timeout: int = 10) -> Screenshot:
        data = base64.b64encode(_png(_secure_frame())).decode()
        return Screenshot(data, W, H, is_sensitive=self.sensitive)

    def get_current_app(self) -> str:
        return "com.eg.android.AlipayGphone"

    def tap(self, x: int, y: int, delay: float | None = None) -> None:
        self.taps.append((x, y))


class _NoLLM:
    """Any model call fails the test: a protected screen must not reach it."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"model called ({name}) on a protected screen")


class _StepAgent(AsyncAgentBase):
    """Captures a screenshot each step like the real agents, then finishes."""

    def _get_default_system_prompt(self, lang: str) -> str:
        return "system"

    def _prepare_initial_context(
        self,
        task: str,
        screenshot_base64: str,
        current_app: str,
        reference_images: list[dict[str, str]] | None = None,
    ) -> None:
        self._context.append({"role": "user", "content": task})

    async def _execute_step(self) -> AsyncGenerator[dict[str, Any], None]:
        self._step_count += 1
        screenshot = await self.device.get_screenshot()
        paused = await self._protected_screen_step(screenshot)
        if paused is not None:
            yield paused
            return
        yield {
            "type": "step",
            "data": {
                "step": self._step_count,
                "action": {"action": "finish"},
                "finished": True,
                "success": True,
                "message": "done",
            },
        }


def _config() -> tuple[ModelConfig, AgentConfig]:
    return (
        ModelConfig(base_url="http://127.0.0.1:9/v1", api_key="k", model_name="m"),
        AgentConfig(device_id="phone-1"),
    )


async def _collect(stream: Any) -> list[dict[str, Any]]:
    return [event async for event in stream]


@pytest.fixture
def managed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOGLM_MANAGED_MODE", "1")


def test_protected_first_screen_pauses_then_resumes(managed_env: None) -> None:
    takeovers: list[str] = []
    device = _Device(sensitive=True)
    agent = _StepAgent(*_config(), device, takeover_callback=takeovers.append)

    events = asyncio.run(_collect(agent.stream("付款")))
    assert [e["type"] for e in events] == ["step", "takeover"]
    assert events[0]["data"]["waiting_for_input"] is True
    assert events[0]["data"]["action"]["action"] == "Take_over"
    assert events[1]["data"]["message"] == PROTECTED_SCREEN_MESSAGE
    assert events[1]["data"]["stop_reason"] == "takeover"
    # The takeover callback (a control plane notice in managed mode) fired.
    assert takeovers == [PROTECTED_SCREEN_MESSAGE]

    # Still protected: pauses again.
    events = asyncio.run(_collect(agent.stream("继续", continue_with="继续")))
    assert [e["type"] for e in events] == ["step", "takeover"]

    device.sensitive = False
    events = asyncio.run(_collect(agent.stream("继续", continue_with="继续")))
    assert [e["type"] for e in events] == ["step", "done"]
    assert events[-1]["data"]["success"] is True
    assert agent._context[1] == {"role": "user", "content": "付款"}


def test_protected_screen_is_ignored_outside_managed_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUTOGLM_MANAGED_MODE", raising=False)
    agent = _StepAgent(*_config(), _Device(sensitive=True), takeover_callback=print)
    events = asyncio.run(_collect(agent.stream("付款")))
    assert [e["type"] for e in events] == ["step", "done"]


@pytest.mark.parametrize(
    "agent_cls", [AsyncGLMAgent, AsyncMAIAgent, AsyncGeminiAgent, AsyncQwenAgent]
)
def test_agents_pause_before_calling_the_model(
    managed_env: None, agent_cls: type[AsyncAgentBase]
) -> None:
    takeovers: list[str] = []
    device = _Device(sensitive=True)
    agent = agent_cls(*_config(), device, takeover_callback=takeovers.append)  # type: ignore[call-arg]
    agent.openai_client = _NoLLM()  # type: ignore[assignment]
    agent._step_count = 1  # a later step: every agent captures a fresh screenshot

    events = asyncio.run(_collect(agent._execute_step()))

    assert events[-1]["type"] == "step"
    assert events[-1]["data"]["waiting_for_input"] is True
    assert events[-1]["data"]["message"] == PROTECTED_SCREEN_MESSAGE
    assert takeovers == [PROTECTED_SCREEN_MESSAGE]
    assert device.taps == []
