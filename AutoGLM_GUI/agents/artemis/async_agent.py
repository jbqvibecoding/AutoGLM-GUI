"""ArtemisAgent - hands a goal to a local Artemis executor.

Artemis (https://github.com/jbqvibecoding/artemis) runs long, multi-app tasks
and checks its own progress. In a managed deployment it runs beside this
runtime (same network namespace and adb server) in its own managed mode:
loopback only, a per-launch API token, the Flash profile, and its own guard
for payment and bank apps -- it drives the phone itself, so this runtime's
``GuardedDevice`` never sees its actions.

This adapter submits the goal over Artemis's HTTP API and turns its
server-sent events into the runtime's event stream.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import Any

import httpx

from AutoGLM_GUI.config import AgentConfig, ModelConfig
from AutoGLM_GUI.logger import logger

ENV_URL = "AUTOGLM_ARTEMIS_URL"
ENV_TOKEN = "AUTOGLM_ARTEMIS_TOKEN"
#: Longest a single Artemis run may take before it is stopped.
RUN_TIMEOUT_SECONDS = 30 * 60
#: Artemis sends a keep-alive every 5 s; a longer silence means it is gone.
READ_TIMEOUT_SECONDS = 60.0
FINAL_ACTION = "report_task_status"


def artemis_endpoint(
    agent_specific_config: Mapping[str, object] | None = None,
) -> tuple[str, str] | None:
    """(base URL, token) from the agent config or the environment, if set."""
    params = agent_specific_config or {}
    url = params.get("artemis_url") or os.getenv(ENV_URL)
    token = params.get("artemis_token") or os.getenv(ENV_TOKEN)
    if not url or not token:
        return None
    return str(url).rstrip("/"), str(token)


async def iter_sse(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, Any]]:
    """Parse ``text/event-stream`` lines into (event, JSON data) pairs."""
    event, data_lines = "message", []
    async for line in lines:
        if line == "":
            if data_lines:
                try:
                    yield event, json.loads("\n".join(data_lines))
                except ValueError:
                    logger.debug(f"[Artemis] Skipping non-JSON {event} event")
            event, data_lines = "message", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].removeprefix(" "))


def _event(kind: str, **data: Any) -> dict[str, Any]:
    return {"type": kind, "data": data}


class ArtemisAgent:
    """Artemis adapter implementing the AsyncAgent protocol."""

    def __init__(
        self,
        model_config: ModelConfig,
        agent_config: AgentConfig,
        device: Any,
        *,
        base_url: str,
        token: str,
        transport: httpx.AsyncBaseTransport | None = None,
        run_timeout: float = RUN_TIMEOUT_SECONDS,
        takeover_callback: Any = None,  # noqa: ARG002
        confirmation_callback: Any = None,  # noqa: ARG002
    ) -> None:
        self.model_config = model_config
        self.agent_config = agent_config
        self._device = device
        self._base_url = base_url
        self._headers = {"Authorization": f"Bearer {token}"}
        self._transport = transport
        self._run_timeout = run_timeout
        self._session_id: str | None = None
        self._step_count = 0
        self._running = False
        self._cancel_requested = False

    def _client(self, **kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            transport=self._transport,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # AsyncAgent protocol
    # ------------------------------------------------------------------

    async def stream(self, task: str) -> AsyncGenerator[dict[str, Any], None]:
        self._step_count = 0
        self._cancel_requested = False
        self._session_id = str(uuid.uuid4())
        self._running = True
        try:
            async with self._client(
                timeout=httpx.Timeout(30.0, read=READ_TIMEOUT_SECONDS)
            ) as client:
                async for event in self._run(client, task, self._session_id):
                    yield event
        except httpx.HTTPError as exc:
            logger.error(f"[Artemis] Connection failed: {exc!r}")
            yield _event("error", message=f"无法连接 Artemis 执行器：{exc}")
        except asyncio.CancelledError:
            await asyncio.shield(self._stop())
            raise
        finally:
            self._running = False

    async def cancel(self) -> None:
        self._cancel_requested = True
        await self._stop()

    async def run(self, task: str) -> str:
        result = ""
        async for event in self.stream(task):
            if event["type"] in ("done", "error", "cancelled"):
                result = str(event["data"].get("message", ""))
        return result

    def reset(self) -> None:
        self._step_count = 0
        self._cancel_requested = False

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def context(self) -> list[dict[str, Any]]:
        return []

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _stop(self) -> None:
        if not self._session_id:
            return
        try:
            async with self._client(timeout=10.0) as client:
                await client.post("/api/stop", json={"session_id": self._session_id})
        except httpx.HTTPError as exc:
            logger.warning(f"[Artemis] Stop request failed: {exc!r}")

    async def _run(
        self, client: httpx.AsyncClient, task: str, session_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._run_timeout
        # Subscribe first so no early event is missed.
        async with client.stream("GET", f"/api/stream/{session_id}") as events:
            if events.status_code != 200:
                yield _event(
                    "error", message=f"Artemis 拒绝了请求（HTTP {events.status_code}）"
                )
                return
            refusal = await self._submit(client, task, session_id)
            if refusal:
                yield _event("error", message=refusal)
                return
            yield _event("thinking", chunk="[Artemis] 任务已提交，正在准备设备…\n")

            final: dict[str, Any] = {}
            denied_app: str | None = None
            async for name, data in iter_sse(events.aiter_lines()):
                if self._cancel_requested:
                    yield _event("cancelled", message="任务已取消")
                    return
                if loop.time() > deadline:
                    await self._stop()
                    yield _event(
                        "error",
                        message=f"Artemis 执行超时（{int(self._run_timeout // 60)} 分钟）",
                    )
                    return
                if not isinstance(data, dict):
                    continue
                if data.get("session_id") not in (None, session_id):
                    continue

                if name == "startup_progress" and data.get("message"):
                    yield _event("thinking", chunk=f"[Artemis] {data['message']}\n")
                elif name == "llm_stream" and data.get("is_thinking"):
                    yield _event("thinking", chunk=str(data.get("chunk", "")))
                elif name == "step_recorded":
                    step = await self._step_event(client, data)
                    if step["data"]["action"]["action"] == FINAL_ACTION:
                        final = step["data"]["action"]["param"]
                    yield step
                elif name == "approval_required":
                    app = data.get("app_name") or data.get("package") or ""
                    yield _event(
                        "approval_required",
                        approval_id=data.get("approval_id"),
                        message=f"允许 agent 在「{app}」中操作？",
                    )
                elif name == "approval_resolved":
                    status = str(data.get("status", ""))
                    if status != "approved":
                        denied_app = str(data.get("app_name") or data.get("package"))
                    yield _event(
                        "approval_resolved",
                        approval_id=data.get("approval_id"),
                        status=status,
                    )
                elif name == "session_ended" and data.get("session_id") == session_id:
                    yield self._final_event(data, final, denied_app)
                    return

            yield _event("error", message="与 Artemis 的连接意外中断")

    async def _submit(
        self, client: httpx.AsyncClient, task: str, session_id: str
    ) -> str | None:
        """Start the run; returns why Artemis refused it, if it did."""
        body = {
            "goal": task,
            "profile": "flash",
            "session_id": session_id,
            "device_serial": getattr(self._device, "device_id", None),
            "ingress": "autoglm",
        }
        resp = await client.post("/api/run", json=body)
        try:
            reply = resp.json()
        except ValueError:
            reply = {}
        if resp.status_code != 200:
            detail = reply.get("detail") if isinstance(reply, dict) else None
            return f"Artemis 拒绝了任务（HTTP {resp.status_code}）：{detail or resp.text[:200]}"
        if isinstance(reply, dict) and reply.get("status") == "rejected":
            return f"Artemis 拒绝了任务：{reply.get('error') or '设备不可用'}"
        return None

    async def _step_event(
        self, client: httpx.AsyncClient, data: dict[str, Any]
    ) -> dict[str, Any]:
        self._step_count += 1
        taken = data.get("action_taken")
        if isinstance(taken, str):
            try:
                taken = json.loads(taken)
            except ValueError:
                taken = {"action": taken}
        if not isinstance(taken, dict):
            taken = {}
        args = taken.get("args") if isinstance(taken.get("args"), dict) else {}
        action: dict[str, Any] = {
            "_metadata": "Artemis",
            "action": str(taken.get("action") or taken.get("name") or "unknown"),
            "param": args,
        }
        thinking = (
            data.get("operator_raw_thinking")
            or data.get("operator_native_thinking")
            or data.get("summary")
            or ""
        )
        return _event(
            "step",
            step=self._step_count,
            thinking=str(thinking),
            action=action,
            success=True,
            finished=action["action"] == FINAL_ACTION,
            message=str(data.get("summary") or ""),
            screenshot=await self._screenshot(client, data),
        )

    async def _screenshot(
        self, client: httpx.AsyncClient, data: dict[str, Any]
    ) -> str | None:
        name = data.get("post_image_name") or data.get("pre_image_name")
        if not name:
            return None
        try:
            resp = await client.get(f"/api/images/{name}")
        except httpx.HTTPError as exc:
            logger.debug(f"[Artemis] No screenshot for step: {exc!r}")
            return None
        if resp.status_code != 200:
            return None
        return base64.b64encode(resp.content).decode("ascii")

    def _final_event(
        self,
        data: dict[str, Any],
        final: dict[str, Any],
        denied_app: str | None,
    ) -> dict[str, Any]:
        status = str(data.get("status") or "")
        explanation = str(final.get("explanation") or "").strip()
        if self._cancel_requested:
            return _event("cancelled", message="任务已取消")
        if denied_app:
            return _event(
                "done",
                message=f"用户未允许 agent 在「{denied_app}」中操作，任务已停止",
                steps=self._step_count,
                success=False,
                stop_reason="approval_denied",
            )
        if status == "completed":
            return _event(
                "done",
                message=explanation or "任务完成",
                steps=self._step_count,
                success=True,
            )
        return _event(
            "done",
            message=explanation or f"Artemis 未完成任务（{status or 'unknown'}）",
            steps=self._step_count,
            success=False,
        )
