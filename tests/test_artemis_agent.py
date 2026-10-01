"""Artemis executor: event translation, failures, env agent type and layered routing."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx
import pytest

from AutoGLM_GUI.agents.artemis.async_agent import ArtemisAgent, artemis_endpoint
from AutoGLM_GUI.config import AgentConfig, ModelConfig

TOKEN = "artemis-token"
SERIAL = "cpa-phone-abc:5555"
JPEG = b"\xff\xd8\xff\xe0fake"


class FakeDevice:
    device_id = SERIAL


def sse(*events: tuple[str, dict[str, Any]]) -> bytes:
    lines = ['event: info\ndata: {"message": "Subscribed"}\n\n', ": comment\n\n"]
    for name, data in events:
        lines.append(f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")
        lines.append("event: keep-alive\ndata: {}\n\n")
    return "".join(lines).encode()


class FakeArtemis:
    """Serves one session's events; ``{sid}`` in event data is the run's session."""

    def __init__(
        self,
        events: list[tuple[str, dict[str, Any]]],
        *,
        run: httpx.Response | None = None,
        stream_status: int = 200,
    ) -> None:
        self.events = events
        self.run = run
        self.stream_status = stream_status
        self.requests: list[httpx.Request] = []
        self.session_id: str | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        path = request.url.path
        if path.startswith("/api/stream/"):
            self.session_id = path.rsplit("/", 1)[1]
            body = sse(
                *[
                    (
                        name,
                        json.loads(json.dumps(data).replace("{sid}", self.session_id)),
                    )
                    for name, data in self.events
                ]
            )
            return httpx.Response(
                self.stream_status,
                content=body,
                headers={"content-type": "text/event-stream"},
            )
        if path == "/api/run":
            return self.run or httpx.Response(
                200, json={"status": "started", "tasks": []}
            )
        if path.startswith("/api/images/"):
            return httpx.Response(200, content=JPEG)
        if path == "/api/stop":
            return httpx.Response(200, json={"status": "stopped"})
        return httpx.Response(404)

    def posted(self, path: str) -> list[dict[str, Any]]:
        return [
            json.loads(r.content)
            for r in self.requests
            if r.method == "POST" and r.url.path == path
        ]


def make_agent(fake: Any, **kwargs: Any) -> ArtemisAgent:
    return ArtemisAgent(
        model_config=ModelConfig(),
        agent_config=AgentConfig(),
        device=FakeDevice(),
        base_url="http://127.0.0.1:8100",
        token=TOKEN,
        transport=httpx.MockTransport(fake),
        **kwargs,
    )


def collect(
    agent: ArtemisAgent, task: str = "查订单并提交售后"
) -> list[dict[str, Any]]:
    async def run() -> list[dict[str, Any]]:
        return [event async for event in agent.stream(task)]

    return asyncio.run(run())


def step(n: int, action: str, args: dict[str, Any], **extra: Any) -> tuple[str, dict]:
    return (
        "step_recorded",
        {
            "session_id": "{sid}",
            "step_number": n,
            "action_taken": {"action": action, "args": args},
            "operator_raw_thinking": f"thinking {n}",
            "summary": f"summary {n}",
            **extra,
        },
    )


def ended(status: str) -> tuple[str, dict]:
    return ("session_ended", {"session_id": "{sid}", "status": status})


def test_completed_run_is_translated() -> None:
    fake = FakeArtemis(
        [
            ("startup_progress", {"session_id": "{sid}", "message": "Connecting"}),
            (
                "llm_stream",
                {"session_id": "{sid}", "chunk": "hmm", "is_thinking": True},
            ),
            (
                "llm_stream",
                {"session_id": "{sid}", "chunk": "text", "is_thinking": False},
            ),
            step(1, "click", {"target": [500, 500]}, post_image_name="abc123"),
            step(
                2,
                "report_task_status",
                {"status": "completed", "explanation": "已提交售后"},
            ),
            ended("completed"),
        ]
    )
    agent = make_agent(fake)
    events = collect(agent)

    [run] = fake.posted("/api/run")
    assert run == {
        "goal": "查订单并提交售后",
        "profile": "flash",
        "session_id": fake.session_id,
        "device_serial": SERIAL,
        "ingress": "autoglm",
    }
    kinds = [e["type"] for e in events]
    assert kinds == ["thinking", "thinking", "thinking", "step", "step", "done"]
    assert events[1]["data"]["chunk"] == "[Artemis] Connecting\n"
    assert events[2]["data"]["chunk"] == "hmm"

    first = events[3]["data"]
    assert first["step"] == 1
    assert first["thinking"] == "thinking 1"
    assert first["action"] == {
        "_metadata": "Artemis",
        "action": "click",
        "param": {"target": [500, 500]},
    }
    assert first["screenshot"] == base64.b64encode(JPEG).decode()
    assert first["finished"] is False
    assert events[4]["data"]["finished"] is True
    assert events[4]["data"]["screenshot"] is None

    assert events[-1]["data"] == {"message": "已提交售后", "steps": 2, "success": True}
    assert agent.step_count == 2
    assert not agent.is_running


def test_other_sessions_are_ignored() -> None:
    fake = FakeArtemis(
        [
            step(1, "click", {}, session_id="someone-else"),
            ("session_ended", {"session_id": "someone-else", "status": "failed"}),
            ended("completed"),
        ]
    )
    events = collect(make_agent(fake))
    assert [e["type"] for e in events] == ["thinking", "done"]
    assert events[-1]["data"]["success"] is True


def test_refused_app_access_ends_the_task() -> None:
    fake = FakeArtemis(
        [
            (
                "approval_required",
                {"session_id": "{sid}", "approval_id": "a1", "app_name": "Bank"},
            ),
            (
                "approval_resolved",
                {
                    "session_id": "{sid}",
                    "approval_id": "a1",
                    "app_name": "Bank",
                    "status": "denied",
                },
            ),
            step(1, "report_task_status", {"status": "failed", "explanation": "x"}),
            ended("cancelled"),
        ]
    )
    events = collect(make_agent(fake))
    required = next(e for e in events if e["type"] == "approval_required")
    assert required["data"] == {
        "approval_id": "a1",
        "message": "允许 agent 在「Bank」中操作？",
    }
    resolved = next(e for e in events if e["type"] == "approval_resolved")
    assert resolved["data"] == {"approval_id": "a1", "status": "denied"}
    assert events[-1] == {
        "type": "done",
        "data": {
            "message": "用户未允许 agent 在「Bank」中操作，任务已停止",
            "steps": 1,
            "success": False,
            "stop_reason": "approval_denied",
        },
    }


def test_failed_run_reports_the_explanation() -> None:
    fake = FakeArtemis(
        [
            step(
                1,
                "report_task_status",
                {"status": "failed", "explanation": "找不到订单"},
            ),
            ended("failed"),
        ]
    )
    events = collect(make_agent(fake))
    assert events[-1]["data"] == {"message": "找不到订单", "steps": 1, "success": False}

    events = collect(make_agent(FakeArtemis([ended("failed")])))
    assert events[-1]["data"]["message"] == "Artemis 未完成任务（failed）"


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (
            httpx.Response(
                403, json={"detail": "Only the 'flash' profile runs in managed mode"}
            ),
            "Artemis 拒绝了任务（HTTP 403）：Only the 'flash' profile runs in managed mode",
        ),
        (
            httpx.Response(200, json={"status": "rejected", "error": "Unknown device"}),
            "Artemis 拒绝了任务：Unknown device",
        ),
    ],
)
def test_refused_runs_are_errors(run: httpx.Response, message: str) -> None:
    events = collect(make_agent(FakeArtemis([ended("completed")], run=run)))
    assert events == [{"type": "error", "data": {"message": message}}]


def test_rejected_stream_is_an_error() -> None:
    events = collect(make_agent(FakeArtemis([], stream_status=401)))
    assert events == [
        {"type": "error", "data": {"message": "Artemis 拒绝了请求（HTTP 401）"}}
    ]


def test_unreachable_artemis_is_an_error() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    [event] = collect(make_agent(down))
    assert event["type"] == "error"
    assert event["data"]["message"].startswith("无法连接 Artemis 执行器")


def test_stream_ending_early_is_an_error() -> None:
    events = collect(make_agent(FakeArtemis([step(1, "click", {})])))
    assert events[-1] == {
        "type": "error",
        "data": {"message": "与 Artemis 的连接意外中断"},
    }


def test_cancel_stops_the_run() -> None:
    fake = FakeArtemis([step(1, "click", {}), step(2, "click", {}), ended("completed")])
    agent = make_agent(fake)

    async def run() -> list[dict[str, Any]]:
        events = []
        async for event in agent.stream("t"):
            events.append(event)
            if event["type"] == "step":
                await agent.cancel()
        return events

    events = asyncio.run(run())
    assert events[-1] == {"type": "cancelled", "data": {"message": "任务已取消"}}
    assert fake.posted("/api/stop") == [{"session_id": fake.session_id}]


def test_overall_timeout_stops_the_run() -> None:
    fake = FakeArtemis([step(1, "click", {}), ended("completed")])
    events = collect(make_agent(fake, run_timeout=0))
    assert events[-1] == {
        "type": "error",
        "data": {"message": "Artemis 执行超时（0 分钟）"},
    }
    assert fake.posted("/api/stop") == [{"session_id": fake.session_id}]


def test_run_returns_the_final_message() -> None:
    fake = FakeArtemis(
        [step(1, "report_task_status", {"explanation": "好了"}), ended("completed")]
    )
    assert asyncio.run(make_agent(fake).run("t")) == "好了"


def test_endpoint_from_params_or_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOGLM_ARTEMIS_URL", raising=False)
    monkeypatch.delenv("AUTOGLM_ARTEMIS_TOKEN", raising=False)
    assert artemis_endpoint() is None
    assert artemis_endpoint({"artemis_url": "http://a/", "artemis_token": "t"}) == (
        "http://a",
        "t",
    )
    monkeypatch.setenv("AUTOGLM_ARTEMIS_URL", "http://127.0.0.1:8100")
    assert artemis_endpoint() is None  # a URL without a token is not enough
    monkeypatch.setenv("AUTOGLM_ARTEMIS_TOKEN", "tok")
    assert artemis_endpoint() == ("http://127.0.0.1:8100", "tok")


def test_factory_creates_artemis_only_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from AutoGLM_GUI.agents.factory import create_agent, is_agent_type_registered

    assert is_agent_type_registered("artemis")
    monkeypatch.delenv("AUTOGLM_ARTEMIS_URL", raising=False)
    monkeypatch.delenv("AUTOGLM_ARTEMIS_TOKEN", raising=False)
    kwargs = {
        "agent_type": "artemis",
        "model_config": ModelConfig(),
        "agent_config": AgentConfig(),
        "device": FakeDevice(),
    }
    with pytest.raises(ValueError, match="AUTOGLM_ARTEMIS_URL"):
        create_agent(agent_specific_config={}, **kwargs)
    monkeypatch.setenv("AUTOGLM_ARTEMIS_URL", "http://127.0.0.1:8100")
    monkeypatch.setenv("AUTOGLM_ARTEMIS_TOKEN", "tok")
    assert isinstance(create_agent(agent_specific_config={}, **kwargs), ArtemisAgent)


def test_agent_type_and_params_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from AutoGLM_GUI.config_manager import UnifiedConfigManager

    manager = UnifiedConfigManager()
    monkeypatch.setenv("AUTOGLM_AGENT_TYPE", "gemini")
    monkeypatch.setenv("AUTOGLM_AGENT_CONFIG_PARAMS", '{"model_family": "gemini"}')
    manager.load_env_config()
    config = manager.get_effective_config()
    assert config.agent_type == "gemini"
    assert config.agent_config_params == {"model_family": "gemini"}

    monkeypatch.setenv("AUTOGLM_AGENT_CONFIG_PARAMS", "[1, 2]")
    manager.load_env_config()
    assert manager.get_effective_config().agent_config_params is None


# ---------------------------------------------------------------- layered


class FakeManager:
    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def get_agent_with_context_async(self, device_id: str, **kwargs: Any) -> Any:
        self.calls.append(("get", {"device_id": device_id, **kwargs}))
        return self.agent

    async def acquire_device_async(self, device_id: str, **kwargs: Any) -> bool:
        self.calls.append(("acquire", {"device_id": device_id, **kwargs}))
        return True

    async def release_device_async(self, device_id: str, **kwargs: Any) -> None:
        self.calls.append(("release", {"device_id": device_id, **kwargs}))


def _layered_artemis(monkeypatch: pytest.MonkeyPatch, fake: FakeArtemis) -> FakeManager:
    from AutoGLM_GUI import managed
    from AutoGLM_GUI.phone_agent_manager import PhoneAgentManager

    monkeypatch.setenv("AUTOGLM_ARTEMIS_URL", "http://127.0.0.1:8100")
    monkeypatch.setenv("AUTOGLM_ARTEMIS_TOKEN", TOKEN)
    manager = FakeManager(make_agent(fake))
    monkeypatch.setattr(
        PhoneAgentManager, "get_instance", classmethod(lambda cls: manager)
    )
    forwarded: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        managed,
        "forward_approval_event",
        lambda kind, data: forwarded.append((kind, data)),
    )
    manager.forwarded = forwarded  # type: ignore[attr-defined]
    return manager


def test_layered_hands_the_subtask_to_artemis(monkeypatch: pytest.MonkeyPatch) -> None:
    from AutoGLM_GUI.layered_agent_service import ARTEMIS_CONTEXT, _chat_artemis

    fake = FakeArtemis(
        [
            (
                "approval_required",
                {"session_id": "{sid}", "approval_id": "a1", "app_name": "Bank"},
            ),
            (
                "approval_resolved",
                {"session_id": "{sid}", "approval_id": "a1", "status": "approved"},
            ),
            step(1, "report_task_status", {"explanation": "转账记录已导出"}),
            ended("completed"),
        ]
    )
    manager = _layered_artemis(monkeypatch, fake)
    reply = json.loads(asyncio.run(_chat_artemis(SERIAL, "导出转账记录")))
    assert reply == {"result": "转账记录已导出", "steps": 1, "success": True}
    assert manager.calls == [
        (
            "get",
            {"device_id": SERIAL, "context": ARTEMIS_CONTEXT, "agent_type": "artemis"},
        ),
        ("acquire", {"device_id": SERIAL, "context": ARTEMIS_CONTEXT}),
        ("release", {"device_id": SERIAL, "context": ARTEMIS_CONTEXT}),
    ]
    assert [kind for kind, _ in manager.forwarded] == [  # type: ignore[attr-defined]
        "approval_required",
        "approval_resolved",
    ]


def test_layered_reports_artemis_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    from AutoGLM_GUI.layered_agent_service import _chat_artemis

    _layered_artemis(monkeypatch, FakeArtemis([ended("failed")]))
    reply = json.loads(asyncio.run(_chat_artemis(SERIAL, "t")))
    assert reply["success"] is False


def test_layered_without_artemis(monkeypatch: pytest.MonkeyPatch) -> None:
    from AutoGLM_GUI.layered_agent_service import (
        ARTEMIS_INSTRUCTIONS,
        _chat_artemis,
        chat,
        planner_instructions,
    )

    monkeypatch.delenv("AUTOGLM_ARTEMIS_URL", raising=False)
    monkeypatch.delenv("AUTOGLM_ARTEMIS_TOKEN", raising=False)
    reply = json.loads(asyncio.run(_chat_artemis(SERIAL, "t")))
    assert reply["success"] is False
    assert "Artemis 执行器未启用" in reply["result"]
    assert ARTEMIS_INSTRUCTIONS not in planner_instructions()

    schema = chat.params_json_schema["properties"]["executor"]
    assert schema["enum"] == ["phone", "artemis"]
    assert schema["default"] == "phone"

    monkeypatch.setenv("AUTOGLM_ARTEMIS_URL", "http://127.0.0.1:8100")
    monkeypatch.setenv("AUTOGLM_ARTEMIS_TOKEN", TOKEN)
    assert planner_instructions().endswith(ARTEMIS_INSTRUCTIONS)
