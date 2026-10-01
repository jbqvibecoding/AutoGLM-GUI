"""Mock OpenAI-compatible LLM Server for integration testing.

This FastAPI server simulates an OpenAI-compatible API endpoint,
returning predefined responses in a round-robin fashion via SSE streaming.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from collections.abc import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Predefined responses for the "点击消息按钮" task
DEFAULT_RESPONSES = [
    # Response A: First request (find and tap message button)
    """用户要求点击屏幕下方的消息按钮。我需要查看当前截图，找到消息按钮的位置。

从截图中可以看到，这是美团app的主界面。在底部导航栏中，我可以看到几个选项：
- 推荐（黄色高亮）
- 视频（有红色数字3的徽章）
- 消息（有红色数字2的徽章）
- 购物车
- 我的

消息按钮在底部导航栏的中间位置，显示为"消息"图标，上面有红色的数字2，表示有2条未读消息。

我需要点击这个消息按钮。根据截图，消息按钮的位置大约在底部导航栏的中间位置，坐标大约是(499, 966)左右。
do(action="Tap", element=[499,966])""",
    # Response B: Second request (finish with success message)
    """好的，我成功点击了消息按钮，现在进入了消息页面。页面显示了各种消息类型，包括：
- 订单动态
- 服务提醒（有2条未读）
- 粉丝福利
- 美团会员
- 美团客服
- 2周前的消息（包括一些加载中的消息）

底部还有一个通知提示"开启通知，获取外卖订单进度"。

任务已经完成，我成功点击了屏幕下方的消息按钮，现在进入了消息页面。
finish(message="已成功点击消息按钮！现在进入了消息页面，可以看到各类消息通知，包括订单动态、服务提醒（有2条未读）、美团会员、美团客服以及2周前的历史消息。")""",
]


# Requests for this model are the managed-mode action guard (see
# AutoGLM_GUI/devices/action_guard.py): they get the configured verdict, and
# do not consume or count against the scripted executor responses.
GUARD_MODEL = "mock-guard-model"
HARMLESS_VERDICT = json.dumps(
    {"sensitive": False, "category": "none", "target": "", "reason": ""}
)


@dataclass
class MockLLMState:
    """Global state for the mock LLM server."""

    request_count: int = 0
    responses: list[str] = field(default_factory=lambda: DEFAULT_RESPONSES.copy())
    guard_verdict: str = HARMLESS_VERDICT
    guard_request_count: int = 0

    def get_next_response(self) -> str:
        """Get next response in round-robin fashion."""
        if not self.responses:
            return "No responses configured"

        idx = self.request_count % len(self.responses)
        self.request_count += 1
        return self.responses[idx]

    def reset(self) -> None:
        """Reset request counts and the guard verdict."""
        self.request_count = 0
        self.guard_request_count = 0
        self.guard_verdict = HARMLESS_VERDICT

    def set_responses(self, responses: list[str]) -> None:
        """Override predefined responses."""
        self.responses = responses
        self.request_count = 0


# Global state instance
state = MockLLMState()


class ChatRequest(BaseModel):
    """OpenAI-compatible chat request."""

    model: str
    messages: list[dict]
    stream: bool = True
    # Other OpenAI params (ignored for mock)


async def stream_response(content: str) -> AsyncGenerator[str, None]:
    """Stream content as OpenAI-compatible SSE chunks.

    Args:
        content: The full response text to stream

    Yields:
        SSE-formatted chunks (data: JSON\\n\\n)
    """
    words = content.split()

    for i, word in enumerate(words):
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "mock-glm-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)  # Simulate network delay

    # Final chunk with finish_reason
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "mock-glm-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def guard_completion(content: str) -> dict:
    """A non-streaming chat completion for the action guard."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": GUARD_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def create_app() -> FastAPI:
    """Create the FastAPI app."""
    app = FastAPI(
        title="Mock OpenAI LLM Server",
        description="Mock LLM server for integration testing",
    )

    # Register routes
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    """Register all routes on the app."""

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        """OpenAI-compatible chat completions endpoint."""

        # Validate streaming requirement (the guard model answers in one piece)
        if not req.stream and req.model != GUARD_MODEL:
            raise HTTPException(
                status_code=400, detail="Only streaming mode is supported"
            )

        # Validate messages array
        if not req.messages:
            raise HTTPException(status_code=400, detail="Messages array is required")

        if req.model == GUARD_MODEL:
            state.guard_request_count += 1
            return guard_completion(state.guard_verdict)

        # Get next response
        response_text = state.get_next_response()

        # Stream response
        from fastapi.responses import StreamingResponse

        return StreamingResponse(
            stream_response(response_text), media_type="text/event-stream"
        )

    # === Test Helper Endpoints ===

    @app.get("/test/stats")
    async def get_stats():
        """Get request statistics."""
        return {
            "request_count": state.request_count,
            "total_responses": len(state.responses),
            "guard_request_count": state.guard_request_count,
        }

    @app.post("/test/guard_verdict")
    async def set_guard_verdict(verdict: dict):
        """Set what the action guard answers: a verdict object, or {"text": raw}."""
        state.guard_verdict = (
            str(verdict["text"]) if "text" in verdict else json.dumps(verdict)
        )
        return {"status": "updated"}

    @app.post("/test/reset")
    async def reset():
        """Reset request counter."""
        state.reset()
        return {"status": "reset", "request_count": 0}

    @app.post("/test/set_responses")
    async def set_responses(responses: list[str]):
        """Set custom responses."""
        state.set_responses(responses)
        return {"status": "updated", "response_count": len(responses)}


# Create app instance
app = create_app()


def run_server(port: int = 18003, log_level: str = "warning"):
    """Run the mock LLM server.

    Args:
        port: Port to listen on (default: 18003)
        log_level: Log level for uvicorn
    """
    uvicorn.run(app, host="127.0.0.1", port=port, log_level=log_level)


if __name__ == "__main__":
    run_server()
