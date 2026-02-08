from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Iterator
import time
import json
import uuid


app = FastAPI(title="Fake Echo LLM (OpenAI-compatible)", version="0.1.0")


# ---- Schemas (minimal) ----
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionsIn(BaseModel):
    model: Optional[str] = Field(default="echo-001")
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


class CompletionsIn(BaseModel):
    model: Optional[str] = Field(default="echo-001")
    prompt: str
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


def _check_auth(authorization: Optional[str]):
    # Accept any Bearer token; require header to be present for realism
    if not authorization or not authorization.lower().startswith("bearer "):
        # stay permissive; do not hard fail to simplify local usage
        return


def _split_default_chunks(text: str) -> List[str]:
    if not text:
        return []
    if len(text) <= 12:
        return [text]
    chunk_size = max(1, len(text) // 3)
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _split_stream_chunks(text: str) -> List[str]:
    """Split text into chunks, deliberately splitting __PII_XXX__ placeholders."""
    marker = "__PII_"
    start = text.find(marker)
    if start < 0:
        return _split_default_chunks(text)

    suffix_start = start + len(marker)
    suffix_end = text.find("__", suffix_start)
    if suffix_end >= 0:
        suffix_end += 2
    else:
        suffix_end = len(text)

    chunks: List[str] = []
    if start > 0:
        chunks.extend(_split_default_chunks(text[:start]))
    chunks.append(text[start:suffix_start])  # "__PII_"
    if suffix_start < suffix_end:
        chunks.append(text[suffix_start:suffix_end])  # "EMAIL_01__"
    if suffix_end < len(text):
        chunks.extend(_split_default_chunks(text[suffix_end:]))

    return [chunk for chunk in chunks if chunk]


def _iter_chat_completions_sse(resp_id: str, model: str, content: str) -> Iterator[str]:
    now = int(time.time())
    for piece in _split_stream_chunks(content):
        payload = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    final_payload = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": now,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def _iter_completions_sse(resp_id: str, model: str, text: str) -> Iterator[str]:
    """SSE generator for /v1/completions stream."""
    now = int(time.time())
    for piece in _split_stream_chunks(text):
        payload = {
            "id": resp_id,
            "object": "text_completion",
            "created": now,
            "model": model,
            "choices": [{"index": 0, "text": piece, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    final_payload = {
        "id": resp_id,
        "object": "text_completion",
        "created": now,
        "model": model,
        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
def list_models(x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": "echo-001",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(inp: ChatCompletionsIn, x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    # Echo last user content; fallback to concat
    last_user = next((m.content for m in reversed(inp.messages) if m.role == "user"), None)
    if last_user is None:
        last_user = "\n\n".join([m.content for m in inp.messages])
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    if inp.stream:
        return StreamingResponse(
            _iter_chat_completions_sse(
                resp_id=resp_id,
                model=inp.model or "echo-001",
                content=last_user,
            ),
            media_type="text/event-stream",
        )
    now = int(time.time())
    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": now,
        "model": inp.model or "echo-001",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": last_user},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/completions")
def completions(inp: CompletionsIn, x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    resp_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    if inp.stream:
        return StreamingResponse(
            _iter_completions_sse(
                resp_id=resp_id,
                model=inp.model or "echo-001",
                text=inp.prompt,
            ),
            media_type="text/event-stream",
        )
    now = int(time.time())
    return {
        "id": resp_id,
        "object": "text_completion",
        "created": now,
        "model": inp.model or "echo-001",
        "choices": [
            {
                "index": 0,
                "text": inp.prompt,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/v1/health")
def health():
    return {"status": "ok"}


