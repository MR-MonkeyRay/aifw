from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union, AsyncIterator
from .one_aifw_api import OneAIFWAPI
from .aifw_utils import cleanup_monthly_logs
from .proxy import (
    ProxyRouter,
    ProxyTransport,
    UpstreamNotConfiguredError,
    StreamLimitError,
    StreamState,
    SSEParser,
    SSEEvent,
    BaseRewriter,
    OpenAIRewriter,
    AnthropicRewriter,
    GeminiRewriter,
)
from .proxy.transport import (
    is_sse_response,
    iter_response_bytes,
    sanitize_upstream_response_headers,
)
import codecs
import httpx
import json
import os
import secrets
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="OneAIFW Service", version="0.2.0")

api = OneAIFWAPI()
# HTTP API key for Authorization header; can be set via env AIFW_HTTP_API_KEY
API_KEY = os.environ.get("AIFW_HTTP_API_KEY") or None

_proxy_transport: Optional[ProxyTransport] = None

def _get_proxy_transport() -> ProxyTransport:
    global _proxy_transport
    if _proxy_transport is None:
        _proxy_transport = ProxyTransport()
    return _proxy_transport


@app.on_event("shutdown")
async def _shutdown_proxy_transport():
    global _proxy_transport
    if _proxy_transport is not None:
        await _proxy_transport.aclose()
        _proxy_transport = None

_PROXY_METHODS = ["GET", "POST", "OPTIONS"]
_PROXY_REWRITER_BY_PROVIDER: Dict[str, BaseRewriter] = {
    "openai": OpenAIRewriter(),
    "anthropic": AnthropicRewriter(),
    "gemini": GeminiRewriter(),
}


def _get_proxy_rewriter(provider: str) -> BaseRewriter:
    return _PROXY_REWRITER_BY_PROVIDER.get(
        provider,
        _PROXY_REWRITER_BY_PROVIDER["openai"],
    )


class ConfigIn(BaseModel):
	maskConfig: Dict[str, bool]


class CallIn(BaseModel):
	text: str
	apiKeyFile: Optional[str] = None
	model: Optional[str] = None
	temperature: Optional[float] = 0.0


class MaskIn(BaseModel):
	text: str
	language: Optional[str] = None


class RestoreIn(BaseModel):
	text: str
	# maskMeta: base64 string of JSON(bytes) for placeholdersMap
	maskMeta: str


def parse_auth_header(auth: Optional[str]) -> Optional[str]:
    if not auth:
        return None
    s = auth.strip()
    if s.lower().startswith("bearer "):
        return s[7:].strip()
    return s


def _mask_auth_for_log(auth: Optional[str]) -> str:
    """Mask authorization header for safe logging."""
    if not auth:
        return "***"
    s = auth.strip()
    if s.lower().startswith("bearer "):
        return "Bearer ***"
    return "***"


def check_api_key(authorization: Optional[str] = Header(None)):
    if not API_KEY:
        return True
    token = parse_auth_header(authorization)
    if not token or not secrets.compare_digest(token, API_KEY):
        reason = "missing_token" if not token else "token_mismatch"
        logger.error("check_api_key unauthorized: reason=%s, auth=%s", reason, _mask_auth_for_log(authorization))
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


def _internal_error_response() -> Dict[str, Any]:
    return {"output": None, "error": {"message": "Internal server error", "code": None}}


# ============== Proxy Helper Functions ==============

def _is_json_content_type(content_type: Optional[str]) -> bool:
    return bool(content_type and "json" in content_type.lower())


def _normalize_mask_meta(mask_meta: Any) -> Any:
    if mask_meta is None:
        return None
    if isinstance(mask_meta, dict):
        metas = mask_meta.get("metas")
        if isinstance(metas, list) and len(metas) == 0:
            return None
        return mask_meta
    if isinstance(mask_meta, list):
        return mask_meta if len(mask_meta) > 0 else None
    return mask_meta


def _proxy_mask_text(text: str) -> Dict[str, Any]:
    return api.mask_text(text=text, language=None)


def _proxy_restore_text(text: str, mask_meta: Any) -> str:
    return api.restore_text(text=text, mask_meta=mask_meta)


def _encode_sse_event(event: SSEEvent) -> bytes:
    lines: List[str] = []
    if event.event is not None:
        lines.append(f"event: {event.event}")
    if event.id is not None:
        lines.append(f"id: {event.id}")
    if event.retry is not None:
        lines.append(f"retry: {event.retry}")
    for line in (event.data or "").split("\n"):
        lines.append(f"data: {line}")
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def _build_stream_flush_payload(provider: str, index: int, text: str) -> Dict[str, Any]:
    if provider == "anthropic":
        return {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        }
    if provider == "gemini":
        return {
            "candidates": [{"index": index, "content": {"parts": [{"text": text}]}}]
        }
    return {
        "id": "aifw-proxy-flush",
        "object": "chat.completion.chunk",
        "choices": [{"index": index, "delta": {"content": text}, "finish_reason": None}],
    }


def _build_stream_flush_events(provider: str, state: StreamState) -> List[SSEEvent]:
    events: List[SSEEvent] = []
    pending = state.flush_all()
    for index in sorted(pending.keys()):
        text = pending.get(index) or ""
        if not text:
            continue
        payload = _build_stream_flush_payload(provider, index, text)
        events.append(SSEEvent(data=json.dumps(payload, ensure_ascii=False)))
    return events


def _rewrite_openai_stream_payload(
    payload: Dict[str, Any], rewriter: BaseRewriter, mask_meta: Any, state: StreamState
) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False
    changed = False
    for position, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        index = choice.get("index")
        stream_index = index if isinstance(index, int) else position
        state.set_current_index(stream_index)

        # /v1/chat/completions stream: choices[].delta.content
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                restored = rewriter.restore_stream_delta(content, state, mask_meta, _proxy_restore_text)
                if restored != content:
                    changed = True
                delta["content"] = restored

        # /v1/completions stream: choices[].text
        text = choice.get("text")
        if isinstance(text, str):
            restored = rewriter.restore_stream_delta(text, state, mask_meta, _proxy_restore_text)
            if restored != text:
                changed = True
            choice["text"] = restored

    return changed


def _rewrite_anthropic_stream_payload(
    payload: Dict[str, Any], rewriter: BaseRewriter, mask_meta: Any, state: StreamState
) -> bool:
    delta = payload.get("delta")
    if not isinstance(delta, dict):
        return False
    text = delta.get("text")
    if not isinstance(text, str):
        return False
    index = payload.get("index")
    stream_index = index if isinstance(index, int) else 0
    state.set_current_index(stream_index)
    restored = rewriter.restore_stream_delta(text, state, mask_meta, _proxy_restore_text)
    changed = restored != text
    delta["text"] = restored
    return changed


def _rewrite_gemini_stream_payload(
    payload: Dict[str, Any], rewriter: BaseRewriter, mask_meta: Any, state: StreamState
) -> bool:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return False
    changed = False
    for candidate_pos, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        candidate_index = candidate.get("index")
        if not isinstance(candidate_index, int):
            candidate_index = candidate_pos
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part_pos, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if not isinstance(text, str):
                continue
            stream_index = candidate_index * 100000 + part_pos
            state.set_current_index(stream_index)
            restored = rewriter.restore_stream_delta(text, state, mask_meta, _proxy_restore_text)
            if restored != text:
                changed = True
            part["text"] = restored
    return changed


def _rewrite_stream_payload(
    provider: str, payload: Dict[str, Any], rewriter: BaseRewriter, mask_meta: Any, state: StreamState
) -> bool:
    if provider == "anthropic":
        return _rewrite_anthropic_stream_payload(payload, rewriter, mask_meta, state)
    if provider == "gemini":
        return _rewrite_gemini_stream_payload(payload, rewriter, mask_meta, state)
    return _rewrite_openai_stream_payload(payload, rewriter, mask_meta, state)


def _process_sse_event(
    provider: str, event: SSEEvent, rewriter: BaseRewriter, mask_meta: Any, state: StreamState
) -> List[bytes]:
    raw_data = (event.data or "").strip()
    if raw_data == "[DONE]":
        out: List[bytes] = []
        for flush_event in _build_stream_flush_events(provider, state):
            out.append(_encode_sse_event(flush_event))
        out.append(_encode_sse_event(event))
        return out
    try:
        payload = json.loads(event.data)
    except Exception:
        return [_encode_sse_event(event)]
    if not isinstance(payload, dict):
        return [_encode_sse_event(event)]
    changed = _rewrite_stream_payload(provider, payload, rewriter, mask_meta, state)
    if not changed:
        return [_encode_sse_event(event)]
    rewritten_event = SSEEvent(
        data=json.dumps(payload, ensure_ascii=False),
        event=event.event,
        id=event.id,
        retry=event.retry,
    )
    return [_encode_sse_event(rewritten_event)]


async def _iter_proxy_stream_response(
    upstream_response: Any, transport: ProxyTransport, provider: str, rewriter: BaseRewriter, mask_meta: Any
) -> AsyncIterator[bytes]:
    parser = SSEParser()
    state = StreamState()
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    try:
        if mask_meta is None:
            async for chunk in iter_response_bytes(upstream_response):
                yield chunk
            return
        async for chunk in iter_response_bytes(upstream_response):
            decoded = decoder.decode(chunk)
            if not decoded:
                continue
            for event in parser.feed(decoded):
                for output in _process_sse_event(provider, event, rewriter, mask_meta, state):
                    yield output
        decoded_tail = decoder.decode(b"", final=True)
        if decoded_tail:
            for event in parser.feed(decoded_tail):
                for output in _process_sse_event(provider, event, rewriter, mask_meta, state):
                    yield output
        for event in parser.flush():
            for output in _process_sse_event(provider, event, rewriter, mask_meta, state):
                yield output
        for flush_event in _build_stream_flush_events(provider, state):
            yield _encode_sse_event(flush_event)
    except StreamLimitError:
        logger.exception("proxy stream length exceeded: provider=%s", provider)
        payload = {"error": {"message": "stream size limit exceeded", "type": "proxy_error"}}
        yield _encode_sse_event(SSEEvent(data=json.dumps(payload, ensure_ascii=False)))
        yield _encode_sse_event(SSEEvent(data="[DONE]"))
    except Exception:
        logger.exception("proxy stream failed: provider=%s", provider)
        payload = {"error": {"message": "stream processing failed", "type": "proxy_error"}}
        yield _encode_sse_event(SSEEvent(data=json.dumps(payload, ensure_ascii=False)))
        yield _encode_sse_event(SSEEvent(data="[DONE]"))
    finally:
        await upstream_response.aclose()


async def _proxy_route_request(request: Request) -> Response:
    try:
        route_result = ProxyRouter(logger=logger).resolve(request.url.path)
    except UpstreamNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    rewriter = _get_proxy_rewriter(route_result.provider)
    request_body = await request.body()
    upstream_body = request_body
    mask_meta: Any = None

    if request_body and _is_json_content_type(request.headers.get("content-type")):
        try:
            request_json = json.loads(request_body.decode("utf-8"))
        except Exception:
            logger.warning("proxy request json parse failed: provider=%s path=%s", route_result.provider, request.url.path)
        else:
            if isinstance(request_json, dict):
                try:
                    rewritten_request_json, mask_meta = rewriter.mask_request(request_json, _proxy_mask_text)
                    mask_meta = _normalize_mask_meta(mask_meta)
                    upstream_body = json.dumps(rewritten_request_json, ensure_ascii=False).encode("utf-8")
                except Exception:
                    logger.exception("proxy request rewrite failed: provider=%s path=%s", route_result.provider, request.url.path)
                    raise HTTPException(status_code=500, detail="Request processing failed")

    # Preserve raw query string to avoid losing repeated keys
    upstream_url = route_result.upstream_url
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"

    transport = _get_proxy_transport()
    try:
        upstream_response = await transport.request(
            method=request.method,
            url=upstream_url,
            headers=request.headers,
            content=upstream_body if request_body else None,
            params=None,
            stream=True,
        )
    except httpx.TimeoutException:
        logger.warning("proxy upstream timeout: provider=%s path=%s", route_result.provider, request.url.path)
        raise HTTPException(status_code=504, detail="Gateway Timeout")
    except httpx.HTTPError:
        logger.exception("proxy upstream request failed: provider=%s path=%s", route_result.provider, request.url.path)
        raise HTTPException(status_code=502, detail="Bad Gateway")
    except Exception:
        logger.exception("proxy upstream unexpected error: provider=%s path=%s", route_result.provider, request.url.path)
        raise HTTPException(status_code=502, detail="Bad Gateway")

    response_headers = sanitize_upstream_response_headers(upstream_response.headers)
    if is_sse_response(upstream_response.headers):
        return StreamingResponse(
            _iter_proxy_stream_response(upstream_response, transport, route_result.provider, rewriter, mask_meta),
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    try:
        response_body = await upstream_response.aread()
    finally:
        await upstream_response.aclose()

    if mask_meta is not None and response_body and _is_json_content_type(upstream_response.headers.get("content-type")):
        try:
            response_json = json.loads(response_body.decode("utf-8"))
        except Exception:
            logger.warning("proxy response json parse failed: provider=%s path=%s", route_result.provider, request.url.path)
        else:
            if isinstance(response_json, dict):
                try:
                    restored_response_json = rewriter.restore_response(response_json, mask_meta, _proxy_restore_text)
                    response_body = json.dumps(restored_response_json, ensure_ascii=False).encode("utf-8")
                except Exception:
                    logger.exception("proxy response rewrite failed: provider=%s path=%s", route_result.provider, request.url.path)

    return Response(content=response_body, status_code=upstream_response.status_code, headers=response_headers)


@app.get("/api/health")
async def health():
	return {"status": "ok"}


@app.post("/api/config")
async def api_config(inp: ConfigIn, authorization: Optional[str] = Header(None)):
	check_api_key(authorization)
	try:
		api.config(mask_config=inp.maskConfig or {})
		return {"output": {"status": "ok"}, "error": None}
	except Exception:
		logger.exception("/api/config failed")
		return _internal_error_response()


@app.post("/api/call")
async def api_call(inp: CallIn, authorization: Optional[str] = Header(None)):
	check_api_key(authorization)
	default_key_file = os.environ.get("AIFW_API_KEY_FILE")
	chosen_key_file = inp.apiKeyFile or default_key_file
	# Server-side monthly log cleanup based on env config
	base_log = os.environ.get("AIFW_LOG_FILE")
	try:
		months = int(os.environ.get("AIFW_LOG_MONTHS_TO_KEEP", "6"))
	except Exception:
		months = 6
	cleanup_monthly_logs(base_log, months)
	try:
		out = api.call(
			text=inp.text,
			api_key_file=chosen_key_file,
			model=inp.model,
			temperature=inp.temperature or 0.0,
		)
		return {"output": {"text": out}, "error": None}
	except Exception:
		logger.exception("/api/call failed")
		return _internal_error_response()


@app.post("/api/mask_text")
async def api_mask_text(inp: MaskIn, authorization: Optional[str] = Header(None)):
	check_api_key(authorization)
	try:
		res = api.mask_text(text=inp.text, language=inp.language)
		return {"output": {"text": res["text"], "maskMeta": res["maskMeta"]}, "error": None}
	except Exception:
		logger.exception("/api/mask_text failed")
		return _internal_error_response()


@app.post("/api/restore_text")
async def api_restore_text(inp: RestoreIn, authorization: Optional[str] = Header(None)):
	check_api_key(authorization)
	try:
		restored = api.restore_text(text=inp.text, mask_meta=inp.maskMeta)
		return {"output": {"text": restored}, "error": None}
	except Exception:
		logger.exception("/api/restore_text failed")
		return _internal_error_response()


@app.post("/api/mask_text_batch")
async def api_mask_text_batch(inp_array: List[MaskIn], authorization: Optional[str] = Header(None)):
	check_api_key(authorization)
	try:
		res_array = []
		for inp in inp_array:
			res_array.append(api.mask_text(text=inp.text, language=inp.language))
		return {"output": res_array, "error": None}
	except Exception:
		logger.exception("/api/mask_text_batch failed")
		return _internal_error_response()


@app.post("/api/restore_text_batch")
async def api_restore_text_batch(inp_array: List[RestoreIn], authorization: Optional[str] = Header(None)):
	check_api_key(authorization)
	try:
		restored_array = []
		for inp in inp_array:
			restored = api.restore_text(text=inp.text, mask_meta=inp.maskMeta)
			restored_array.append({"text": restored})
		return {"output": restored_array, "error": None}
	except Exception:
		logger.exception("/api/restore_text_batch failed")
		return _internal_error_response()


# ============== Proxy Routes ==============

@app.api_route("/v1", methods=_PROXY_METHODS)
@app.api_route("/v1/{path:path}", methods=_PROXY_METHODS)
async def proxy_v1(request: Request, path: str = "", authorization: Optional[str] = Header(None)):
    check_api_key(authorization)
    return await _proxy_route_request(request)


@app.api_route("/v1beta", methods=_PROXY_METHODS)
@app.api_route("/v1beta/{path:path}", methods=_PROXY_METHODS)
async def proxy_v1beta(request: Request, path: str = "", authorization: Optional[str] = Header(None)):
    check_api_key(authorization)
    return await _proxy_route_request(request)