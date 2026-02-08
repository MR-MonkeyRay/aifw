#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for transparent proxy:
- OpenAI /v1/chat/completions non-stream: mask/restore roundtrip
- OpenAI /v1/chat/completions stream: no half placeholder in output
"""

import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _load_apps():
    _ensure_repo_root_on_path()
    fake_echo = importlib.import_module("cli.python.services.fake_llm.echo_server")
    proxy_main = importlib.import_module("cli.python.services.app.main")
    return fake_echo, proxy_main


class RecordingASGITransport(httpx.AsyncBaseTransport):
    """ASGI transport that records all requests for inspection."""

    def __init__(self, app: Any):
        self._transport = ASGITransport(app=app)
        self.requests: List[Dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        self.requests.append({
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "headers": dict(request.headers),
            "body": body,
        })
        forwarded_request = httpx.Request(
            method=request.method,
            url=request.url,
            headers=request.headers,
            content=body,
        )
        return await self._transport.handle_async_request(forwarded_request)

    async def aclose(self) -> None:
        await self._transport.aclose()


def _to_text(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _iter_sse_data_lines(response) -> List[str]:
    data_lines: List[str] = []
    for raw_line in response.iter_lines():
        line = _to_text(raw_line).strip()
        if not line or not line.startswith("data:"):
            continue
        data_lines.append(line[5:].strip())
    return data_lines


@pytest.fixture
def fake_upstream_url() -> str:
    return "http://fake-upstream"


@pytest.fixture
def proxy_env(monkeypatch: pytest.MonkeyPatch, fake_upstream_url: str):
    fake_echo, proxy_main = _load_apps()

    recorder = RecordingASGITransport(fake_echo.app)

    async def _patched_get_client(self):
        if self._client is None:
            self._client = AsyncClient(
                transport=recorder,
                base_url=fake_upstream_url,
                timeout=self._timeout_secs,
                follow_redirects=False,
            )
        return self._client

    monkeypatch.setenv("AIFW_UPSTREAM_OPENAI_BASE_URL", fake_upstream_url)
    monkeypatch.setattr(proxy_main.ProxyTransport, "_get_client", _patched_get_client)

    try:
        proxy_main.api.config({"maskAll": True})
    except Exception:
        pass

    client = TestClient(proxy_main.app)
    try:
        yield client, recorder
    finally:
        client.close()


@pytest.fixture
def proxy_client(proxy_env):
    client, _ = proxy_env
    return client


@pytest.fixture
def upstream_recorder(proxy_env):
    _, recorder = proxy_env
    return recorder


def test_openai_chat_completions_non_stream_mask_restore(
    proxy_client: TestClient, upstream_recorder: RecordingASGITransport
):
    """
    Test non-stream /v1/chat/completions:
    - Upstream receives masked text (contains __PII_)
    - Client receives restored text (contains original PII)
    """
    original_text = "请联系我：alice@example.com，电话：+1-202-555-0188。"
    payload = {
        "model": "echo-001",
        "messages": [{"role": "user", "content": original_text}],
        "stream": False,
    }

    response = proxy_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    body = response.json()
    restored_text = body["choices"][0]["message"]["content"]

    # Client should receive restored text with original PII
    assert "alice@example.com" in restored_text
    assert "+1-202-555-0188" in restored_text
    assert "__PII_" not in restored_text

    # Upstream should have received masked text
    upstream_calls = [r for r in upstream_recorder.requests if r["path"] == "/v1/chat/completions"]
    assert upstream_calls, "upstream /v1/chat/completions should be called"

    upstream_json = json.loads(upstream_calls[-1]["body"].decode("utf-8"))
    upstream_text = upstream_json["messages"][-1]["content"]

    assert "__PII_" in upstream_text
    assert "alice@example.com" not in upstream_text


def test_openai_chat_completions_stream_no_half_placeholder(proxy_client: TestClient):
    """
    Test stream /v1/chat/completions:
    - No half placeholder in streamed output
    - Final concatenated text equals original
    """
    original_text = "请联系我：alice@example.com，电话：+1-202-555-0188。"
    payload = {
        "model": "echo-001",
        "messages": [{"role": "user", "content": original_text}],
        "stream": True,
    }

    streamed_chunks: List[str] = []
    # Pattern to detect half placeholder like "EMAIL_01__" without "__PII_" prefix
    half_placeholder_re = re.compile(r"^[A-Z_]+_\d+__")

    with proxy_client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        for data in _iter_sse_data_lines(response):
            if data == "[DONE]":
                break

            event = json.loads(data)
            choices = event.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if not piece:
                continue

            # No half placeholder should appear
            assert "__PII_" not in piece, f"Half placeholder found: {piece}"
            assert half_placeholder_re.search(piece) is None, f"Half placeholder suffix found: {piece}"
            streamed_chunks.append(piece)

    restored_text = "".join(streamed_chunks)
    assert restored_text == original_text


def test_openai_completions_stream_no_half_placeholder(proxy_client: TestClient):
    """
    Test stream /v1/completions:
    - No half placeholder in streamed output
    - Final concatenated text equals original
    """
    original_text = "请联系我：alice@example.com，电话：+1-202-555-0188。"
    payload = {
        "model": "echo-001",
        "prompt": original_text,
        "stream": True,
    }

    streamed_chunks: List[str] = []
    half_placeholder_re = re.compile(r"^[A-Z_]+_\d+__")

    with proxy_client.stream("POST", "/v1/completions", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        for data in _iter_sse_data_lines(response):
            if data == "[DONE]":
                break

            event = json.loads(data)
            choices = event.get("choices") or []
            if not choices:
                continue

            # /v1/completions uses choices[].text instead of delta.content
            piece = choices[0].get("text")
            if not piece:
                continue

            # No half placeholder should appear
            assert "__PII_" not in piece, f"Half placeholder found: {piece}"
            assert half_placeholder_re.search(piece) is None, f"Half placeholder suffix found: {piece}"
            streamed_chunks.append(piece)

    restored_text = "".join(streamed_chunks)
    assert restored_text == original_text


def test_openai_completions_non_stream_mask_restore(
    proxy_client: TestClient, upstream_recorder: RecordingASGITransport
):
    """
    Test non-stream /v1/completions:
    - Upstream receives masked text (contains __PII_)
    - Client receives restored text (contains original PII)
    """
    original_text = "请联系我：alice@example.com，电话：+1-202-555-0188。"
    payload = {
        "model": "echo-001",
        "prompt": original_text,
        "stream": False,
    }

    response = proxy_client.post("/v1/completions", json=payload)
    assert response.status_code == 200

    body = response.json()
    restored_text = body["choices"][0]["text"]

    # Client should receive restored text with original PII
    assert "alice@example.com" in restored_text
    assert "+1-202-555-0188" in restored_text
    assert "__PII_" not in restored_text

    # Upstream should have received masked text
    upstream_calls = [r for r in upstream_recorder.requests if r["path"] == "/v1/completions"]
    assert upstream_calls, "upstream /v1/completions should be called"

    upstream_json = json.loads(upstream_calls[-1]["body"].decode("utf-8"))
    upstream_text = upstream_json["prompt"]

    assert "__PII_" in upstream_text
    assert "alice@example.com" not in upstream_text
