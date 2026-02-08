from __future__ import annotations

import importlib
import os
from typing import Any, AsyncIterator, Dict, Mapping, Optional, Set

PROXY_TIMEOUT_SECS_ENV = "AIFW_PROXY_TIMEOUT_SECS"
DEFAULT_PROXY_TIMEOUT_SECS = 300.0

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def get_proxy_timeout_secs(environ: Optional[Mapping[str, str]] = None) -> float:
    source = environ if environ is not None else os.environ
    raw = source.get(PROXY_TIMEOUT_SECS_ENV, str(int(DEFAULT_PROXY_TIMEOUT_SECS)))
    try:
        value = float(raw)
        if value <= 0:
            return DEFAULT_PROXY_TIMEOUT_SECS
        return value
    except (TypeError, ValueError):
        return DEFAULT_PROXY_TIMEOUT_SECS


def strip_hop_by_hop_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    connection_tokens: Set[str] = set()
    for name, value in headers.items():
        lower_name = str(name).lower()
        if lower_name != "connection":
            continue
        for token in str(value).split(","):
            normalized = token.strip().lower()
            if normalized:
                connection_tokens.add(normalized)

    filtered: Dict[str, str] = {}
    for name, value in headers.items():
        header_name = str(name)
        lower_name = header_name.lower()
        if lower_name in _HOP_BY_HOP_HEADERS:
            continue
        if lower_name in connection_tokens:
            continue
        if lower_name.startswith("proxy-"):
            continue
        if lower_name == "host":
            continue
        if lower_name == "content-length":
            continue
        filtered[header_name] = str(value)
    return filtered


def sanitize_upstream_response_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    return strip_hop_by_hop_headers(headers)


def is_sse_response(headers: Mapping[str, str]) -> bool:
    content_type = ""
    for key, value in headers.items():
        if str(key).lower() == "content-type":
            content_type = str(value).lower()
            break
    return "text/event-stream" in content_type


def _load_httpx() -> Any:
    return importlib.import_module("httpx")


class ProxyTransport:
    def __init__(self, timeout_secs: Optional[float] = None):
        self._timeout_secs = (
            timeout_secs if timeout_secs is not None else get_proxy_timeout_secs()
        )
        self._client: Optional[Any] = None

    async def __aenter__(self) -> "ProxyTransport":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    async def _get_client(self) -> Any:
        if self._client is None:
            httpx = _load_httpx()
            self._client = httpx.AsyncClient(
                timeout=self._timeout_secs,
                follow_redirects=False,
            )
        return self._client

    async def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        *,
        content: Optional[bytes] = None,
        params: Optional[Mapping[str, str]] = None,
        stream: bool = False,
    ) -> Any:
        client = await self._get_client()
        forwarded_headers = strip_hop_by_hop_headers(headers)
        request = client.build_request(
            method=method.upper(),
            url=url,
            headers=forwarded_headers,
            params=params,
            content=content,
        )
        return await client.send(request, stream=stream)

    async def aclose(self) -> None:
        if self._client is None:
            return
        await self._client.aclose()
        self._client = None


async def iter_response_bytes(
    response: Any,
    chunk_size: int = 8192,
) -> AsyncIterator[bytes]:
    async for chunk in response.aiter_bytes(chunk_size):
        if chunk:
            yield chunk
