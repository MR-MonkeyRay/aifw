"""Transparent proxy building blocks."""

from .router import (
    ANTHROPIC_BASE_URL_ENV,
    GEMINI_BASE_URL_ENV,
    OPENAI_BASE_URL_ENV,
    ProxyRouter,
    RouteResult,
    UpstreamNotConfiguredError,
    resolve_upstream,
)
from .streaming import SSEEvent, SSEParser, StreamLimitError, StreamState
from .transport import ProxyTransport, get_proxy_timeout_secs
from .rewriter import AnthropicRewriter, BaseRewriter, GeminiRewriter, OpenAIRewriter

__all__ = [
    "ANTHROPIC_BASE_URL_ENV",
    "GEMINI_BASE_URL_ENV",
    "OPENAI_BASE_URL_ENV",
    "ProxyRouter",
    "RouteResult",
    "UpstreamNotConfiguredError",
    "resolve_upstream",
    "SSEEvent",
    "SSEParser",
    "StreamLimitError",
    "StreamState",
    "ProxyTransport",
    "get_proxy_timeout_secs",
    "BaseRewriter",
    "OpenAIRewriter",
    "AnthropicRewriter",
    "GeminiRewriter",
]
