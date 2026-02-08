from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

OPENAI_BASE_URL_ENV = "AIFW_UPSTREAM_OPENAI_BASE_URL"
ANTHROPIC_BASE_URL_ENV = "AIFW_UPSTREAM_ANTHROPIC_BASE_URL"
GEMINI_BASE_URL_ENV = "AIFW_UPSTREAM_GEMINI_BASE_URL"

_OPENAI_PROVIDER = "openai"
_ANTHROPIC_PROVIDER = "anthropic"
_GEMINI_PROVIDER = "gemini"

_OPENAI_EXACT_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/models",
}
_ANTHROPIC_EXACT_PATHS = {"/v1/messages", "/v1/complete"}


@dataclass(frozen=True)
class RouteResult:
    provider: str
    base_url: str
    upstream_url: str


class UpstreamNotConfiguredError(RuntimeError):
    status_code = 503

    def __init__(self, provider: str):
        super().__init__(f"upstream not configured for provider: {provider}")
        self.provider = provider


class ProxyRouter:
    def __init__(
        self,
        environ: Optional[Mapping[str, str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._environ = environ if environ is not None else os.environ
        self._logger = logger if logger is not None else logging.getLogger(__name__)

    def resolve(self, path: str) -> RouteResult:
        normalized_path = self._normalize_path(path)
        provider, fallback_to_openai = self._resolve_provider(normalized_path)
        base_url = self._get_base_url(provider)

        if not base_url:
            raise UpstreamNotConfiguredError(provider)

        if fallback_to_openai:
            self._logger.warning(
                "proxy route fallback to OpenAI upstream: path=%s",
                normalized_path,
            )

        return RouteResult(
            provider=provider,
            base_url=base_url,
            upstream_url=self._join_base_and_path(base_url, normalized_path),
        )

    def _get_base_url(self, provider: str) -> str:
        env_name = self._provider_env_name(provider)
        return self._environ.get(env_name, "").strip().rstrip("/")

    @staticmethod
    def _provider_env_name(provider: str) -> str:
        if provider == _OPENAI_PROVIDER:
            return OPENAI_BASE_URL_ENV
        if provider == _ANTHROPIC_PROVIDER:
            return ANTHROPIC_BASE_URL_ENV
        if provider == _GEMINI_PROVIDER:
            return GEMINI_BASE_URL_ENV
        return OPENAI_BASE_URL_ENV

    @staticmethod
    def _resolve_provider(path: str) -> Tuple[str, bool]:
        if path == "/v1beta" or path.startswith("/v1beta/"):
            return _GEMINI_PROVIDER, False

        if path in _ANTHROPIC_EXACT_PATHS:
            return _ANTHROPIC_PROVIDER, False

        if path in _OPENAI_EXACT_PATHS:
            return _OPENAI_PROVIDER, False

        if path == "/v1" or path.startswith("/v1/"):
            return _OPENAI_PROVIDER, True

        return _OPENAI_PROVIDER, False

    @staticmethod
    def _normalize_path(path: str) -> str:
        raw = (path or "").split("?", 1)[0]
        if not raw.startswith("/"):
            raw = f"/{raw}"

        if len(raw) > 1:
            raw = raw.rstrip("/")
            if not raw:
                raw = "/"
        return raw

    @staticmethod
    def _join_base_and_path(base_url: str, path: str) -> str:
        return f"{base_url.rstrip('/')}{path}"


def resolve_upstream(
    path: str,
    environ: Optional[Mapping[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> RouteResult:
    return ProxyRouter(environ=environ, logger=logger).resolve(path)
