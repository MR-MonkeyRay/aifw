from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MAX_STREAM_CHARS_ENV = "AIFW_PROXY_MAX_STREAM_CHARS"
DEFAULT_MAX_STREAM_CHARS = 1_000_000

PLACEHOLDER_PREFIX = "__PII_"
PLACEHOLDER_PATTERN = re.compile(r"__PII_[A-Z_]+_\d+__")
PLACEHOLDER_PREFIX_CANDIDATE_PATTERN = re.compile(r"__PII_[A-Z0-9_]*$")


def get_max_stream_chars() -> int:
    raw = os.environ.get(MAX_STREAM_CHARS_ENV, str(DEFAULT_MAX_STREAM_CHARS))
    try:
        value = int(raw)
        if value <= 0:
            return DEFAULT_MAX_STREAM_CHARS
        return value
    except (TypeError, ValueError):
        return DEFAULT_MAX_STREAM_CHARS


class StreamLimitError(RuntimeError):
    pass


@dataclass
class SSEEvent:
    data: str
    event: Optional[str] = None
    id: Optional[str] = None
    retry: Optional[int] = None


class SSEParser:
    def __init__(self):
        self._buffer = ""

    def feed(self, chunk: str) -> List[SSEEvent]:
        if not chunk:
            return []

        normalized = chunk.replace("\r\n", "\n").replace("\r", "\n")
        self._buffer += normalized

        events: List[SSEEvent] = []
        while True:
            boundary = self._buffer.find("\n\n")
            if boundary < 0:
                break
            block = self._buffer[:boundary]
            self._buffer = self._buffer[boundary + 2 :]
            event = self._parse_block(block)
            if event is not None:
                events.append(event)
        return events

    def flush(self) -> List[SSEEvent]:
        if not self._buffer:
            return []

        block = self._buffer
        self._buffer = ""
        event = self._parse_block(block)
        if event is None:
            return []
        return [event]

    @staticmethod
    def _parse_block(block: str) -> Optional[SSEEvent]:
        data_lines: List[str] = []
        event_name: Optional[str] = None
        event_id: Optional[str] = None
        retry: Optional[int] = None
        has_non_comment = False

        for line in block.split("\n"):
            if not line:
                continue
            if line.startswith(":"):
                continue

            has_non_comment = True
            if ":" in line:
                field, value = line.split(":", 1)
                if value.startswith(" "):
                    value = value[1:]
            else:
                field, value = line, ""

            if field == "data":
                data_lines.append(value)
            elif field == "event":
                event_name = value
            elif field == "id":
                event_id = value
            elif field == "retry":
                try:
                    retry = max(0, int(value))
                except ValueError:
                    continue

        if not has_non_comment:
            return None

        return SSEEvent(
            data="\n".join(data_lines),
            event=event_name,
            id=event_id,
            retry=retry,
        )


def contains_placeholder(text: str) -> bool:
    return PLACEHOLDER_PATTERN.search(text) is not None


def has_unclosed_placeholder_prefix(text: str) -> bool:
    return safe_prefix_length(text) < len(text)


def _find_trailing_prefix_fragment_start(text: str) -> int:
    """
    Find start of trailing fragment that could be start of __PII_.

    Only consider it a fragment if it's NOT part of a complete placeholder ending.
    E.g., "hello__" at end could be start of "__PII_", but "__PII_EMAIL_01__" ending
    with "__" is complete and should NOT be treated as a fragment.
    """
    # Check if text ends with a complete placeholder - if so, no fragment
    if PLACEHOLDER_PATTERN.search(text):
        # Find the last complete placeholder
        last_match = None
        for match in PLACEHOLDER_PATTERN.finditer(text):
            last_match = match
        if last_match and last_match.end() == len(text):
            # Text ends with a complete placeholder, no trailing fragment
            return len(text)

    # Look for trailing fragments that could be start of __PII_
    max_fragment_len = min(len(text), len(PLACEHOLDER_PREFIX) - 1)
    for fragment_len in range(max_fragment_len, 0, -1):
        fragment = PLACEHOLDER_PREFIX[:fragment_len]
        if text.endswith(fragment):
            # Make sure this isn't the end of a complete placeholder
            # by checking if there's a complete placeholder ending right before
            potential_start = len(text) - fragment_len
            prefix = text[:potential_start]
            # If prefix ends with "__" from a placeholder, this fragment is actually
            # the start of a new placeholder, not part of the old one
            if prefix and prefix.endswith("__"):
                # Check if there's a complete placeholder ending at prefix
                last_match = None
                for match in PLACEHOLDER_PATTERN.finditer(prefix):
                    last_match = match
                if last_match and last_match.end() == len(prefix):
                    # The "__" is part of a complete placeholder, so the trailing
                    # fragment is indeed a new potential placeholder start
                    return potential_start
            return potential_start
    return len(text)


def safe_prefix_length(text: str) -> int:
    if not text:
        return 0

    unsafe_start = len(text)

    start = text.find(PLACEHOLDER_PREFIX)
    while start >= 0:
        suffix = text[start:]
        full_match = PLACEHOLDER_PATTERN.match(suffix)
        if full_match is None and PLACEHOLDER_PREFIX_CANDIDATE_PATTERN.fullmatch(suffix):
            unsafe_start = min(unsafe_start, start)
        start = text.find(PLACEHOLDER_PREFIX, start + 1)

    trailing_fragment_start = _find_trailing_prefix_fragment_start(text)
    unsafe_start = min(unsafe_start, trailing_fragment_start)

    return unsafe_start


@dataclass
class StreamState:
    max_chars: int = field(default_factory=get_max_stream_chars)
    total_chars: int = 0
    buffers: Dict[int, str] = field(default_factory=dict)
    current_index: int = 0

    def set_current_index(self, index: int) -> None:
        self.current_index = index

    def append(self, index: int, text: str) -> None:
        if not text:
            return

        next_total = self.total_chars + len(text)
        if next_total > self.max_chars:
            raise StreamLimitError(
                f"stream length exceeded limit ({self.max_chars}), "
                f"env={MAX_STREAM_CHARS_ENV}"
            )

        self.total_chars = next_total
        self.buffers[index] = self.buffers.get(index, "") + text

    def pop_safe_prefix(self, index: int) -> str:
        content = self.buffers.get(index, "")
        if not content:
            return ""

        safe_len = safe_prefix_length(content)
        if safe_len <= 0:
            return ""

        output = content[:safe_len]
        self.buffers[index] = content[safe_len:]
        return output

    def flush(self, index: int) -> str:
        return self.buffers.pop(index, "")

    def flush_all(self) -> Dict[int, str]:
        data = dict(self.buffers)
        self.buffers.clear()
        return data
