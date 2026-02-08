from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple, Union

from ..streaming import StreamState

MaskFnResult = Union[Tuple[str, Any], Mapping[str, Any]]
MaskFn = Callable[[str], MaskFnResult]
RestoreFn = Callable[[str, Any], str]


class BaseRewriter(ABC):
    @abstractmethod
    def mask_request(self, body: Dict[str, Any], mask_fn: MaskFn) -> Tuple[Dict[str, Any], Any]:
        raise NotImplementedError

    @abstractmethod
    def restore_response(
        self,
        body: Dict[str, Any],
        mask_meta: Any,
        restore_fn: RestoreFn,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def restore_stream_delta(
        self,
        delta: str,
        state: StreamState,
        mask_meta: Any,
        restore_fn: RestoreFn,
    ) -> str:
        """
        Restore stream delta with proper handling of cross-chunk placeholders.

        Algorithm:
        1. Accumulate masked text (delta) into buffer
        2. Get safe prefix from accumulated masked text (no half placeholder)
        3. Restore the safe prefix
        4. Output the new restored portion (incremental)
        """
        index = getattr(state, "current_index", 0)
        # Accumulate the raw masked delta (NOT restored yet)
        state.append(index, delta)
        # Get safe prefix of accumulated masked text
        safe_masked = state.pop_safe_prefix(index)
        if not safe_masked:
            return ""
        # Restore the safe masked prefix
        restored = self._restore_text(safe_masked, mask_meta, restore_fn)
        return restored

    @staticmethod
    def clone_body(body: Dict[str, Any]) -> Dict[str, Any]:
        return deepcopy(body)

    @staticmethod
    def _apply_mask_fn(mask_fn: MaskFn, text: str) -> Tuple[str, Any]:
        result = mask_fn(text)
        if isinstance(result, tuple) and len(result) == 2:
            return str(result[0]), result[1]
        if isinstance(result, Mapping):
            masked_text = result.get("text", text)
            return str(masked_text), result.get("maskMeta")
        raise TypeError("mask_fn must return (masked_text, mask_meta) or {'text', 'maskMeta'}")

    @classmethod
    def _restore_text(cls, text: str, mask_meta: Any, restore_fn: RestoreFn) -> str:
        restored = text
        for single_meta in cls._iter_mask_meta(mask_meta):
            if single_meta is None:
                continue
            restored = restore_fn(restored, single_meta)
        return restored

    @staticmethod
    def _pack_mask_meta(mask_metas: List[Any]) -> Dict[str, List[Any]]:
        return {"metas": mask_metas}

    @staticmethod
    def _iter_mask_meta(mask_meta: Any) -> Iterable[Any]:
        if mask_meta is None:
            return []
        if isinstance(mask_meta, Mapping):
            metas = mask_meta.get("metas")
            if isinstance(metas, list):
                return metas
        if isinstance(mask_meta, list):
            return mask_meta
        return [mask_meta]
