from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .base import BaseRewriter, MaskFn, RestoreFn


class GeminiRewriter(BaseRewriter):
    def mask_request(self, body: Dict[str, Any], mask_fn: MaskFn) -> Tuple[Dict[str, Any], Any]:
        masked_body = self.clone_body(body)
        mask_metas: List[Any] = []

        contents = masked_body.get("contents")
        if isinstance(contents, list):
            for item in contents:
                if not isinstance(item, dict):
                    continue
                parts = item.get("parts")
                if isinstance(parts, list):
                    self._mask_parts(parts, mask_fn, mask_metas)

        system_instruction = masked_body.get("systemInstruction")
        if isinstance(system_instruction, dict):
            system_parts = system_instruction.get("parts")
            if isinstance(system_parts, list):
                self._mask_parts(system_parts, mask_fn, mask_metas)

        return masked_body, self._pack_mask_meta(mask_metas)

    def restore_response(
        self,
        body: Dict[str, Any],
        mask_meta: Any,
        restore_fn: RestoreFn,
    ) -> Dict[str, Any]:
        restored_body = self.clone_body(body)
        candidates = restored_body.get("candidates")
        if not isinstance(candidates, list):
            return restored_body

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if isinstance(parts, list):
                self._restore_parts(parts, mask_meta, restore_fn)

        return restored_body

    def _mask_text(self, text: str, mask_fn: MaskFn, mask_metas: List[Any]) -> str:
        masked_text, single_meta = self._apply_mask_fn(mask_fn, text)
        if single_meta is not None:
            mask_metas.append(single_meta)
        return masked_text

    def _mask_parts(self, parts: List[Any], mask_fn: MaskFn, mask_metas: List[Any]) -> None:
        for part in parts:
            if not isinstance(part, dict):
                continue
            text_value = part.get("text")
            if isinstance(text_value, str):
                part["text"] = self._mask_text(text_value, mask_fn, mask_metas)

    def _restore_parts(self, parts: List[Any], mask_meta: Any, restore_fn: RestoreFn) -> None:
        for part in parts:
            if not isinstance(part, dict):
                continue
            text_value = part.get("text")
            if isinstance(text_value, str):
                part["text"] = self._restore_text(text_value, mask_meta, restore_fn)
