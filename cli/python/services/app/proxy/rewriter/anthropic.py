from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .base import BaseRewriter, MaskFn, RestoreFn


class AnthropicRewriter(BaseRewriter):
    def mask_request(self, body: Dict[str, Any], mask_fn: MaskFn) -> Tuple[Dict[str, Any], Any]:
        masked_body = self.clone_body(body)
        mask_metas: List[Any] = []

        system_value = masked_body.get("system")
        if isinstance(system_value, str):
            masked_body["system"] = self._mask_text(system_value, mask_fn, mask_metas)
        elif isinstance(system_value, list):
            self._mask_text_blocks(system_value, mask_fn, mask_metas)

        messages = masked_body.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    message["content"] = self._mask_text(content, mask_fn, mask_metas)
                elif isinstance(content, list):
                    self._mask_text_blocks(content, mask_fn, mask_metas)

        return masked_body, self._pack_mask_meta(mask_metas)

    def restore_response(
        self,
        body: Dict[str, Any],
        mask_meta: Any,
        restore_fn: RestoreFn,
    ) -> Dict[str, Any]:
        restored_body = self.clone_body(body)
        content = restored_body.get("content")
        if isinstance(content, list):
            self._restore_text_blocks(content, mask_meta, restore_fn)
        return restored_body

    def _mask_text(self, text: str, mask_fn: MaskFn, mask_metas: List[Any]) -> str:
        masked_text, single_meta = self._apply_mask_fn(mask_fn, text)
        if single_meta is not None:
            mask_metas.append(single_meta)
        return masked_text

    def _mask_text_blocks(
        self,
        blocks: List[Any],
        mask_fn: MaskFn,
        mask_metas: List[Any],
    ) -> None:
        for block in blocks:
            if not isinstance(block, dict):
                continue
            text_value = block.get("text")
            if isinstance(text_value, str):
                block["text"] = self._mask_text(text_value, mask_fn, mask_metas)

    def _restore_text_blocks(
        self,
        blocks: List[Any],
        mask_meta: Any,
        restore_fn: RestoreFn,
    ) -> None:
        for block in blocks:
            if not isinstance(block, dict):
                continue
            text_value = block.get("text")
            if isinstance(text_value, str):
                block["text"] = self._restore_text(text_value, mask_meta, restore_fn)
