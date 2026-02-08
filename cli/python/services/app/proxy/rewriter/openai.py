from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .base import BaseRewriter, MaskFn, RestoreFn


class OpenAIRewriter(BaseRewriter):
    def mask_request(self, body: Dict[str, Any], mask_fn: MaskFn) -> Tuple[Dict[str, Any], Any]:
        masked_body = self.clone_body(body)
        mask_metas: List[Any] = []

        # /v1/chat/completions: messages[].content
        messages = masked_body.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    message["content"] = self._mask_text(content, mask_fn, mask_metas)
                elif isinstance(content, list):
                    self._mask_parts(content, mask_fn, mask_metas)

        # /v1/completions: prompt
        prompt = masked_body.get("prompt")
        if isinstance(prompt, str):
            masked_body["prompt"] = self._mask_text(prompt, mask_fn, mask_metas)

        return masked_body, self._pack_mask_meta(mask_metas)

    def restore_response(
        self,
        body: Dict[str, Any],
        mask_meta: Any,
        restore_fn: RestoreFn,
    ) -> Dict[str, Any]:
        restored_body = self.clone_body(body)
        choices = restored_body.get("choices")
        if not isinstance(choices, list):
            return restored_body

        for choice in choices:
            if not isinstance(choice, dict):
                continue

            # /v1/chat/completions: choices[].message.content
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    message["content"] = self._restore_text(content, mask_meta, restore_fn)
                elif isinstance(content, list):
                    self._restore_parts(content, mask_meta, restore_fn)

            # /v1/chat/completions stream: choices[].delta.content
            delta = choice.get("delta")
            if isinstance(delta, dict):
                delta_content = delta.get("content")
                if isinstance(delta_content, str):
                    delta["content"] = self._restore_text(delta_content, mask_meta, restore_fn)
                elif isinstance(delta_content, list):
                    self._restore_parts(delta_content, mask_meta, restore_fn)

            # /v1/completions: choices[].text
            text = choice.get("text")
            if isinstance(text, str):
                choice["text"] = self._restore_text(text, mask_meta, restore_fn)

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

            input_text = part.get("input_text")
            if isinstance(input_text, str):
                part["input_text"] = self._mask_text(input_text, mask_fn, mask_metas)

    def _restore_parts(self, parts: List[Any], mask_meta: Any, restore_fn: RestoreFn) -> None:
        for part in parts:
            if not isinstance(part, dict):
                continue

            text_value = part.get("text")
            if isinstance(text_value, str):
                part["text"] = self._restore_text(text_value, mask_meta, restore_fn)

            input_text = part.get("input_text")
            if isinstance(input_text, str):
                part["input_text"] = self._restore_text(input_text, mask_meta, restore_fn)
