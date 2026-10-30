"""Judge backends: a local OpenAI-compatible VLM and the sheet-file exchange for hosted judges.

``SheetFileBackend`` is how a hosted model (for example a Haiku subagent that can view image
files) acts as the judge without an API key in this process: ``judge`` writes
``pending/<sheet_id>.png`` and ``pending/<sheet_id>.prompt.txt``, returns None, and a later run
reads ``verdicts/<sheet_id>.json`` that the judge wrote.
"""

from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING, Any

import httpx

from scripts.research.judge_protocol import JudgeRequest, PrivacyRefusalError, RawReply
from src.services.labeling.vlm_client import extract_message_content
from src.services.labeling.vlm_url_policy import compute_locality, sends_images_externally


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _finish_reason(doc: dict[str, Any]) -> str | None:
    try:
        return str(doc['choices'][0]['finish_reason'])
    except (KeyError, IndexError, TypeError):
        return None


class LocalVlmBackend:
    """A VLM served by this stack's vLLM (or any local OpenAI-compatible server)."""

    hosted = False
    name = 'local-vlm'

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        api_key: str = '',
        timeout_s: float = 300.0,
        max_tokens: int = 2048,
        post: Callable[..., Any] = httpx.post,
    ) -> None:
        if sends_images_externally(compute_locality(base_url)):
            raise PrivacyRefusalError(
                f'{base_url} is an external endpoint; local judges stay local'
            )
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        self._timeout = timeout_s
        self._max_tokens = max_tokens
        self._post = post

    def judge(self, request: JudgeRequest) -> RawReply:
        image = base64.b64encode(request.png).decode('ascii')
        body = {
            'model': self.model,
            'temperature': 0,
            'max_tokens': self._max_tokens,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {'url': f'data:image/png;base64,{image}'},
                        },
                        {'type': 'text', 'text': request.prompt},
                    ],
                }
            ],
        }
        started = time.monotonic()
        resp = self._post(
            f'{self.base_url}/chat/completions',
            json=body,
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        doc = resp.json()
        usage = doc.get('usage') or {}
        return RawReply(
            text=extract_message_content(doc),
            input_tokens=usage.get('prompt_tokens'),
            output_tokens=usage.get('completion_tokens'),
            latency_ms=(time.monotonic() - started) * 1000,
            request_id=resp.headers.get('x-request-id'),
            truncated=_finish_reason(doc) == 'length',
        )


class SheetFileBackend:
    """Hosted judge through files: export sheets, let the judge answer, read the JSON back."""

    hosted = True
    name = 'sheet-file'

    def __init__(self, exchange_dir: Path, *, model: str) -> None:
        self.model = model
        self._pending = exchange_dir / 'pending'
        self._verdicts = exchange_dir / 'verdicts'

    def judge(self, request: JudgeRequest) -> RawReply | None:
        sheet_id = request.manifest['sheet_id']
        answer = self._verdicts / f'{sheet_id}.json'
        if answer.is_file():
            return RawReply(text=answer.read_text(), request_id=f'file:{sheet_id}')
        self._pending.mkdir(parents=True, exist_ok=True)
        self._verdicts.mkdir(parents=True, exist_ok=True)
        (self._pending / f'{sheet_id}.png').write_bytes(request.png)
        (self._pending / f'{sheet_id}.prompt.txt').write_text(request.prompt + '\n')
        return None
