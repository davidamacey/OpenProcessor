"""The VLM endpoint probe (W9.4): does this endpoint list its model, see
images, honour ``response_format``, and accept its own image cap?

Never run implicitly (only ``?probe=true`` and ``POST .../probe`` call it).
It sends **only synthetic images** (solid-colour 64x64 JPEGs made in
memory), never user data, so probing an external endpoint needs no
acknowledgement. Each request has a 10 s timeout (the image calls
``min(timeout_s, 60)``), redirects are never followed, and the auth header
is built in memory and never logged or echoed (an upstream message that
happens to contain the key has it masked).
"""

from __future__ import annotations

import io
import json
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from PIL import Image

from src.core.logging import get_logger
from src.services.labeling.vlm_client import (
    build_auth_headers,
    extract_message_content,
    extract_reasoning_content,
)
from src.services.labeling.vlm_endpoint_body import FIELD_RANGES, VlmEndpointBody, VlmProbeRecord


logger = get_logger(__name__)

STEP_TIMEOUT_S = 10.0
MAX_CONCURRENT_PROBES = 2
_ACTIVE_PROBES = 0
_UPSTREAM_MESSAGE_CHARS = 500


class ProbeBusyError(RuntimeError):
    """More than :data:`MAX_CONCURRENT_PROBES` probes are running in this
    process (``429 probe_busy``)."""


class _Stop(Exception):  # noqa: N818 - internal control flow, not an error type
    """A step failed hard enough that later steps cannot say anything."""


def _issue(
    code: str,
    severity: str,
    message: str,
    *,
    field: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        'code': code,
        'severity': severity,
        'field': field,
        'message': message,
        'detail': detail or {},
        'bypassable': False,
    }


def _solid_jpeg(rgb: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), rgb).save(buf, format='JPEG', quality=80)
    return buf.getvalue()


def _image_part(jpeg: bytes) -> dict[str, Any]:
    import base64

    b64 = base64.b64encode(jpeg).decode()
    return {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}


def _mask(text: str, secret: str | None) -> str:
    if secret and secret in text:
        text = text.replace(secret, '***')
    return text[:_UPSTREAM_MESSAGE_CHARS]


def _has_json_object(text: str) -> bool:
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        return False
    try:
        return isinstance(json.loads(text[start : end + 1]), dict)
    except ValueError:
        return False


class _Probe:
    def __init__(
        self, body: VlmEndpointBody, *, api_key: str | None, client: httpx.AsyncClient
    ) -> None:
        self.body = body
        self.key = api_key
        self.client = client
        self.headers = build_auth_headers(api_key or 'EMPTY')
        self.issues: list[dict[str, Any]] = []
        self.result: dict[str, Any] = {}
        self.image_timeout = min(body.timeout_s, 60.0)

    def issue(self, *args: Any, **kwargs: Any) -> None:
        self.issues.append(_issue(*args, **kwargs))

    def fail(self, *args: Any, **kwargs: Any) -> _Stop:
        self.issue(*args, severity='error', **kwargs)
        return _Stop()

    async def _send(
        self, method: str, path: str, *, timeout: float, payload: dict[str, Any] | None = None
    ) -> httpx.Response:
        url = f'{self.body.base_url}{path}'
        try:
            return await self.client.request(
                method,
                url,
                headers=self.headers,
                json=payload,
                timeout=timeout,
                follow_redirects=False,
            )
        except httpx.TimeoutException as exc:
            raise self.fail(
                'vlm_timeout', message=f'{path} did not answer within {timeout:.0f}s'
            ) from exc
        except httpx.HTTPError as exc:
            raise self.fail(
                'vlm_unreachable',
                message=f'{path} could not be reached ({type(exc).__name__})',
            ) from exc

    def _http_failure(self, resp: httpx.Response, path: str) -> _Stop:
        if resp.status_code in (401, 403):
            return self.fail(
                'vlm_auth_failed',
                message='The endpoint rejected the credentials.',
                detail={'status': resp.status_code},
            )
        return self.fail(
            'vlm_http_error',
            message=f'{path} answered HTTP {resp.status_code}.',
            detail={'status': resp.status_code},
        )

    async def list_models(self) -> None:
        started = time.monotonic()
        resp = await self._send('GET', '/models', timeout=STEP_TIMEOUT_S)
        self.result['latency_ms'] = round((time.monotonic() - started) * 1000, 1)
        if resp.status_code == 404:
            self.issue(
                'vlm_models_endpoint_missing',
                'warning',
                message='The endpoint has no /models listing (some hosted APIs lack it).',
            )
            return
        if resp.status_code != 200:
            raise self._http_failure(resp, '/models')
        try:
            entries = [e for e in (resp.json().get('data') or []) if isinstance(e, dict)]
        except ValueError:
            entries = []
        listed = [str(e.get('id')) for e in entries if e.get('id')]
        self.result['models_listed'] = listed
        match = next((e for e in entries if e.get('id') == self.body.model), None)
        self.result['model_listed'] = match is not None
        if match is None:
            raise self.fail(
                'vlm_model_not_listed',
                message=f'The endpoint does not serve {self.body.model!r}.',
                field='model',
                detail={'available': listed},
            )
        self.result['root'] = match.get('root') if isinstance(match.get('root'), str) else None
        mml = match.get('max_model_len')
        self.result['max_model_len'] = mml if isinstance(mml, int) else None

    def _chat(self, content: Any, *, max_tokens: int, json_mode: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'model': self.body.model,
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens,
            'temperature': 0.0,
        }
        if json_mode:
            payload['response_format'] = {'type': 'json_object'}
        return payload

    async def _prompt_tokens(self, resp: httpx.Response) -> int | None:
        try:
            value = (resp.json().get('usage') or {}).get('prompt_tokens')
        except ValueError:
            return None
        return value if isinstance(value, int) else None

    async def text_baseline(self) -> int | None:
        resp = await self._send(
            'POST',
            '/chat/completions',
            timeout=STEP_TIMEOUT_S,
            payload=self._chat('Reply with the single word OK', max_tokens=4),
        )
        if resp.status_code != 200:
            raise self._http_failure(resp, '/chat/completions')
        return await self._prompt_tokens(resp)

    async def one_image(self, baseline: int | None) -> None:
        content = [
            {
                'type': 'text',
                'text': 'Reply with JSON {"color": <the colour of the square>}',
            },
            _image_part(_solid_jpeg((255, 0, 0))),
        ]
        resp = await self._send(
            'POST',
            '/chat/completions',
            timeout=self.image_timeout,
            payload=self._chat(content, max_tokens=64, json_mode=True),
        )
        self.result['json_mode_supported'] = True
        if resp.status_code == 400:
            text = resp.text.lower()
            if 'response_format' in text or 'json_object' in text:
                self.result['json_mode_supported'] = False
                self.issue(
                    'vlm_json_mode_unsupported',
                    'warning',
                    message='The endpoint rejects response_format=json_object; it is left off.',
                )
                resp = await self._send(
                    'POST',
                    '/chat/completions',
                    timeout=self.image_timeout,
                    payload=self._chat(content, max_tokens=64),
                )
        if resp.status_code == 400:
            self.result['vision_ok'] = False
            raise self.fail(
                'vlm_no_vision',
                message='The endpoint rejected an image (it does not accept images).',
                detail={'upstream_message': _mask(resp.text, self.key)},
            )
        if resp.status_code != 200:
            raise self._http_failure(resp, '/chat/completions')
        try:
            data = resp.json()
        except ValueError:
            data = {}
        with_image = await self._prompt_tokens(resp)
        if baseline is not None and with_image is not None:
            self.result['image_tokens'] = max(0, with_image - baseline)
        self.result['reasoning_channel'] = bool(extract_reasoning_content(data))
        reply = extract_message_content(data) or extract_reasoning_content(data)
        if not _has_json_object(reply):
            self.issue(
                'vlm_reply_unparseable',
                'warning',
                message='The endpoint did not answer the JSON test prompt with JSON.',
            )
        self.result['vision_ok'] = 'red' in reply.lower()
        if not self.result['vision_ok']:
            self.issue(
                'vlm_vision_answer_wrong',
                'warning',
                message='The endpoint accepted an image but did not name its colour.',
            )

    async def many_images(self) -> None:
        # Defence in depth: the callers refuse an out-of-range body first.
        cap = min(self.body.max_images_per_call, int(FIELD_RANGES['max_images_per_call'][1]))
        if cap <= 1:
            return
        content: list[dict[str, Any]] = [{'type': 'text', 'text': 'Reply with the single word OK'}]
        content.extend(_image_part(_solid_jpeg((0, 0, 255))) for _ in range(cap))
        resp = await self._send(
            'POST',
            '/chat/completions',
            timeout=self.image_timeout,
            payload=self._chat(content, max_tokens=4),
        )
        if 400 <= resp.status_code < 500 and resp.status_code not in (401, 403):
            self.result['max_images_ok'] = False
            raise self.fail(
                'vlm_max_images_exceeds_server',
                message=f'The endpoint rejected a call with {cap} images.',
                field='max_images_per_call',
                detail={
                    'max_images_per_call': cap,
                    'upstream_message': _mask(resp.text, self.key),
                },
            )
        if resp.status_code != 200:
            raise self._http_failure(resp, '/chat/completions')
        self.result['max_images_ok'] = True


async def probe_endpoint(
    body: VlmEndpointBody, *, api_key: str | None, client: httpx.AsyncClient | None = None
) -> VlmProbeRecord:
    """Probe ``body`` and return the record (never raises for an endpoint
    that misbehaves: that is what the record's ``issues`` say). Raises
    :class:`ProbeBusyError` when the per-process cap is reached."""
    global _ACTIVE_PROBES  # noqa: PLW0603 - per-process concurrency cap
    if _ACTIVE_PROBES >= MAX_CONCURRENT_PROBES:
        raise ProbeBusyError
    _ACTIVE_PROBES += 1
    owned = client is None
    http = client or httpx.AsyncClient(follow_redirects=False)
    probe = _Probe(body, api_key=api_key, client=http)
    started = time.monotonic()
    try:
        try:
            await probe.list_models()
            baseline = await probe.text_baseline()
            await probe.one_image(baseline)
            await probe.many_images()
        except _Stop:
            pass
    finally:
        _ACTIVE_PROBES -= 1
        if owned:
            await http.aclose()
    total_ms = round((time.monotonic() - started) * 1000, 1)
    ok = not any(i['severity'] == 'error' for i in probe.issues)
    return VlmProbeRecord(
        ok=ok,
        probed_at=datetime.now(UTC).isoformat(),
        latency_ms=probe.result.pop('latency_ms', total_ms),
        issues=probe.issues,
        **probe.result,
    )


__all__ = ['MAX_CONCURRENT_PROBES', 'STEP_TIMEOUT_S', 'ProbeBusyError', 'probe_endpoint']
