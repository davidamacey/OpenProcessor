"""Deterministic OpenAI-compatible vision chat endpoint for the live harness.

The application's VLM transport layer (``src/services/labeling/vlm_client.py``)
only needs an endpoint that answers ``POST {base}/chat/completions`` with an
OpenAI-shaped ``{"choices": [{"message": {"content": ...}}]}`` envelope. This
server does exactly that, with a *fixed* reply per prompt kind, so a live test
can assert the exact string that came back out the other end of the pipeline.

What this proves and what it does not
-------------------------------------
It proves **wiring and persistence**: that the router builds a well-formed
request, that the reply is parsed by the labeler's parsers, and that the parsed
verdict reaches OpenSearch in the documented fields. It proves **nothing at all**
about label quality — the answers here are constants.

Prompt-kind routing keys off distinctive fragments of the built-in prompt pack
(``src/services/labeling/vlm_prompts.py::GENERIC_ITEM_PACK``) rather than
on request order, so parallel chunked calls are answered correctly.

Control surface (test-only, not part of any real API):

* ``GET  /health``           -> ``{"status": "ok"}``
* ``GET  /__stats``          -> per-prompt-kind call counts
* ``POST /__control``        -> override reply knobs, e.g.
  ``{"region_verdict_key": "is_plate"}`` to make the fake answer with a
  legacy/off-spec key instead of the one the prompt pack actually asks
  for, or ``{"region_visible": true}``.
* ``POST /__reset``          -> restore defaults and zero the counters.
* ``GET  /__seen``           -> the sha256 of every image received, oldest first (at most
  ``SEEN_MAX``), so a test can learn which hash the product sends for a given crop (ask for
  one crop, read the last entry).

Two classification modes for the auto-label loop (both test-only):

* ``class_cycle``: a list of class names. The n-th image the fake is asked about (counted
  across calls since the last reset) answers ``class_cycle[n % len]``, so a cluster sampled
  through it shows dissent and fails the loop's gate. ``null`` restores the fixed answer.
* ``answers``: ``{sha256-of-the-image-bytes: class name}``. An image whose hash is in the map
  answers that class, any other image falls back to ``class_cycle`` or the fixed answer. The
  wire carries no crop id, so the key is the image hash.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


LISTEN_PORT = int(os.environ.get('FAKE_VLM_PORT', '8000'))

# Defaults, overridable at runtime through POST /__control.
DEFAULTS: dict[str, Any] = {
    # Class name the classification prompts answer with. Must exist in the
    # seeded class registry or the reply lands in the `vlm_unmatched` cohort.
    'class_name': os.environ.get('FAKE_VLM_CLASS', 'box'),
    'class_confidence': 'high',
    'proposed_class': '',
    # Loop modes, see the module docstring.
    'class_cycle': None,
    'answers': None,
    # Region verify.
    'region_is_region': True,
    'region_confidence': 'high',
    'region_reason': 'fake verifier: fixed affirmative',
    'region_text': os.environ.get('FAKE_VLM_REGION_TEXT', 'FAKE-LABEL-0042'),
    'region_text_confidence': 'high',
    # Key the verdict object uses. The built-in prompt pack asks the model
    # for `is_region`, and the labeler's parser reads `is_region` too — this
    # default matches the shipped prompt. Tests can flip this to a legacy
    # key (e.g. `is_plate`) to prove the parser no longer accepts it.
    'region_verdict_key': 'is_region',
    # Region-visibility pre-filter. Defaults to an explicit negative: that
    # endpoint fails open to True, so only a negative reply distinguishes
    # "the fake was understood" from "the fake was ignored".
    'region_visible': os.environ.get('FAKE_VLM_REGION_VISIBLE', '') == 'true',
    # Per-box `region_bbox_correct` of a combined reply, cycled over the boxes
    # offered (e.g. [True, True, False]). None leaves every box without a
    # verdict, which the worker records as `verifier_no_verdict`.
    'combined_box_verdicts': [v == 'true' for v in os.environ['FAKE_VLM_BOX_VERDICTS'].split(',')]
    if os.environ.get('FAKE_VLM_BOX_VERDICTS')
    else None,
}

SEEN_MAX = 5000

_state: dict[str, Any] = dict(DEFAULTS)
_counts: dict[str, int] = {}
_seen: deque[str] = deque(maxlen=SEEN_MAX)
_cycle: dict[str, int] = {'position': 0}


def _count(kind: str) -> None:
    _counts[kind] = _counts.get(kind, 0) + 1


def _n_images(messages: list[dict[str, Any]]) -> int:
    n = 0
    for message in messages:
        content = message.get('content')
        if isinstance(content, list):
            n += sum(1 for part in content if part.get('type') == 'image_url')
    return n


def _image_hashes(messages: list[dict[str, Any]]) -> list[str]:
    """sha256 of each image's decoded bytes, in message order ('' for a non data-URL image)."""
    hashes: list[str] = []
    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get('type') != 'image_url':
                continue
            url = str((part.get('image_url') or {}).get('url') or '')
            try:
                raw = base64.b64decode(url.split(',', 1)[1], validate=False)
            except (IndexError, binascii.Error):
                hashes.append('')
                continue
            hashes.append(hashlib.sha256(raw).hexdigest())
    return hashes


def _all_text(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for message in messages:
        content = message.get('content')
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(part.get('text', '') for part in content if part.get('type') == 'text')
    return '\n'.join(chunks)


def _classify_prompt(text: str) -> str:
    """Map a request's prompt text onto one of the labeler's prompt kinds."""
    lowered = text.lower()
    if 'is a labeled sub-region visible' in lowered or 'contains a visible labeled' in lowered:
        return 'region_visible'
    if 'respond as a json array' in lowered and 'is_region' in lowered:
        return 'region_batch'
    if 'exactly one json object' in lowered or 'real printed label region' in lowered:
        return 'region_single'
    if 'proposed_class' in lowered:
        return 'class_open'
    if '"class"' in lowered or 'class names:' in lowered:
        return 'class_closed'
    if 'region_visible' in lowered and 'class_id' in lowered:
        return 'combined'
    return 'unknown'


def _class_for(image_hash: str) -> str:
    """The class the next image answers: its posted answer, else the cycle, else the fixed class."""
    answers = _state['answers']
    if answers and image_hash in answers:
        return str(answers[image_hash])
    cycle = _state['class_cycle']
    if cycle:
        name = str(cycle[_cycle['position'] % len(cycle)])
        _cycle['position'] += 1
        return name
    return str(_state['class_name'])


def _class_entries(n_images: int, image_hashes: list[str] | None = None) -> list[dict[str, Any]]:
    hashes = image_hashes or []
    return [
        {
            'img': i,
            'class': _class_for(hashes[i - 1] if i - 1 < len(hashes) else ''),
            'confidence': _state['class_confidence'],
            'proposed_class': _state['proposed_class'],
        }
        for i in range(1, n_images + 1)
    ]


def _region_entry(index: int | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        _state['region_verdict_key']: _state['region_is_region'],
        'confidence': _state['region_confidence'],
        'reason': _state['region_reason'],
        'text': _state['region_text'],
        'text_confidence': _state['region_text_confidence'],
    }
    if index is not None:
        entry = {'img': index, **entry}
    return entry


# the combined reply's
# per-box verdicts are nested under `region_boxes` (the numbered-
# overlay list shape), not flat `region_bbox_correct`/`region_text`/
# `region_confidence` top-level keys -- the box_verdicts() parser raises
# MultiRegionKeysMissingError on the old flat shape, which the pipeline
# treats as a parse failure (retried to the no-verdict cap, then written
# rejected). ``n_boxes`` is read off the prompt text itself (the numbered-
# overlay sentence or the coordinate fallback both name the count),
# defaulting to 1 for a prompt with no boxes offered at all.
_OVERLAY_N_RE = re.compile(r'\(1 to (\d+)\)')
_COORD_N_RE = re.compile(r'(\d+)=\[')


def _n_region_boxes(text: str) -> int:
    m = _OVERLAY_N_RE.search(text)
    if m:
        return int(m.group(1))
    nums = [int(n) for n in _COORD_N_RE.findall(text)]
    return max(nums) if nums else 1


def _box_verdict(box: int) -> bool | None:
    verdicts = _state['combined_box_verdicts']
    return verdicts[(box - 1) % len(verdicts)] if verdicts else None


def _reply_for(
    kind: str, n_images: int, text: str = '', image_hashes: list[str] | None = None
) -> str:
    if kind == 'region_visible':
        results = [{'img': i, 'visible': _state['region_visible']} for i in range(1, n_images + 1)]
        return json.dumps({'results': results})
    if kind == 'region_batch':
        return json.dumps([_region_entry(i) for i in range(1, n_images + 1)])
    if kind == 'region_single':
        return json.dumps(_region_entry(None))
    if kind in ('class_open', 'class_closed'):
        return json.dumps(_class_entries(n_images, image_hashes))
    if kind == 'combined':
        n_boxes = _n_region_boxes(text)
        results = [
            {
                'img': i,
                'class_id': None,
                'class_confidence': _state['class_confidence'],
                'region_visible': _state['region_visible'],
                'region_boxes': [
                    {
                        'box': b,
                        'region_bbox_correct': _box_verdict(b),
                        'region_text': None,
                        'region_confidence': None,
                    }
                    for b in range(1, n_boxes + 1)
                ],
            }
            for i in range(1, n_images + 1)
        ]
        return json.dumps({'results': results})
    return 'OK'


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002  # overrides BaseHTTPRequestHandler.log_message
        # One compact line per request; the default logger is too chatty
        # for a container whose logs are read during a failing test.
        print(f'fake-vlm {self.command} {self.path} {format % args}')

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get('Content-Length') or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def do_GET(self) -> None:
        if self.path == '/health':
            self._send(200, {'status': 'ok'})
            return
        if self.path == '/__seen':
            self._send(200, {'hashes': list(_seen)})
            return
        if self.path == '/__stats':
            self._send(200, {'counts': dict(_counts), 'state': dict(_state)})
            return
        self._send(404, {'error': 'not found'})

    def do_POST(self) -> None:
        if self.path == '/__control':
            payload = self._read_json()
            unknown = sorted(set(payload) - set(DEFAULTS))
            if unknown:
                self._send(400, {'error': f'unknown control keys: {unknown}'})
                return
            _state.update(payload)
            self._send(200, {'state': dict(_state)})
            return
        if self.path == '/__reset':
            _state.clear()
            _state.update(DEFAULTS)
            _counts.clear()
            _seen.clear()
            _cycle['position'] = 0
            self._send(200, {'state': dict(_state)})
            return
        if re.fullmatch(r'/v\d+/chat/completions', self.path):
            payload = self._read_json()
            messages = payload.get('messages') or []
            n_images = max(1, _n_images(messages))
            text = _all_text(messages)
            kind = _classify_prompt(text)
            _count(kind)
            hashes = _image_hashes(messages)
            _seen.extend(hashes)
            content = _reply_for(kind, n_images, text, hashes)
            self._send(
                200,
                {
                    'id': 'fake-vlm-completion',
                    'object': 'chat.completion',
                    'model': payload.get('model') or 'fake-vlm',
                    'choices': [
                        {
                            'index': 0,
                            'finish_reason': 'stop',
                            'message': {'role': 'assistant', 'content': content},
                        }
                    ],
                    'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                },
            )
            return
        self._send(404, {'error': 'not found'})


def main() -> None:
    # Binds all interfaces: this process only ever runs inside a
    # throwaway harness container on an internal compose network.
    server = ThreadingHTTPServer(('0.0.0.0', LISTEN_PORT), Handler)  # nosec B104
    print(f'fake-vlm listening on :{LISTEN_PORT}')
    server.serve_forever()


if __name__ == '__main__':
    main()
