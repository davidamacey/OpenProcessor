"""The live harness's fake VLM: the two classification modes the auto-label loop's live test uses.

``class_cycle`` makes a sampled cluster show dissent; ``answers`` replays a posted
image-hash to class map; ``/__seen`` lets a test learn the hash the product sends for a crop.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest


if TYPE_CHECKING:
    from collections.abc import Iterator


FAKE_VLM = Path(__file__).resolve().parents[1] / 'docker' / 'test' / 'fake_vlm.py'
CLASS_PROMPT = 'Class names: box, envelope, tube. Answer with "class" for each image.'


def _load() -> Any:
    spec = importlib.util.spec_from_file_location('fake_vlm_under_test', FAKE_VLM)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake() -> Iterator[str]:
    module = _load()
    server = ThreadingHTTPServer(('127.0.0.1', 0), module.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _call(base: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:  # loopback test server
        return json.loads(resp.read())


def _classify(base: str, images: list[bytes]) -> list[str]:
    content: list[dict[str, Any]] = [{'type': 'text', 'text': CLASS_PROMPT}]
    for raw in images:
        url = 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()
        content.append({'type': 'image_url', 'image_url': {'url': url}})
    reply = _call(
        base,
        '/v1/chat/completions',
        {'model': 'm', 'messages': [{'role': 'user', 'content': content}]},
    )
    entries = json.loads(reply['choices'][0]['message']['content'])
    return [e['class'] for e in entries]


def test_the_fixed_class_is_the_default(fake: str) -> None:
    assert _classify(fake, [b'a', b'b']) == ['box', 'box']


def test_class_cycle_walks_the_list_across_calls_and_reset_restarts_it(fake: str) -> None:
    _call(fake, '/__control', {'class_cycle': ['box', 'box', 'envelope']})
    assert _classify(fake, [b'a', b'b']) == ['box', 'box']
    assert _classify(fake, [b'c', b'd']) == ['envelope', 'box']
    _call(fake, '/__reset', {})
    assert _classify(fake, [b'a']) == ['box']
    _call(fake, '/__control', {'class_cycle': ['tube']})
    assert _classify(fake, [b'a']) == ['tube']
    _call(fake, '/__control', {'class_cycle': None})
    assert _classify(fake, [b'a']) == ['box']


def test_answers_replay_by_image_hash_and_fall_back_for_other_images(fake: str) -> None:
    known = hashlib.sha256(b'crop-one').hexdigest()
    _call(fake, '/__control', {'answers': {known: 'tube'}})
    assert _classify(fake, [b'crop-one', b'crop-two']) == ['tube', 'box']
    _call(fake, '/__control', {'class_cycle': ['envelope']})
    assert _classify(fake, [b'crop-one', b'crop-two']) == ['tube', 'envelope']


def test_seen_lists_the_hash_of_each_image_in_order(fake: str) -> None:
    _classify(fake, [b'one', b'two'])
    _classify(fake, [b'three'])
    hashes = _call(fake, '/__seen')['hashes']
    assert hashes == [hashlib.sha256(x).hexdigest() for x in (b'one', b'two', b'three')]
    _call(fake, '/__reset', {})
    assert _call(fake, '/__seen')['hashes'] == []
