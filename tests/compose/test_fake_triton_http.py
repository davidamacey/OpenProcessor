"""The harness Triton answers the HTTP calls the config vocabulary makes: readiness and the
repository index (an empty repository), nothing else.

Without them ``GET /config/vocabulary`` is a 503 ``triton_unavailable`` in the harness.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[2] / 'docker' / 'test' / 'fake_triton.py'


@pytest.fixture
def base_url():
    spec = importlib.util.spec_from_file_location('fake_triton_harness', PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server = ThreadingHTTPServer(('127.0.0.1', 0), module._Http)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{server.server_address[1]}'
    server.shutdown()
    server.server_close()


def test_the_repository_index_lists_the_served_and_listed_only_models(base_url: str) -> None:
    request = urllib.request.Request(f'{base_url}/v2/repository/index', method='POST')
    with urllib.request.urlopen(request, timeout=5) as response:  # nosec B310 - loopback
        assert response.status == 200
        listed = {row['name']: row for row in json.loads(response.read())}
    assert {'yolov11_small_trt_end2end', 'ocr_pipeline', 'paddleocr_det_trt'} <= set(listed)
    assert {row['state'] for row in listed.values()} == {'READY'}


def test_readiness_is_ok_and_anything_else_is_unimplemented(base_url: str) -> None:
    with urllib.request.urlopen(f'{base_url}/v2/health/ready', timeout=5) as response:  # nosec B310
        assert response.status == 200
    with pytest.raises(urllib.error.HTTPError) as missing:
        urllib.request.urlopen(f'{base_url}/v2/models/x/config', timeout=5)  # nosec B310
    assert missing.value.code == 404
