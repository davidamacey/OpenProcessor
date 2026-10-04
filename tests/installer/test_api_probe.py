"""The installer's functional API probe against the real response shapes."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

# Real GET /curation/health shape (src/routers/curation/global_status.py).
HEALTH_OK = {
    'status': 'ok',
    'triton': {'reachable': True, 'detail': ''},
    'opensearch': {'reachable': True},
    'vlm': {'reachable': True, 'model': 'local-vlm'},
    'mlflow_public_url': 'http://localhost:4609',
    'version': '0.4.0',
    'api_version': 'v1',
}
HEALTH_DEGRADED_NO_VLM = {**HEALTH_OK, 'status': 'degraded', 'vlm': {'reachable': False}}


def _probe_source() -> str:
    text = (REPO_ROOT / 'setup-openprocessor.sh').read_text()
    m = re.search(r"_API_PROBE_PY='\n(.*?)\n'\n", text, re.S)
    assert m, '_API_PROBE_PY not found'
    return m.group(1)


def _run_probe(health: dict, curation: str = '1', vlm: str = '1') -> subprocess.CompletedProcess:
    class H(BaseHTTPRequestHandler):
        def _send(self, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            self._send(health)

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get('Content-Length', 0)))
            self._send({'embedding': [0.0] * 512})

        def log_message(self, *a: object) -> None:
            pass

    srv = HTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        src = _probe_source().replace(
            'http://127.0.0.1:8000', f'http://127.0.0.1:{srv.server_port}'
        )
        return subprocess.run(
            [sys.executable, '-c', src, curation, vlm],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        srv.shutdown()


pytest.importorskip('cv2')


def test_probe_passes_on_real_ok_health_shape() -> None:
    r = _run_probe(HEALTH_OK)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'registry' not in r.stdout


def test_probe_degraded_without_vlm_fails_cleanly_not_keyerror() -> None:
    r = _run_probe(HEALTH_DEGRADED_NO_VLM)
    assert r.returncode == 1
    assert "'registry'" not in r.stdout + r.stderr
