"""Semantic search needs Triton's text model, and nothing else answers for it (ES-C5).

With the text model not ready the route refuses naming it, the refusal never imports ``torch`` or
the tokenizer's package, and the settings that selected the retired in-process backends fail at
startup.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.clients.pe_encoder import PEEncoder
from src.config.retired_env import RETIRED_ENV, reject_retired_env


class _NotReadyClient:
    def is_model_ready(self, _name: str) -> bool:
        return False


def _opensearch() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    monkeypatch.setenv('OP_SEMANTIC_SEARCH_ENABLED', 'true')
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.state.pe_encoder = PEEncoder(triton_client_factory=_NotReadyClient)
    app.dependency_overrides[_raw_opensearch_dep] = _opensearch
    return TestClient(app)


def test_search_refuses_naming_the_triton_text_model(client: TestClient) -> None:
    resp = client.get('/curation/projects/default/search/text', params={'q': 'white pickup truck'})
    assert resp.status_code == 503
    assert "Triton's text model 'pe_text_encoder'" in resp.json()['detail']


def test_the_refusal_imports_no_torch_and_no_tokenizer_package() -> None:
    # What the refusal itself imports: the app's own import chain may already hold torch.
    code = textwrap.dedent(
        """
        import sys

        from src.clients.pe_encoder import PEEncoder, TextModelNotReady

        class _Client:
            def is_model_ready(self, name):
                return False

        enc = PEEncoder(triton_client_factory=_Client)
        before = set(sys.modules)
        try:
            enc.warm_text_encoder()
        except TextModelNotReady:
            pass
        else:
            raise SystemExit('warm-up must refuse')
        leaked = sorted(
            m for m in set(sys.modules) - before if m.split('.')[0] in ('torch', 'core')
        )
        raise SystemExit(f'the refusal imported {leaked[:5]}' if leaked else 0)
        """
    )
    done = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize(
    'name', ['OP_PE_TEXT_BACKEND', 'OP_PE_TEXT_ONNX_PATH', 'OP_PE_TEXT_ORT_THREADS']
)
def test_the_retired_in_process_backend_settings_fail_at_startup(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    assert RETIRED_ENV[name].startswith('removed:')
    monkeypatch.setenv(name, 'onnx')
    with pytest.raises(RuntimeError, match=name):
        reject_retired_env()
