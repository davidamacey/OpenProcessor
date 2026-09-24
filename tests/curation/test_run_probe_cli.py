"""Tests for the probe-inference backfill driver (``scripts/curation/run_probe.py``)
and the resume / page-size plumbing it relies on in
:mod:`src.services.curation.probe_predictions`.

No real model or cluster: ``_build_predictor`` is monkeypatched to a canned
predictor and OpenSearch is an in-memory fake that genuinely evaluates the
``must_not`` term clauses of the query it receives, so the resume filter is
exercised end to end rather than asserted structurally.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image


if TYPE_CHECKING:
    from types import ModuleType


def _load_script() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / 'scripts' / 'curation' / 'run_probe.py'
    spec = importlib.util.spec_from_file_location('curation_run_probe_cli_test', script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeOpenSearch:
    """Scroll/update/count double that applies ``must_not`` term clauses."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = {d['crop_id']: d for d in docs}
        self.search_bodies: list[dict[str, Any]] = []
        self.count_bodies: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.refreshed: list[str] = []
        self.closed = False
        self.indices = self

    def _matches(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        must_not = query.get('bool', {}).get('must_not', [])
        out = []
        for doc in self.docs.values():
            excluded = False
            for clause in must_not:
                ((field, value),) = clause['term'].items()
                if doc.get(field) == value:
                    excluded = True
            if not excluded:
                out.append(doc)
        return out

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        self.count_bodies.append(body)
        return {'count': len(self._matches(body['query']))}

    async def search(self, index: str, body: dict[str, Any], scroll: str | None = None) -> dict:  # noqa: ARG002
        self.search_bodies.append(body)
        hits = [{'_id': d['crop_id'], '_source': d} for d in self._matches(body['query'])]
        return {'_scroll_id': 'scroll-1', 'hits': {'hits': hits[: body['size']]}}

    async def scroll(self, scroll_id: str, scroll: str) -> dict:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, scroll_id: str) -> None:  # noqa: ARG002
        return None

    async def update(self, index: str, id: str, body: dict[str, Any]) -> None:  # noqa: A002, ARG002
        self.updates.append({'id': id, 'doc': body['doc']})
        self.docs[id].update(body['doc'])

    async def refresh(self, *, index: str) -> None:
        self.refreshed.append(index)

    async def close(self) -> None:
        self.closed = True


def _docs() -> list[dict[str, Any]]:
    return [
        {
            'crop_id': 'a',
            'image_path': 'a.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_name': 'car',
            'probe_model_version': 'run-7',
        },
        {'crop_id': 'b', 'image_path': 'b.jpg', 'bbox_norm': [0.0, 0.0, 1.0, 1.0]},
        {
            'crop_id': 'c',
            'image_path': 'c.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'test_holdout': True,
        },
    ]


@pytest.fixture
def canned_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    from src.services.curation import probe_predictions as pp

    image = tmp_path / 'frame.jpg'
    Image.new('RGB', (32, 32), (1, 2, 3)).save(image)
    monkeypatch.setattr(
        pp,
        '_build_predictor',
        lambda model_path, architecture: ((lambda crop: ('car', 0.9, 0.1, 0.8)), 'x'),  # noqa: ARG005
    )
    monkeypatch.setattr(pp, '_resolve_image', lambda image_path, *, config: image)  # noqa: ARG005
    return image


# --------------------------------------------------------------------------- service


@pytest.mark.asyncio
async def test_resume_skips_items_already_scored_by_this_version(canned_probe, tmp_path):
    from src.services.curation.probe_predictions import run_probe_inference

    fake = _FakeOpenSearch(_docs())
    n = await run_probe_inference(
        tmp_path / 'm.onnx', fake, model_version='run-7', resume=True, page_size=50
    )

    assert n == 1
    assert [u['id'] for u in fake.updates] == ['b']
    assert fake.search_bodies[0]['size'] == 50


@pytest.mark.asyncio
async def test_without_resume_every_non_holdout_item_is_rescored(canned_probe, tmp_path):
    from src.services.curation.probe_predictions import run_probe_inference

    fake = _FakeOpenSearch(_docs())
    n = await run_probe_inference(tmp_path / 'm.onnx', fake, model_version='run-7')

    assert n == 2
    assert sorted(u['id'] for u in fake.updates) == ['a', 'b']


@pytest.mark.asyncio
async def test_count_probe_candidates_honours_holdout_and_resume():
    from src.services.curation.probe_predictions import count_probe_candidates

    fake = _FakeOpenSearch(_docs())
    assert await count_probe_candidates(fake) == 2
    assert await count_probe_candidates(fake, skip_version='run-7') == 1


# --------------------------------------------------------------------------- CLI


def test_default_model_version():
    mod = _load_script()
    assert mod.default_model_version(Path('/runs/probe-3/weights/best.onnx')) == 'probe-3'
    assert mod.default_model_version(Path('/models/detector_v2.onnx')) == 'detector_v2'


def test_cli_defaults_to_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_script()
    captured: dict[str, Any] = {}

    async def fake_async_main(args):
        captured['args'] = args
        return 0

    monkeypatch.setattr(mod, '_async_main', fake_async_main)
    monkeypatch.setattr(sys, 'argv', ['run_probe.py', '--model', 'm.onnx'])
    assert mod.main() == 0
    assert captured['args'].dry_run is True
    assert captured['args'].architecture == 'yolo11'


def _patch_client(monkeypatch, mod, fake):
    monkeypatch.setattr(mod, 'AsyncOpenSearch', lambda **_kw: fake)


def test_cli_dry_run_counts_without_scoring(monkeypatch, tmp_path, capsys) -> None:
    mod = _load_script()
    model = tmp_path / 'm.onnx'
    model.write_bytes(b'')
    fake = _FakeOpenSearch(_docs())
    _patch_client(monkeypatch, mod, fake)

    async def must_not_run(*_a, **_kw):
        raise AssertionError('dry-run must not load the model or write')

    monkeypatch.setattr(mod, 'run_probe_inference', must_not_run)
    monkeypatch.setattr(sys, 'argv', ['run_probe.py', '--model', str(model)])

    assert mod.main() == 0
    assert 'candidates=2' in capsys.readouterr().out
    assert fake.updates == []
    assert fake.closed


def test_cli_apply_calls_the_service_with_resume(monkeypatch, tmp_path) -> None:
    mod = _load_script()
    model = tmp_path / 'weights' / 'best.onnx'
    model.parent.mkdir()
    model.write_bytes(b'')
    fake = _FakeOpenSearch(_docs())
    _patch_client(monkeypatch, mod, fake)
    calls: list[dict[str, Any]] = []

    async def fake_run(model_path, client, **kwargs):
        calls.append({'model_path': model_path, 'client': client, **kwargs})
        return 1

    monkeypatch.setattr(mod, 'run_probe_inference', fake_run)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'run_probe.py',
            '--model',
            str(model),
            '--architecture',
            'v6',
            '--model-version',
            'run-7',
            '--resume',
            '--page-size',
            '64',
            '--limit',
            '10',
            '--apply',
        ],
    )

    assert mod.main() == 0
    assert len(calls) == 1
    call = calls[0]
    assert call['client'] is fake
    assert call['architecture'] == 'v6'
    assert call['model_version'] == 'run-7'
    assert call['resume'] is True
    assert call['page_size'] == 64
    assert call['max_crops'] == 10
    assert fake.refreshed


def test_cli_missing_model_fails(monkeypatch, tmp_path) -> None:
    mod = _load_script()
    monkeypatch.setattr(sys, 'argv', ['run_probe.py', '--model', str(tmp_path / 'nope.onnx')])
    assert mod.main() == 1
