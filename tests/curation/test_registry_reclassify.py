"""Tests for the registry-growth reclassification loop
(:mod:`src.services.curation.registry_reclassify` and its operator script
``scripts/curation/reclassify_after_registry_growth.py``).

OpenSearch is :class:`tests.curation.query_fakes.QueryFakeOpenSearch`, which
really evaluates the query, ``search_after`` paging and the OCC bulk write,
so selection, guards, idempotency and resume are all exercised for real.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch import ClassRegistry
from src.config import CurationConfig
from src.services.curation.registry_reclassify import UnmatchedLabelSource, reclassify_unmatched
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


if TYPE_CHECKING:
    from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
ITEMS = 'test_items'
CFG = CurationConfig(items_index=ITEMS)
VLM = UnmatchedLabelSource('vlm')
PACK = dataclasses.replace(GENERIC_ITEM_PACK, synonyms={'thingamajig': 'gadget'})


@pytest.fixture
def registry(tmp_path: Path) -> ClassRegistry:
    path = tmp_path / 'class_registry.json'
    path.write_text(
        json.dumps(
            {
                'version': 1,
                'classes': [
                    {'class_id': 0, 'class_name': 'widget'},
                    {'class_id': 1, 'class_name': 'gadget'},
                    {'class_id': 2, 'class_name': 'retired', 'deprecated': True},
                ],
            }
        )
    )
    return ClassRegistry(path)


def _unmatched(raw: str, **extra: Any) -> dict[str, Any]:
    return {
        'class_source': 'vlm_unmatched',
        'label_source': 'vlm',
        'vlm_raw_label': raw,
        'vlm_confidence': 'high',
        'cluster_id': 10004,
        'cluster_subid': '10004a',
        **extra,
    }


def _corpus() -> dict[str, dict[str, Any]]:
    docs = {
        # normalised match; carries a prior (detector) class that must be preserved in history
        'u1': _unmatched('Widget', class_id=1, class_name='gadget'),
        'u2': _unmatched('thingamajig'),  # synonym match
        'u3': _unmatched('no such thing'),
        'u4': _unmatched('widget', class_validated=True),
        'u5': _unmatched('widget', test_holdout=True),
        'u6': _unmatched('retired'),  # deprecated class — never a target
        'u7': {**_unmatched('widget'), 'class_source': 'alt_unmatched'},
        'u8': _unmatched('Widget', vlm_confidence='low'),  # low conf: exact only
        'u9': _unmatched('widget', vlm_confidence='low'),
    }
    return {doc_id: {'crop_id': doc_id, **doc} for doc_id, doc in docs.items()}


def _fake() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch({ITEMS: _corpus()})


async def _run(fake, registry, **kw):
    return await reclassify_unmatched(
        fake, source=VLM, registry=registry, pack=PACK, config=CFG, **kw
    )


@pytest.mark.asyncio
async def test_apply_promotes_only_resolvable_unguarded_items(registry):
    fake = _fake()
    result = await _run(fake, registry, dry_run=False)

    items = fake.docs(ITEMS)
    converted = {i for i, d in items.items() if d['class_source'] == 'vlm_reclassified'}
    assert converted == {'u1', 'u2', 'u9'}
    assert result.converted == 3
    assert result.by_class == {'widget': 2, 'gadget': 1}

    u1 = items['u1']
    assert (u1['class_id'], u1['class_name'], u1['cluster_id']) == (0, 'widget', 0)
    assert u1['cluster_subid'] is None
    assert u1['label_source'] == 'vlm'
    assert 'class_validated' not in u1 or u1['class_validated'] is not True
    [prior] = u1['class_id_history']
    assert (prior['class_id'], prior['writer']) == (1, 'registry_reclassify')
    assert items['u2']['class_id'] == 1

    original = _corpus()
    for untouched in ('u3', 'u4', 'u5', 'u6', 'u7', 'u8'):
        assert items[untouched] == original[untouched], untouched
    assert fake.indices.refreshed == [ITEMS]


@pytest.mark.asyncio
async def test_dry_run_counts_without_writing(registry):
    fake = _fake()
    before = json.dumps(fake.docs(ITEMS), sort_keys=True)
    result = await _run(fake, registry, dry_run=True)

    assert result.matched == 3
    assert result.converted == 0
    assert fake.bulk_calls == 0
    assert json.dumps(fake.docs(ITEMS), sort_keys=True) == before


@pytest.mark.asyncio
async def test_second_run_is_a_noop(registry):
    fake = _fake()
    await _run(fake, registry, dry_run=False)
    again = await _run(fake, registry, dry_run=False)
    assert again.matched == 0
    assert again.converted == 0


@pytest.mark.asyncio
async def test_search_after_cursor_resumes_an_interrupted_run(registry):
    fake = _fake()
    first = await _run(fake, registry, dry_run=False, page_size=2, max_pages=1)
    assert first.pages == 1
    assert first.last_cursor is not None

    rest = await _run(fake, registry, dry_run=False, page_size=2, search_after=first.last_cursor)
    assert first.converted + rest.converted == 3
    converted = {i for i, d in fake.docs(ITEMS).items() if d['class_source'] == 'vlm_reclassified'}
    assert converted == {'u1', 'u2', 'u9'}


@pytest.mark.asyncio
async def test_other_prefix_is_selected_independently(registry):
    fake = _fake()
    await reclassify_unmatched(
        fake,
        source=UnmatchedLabelSource('alt'),
        registry=registry,
        pack=PACK,
        config=CFG,
        dry_run=False,
    )
    # alt_* uses alt_raw_label, which u7 does not carry -> nothing to do.
    assert fake.docs(ITEMS)['u7']['class_source'] == 'alt_unmatched'

    fake.docs(ITEMS)['u7']['alt_raw_label'] = 'widget'
    await reclassify_unmatched(
        fake,
        source=UnmatchedLabelSource('alt'),
        registry=registry,
        pack=PACK,
        config=CFG,
        dry_run=False,
    )
    assert fake.docs(ITEMS)['u7']['class_source'] == 'alt_reclassified'
    assert fake.docs(ITEMS)['u1']['class_source'] == 'vlm_unmatched'


def test_prefix_is_validated():
    with pytest.raises(ValueError, match='invalid label prefix'):
        UnmatchedLabelSource('bad prefix')


# --------------------------------------------------------------------------- CLI


def _load_script() -> ModuleType:
    path = REPO_ROOT / 'scripts' / 'curation' / 'reclassify_after_registry_growth.py'
    spec = importlib.util.spec_from_file_location('curation_reclassify_cli_test', path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_to_dry_run_with_vlm_prefix(monkeypatch):
    mod = _load_script()
    captured: dict[str, Any] = {}

    async def fake_async_main(args):
        captured['args'] = args
        return 0

    monkeypatch.setattr(mod, '_async_main', fake_async_main)
    monkeypatch.setattr(sys, 'argv', ['reclassify_after_registry_growth.py'])
    assert mod.main() == 0
    assert captured['args'].dry_run is True
    assert captured['args'].label_prefix is None


def test_cli_apply_runs_every_prefix(monkeypatch, registry):
    mod = _load_script()
    fake = _fake()
    fake.docs(ITEMS)['u7']['alt_raw_label'] = 'widget'
    monkeypatch.setattr(mod, 'AsyncOpenSearch', lambda **_kw: _Closable(fake))
    monkeypatch.setattr(mod, 'get_curation_config', lambda: CFG)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'x',
            '--registry',
            str(registry.path),
            '--label-prefix',
            'vlm',
            '--label-prefix',
            'alt',
            '--apply',
        ],
    )
    assert mod.main() == 0
    sources = {d['class_source'] for d in fake.docs(ITEMS).values()}
    assert 'vlm_reclassified' in sources
    assert 'alt_reclassified' in sources


def test_cli_refuses_empty_registry(monkeypatch, tmp_path):
    mod = _load_script()
    monkeypatch.setattr(sys, 'argv', ['x', '--registry', str(tmp_path / 'missing.json'), '--apply'])
    assert mod.main() == 2


def test_cli_start_after_requires_single_prefix(monkeypatch):
    mod = _load_script()
    monkeypatch.setattr(
        sys,
        'argv',
        ['x', '--label-prefix', 'a', '--label-prefix', 'b', '--start-after', 'c1'],
    )
    with pytest.raises(SystemExit):
        mod.main()


def test_review_router_points_at_the_real_script():
    """``GET /review/unmatched_terms`` tells operators to run a
    reclassification script — that pointer must name a file that exists."""
    source = (REPO_ROOT / 'src' / 'routers' / 'curation' / 'review.py').read_text()
    ref = 'scripts/curation/reclassify_after_registry_growth.py'
    assert ref in source
    assert (REPO_ROOT / ref).is_file()


class _Closable:
    """Wraps the fake with the ``close()`` the script calls in ``finally``."""

    def __init__(self, inner: QueryFakeOpenSearch) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def close(self) -> None:
        return None
