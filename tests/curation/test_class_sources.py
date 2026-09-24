"""``class_source`` catalog (``GET /class_sources``) and the VLM class
suggestion derived from it (``vlm_proposed_class_*`` on every wire item).

The catalog must list every value the codebase can write. The scan below
walks every ``class_source`` write in ``src/`` and ``scripts/`` so a new
writer that forgets to register its value fails here.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.curation import _common
from src.services.curation.class_sources import (
    CLASS_SOURCE_ROLES,
    class_source_catalog,
    vlm_suggestion,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key in list(os.environ):
        if key.startswith(('OP_INGEST_', 'OP_DETECTION_')):
            monkeypatch.delenv(key)
    return monkeypatch


def _ids(catalog: list[dict[str, str]]) -> list[str]:
    return [e['id'] for e in catalog]


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_proposer_primary_with_secondary(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv('OP_INGEST_PRIMARY_NAME', 'coco_yolo11')
    clean_env.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'yolov11_small_trt_end2end')
    clean_env.setenv('OP_INGEST_SECONDARY_NAME', 'v6')
    clean_env.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'item_classifier_trt')
    catalog = class_source_catalog()
    by_id = {e['id']: e for e in catalog}
    assert by_id['coco_yolo11_proposal']['role'] == 'proposal'
    assert 'yolov11_small_trt_end2end' in by_id['coco_yolo11_proposal']['label']
    assert by_id['v6_model']['role'] == 'model'
    assert 'item_classifier_trt' in by_id['v6_model']['label']
    # A non-assigning primary never writes these.
    assert 'coco_yolo11_model' not in by_id
    assert 'coco_yolo11_low_conf' not in by_id


def test_assigning_primary_writes_model_and_low_conf(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv('OP_INGEST_PRIMARY_NAME', 'det')
    clean_env.setenv('OP_INGEST_PRIMARY_ASSIGNS_CLASS', 'true')
    by_id = {e['id']: e for e in class_source_catalog()}
    assert by_id['det_model']['role'] == 'model'
    assert by_id['det_low_conf']['role'] == 'low_conf'
    assert by_id['det_proposal']['role'] == 'proposal'
    # Label falls back to the profile name without a detector model.
    assert 'det' in by_id['det_model']['label']


def test_secondary_requires_a_detector_model(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv('OP_INGEST_SECONDARY_NAME', 'v6')
    assert 'v6_model' not in _ids(class_source_catalog())


def test_fixed_writer_values_and_roles(clean_env: pytest.MonkeyPatch) -> None:
    by_id = {e['id']: e['role'] for e in class_source_catalog()}
    assert by_id['vlm'] == 'vlm'
    assert by_id['vlm_unmatched'] == 'vlm_unmatched'
    assert by_id['vlm_new_class_pending'] == 'vlm_new_class_pending'
    assert by_id['vlm_reclassified'] == 'vlm_reclassified'
    assert by_id['cluster_majority_agreement'] == 'cluster'
    assert by_id['human'] == 'human'
    assert by_id['human_move'] == 'human'
    assert by_id['class_merge'] == 'merge'
    assert by_id['external_label'] == 'label_import'
    assert by_id['unlabeled_proposal'] == 'proposal'


def test_catalog_entries_are_well_formed(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv('OP_INGEST_PRIMARY_ASSIGNS_CLASS', 'true')
    clean_env.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'clf')
    catalog = class_source_catalog()
    ids = _ids(catalog)
    assert len(ids) == len(set(ids))
    for entry in catalog:
        assert set(entry) == {'id', 'label', 'role', 'short_label'}
        assert entry['role'] in CLASS_SOURCE_ROLES
        assert entry['label']


def test_class_sources_endpoint(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv('OP_INGEST_PRIMARY_NAME', 'coco_yolo11')
    clean_env.setenv('OP_INGEST_SECONDARY_NAME', 'v6')
    clean_env.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'item_classifier_trt')
    from src.routers.curation import router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    with TestClient(app) as client:
        r = client.get(f'{_common.config.api_prefix}/class_sources')
    assert r.status_code == 200, r.text
    assert r.json() == {'class_sources': class_source_catalog()}
    assert 'v6_model' in _ids(r.json()['class_sources'])


# ---------------------------------------------------------------------------
# Every class_source the codebase writes is in the catalog
# ---------------------------------------------------------------------------

# Seeds a synthetic live-test dataset with its own fixed cohort values; not
# a deployment writer.
_SCAN_EXCLUDED = {'scripts/curation/seed_live_harness.py'}

_QUERY_KEYS = frozenset({'term', 'terms', 'match', 'match_phrase', 'prefix', 'wildcard'})

# Non-literal writes, by source text. A value -> the catalog id it writes;
# None -> a pass-through/default that re-writes an existing value (or a
# caller-chosen one) rather than originating a new vocabulary entry.
_DYNAMIC_WRITES: dict[str, str | None] = {
    'CLUSTER_MAJORITY_CLASS_SOURCE': 'cluster_majority_agreement',
    # Registry reclassification; the only unmatched source written is
    # 'vlm_unmatched' (default prefix 'vlm').
    'source.reclassified_source': 'vlm_reclassified',
    # PUT /crops/{id}/label, PUT /crops/batch_label: caller-chosen,
    # default 'human'.
    'payload.label_source': 'human',
    # Label import: caller-chosen, default 'external_label'.
    'label_source': 'external_label',
    'item.class_source': None,
    'class_source': None,
    "doc.get('class_source', '')": None,
    "src.get('class_source', '')": None,
    "str(src.get('class_source') or '')": None,
    "current_source.get('class_source')": None,
    "(h.get('_source') or {}).get('class_source')": None,
}


def _written_values(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.IfExp):
        return _written_values(node.body) + _written_values(node.orelse)
    return [node]


def _class_source_writes() -> list[tuple[str, int, ast.expr]]:
    out: list[tuple[str, int, ast.expr]] = []
    for root in ('src', 'scripts'):
        for path in sorted((REPO_ROOT / root).rglob('*.py')):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in _SCAN_EXCLUDED:
                continue
            tree = ast.parse(path.read_text())
            query_dicts: set[int] = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.Dict):
                    for k, v in zip(n.keys, n.values, strict=True):
                        if isinstance(k, ast.Constant) and k.value in _QUERY_KEYS:
                            query_dicts.add(id(v))
            for n in ast.walk(tree):
                values: list[ast.expr] = []
                if isinstance(n, ast.Dict) and id(n) not in query_dicts:
                    values = [
                        v
                        for k, v in zip(n.keys, n.values, strict=True)
                        if isinstance(k, ast.Constant) and k.value == 'class_source'
                    ]
                elif isinstance(n, ast.Call):
                    values = [kw.value for kw in n.keywords if kw.arg == 'class_source']
                elif isinstance(n, ast.Assign):
                    values = [
                        n.value
                        for t in n.targets
                        if isinstance(t, ast.Attribute) and t.attr == 'class_source'
                    ]
                elif isinstance(n, ast.AnnAssign) and n.value is not None:
                    t = n.target
                    if (isinstance(t, ast.Attribute) and t.attr == 'class_source') or (
                        isinstance(t, ast.Name) and t.id == 'class_source'
                    ):
                        values = [n.value]
                for v in values:
                    out.extend((rel, v.lineno, w) for w in _written_values(v))
    return out


def test_scan_finds_the_known_writers() -> None:
    """Sanity: the scanner sees the writers it exists to police."""
    literals = {
        w.value
        for _, _, w in _class_source_writes()
        if isinstance(w, ast.Constant) and isinstance(w.value, str)
    }
    assert {'vlm', 'vlm_unmatched', 'vlm_new_class_pending', 'human_move'} <= literals


def test_every_written_class_source_is_in_the_catalog(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv('OP_INGEST_PRIMARY_NAME', 'prim')
    clean_env.setenv('OP_INGEST_PRIMARY_ASSIGNS_CLASS', 'true')
    clean_env.setenv('OP_INGEST_SECONDARY_NAME', 'sec')
    clean_env.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'clf')
    ids = set(_ids(class_source_catalog()))
    ingest_ids = {'prim_proposal', 'prim_low_conf', 'prim_model', 'sec_model'}
    problems: list[str] = []
    for rel, line, value in _class_source_writes():
        where = f'{rel}:{line}'
        if isinstance(value, ast.Dict | ast.Tuple):
            continue  # index mapping / field-group tuple, not a value
        if isinstance(value, ast.Constant):
            if value.value in (None, ''):
                continue
            if value.value not in ids:
                problems.append(f'{where}: {value.value!r} not in catalog')
        elif isinstance(value, ast.JoinedStr):
            tail = value.values[-1]
            suffix = tail.value if isinstance(tail, ast.Constant) else None
            if not any(i.endswith(str(suffix)) for i in ingest_ids & ids):
                problems.append(f'{where}: f-string suffix {suffix!r} not in catalog')
        else:
            text = ast.unparse(value)
            if text not in _DYNAMIC_WRITES:
                problems.append(f'{where}: unrecognised dynamic write {text!r}')
            elif (target := _DYNAMIC_WRITES[text]) is not None and target not in ids:
                problems.append(f'{where}: {text} writes {target!r}, not in catalog')
    assert not problems, '\n'.join(problems)


# ---------------------------------------------------------------------------
# VLM suggestion
# ---------------------------------------------------------------------------


def _doc(**kw: Any) -> dict[str, Any]:
    return {'class_id': 4, 'class_name': 'widget', 'class_validated': False, **kw}


@pytest.mark.parametrize('source', ['vlm', 'vlm_reclassified'])
def test_unvalidated_vlm_label_is_a_suggestion(source: str) -> None:
    assert vlm_suggestion(_doc(class_source=source)) == (4, 'widget')


def test_new_class_proposal_has_name_but_no_id() -> None:
    doc = _doc(class_source='vlm_new_class_pending', vlm_proposed_class='gizmo')
    assert vlm_suggestion(doc) == (None, 'gizmo')


@pytest.mark.parametrize(
    'doc',
    [
        _doc(class_source='vlm', class_validated=True),
        _doc(class_source='human', class_validated=True),
        _doc(class_source='vlm_unmatched', vlm_raw_class='mystery'),
        _doc(class_source='cluster_majority_agreement'),
        _doc(class_source='sec_model'),
        _doc(class_source='vlm', class_id=None),
        # Stale proposal field left over from an earlier pass.
        _doc(class_source='human', class_validated=True, vlm_proposed_class='gizmo'),
        _doc(class_source='vlm_new_class_pending'),
        {},
    ],
)
def test_no_suggestion(doc: dict[str, Any]) -> None:
    assert vlm_suggestion(doc) == (None, None)


def test_every_catalog_entry_has_a_short_badge_label() -> None:
    from src.services.curation.class_sources import class_source_catalog

    for entry in class_source_catalog():
        assert entry['short_label'], entry
        assert len(entry['short_label'].split()) <= 2, entry
