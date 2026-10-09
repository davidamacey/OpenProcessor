"""Class-identity E2E (W10.17, the owner's "Class identity invariant"
section): class numbering is local to each boundary; the NAME is the
identity. This test walks import -> export (real exporter) -> stub-train
(class_remap.json) -> promote (labels.txt) -> predict, and asserts a
SPECIFIC box's geometry stays paired with its SPECIFIC class name at
every hop — nothing ever crosses a boundary by raw index.

Two public-style YOLO fixtures share class names but use DIFFERENT
``data.yaml`` index orders, and fixture B adds one extra class
(``bus``) fixture A never saw. The pre-W10 bug (index-based mapping)
would silently swap ``car``/``truck`` here.

W10 fix-pass note (Opus review 2026-09-28, finding M4): the prior
version of this test passed even with ``yolo._names_list`` (now
``_names_map``) monkeypatched to return class names reversed — every
hop it asserted either compared two values both derived from the same
map (so a swap always agreed with itself) or used test-local code that
never touched a real writer/reader. This version:

* asserts, per box, that the SPECIFIC geometry written for that box
  resolves to the SPECIFIC class name the fixture assigned it (never
  an aggregate count or set-membership check);
* exports through the real production exporter
  (:class:`~src.services.curation.export.GenericYoloExportService`)
  against the same in-memory index the import wrote, then re-parses
  the label rows it wrote plus ``data.yaml`` and re-ties each row's
  geometry back to its fixture name;
* derives the dense id map from the real export's
  ``class_registry.json`` artifact (``export_id_map``), not test-local
  code, for the stub-train/promote/predict hops.

Structured as composable steps (module-level functions, not one
monolithic test) so P4 (combine-projects, W10.19) can insert an
additional "combine A+B into project C" hop between
:func:`step_import_fixture` and the export hop without rewriting this
test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from curation.dataset_import.harness import Harness

from src.config.curation import CurationConfig
from src.services.curation.dataset_import.mapping import ClassMappingEntry
from src.services.curation.export import GenericYoloExportService
from src.services.detection.geometry import crop_id as _crop_id
from src.services.training.class_remap import resolve_class_remap
from src.services.training.yolo_triton_config import render_labels_file


if TYPE_CHECKING:
    from src.clients.curation_opensearch.registry import ClassRegistry


# =============================================================================
# Step 1: build two fixtures with the SAME class names in DIFFERENT index
# orders, fixture B carrying one extra class. ``boxes`` uses class NAME,
# not index, and every row's normalized bbox is recorded so later hops
# can tie a specific geometry back to its intended name.
# =============================================================================


def _row_to_bbox(row: str) -> tuple[str, tuple[float, float, float, float]]:
    cls_name, cx_s, cy_s, w_s, h_s = row.split()
    cx, cy, w, h = float(cx_s), float(cy_s), float(w_s), float(h_s)
    return cls_name, (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _write_yolo_fixture(
    root: Path, *, names_by_index: dict[int, str], boxes: dict[str, list[str]]
) -> dict[tuple[float, ...], str]:
    """Writes the fixture; returns ``{rounded_bbox: expected_class_name}``
    for every box, keyed by the SAME 6-decimal rounding the export writer
    uses, so later hops can look a written row's geometry straight up."""
    root.mkdir(parents=True, exist_ok=True)
    names_yaml = '\n'.join(f'  {i}: {n}' for i, n in sorted(names_by_index.items()))
    (root / 'data.yaml').write_text(f'train: images/train\nnames:\n{names_yaml}\n')
    name_to_index = {n: i for i, n in names_by_index.items()}
    (root / 'images/train').mkdir(parents=True, exist_ok=True)
    (root / 'labels/train').mkdir(parents=True, exist_ok=True)
    from PIL import Image

    expected: dict[tuple[float, ...], str] = {}
    for stem, rows in boxes.items():
        Image.new(
            'RGB', (100, 100), color=tuple(ord(c) * 7 % 256 for c in (stem + 'xxx')[:3])
        ).save(root / f'images/train/{stem}.jpg', format='JPEG')
        resolved_rows = []
        for row in rows:
            cls_name, cx, cy, w, h = row.split()
            resolved_rows.append(f'{name_to_index[cls_name]} {cx} {cy} {w} {h}')
            _, bbox = _row_to_bbox(row)
            expected[tuple(round(v, 6) for v in bbox)] = cls_name
        (root / f'labels/train/{stem}.txt').write_text('\n'.join(resolved_rows) + '\n')
    return expected


def fixture_a(root: Path) -> dict[tuple[float, ...], str]:
    """index order: {0: truck, 1: car} — reversed vs. fixture B."""
    return _write_yolo_fixture(
        root,
        names_by_index={0: 'truck', 1: 'car'},
        boxes={
            'a1': ['car 0.3 0.3 0.2 0.2', 'truck 0.7 0.7 0.2 0.2'],
        },
    )


def fixture_b(root: Path) -> dict[tuple[float, ...], str]:
    """index order: {0: car, 1: truck, 2: bus} — different order, plus a
    class fixture A never had."""
    return _write_yolo_fixture(
        root,
        names_by_index={0: 'car', 1: 'truck', 2: 'bus'},
        boxes={
            'b1': ['car 0.2 0.2 0.1 0.1', 'truck 0.5 0.5 0.1 0.1', 'bus 0.8 0.8 0.1 0.1'],
        },
    )


# =============================================================================
# Step 2: import each fixture through the real importer, mapping by name
# (never by index): the suggestion ladder maps what the registry knows and
# the rest is created.
# =============================================================================


async def run_import(h: Harness, root: Path, *, name: str):
    probe = h.prepare(h.request(root))
    entries = []
    for dataset_class, s in probe.suggestions.items():
        if s.action == 'map':
            entries.append(
                ClassMappingEntry(dataset_class=dataset_class, action='map', class_id=s.class_id)
            )
        else:
            entries.append(
                ClassMappingEntry(
                    dataset_class=dataset_class, action='create', new_class_name=dataset_class
                )
            )
    store, _ = await h.run(h.request(root, entries, name=name))
    assert store.job.read()['status'] == 'completed'
    return store


def assert_written_items_match_fixture_geometry(
    h: Harness, root: Path, expected: dict[tuple[float, ...], str]
) -> None:
    """Tie each SPECIFIC box's geometry to its SPECIFIC written class
    name, by recomputing the same crop_id the importer used
    (image_id + bbox) and reading that exact document back. A name-set
    or aggregate-count assertion would pass even if two boxes' names
    were swapped; this cannot, because it addresses one document per
    fixture box by its geometry-derived id."""
    image_ids = {
        d['import_source_stem']: d['image_id']
        for d in h.images.values()
        if d.get('image_path', '').startswith(str(root))
    }
    assert image_ids, 'fixture wrote no images'
    matched = 0
    for bbox, expected_name in expected.items():
        doc = next(
            (
                h.items[cid]
                for image_id in image_ids.values()
                if (cid := _crop_id(image_id, list(bbox))) in h.items
            ),
            None,
        )
        assert doc is not None, f'no item doc found for fixture box {bbox} ({expected_name})'
        assert doc['class_name'] == expected_name, (
            f'box {bbox} imported as {doc["class_name"]!r}, expected {expected_name!r}'
        )
        matched += 1
    assert matched == len(expected)


# =============================================================================
# Step 3: export through the REAL production exporter, then re-parse the
# label rows + data.yaml it wrote and re-tie each row's geometry back to
# its fixture name.
# =============================================================================


async def step_real_export(
    h: Harness, registry: ClassRegistry, tmp_path: Path
) -> tuple[Path, dict[int, int]]:
    cfg = CurationConfig(
        items_index=h.cfg.items_index,
        images_index=h.cfg.images_index,
        export_root=tmp_path / 'exports',
    )
    service = GenericYoloExportService(h.os, config=cfg, registry=registry)
    result = await service.export_dataset(version_tag='e2e', copy_images=False)
    export_dir = Path(result.export_dir)
    class_registry_payload = json.loads((export_dir / 'class_registry.json').read_text())
    dense_mapping = {int(k): v for k, v in class_registry_payload['export_id_map'].items()}
    return export_dir, dense_mapping


def assert_exported_labels_match_fixture_geometry(
    export_dir: Path, expected_by_bbox: dict[tuple[float, ...], str]
) -> None:
    """Re-read the real exporter's own written artifacts (never test-local
    code) and re-tie each label row's geometry back to its fixture name."""
    data_yaml = (export_dir / 'data.yaml').read_text()
    names_line = next(line for line in data_yaml.splitlines() if line.startswith('names:'))
    names: list[str] = json.loads(names_line.removeprefix('names:').strip())

    matched = 0
    for label_file in sorted(export_dir.glob('labels/*/*.txt')):
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            cls_idx_str, cx_s, cy_s, w_s, h_s = line.split()
            cls_idx = int(cls_idx_str)
            cx, cy, w, h = float(cx_s), float(cy_s), float(w_s), float(h_s)
            bbox = (
                round(cx - w / 2.0, 6),
                round(cy - h / 2.0, 6),
                round(cx + w / 2.0, 6),
                round(cy + h / 2.0, 6),
            )
            expected_name = expected_by_bbox.get(bbox)
            assert expected_name is not None, f'exported row {bbox} matches no fixture box'
            assert names[cls_idx] == expected_name, (
                f'exported row {bbox}: names[{cls_idx}] == {names[cls_idx]!r}, '
                f'expected {expected_name!r}'
            )
            matched += 1
    assert matched == len(expected_by_bbox)


# =============================================================================
# Step 4: "stub-train" — write the class_remap.json shape
# docker/trainer/dataset_prep.py's write_full_class_remap produces (no
# real training: this exercises the CONTRACT, not the trainer). The
# dense mapping itself now comes from the real exporter's
# class_registry.json (step_real_export), not test-local code.
# =============================================================================


def _targets(store) -> list[dict]:
    raw = store.read_mapping()['targets']
    return [{'dataset_class': k, **v} for k, v in raw.items()]


def step_stub_train_manifest(dense_mapping: dict[int, int], names: list[str]) -> dict:
    return {
        'lineage': {
            'class_remap': {
                'original_to_new': {str(k): v for k, v in dense_mapping.items()},
                'new_to_original': {str(v): k for k, v in dense_mapping.items()},
                'names': names,
                'single_cls': False,
            }
        }
    }


# =============================================================================
# Step 5: "promote" — resolve_class_remap() + render_labels_file(), the
# real functions triton_promote.TritonPromoteService.promote() calls.
# =============================================================================


def step_promote_labels(manifest: dict, registry: ClassRegistry) -> tuple[dict[int, str], dict]:
    remap = resolve_class_remap(
        job_id='stub_job', checkpoint_path=Path('/tmp/does-not-exist/best.onnx'), manifest=manifest
    )
    reg = registry.load()
    class_name_by_id = {c.class_id: c.class_name for c in reg.classes}
    class_id_to_name = {
        dense_id: class_name_by_id[registry_id] for registry_id, dense_id in remap.mapping.items()
    }
    labels_text = render_labels_file(class_id_to_name)
    return class_id_to_name, {'remap': remap, 'labels_text': labels_text}


# =============================================================================
# Step 6: "predict" — a raw model output (dense id) resolves back to the
# project (class_id, class_name) via the PRODUCTION remap's own inverse
# (``remap.mapping`` / ``remap.names`` are resolve_class_remap's real
# output; only the dict-inversion arithmetic here is test-local, and it
# operates on production data, not a test-built substitute).
# =============================================================================


def step_predict(dense_class_id: int, remap) -> tuple[int, str | None, int]:
    new_to_original = {v: k for k, v in remap.mapping.items()}
    registry_class_id = new_to_original[dense_class_id]
    name = remap.names[dense_class_id] if remap.names else None
    return registry_class_id, name, dense_class_id


@pytest.mark.asyncio
async def test_class_identity_holds_at_every_hop(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path / 'state', monkeypatch, root=tmp_path)
    registry = h.registry

    root_a = tmp_path / 'fixture_a'
    root_b = tmp_path / 'fixture_b'
    expected_a = fixture_a(root_a)
    expected_b = fixture_b(root_b)
    expected_all = {**expected_a, **expected_b}

    store_a = await run_import(h, root_a, name='imp_a')
    assert store_a.job.read()['report']['items_created'] == 2
    store_b = await run_import(h, root_b, name='imp_b')
    assert store_b.job.read()['report']['items_created'] == 3
    resolved_a = {t['dataset_class']: t for t in _targets(store_a)}
    resolved_b = {t['dataset_class']: t for t in _targets(store_b)}

    reg = registry.load()
    names_by_id = {c.class_id: c.class_name for c in reg.classes}
    assert set(names_by_id.values()) == {'car', 'truck', 'bus'}
    # The two imports' resolved targets must agree on car/truck's ids
    # (fixture B did not recreate them despite a different data.yaml order).
    assert resolved_a['car']['class_id'] == resolved_b['car']['class_id']
    assert resolved_a['truck']['class_id'] == resolved_b['truck']['class_id']

    # M4 fix: tie EACH box's geometry to its SPECIFIC written class name,
    # not an aggregate name-set / count check that a car<->truck swap
    # would still pass.
    assert_written_items_match_fixture_geometry(h, root_a, expected_a)
    assert_written_items_match_fixture_geometry(h, root_b, expected_b)

    # M4 fix: export through the real production exporter, not test-local
    # dense-mapping code.
    export_dir, dense_mapping = await step_real_export(h, registry, tmp_path)
    assert_exported_labels_match_fixture_geometry(export_dir, expected_all)

    dense_names: list[str] = [''] * len(dense_mapping)
    for registry_id, dense_id in dense_mapping.items():
        dense_names[dense_id] = names_by_id[registry_id]

    manifest = step_stub_train_manifest(dense_mapping, dense_names)
    class_id_to_name, promote_ctx = step_promote_labels(manifest, registry)
    remap = promote_ctx['remap']

    # labels.txt (dense order) must name the SAME class the registry does
    # for every registry id the remap covers.
    labels_lines = promote_ctx['labels_text'].splitlines()
    for registry_id, dense_id in remap.mapping.items():
        assert labels_lines[dense_id] == names_by_id[registry_id]

    # predict: round-trip every class through a raw dense id.
    for class_name in ('car', 'truck', 'bus'):
        registry_id = next(cid for cid, n in names_by_id.items() if n == class_name)
        dense_id = dense_mapping[registry_id]
        predicted_id, predicted_name, _ = step_predict(dense_id, remap)
        assert predicted_id == registry_id
        assert predicted_name == class_name
        assert class_id_to_name[dense_id] == class_name
