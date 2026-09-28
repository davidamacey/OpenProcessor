"""Class-identity E2E (W10.17, the owner's "Class identity invariant"
section): class numbering is local to each boundary; the NAME is the
identity. This test walks import -> export (dense remap) -> stub-train
(class_remap.json) -> promote (labels.txt) -> predict, and asserts
``(class_id, class_name)`` stays correctly paired at every hop — nothing
ever crosses a boundary by raw index.

Two public-style YOLO fixtures share class names but use DIFFERENT
``data.yaml`` index orders, and fixture B adds one extra class
(``bus``) fixture A never saw. The pre-W10 bug (index-based mapping)
would silently swap ``car``/``truck`` here.

Structured as composable steps (module-level functions, not one
monolithic test) so P4 (combine-projects, W10.19) can insert an
additional "combine A+B into project C" hop between
:func:`step_import_fixture` and :func:`step_export_dense_mapping`
without rewriting this test.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.clients.curation_opensearch import ClassRegistry
from src.services.curation.dataset_import.job import (
    import_dataset,
    materialize_created_classes,
    registry_class_views,
)
from src.services.curation.dataset_import.mapping import (
    ClassMappingEntry,
    ResolvedMapping,
    resolve_mapping,
    suggest_mapping,
)
from src.services.curation.dataset_import.yolo import scan_yolo
from src.services.training.triton_promote import resolve_class_remap
from src.services.training.yolo_triton_config import render_labels_file


if TYPE_CHECKING:
    from src.services.curation.dataset_import.scan import DatasetScan


def _import_fakes():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingest_fakes import FakeOpenSearch

    return FakeOpenSearch


IMAGES_INDEX = 'op_curation_images'
ITEMS_INDEX = 'op_curation_items'


# =============================================================================
# Step 1: build two fixtures with the SAME class names in DIFFERENT index
# orders, fixture B carrying one extra class.
# =============================================================================


def _write_yolo_fixture(
    root: Path, *, names_by_index: dict[int, str], boxes: dict[str, list[str]]
) -> None:
    """``boxes``: {image_stem: [label rows]}, using CLASS NAME not index —
    resolved to this fixture's own index at write time, so the same
    logical box set can be written under two different orderings."""
    root.mkdir(parents=True, exist_ok=True)
    names_yaml = '\n'.join(f'  {i}: {n}' for i, n in sorted(names_by_index.items()))
    (root / 'data.yaml').write_text(f'train: images/train\nnames:\n{names_yaml}\n')
    name_to_index = {n: i for i, n in names_by_index.items()}
    (root / 'images/train').mkdir(parents=True, exist_ok=True)
    (root / 'labels/train').mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for stem, rows in boxes.items():
        Image.new('RGB', (100, 100), color='blue').save(
            root / f'images/train/{stem}.jpg', format='JPEG'
        )
        resolved_rows = []
        for row in rows:
            cls_name, cx, cy, w, h = row.split()
            resolved_rows.append(f'{name_to_index[cls_name]} {cx} {cy} {w} {h}')
        (root / f'labels/train/{stem}.txt').write_text('\n'.join(resolved_rows) + '\n')


def fixture_a(root: Path) -> None:
    """index order: {0: truck, 1: car} — reversed vs. fixture B."""
    _write_yolo_fixture(
        root,
        names_by_index={0: 'truck', 1: 'car'},
        boxes={
            'a1': ['car 0.3 0.3 0.2 0.2', 'truck 0.7 0.7 0.2 0.2'],
        },
    )


def fixture_b(root: Path) -> None:
    """index order: {0: car, 1: truck, 2: bus} — different order, plus a
    class fixture A never had."""
    _write_yolo_fixture(
        root,
        names_by_index={0: 'car', 1: 'truck', 2: 'bus'},
        boxes={
            'b1': ['car 0.2 0.2 0.1 0.1', 'truck 0.5 0.5 0.1 0.1', 'bus 0.8 0.8 0.1 0.1'],
        },
    )


# =============================================================================
# Step 2: import each fixture, mapping by name (never by index).
# =============================================================================


def step_import_fixture(
    root: Path, registry: ClassRegistry, opensearch, *, import_id: str
) -> tuple[DatasetScan, ResolvedMapping]:
    scan = scan_yolo(root)
    views = registry_class_views(registry)
    dataset_classes = [c for c, n in scan.class_box_counts.items() if n > 0]
    suggestions = {c: suggest_mapping(c, registry_classes=views) for c in dataset_classes}
    entries = []
    for dataset_class in dataset_classes:
        s = suggestions[dataset_class]
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
    resolved = resolve_mapping(dataset_classes, entries, registry_classes=views)
    assert resolved.ok, resolved.errors
    materialize_created_classes(resolved, registry)
    return scan, resolved


async def run_import(root: Path, registry: ClassRegistry, opensearch, *, import_id: str):
    scan, resolved = step_import_fixture(root, registry, opensearch, import_id=import_id)
    report = await import_dataset(
        opensearch,
        scan,
        resolved,
        import_id=import_id,
        images_index=IMAGES_INDEX,
        items_index=ITEMS_INDEX,
    )
    return report, resolved


# =============================================================================
# Step 3: "export" — the dense id assignment a real exporter would freeze
# into class_remap.json (project class_id -> dense id, sorted by class_id;
# this is export.py's export_id_map convention, W10.2.2).
# =============================================================================


def step_export_dense_mapping(registry: ClassRegistry) -> tuple[dict[int, int], list[str]]:
    reg = registry.load()
    non_deprecated = sorted((c for c in reg.classes if not c.deprecated), key=lambda c: c.class_id)
    dense_mapping = {c.class_id: i for i, c in enumerate(non_deprecated)}
    names = [c.class_name for c in non_deprecated]
    return dense_mapping, names


# =============================================================================
# Step 4: "stub-train" — write the class_remap.json shape
# docker/trainer/dataset_prep.py's write_full_class_remap produces (no
# real training: this exercises the CONTRACT, not the trainer).
# =============================================================================


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
# project (class_id, class_name) via the remap's inverse, never a guess.
# =============================================================================


def step_predict(dense_class_id: int, remap) -> tuple[int, str | None, int]:
    new_to_original = {v: k for k, v in remap.mapping.items()}
    registry_class_id = new_to_original[dense_class_id]
    name = remap.names[dense_class_id] if remap.names else None
    return registry_class_id, name, dense_class_id


@pytest.mark.asyncio
async def test_class_identity_holds_at_every_hop(tmp_path: Path) -> None:
    FakeOpenSearch = _import_fakes()
    opensearch = FakeOpenSearch()
    registry = ClassRegistry(path=tmp_path / 'class_registry.json')

    root_a = tmp_path / 'fixture_a'
    root_b = tmp_path / 'fixture_b'
    fixture_a(root_a)
    fixture_b(root_b)

    report_a, resolved_a = await run_import(root_a, registry, opensearch, import_id='imp_a')
    assert report_a.items_created == 2
    report_b, resolved_b = await run_import(root_b, registry, opensearch, import_id='imp_b')
    assert report_b.items_created == 3

    reg = registry.load()
    names_by_id = {c.class_id: c.class_name for c in reg.classes}
    assert set(names_by_id.values()) == {'car', 'truck', 'bus'}
    # The two imports' resolved targets must agree on car/truck's ids
    # (fixture B did not recreate them despite a different data.yaml order).
    assert resolved_a.targets['car'].class_id == resolved_b.targets['car'].class_id
    assert resolved_a.targets['truck'].class_id == resolved_b.targets['truck'].class_id

    # Every written item's (class_id, class_name) pair matches the registry.
    for doc in opensearch.items.values():
        assert names_by_id[doc['class_id']] == doc['class_name']

    dense_mapping, dense_names = step_export_dense_mapping(registry)
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
