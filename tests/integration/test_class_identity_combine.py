"""The class identity invariant across a combine (any_domain_plan.md "Class
identity invariant", boundary: Combine): two projects number the same class
names differently; the combined project owns its own ids, and a box keeps its
NAME through combine -> export -> train remap -> promote -> predict.

Extends ``test_class_identity_e2e``: it reuses that test's export, remap,
promote and predict hops on the combined project, and ties every box's
geometry (not an aggregate) to its class name at the combine hop.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from projects.combine.world import World, run_job

from integration.test_class_identity_e2e import (
    _row_to_bbox,
    assert_exported_labels_match_fixture_geometry,
    step_predict,
    step_promote_labels,
    step_real_export,
    step_stub_train_manifest,
)
from src.clients.curation_opensearch import ClassRegistry
from src.services.projects.combine import service


if TYPE_CHECKING:
    from pathlib import Path


# project -> its registry order -> the boxes of one frame, by class NAME.
A_ORDER = ['truck', 'car']
B_ORDER = ['car', 'truck', 'bus']
A_BOXES = ['car 0.3 0.3 0.2 0.2', 'truck 0.7 0.7 0.2 0.2']
B_BOXES = ['car 0.2 0.2 0.1 0.1', 'truck 0.5 0.5 0.1 0.1', 'bus 0.8 0.8 0.1 0.1']
VALIDATED = {'class_validated': True, 'class_source': 'human', 'label_source': 'human'}


def populate(world: World, slug: str, rows: list[str]) -> dict[tuple[float, ...], str]:
    expected: dict[tuple[float, ...], str] = {}
    specs = []
    for row in rows:
        name, bbox = _row_to_bbox(row)
        specs.append({'cls': name, 'bbox': list(bbox), **VALIDATED})
        expected[tuple(round(v, 6) for v in bbox)] = name
    world.add_image(slug, items=specs)
    return expected


@pytest.mark.asyncio
async def test_class_identity_holds_through_a_combine(tmp_path: Path, monkeypatch) -> None:
    world = World(tmp_path, monkeypatch)
    world.project('proj-a', A_ORDER)
    world.project('proj-b', B_ORDER)
    # The two registries really do number the same names differently.
    assert world.registry_ids('proj-a')['car'] != world.registry_ids('proj-b')['car']
    expected = {**populate(world, 'proj-a', A_BOXES), **populate(world, 'proj-b', B_BOXES)}

    # The mapping is W10's own name-based suggestion, accepted as it stands.
    request = world.request(['proj-a', 'proj-b'], {})
    preview, _ = await service.preview(world.fake, request)
    request = world.request(['proj-a', 'proj-b'], {
        p: [r.model_dump(exclude_none=True) for r in rows]
        for p, rows in preview.suggested_mapping.items()
    })  # fmt: skip
    store, target = await run_job(world, request)
    assert store.job.read()['status'] == 'completed'

    registry = ClassRegistry(path=target.resources.class_registry_path)
    names_by_id = {c.class_id: c.class_name for c in registry.load().classes}
    assert set(names_by_id.values()) == {'car', 'truck', 'bus'}  # one class per name

    # Combine hop: tie each SPECIFIC box's geometry to its SPECIFIC class name
    # AND to the target's own id for that name.
    items = list(world.items('combined').values())
    assert len(items) == len(expected)
    for doc in items:
        bbox = tuple(round(v, 6) for v in doc['bbox_norm'])
        assert doc['class_name'] == expected[bbox], bbox
        assert names_by_id[doc['class_id']] == expected[bbox], bbox

    # Export / remap / promote / predict on the combined project.
    h = SimpleNamespace(
        os=world.fake,
        cfg=SimpleNamespace(
            items_index=world.items_index('combined'), images_index=world.images_index('combined')
        ),
    )
    export_dir, dense_mapping = await step_real_export(h, registry, tmp_path)  # type: ignore[arg-type]
    assert_exported_labels_match_fixture_geometry(export_dir, expected)

    dense_names = [''] * len(dense_mapping)
    for registry_id, dense_id in dense_mapping.items():
        dense_names[dense_id] = names_by_id[registry_id]
    manifest = step_stub_train_manifest(dense_mapping, dense_names)
    class_id_to_name, promote = step_promote_labels(manifest, registry)
    remap = promote['remap']
    labels = promote['labels_text'].splitlines()
    for registry_id, dense_id in remap.mapping.items():
        assert labels[dense_id] == names_by_id[registry_id]
    for name in ('car', 'truck', 'bus'):
        registry_id = next(cid for cid, n in names_by_id.items() if n == name)
        dense_id = dense_mapping[registry_id]
        assert step_predict(dense_id, remap)[:2] == (registry_id, name)
        assert class_id_to_name[dense_id] == name
