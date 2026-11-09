"""ND-10: the 0.6.0 near-duplicate values are cleared, every project gets its profile, and the
report names the projects that should run the source backfill."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from src.services.upgrade.from_060.steps import near_duplicates as step


if TYPE_CHECKING:
    from tests.upgrade._v060 import V060Install

pytestmark = [pytest.mark.db, pytest.mark.asyncio]

ITEMS = 'op_prj_default__items'


def _dirty(v060: V060Install) -> int:
    """Put the 0.6.0 values on every item of the default project and on a core document."""
    docs = v060.os.docs(ITEMS)
    for doc in docs.values():
        doc.update(dup_group_id='dup_3', dup_group_size=4, dup_is_representative=True)
    v060.os.indexes[step.CORE_INDEX] = {
        'docs': {
            'core1': {'duplicate_group_id': 'g', 'is_duplicate_primary': False, 'x': 1},
            'core2': {'x': 2},
        },
        'mappings': {},
        'settings': {},
    }
    return len(docs)


async def test_the_values_leave_every_document_and_the_rest_of_it_stays(
    make_runner, clean_db, v060
) -> None:
    items = _dirty(v060)
    result = await make_runner().run()

    assert 'near_duplicates' in result.steps_run
    for doc in v060.os.docs(ITEMS).values():
        assert not [k for k in doc if k in step.ITEM_FIELDS]
    core = v060.os.docs(step.CORE_INDEX)
    assert core['core1'] == {'x': 1}
    assert core['core2'] == {'x': 2}
    assert items > 0


async def test_every_project_has_the_default_profile_and_no_run(make_runner, clean_db) -> None:
    await make_runner().run()
    with clean_db.engine('super').connect() as conn:
        profiles = conn.execute(
            text(
                'SELECT p.slug, d.name, r.revision FROM config_documents d JOIN projects p '
                'ON p.id = d.project_id JOIN config_revisions r ON r.config_id = d.id '
                "WHERE d.kind = 'near_dup_profile' ORDER BY 1"
            )
        ).all()
        runs = conn.execute(text('SELECT count(*) FROM near_dup_runs')).scalar_one()
    assert [(slug, name, rev) for slug, name, rev in profiles] == [
        ('default', 'default', 1),
        ('demo', 'default', 1),
        ('lean', 'default', 1),
    ]
    assert runs == 0


async def test_the_report_lists_the_projects_for_the_source_backfill(
    make_runner, clean_db, v060
) -> None:
    _dirty(v060)
    await make_runner().run()
    with clean_db.engine('super').connect() as conn:
        report = conn.execute(
            text("SELECT report FROM upgrade_runs WHERE step = 'load'")
        ).scalar_one()
    notes = report['notes']['near_duplicates']
    assert notes['start_source_backfill'] == ['default', 'demo', 'lean']
    assert notes['levels_needing_a_full_build'] == ['image', 'item']
    assert notes['values_cleared_by_index'][ITEMS] > 0
    assert notes['values_cleared_by_index'][step.CORE_INDEX] == 1
    assert report['counts']['near_dup_profiles'] == 3


async def test_a_second_clear_starts_no_task(v060) -> None:
    from types import SimpleNamespace

    _dirty(v060)
    ctx = SimpleNamespace(writer=v060.writer)
    first = await step.clear_index(ctx, ITEMS, step.ITEM_FIELDS)
    calls_after_first = list(v060.os.calls)
    second = await step.clear_index(ctx, ITEMS, step.ITEM_FIELDS)
    assert first > 0
    assert second == 0
    assert v060.os.calls[len(calls_after_first) :] == ['count']


HASH_A = 'a' * 32
HASH_B = 'b' * 32


async def test_core_images_load_their_content_key_and_wait_for_one_grouping(
    make_runner, clean_db, v060
) -> None:
    v060.os.indexes[step.CORE_INDEX] = {
        'docs': {
            'c1': {'imohash': HASH_A, 'image_path': '/img/c1.jpg', 'file_size_bytes': 10},
            'c2': {'imohash': HASH_B, 'image_path': '/img/c2.jpg', 'file_size_bytes': 11},
            'c3': {'image_path': '/img/c3.jpg'},
        },
        'mappings': {},
        'settings': {},
    }
    await make_runner().run()
    with clean_db.engine('super').connect() as conn:
        contents = conn.execute(
            text(
                'SELECT image_id, imohash, image_path, size_bytes FROM image_contents '
                'WHERE project_id IS NULL ORDER BY 1'
            )
        ).all()
        pending = (
            conn.execute(
                text(
                    "SELECT member_id FROM near_dup_pending WHERE level = 'core' "
                    'AND project_id IS NULL ORDER BY 1'
                )
            )
            .scalars()
            .all()
        )
        report = conn.execute(
            text("SELECT report FROM upgrade_runs WHERE step = 'load'")
        ).scalar_one()
    assert [tuple(r) for r in contents] == [
        ('c1', HASH_A, '/img/c1.jpg', 10),
        ('c2', HASH_B, '/img/c2.jpg', 11),
    ]
    assert pending == ['c1', 'c2', 'c3']
    notes = report['notes']['near_duplicates']
    assert notes['core_images_without_imohash'] == ['c3']
    assert notes['start_core_grouping'] is True
