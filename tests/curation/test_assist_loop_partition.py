"""The loop's pinned partition: roots read from ``cluster_id``, a snapshot per root in the
job directory, and the reader seam a versioned partition will implement.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.services.curation.assist.scope import membership_digest
from src.services.curation.assist_loop.partition import (
    ClusterFieldPartition,
    PartitionScope,
    read_snapshot,
    snapshot_line,
    write_snapshot,
)
from src.services.curation.class_write_guard import class_state_token


pytestmark = pytest.mark.asyncio

ITEMS = base_curation_config().items_index


def _item(crop_id: str, cluster_id: int, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'cluster_id': cluster_id,
        'class_id': None,
        'class_name': None,
        'class_source': 'item_proposal',
        'class_validated': False,
        **extra,
    }


def _world() -> QueryFakeOpenSearch:
    docs = [
        *(_item(f'a{i:02d}', 10001) for i in range(5)),
        _item(
            'a-human',
            10001,
            class_id=3,
            class_name='widget',
            class_source='human',
            class_validated=True,
        ),
        _item(
            'a-valid',
            10001,
            class_id=3,
            class_name='widget',
            class_source='vlm',
            class_validated=True,
        ),
        _item('a-frozen', 10001, test_holdout=True),
        _item('a-shelved', 10001, class_excluded=True),
        *(_item(f'b{i:02d}', 10002) for i in range(3)),
        *(
            _item(f'k{i:02d}', 4, class_id=4, class_name='gadget', class_source='vlm')
            for i in range(6)
        ),
        _item('noise', -1),
    ]
    return QueryFakeOpenSearch({ITEMS: {d['crop_id']: d for d in docs}})


async def test_roots_are_the_candidate_clusters_at_least_min_size() -> None:
    reader = ClusterFieldPartition(_world())  # type: ignore[arg-type]

    roots = await reader.roots(PartitionScope(cluster_ids=None, min_root_size=4))

    assert [(r.key, r.cluster_id) for r in roots] == [('r10001', 10001)]


async def test_named_clusters_are_roots_whatever_their_kind() -> None:
    reader = ClusterFieldPartition(_world())  # type: ignore[arg-type]

    roots = await reader.roots(PartitionScope(cluster_ids=[4, 10002, 99999], min_root_size=1))

    assert [r.key for r in roots] == ['r4', 'r10002']


async def test_a_root_snapshots_every_member_with_the_state_the_lock_rule_gives() -> None:
    reader = ClusterFieldPartition(_world())  # type: ignore[arg-type]
    (root,) = await reader.roots(PartitionScope(cluster_ids=[10001], min_root_size=1))

    states = {m.crop_id: m.state for m in root.members}
    assert states == {
        'a00': 'eligible',
        'a01': 'eligible',
        'a02': 'eligible',
        'a03': 'eligible',
        'a04': 'eligible',
        'a-human': 'locked',
        'a-valid': 'locked',
        'a-frozen': 'holdout',
        'a-shelved': 'excluded',
    }
    assert [m.crop_id for m in root.members] == sorted(states)
    assert root.version == membership_digest(list(states))
    assert root.eligible_ids == [f'a0{i}' for i in range(5)]
    human = next(m for m in root.members if m.crop_id == 'a-human')
    assert human.class_name == 'widget'


async def test_only_a_person_or_a_validated_import_is_ground_truth() -> None:
    reader = ClusterFieldPartition(_world())  # type: ignore[arg-type]
    (root,) = await reader.roots(PartitionScope(cluster_ids=[10001], min_root_size=1))

    truth = {m.crop_id for m in root.members if m.truth}

    assert truth == {'a-human'}, 'a class an automated writer validated is not evidence'


async def test_the_token_is_the_class_state_the_member_was_read_in() -> None:
    world = _world()
    (root,) = await ClusterFieldPartition(world).roots(  # type: ignore[arg-type]
        PartitionScope(cluster_ids=[10002], min_root_size=1)
    )

    for member in root.members:
        assert member.token == class_state_token(world.docs(ITEMS)[member.crop_id])


async def test_root_of_names_the_cluster_each_item_is_in_now() -> None:
    reader = ClusterFieldPartition(_world())  # type: ignore[arg-type]

    got = await reader.root_of(['a00', 'b01', 'k00', 'noise', 'missing'])

    assert got == {'a00': 'r10001', 'b01': 'r10002', 'k00': 'r4'}


async def test_a_snapshot_round_trips_through_the_job_directory(tmp_path: Any) -> None:
    reader = ClusterFieldPartition(_world())  # type: ignore[arg-type]
    (root,) = await reader.roots(PartitionScope(cluster_ids=[10001], min_root_size=1))

    line = write_snapshot(tmp_path, root)

    assert line == {
        'root': 'r10001',
        'cluster_id': 10001,
        'n': 9,
        'digest': root.version,
        'members_file': 'members/r10001.json',
    }
    assert snapshot_line(root) == line
    assert json.loads((tmp_path / 'members' / 'r10001.json').read_text())['root'] == 'r10001'
    assert read_snapshot(tmp_path, 'r10001') == root
    assert not list((tmp_path / 'members').glob('*.tmp'))


async def test_a_corrupt_snapshot_fails_closed(tmp_path: Any) -> None:
    (tmp_path / 'members').mkdir()
    (tmp_path / 'members' / 'r1.json').write_text('{"root": "r1", "members": [')

    with pytest.raises(ValueError, match='snapshot'):
        read_snapshot(tmp_path, 'r1')


async def test_a_missing_snapshot_fails_closed(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match='snapshot'):
        read_snapshot(tmp_path, 'r1')


@pytest.mark.parametrize('bad', ['../x', 'r1/../../x', 'x1', ''])
async def test_a_root_key_cannot_name_another_path(tmp_path: Any, bad: str) -> None:
    with pytest.raises(ValueError, match='path'):
        read_snapshot(tmp_path, bad)


async def test_the_survey_counts_exactly_what_a_snapshot_would_make_eligible() -> None:
    from src.domain.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE

    world = _world()
    world.docs(ITEMS)['a-import-suggestion'] = _item(
        'a-import-suggestion', 10001, class_source=LABEL_IMPORT_CLASS_SOURCE, class_name='widget'
    )
    world.docs(ITEMS)['a-import-valid'] = _item(
        'a-import-valid',
        10001,
        class_source=LABEL_IMPORT_CLASS_SOURCE,
        class_name='widget',
        class_validated=True,
    )
    reader = ClusterFieldPartition(world)  # type: ignore[arg-type]
    scope = PartitionScope(cluster_ids=[10001, 10002, 4], min_root_size=1)

    survey = await reader.survey(scope)
    roots = await reader.roots(scope)

    assert [(s.cluster_id, s.members, s.eligible) for s in survey] == [
        (r.cluster_id, len(r.members), len(r.eligible_ids)) for r in roots
    ]
    assert [s.cluster_id for s in survey] == [4, 10001, 10002]
    assert [s.eligible for s in survey] == [6, 6, 3]
    assert [s.members for s in survey] == [6, 11, 3]


async def test_the_survey_drops_clusters_under_the_minimum_size() -> None:
    reader = ClusterFieldPartition(_world())  # type: ignore[arg-type]

    survey = await reader.survey(PartitionScope(cluster_ids=None, min_root_size=4))

    assert [s.cluster_id for s in survey] == [10001]
