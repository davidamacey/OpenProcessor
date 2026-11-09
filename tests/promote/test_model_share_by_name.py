"""Sharing a promoted model writes `model_shares`; the consumer maps classes by name only."""

from __future__ import annotations

import asyncio

import pytest
from fixtures.job_rows import make_project, sql
from promote.test_promote_job import promote
from sqlalchemy import select

from src.clients.curation_opensearch import ClassRegistryFile, RegistryClassEntry
from src.db.models.training import Model
from src.db.session import async_session_factory, project_tx
from src.services.training import model_registry
from src.services.training.model_classes import (
    is_model_shared,
    model_class_mapping,
    model_owner_project,
    model_sharing_revision,
)


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def share(owner, name, consumers, *, shared=True, expected=1):
    return await model_registry.set_sharing(
        owner,
        name,
        shared=shared,
        expected_revision=expected,
        consumers=consumers,
        requested_by='test',
    )


async def test_a_shared_model_reaches_the_consumer_only(pg, job_env) -> None:
    owner, consumer, other = make_project(pg), make_project(pg), make_project(pg)
    await promote(owner, 'own__shared_det', 1, labels=('car', 'Truck', 'van'))

    assert await share(owner, 'own__shared_det', [consumer]) == 2

    async with async_session_factory()() as session, project_tx(session, consumer):
        seen = (await session.execute(select(Model.name))).scalars().all()
    assert seen == ['own__shared_det']  # the shared-read policy shows the model, nothing else
    async with async_session_factory()() as session, project_tx(session, other):
        assert (await session.execute(select(Model.name))).scalars().all() == []
    slug = sql(pg, 'SELECT slug FROM projects WHERE id = %s', consumer)[0][0]
    shared = await model_registry.shared_with(slug)
    assert [m.promotion.triton_model_name for m in shared] == ['own__shared_det']


async def test_the_consumer_maps_the_model_classes_by_name_not_by_position(pg, job_env) -> None:
    owner, consumer = make_project(pg, 'maporig'), make_project(pg, 'mapuser')
    await promote(owner, 'maporig__det', 1, labels=('car', 'Truck', 'van'))
    await share(owner, 'maporig__det', [consumer])
    registry = ClassRegistryFile(
        classes=[
            RegistryClassEntry(class_id=7, class_name='truck'),
            RegistryClassEntry(class_id=2, class_name='car'),
            RegistryClassEntry(class_id=9, class_name='bus'),
        ]
    )

    mapping = await model_class_mapping('maporig__det', registry, project='mapuser')

    by_name = {e.model_name: (e.class_id, e.match) for e in mapping.entries}
    assert by_name == {
        'car': (2, 'exact'),
        'Truck': (7, 'case_insensitive'),
        'van': (None, 'none'),
    }
    assert mapping.model_project == 'maporig'
    assert mapping.not_covered == ('bus',)
    assert await model_owner_project('maporig__det') == 'maporig'
    assert await is_model_shared('maporig__det', 'mapuser') is True
    assert await is_model_shared('maporig__det', 'nobody') is False


async def test_the_revision_only_grows_so_a_stale_writer_after_share_and_unshare_is_refused(
    pg, job_env
) -> None:
    owner, consumer = make_project(pg), make_project(pg)
    await promote(owner, 'own__rev_det', 1)

    with pytest.raises(model_registry.SharingRevisionConflictError) as stale:
        await share(owner, 'own__rev_det', [consumer], expected=99)
    assert stale.value.current_revision == 1
    assert await share(owner, 'own__rev_det', [consumer]) == 2
    assert await share(owner, 'own__rev_det', [consumer], shared=False, expected=2) == 3
    assert sql(
        pg,
        "SELECT count(*) FROM model_shares s JOIN models m ON m.id = s.model_id WHERE m.name = 'own__rev_det'",
    ) == [(0,)]
    assert await model_sharing_revision('own__rev_det') == 3

    # The state is private again, exactly as when the stale writer read revision 1.
    with pytest.raises(model_registry.SharingRevisionConflictError) as aba:
        await share(owner, 'own__rev_det', [consumer], expected=1)
    assert aba.value.current_revision == 3
    assert await model_sharing_revision('own__rev_det') == 3


async def test_two_concurrent_writers_on_one_revision_one_wins_one_is_refused(pg, job_env) -> None:
    owner, consumer, other = make_project(pg), make_project(pg), make_project(pg)
    await promote(owner, 'own__race_det', 1)

    # Both write on revision 1. Whichever takes the row lock first moves the revision to 2 and
    # the other is refused.
    results = await asyncio.gather(
        share(owner, 'own__race_det', [consumer]),
        share(owner, 'own__race_det', [other]),
        return_exceptions=True,
    )

    conflicts = [r for r in results if isinstance(r, model_registry.SharingRevisionConflictError)]
    assert len(conflicts) == 1
    assert len([r for r in results if isinstance(r, int)]) == 1


async def test_only_the_owner_can_share(pg, job_env) -> None:
    owner, stranger, consumer = make_project(pg), make_project(pg), make_project(pg)
    await promote(owner, 'own__locked_det', 1)

    with pytest.raises(model_registry.ModelNotOwnedError):
        await share(stranger, 'own__locked_det', [consumer])
    assert sql(
        pg,
        'SELECT count(*) FROM model_shares s JOIN models m ON m.id = s.model_id '
        "WHERE m.name = 'own__locked_det'",
    ) == [(0,)]
