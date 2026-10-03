"""The ``open_vocab`` clone axis: stored prompt sets and the activation of the
one in use are copied into the target's own configs index; the ACTIVATED
revision's body is what becomes active, and an occupied target is refused."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from curation._fake_config_opensearch import FakeConfigOpenSearch
from fastapi import HTTPException

from src.config import get_curation_config
from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.routers.curation._project_models import CLONEABLE_AXES
from src.services.config_store.index import activate, get_activation, save_config
from src.services.config_store.store import get_config_store, reset_config_stores
from src.services.projects.clone import _apply_clone
from src.services.projects.clone_open_vocab import validate_open_vocab_clone


if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unbound


def _record(slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    reset_config_stores()
    yield
    reset_config_stores()


def _body(prompt: str) -> dict[str, Any]:
    return {'targets': [{'prompt': prompt, 'class_name': prompt}]}


async def _seed_source(client: Any, source: ProjectRecord) -> None:
    with bind_project(source):
        idx = get_curation_config().configs_index
        first = await save_config(
            client,
            idx,
            kind='open_vocab_set',
            name='cones',
            body=_body('cone'),
            expected_revision=None,
        )
        await activate(
            client,
            idx,
            axis='open_vocab',
            name='cones',
            revision=first['revision'],
            expected_active=None,
        )
        # Saved again after activation: current diverges from the activated revision.
        await save_config(
            client,
            idx,
            kind='open_vocab_set',
            name='cones',
            body=_body('cup'),
            expected_revision=first['revision'],
        )
        await save_config(
            client,
            idx,
            kind='open_vocab_set',
            name='other',
            body=_body('bin'),
            expected_revision=None,
        )


def test_open_vocab_is_a_cloneable_axis() -> None:
    assert 'open_vocab' in CLONEABLE_AXES


@pytest.mark.asyncio
async def test_clone_copies_sets_and_activates_the_activated_body() -> None:
    client = FakeConfigOpenSearch()
    source, target = _record('alpha'), _record('beta')
    await _seed_source(client, source)

    await _apply_clone(
        client, target_record=target, source=source, axes=['open_vocab'], target_activations=None
    )

    with bind_project(target):
        idx = get_curation_config().configs_index
        activation = await get_activation(client, idx, 'open_vocab')
        store = get_config_store()
        await store.refresh(client)
        sets = store.current.open_vocab_sets
        pinned = store.current.active_open_vocab_body
    assert sorted(sets) == ['cones', 'other']
    assert activation is not None
    assert activation['name'] == 'cones'
    assert pinned is not None
    assert pinned.body == _body('cone')  # the activated revision, not the later save
    # Same as the prompt-pack axis: a second revision carries the activated body.
    assert sets['cones'].body == _body('cone')
    assert sets['cones'].revision == 2
    assert (sets['other'].cloned_from or '').startswith('alpha:other@')

    # The source is untouched.
    with bind_project(source):
        src_activation = await get_activation(
            client, get_curation_config().configs_index, 'open_vocab'
        )
    assert src_activation is not None
    assert src_activation['revision'] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('occupied', ['set', 'activation'])
async def test_clone_refuses_an_occupied_target(occupied: str) -> None:
    client = FakeConfigOpenSearch()
    target = _record('delta')
    with bind_project(target):
        idx = get_curation_config().configs_index
        doc = await save_config(
            client,
            idx,
            kind='open_vocab_set',
            name='mine',
            body=_body('x'),
            expected_revision=None,
        )
        if occupied == 'activation':
            await activate(
                client,
                idx,
                axis='open_vocab',
                name='mine',
                revision=doc['revision'],
                expected_active=None,
            )
            from src.services.config_store.index import delete_config

            await delete_config(
                client, idx, kind='open_vocab_set', name='mine', expected_revision=doc['revision']
            )
    reset_config_stores()

    with pytest.raises(HTTPException) as exc:
        await validate_open_vocab_clone(client, target_record=target)
    assert exc.value.status_code == 409
    assert exc.value.detail['error'] == 'target_not_empty'
