"""The ``open_vocab`` clone axis: stored prompt sets and the activation of the
one in use are copied into the target's own configs index; the ACTIVATED
revision's body is what becomes active, and an occupied target is refused."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from projects.clone_flow_fakes import activate_config, active_ref, save
from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import CLONEABLE_AXES, ProjectRecord, resources_for_new
from src.core.exceptions import RequestRejectedError
from src.services.config_store.store import get_config_store, reset_config_stores
from src.services.projects.clone import _apply_clone
from src.services.projects.clone_named_axis import OPEN_VOCAB, validate_named_axis_clone


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


async def _seed_source(source: ProjectRecord) -> None:
    first = await save(None, source, 'open_vocab_set', 'cones', _body('cone'))
    await activate_config(None, source, 'open_vocab', 'cones', first.revision)
    # Saved again after activation: current diverges from the activated revision.
    await save(None, source, 'open_vocab_set', 'cones', _body('cup'), expected=first.revision)
    await save(None, source, 'open_vocab_set', 'other', _body('bin'))


def test_open_vocab_is_a_cloneable_axis() -> None:
    assert 'open_vocab' in CLONEABLE_AXES


@pytest.mark.asyncio
async def test_clone_copies_sets_and_activates_the_activated_body() -> None:
    source, target = _record('alpha'), _record('beta')
    await _seed_source(source)

    await _apply_clone(
        target_record=target, source=source, axes=['open_vocab'], target_activations=None
    )

    activation = await active_ref(target, 'open_vocab')
    with bind_project(target):
        store = get_config_store()
        await store.refresh()
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
    src_activation = await active_ref(source, 'open_vocab')
    assert src_activation is not None
    assert src_activation['revision'] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('occupied', ['set', 'activation'])
async def test_clone_refuses_an_occupied_target(occupied: str) -> None:
    target = _record('delta')
    doc = await save(None, target, 'open_vocab_set', 'mine', _body('x'))
    if occupied == 'activation':
        await activate_config(None, target, 'open_vocab', 'mine', doc.revision)
    reset_config_stores()

    with pytest.raises(RequestRejectedError) as exc:
        await validate_named_axis_clone(axis=OPEN_VOCAB, target_record=target)
    assert exc.value.status_code == 409
    assert exc.value.detail['error'] == 'target_not_empty'
