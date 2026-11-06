"""The ``ingest_profile`` clone axis: stored profiles and the activation of the
one in use are copied; the ACTIVATED revision's body becomes active; an occupied
target is refused."""

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
from src.services.projects.clone_named_axis import INGEST_PROFILE, validate_named_axis_clone


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


def _body(model: str) -> dict[str, Any]:
    return {'primary': {'model': model}}


async def _seed_source(source: ProjectRecord) -> None:
    first = await save(None, source, 'ingest_profile', 'cars', _body('m1'))
    await activate_config(None, source, 'ingest_profile', 'cars', first.revision)
    await save(None, source, 'ingest_profile', 'cars', _body('m2'), expected=first.revision)


def test_ingest_profile_is_a_cloneable_axis() -> None:
    assert 'ingest_profile' in CLONEABLE_AXES


@pytest.mark.asyncio
async def test_clone_copies_profiles_and_activates_the_activated_body() -> None:
    source, target = _record('alpha'), _record('beta')
    await _seed_source(source)

    await _apply_clone(
        target_record=target, source=source, axes=['ingest_profile'], target_activations=None
    )

    activation = await active_ref(target, 'ingest_profile')
    with bind_project(target):
        store = get_config_store()
        await store.refresh()
        pinned = store.current.active_ingest_profile_body
    assert activation is not None
    assert activation['name'] == 'cars'
    assert pinned is not None
    assert pinned.body == _body('m1')  # the activated revision, not the later save


@pytest.mark.asyncio
async def test_an_occupied_target_is_refused() -> None:
    target = _record('beta')
    await save(None, target, 'ingest_profile', 'mine', _body('m'))
    reset_config_stores()
    with pytest.raises(RequestRejectedError) as exc:
        await validate_named_axis_clone(axis=INGEST_PROFILE, target_record=target)
    assert exc.value.status_code == 409
    assert exc.value.detail['error'] == 'target_not_empty'
