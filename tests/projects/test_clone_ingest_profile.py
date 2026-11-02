"""The ``ingest_profile`` clone axis: stored profiles and the activation of the
one in use are copied; the ACTIVATED revision's body becomes active; an occupied
target is refused."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from curation._fake_config_opensearch import FakeConfigOpenSearch

from src.config import get_curation_config
from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import CLONEABLE_AXES, ProjectRecord, resources_for_new
from src.core.exceptions import RequestRejectedError
from src.services.config_store.index import activate, get_activation, save_config
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


async def _seed_source(client: Any, source: ProjectRecord) -> None:
    with bind_project(source):
        idx = get_curation_config().configs_index
        first = await save_config(
            client,
            idx,
            kind='ingest_profile',
            name='cars',
            body=_body('m1'),
            expected_revision=None,
        )
        await activate(
            client,
            idx,
            axis='ingest_profile',
            name='cars',
            revision=first['revision'],
            expected_active=None,
        )
        await save_config(
            client,
            idx,
            kind='ingest_profile',
            name='cars',
            body=_body('m2'),
            expected_revision=first['revision'],
        )


def test_ingest_profile_is_a_cloneable_axis() -> None:
    assert 'ingest_profile' in CLONEABLE_AXES


@pytest.mark.asyncio
async def test_clone_copies_profiles_and_activates_the_activated_body() -> None:
    client = FakeConfigOpenSearch()
    source, target = _record('alpha'), _record('beta')
    await _seed_source(client, source)

    await _apply_clone(
        client,
        target_record=target,
        source=source,
        axes=['ingest_profile'],
        target_activations=None,
    )

    with bind_project(target):
        idx = get_curation_config().configs_index
        activation = await get_activation(client, idx, 'ingest_profile')
        store = get_config_store()
        await store.refresh(client)
        pinned = store.current.active_ingest_profile_body
    assert activation is not None
    assert activation['name'] == 'cars'
    assert pinned is not None
    assert pinned.body == _body('m1')  # the activated revision, not the later save


@pytest.mark.asyncio
async def test_an_occupied_target_is_refused() -> None:
    client = FakeConfigOpenSearch()
    target = _record('beta')
    with bind_project(target):
        await save_config(
            client,
            get_curation_config().configs_index,
            kind='ingest_profile',
            name='mine',
            body=_body('m'),
            expected_revision=None,
        )
    reset_config_stores()
    with pytest.raises(RequestRejectedError) as exc:
        await validate_named_axis_clone(client, axis=INGEST_PROFILE, target_record=target)
    assert exc.value.status_code == 409
    assert exc.value.detail['error'] == 'target_not_empty'
