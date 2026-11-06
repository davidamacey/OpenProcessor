"""CLONEABLE_AXES includes ``activations``: cloning copies the source project's active
prompt-pack/region-profile config bodies plus the activation itself into the target
project's own (isolated) configs index, and refuses a profile whose detector is private to
the source project."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from config_store import memory_config

from projects.clone_flow_fakes import (  # noqa: F401 - isolated_state is an autouse fixture
    Fake,
    activate_config,
    active_ref,
    clone_into,
    isolated_state,
    patch_lifecycle,
    save,
)
from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import CLONEABLE_AXES, ProjectRecord, resources_for_new
from src.services.config_store.store import reset_config_stores
from src.services.projects.clone import _apply_clone


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


def test_activations_is_cloneable() -> None:
    assert 'activations' in CLONEABLE_AXES


@pytest.mark.asyncio
async def test_clone_activations_copies_active_pack_and_profile() -> None:
    client = Fake()
    source = _record('alpha')
    target = _record('beta')

    pack = await save(client, source, 'prompt_pack', 'wheel', {'class_system': 'wheel-pack'})
    await activate_config(client, source, 'prompt_pack', 'wheel', pack.revision)
    profile = await save(
        client, source, 'region_profile', 'wheel_profile', {'detector_model': 'wheel_v1'}
    )
    await activate_config(client, source, 'detection_profile', 'wheel_profile', profile.revision)

    await _apply_clone(
        target_record=target, source=source, axes=['activations'], target_activations=None
    )

    assert (await active_ref(target, 'prompt_pack'))['name'] == 'wheel'
    assert (await active_ref(target, 'detection_profile'))['name'] == 'wheel_profile'
    # Source untouched (still its own activation, never overwritten).
    assert (await active_ref(source, 'prompt_pack'))['name'] == 'wheel'


@pytest.mark.asyncio
async def test_clone_activations_copies_the_activated_revision_not_the_current_body() -> None:
    """A pack saved again after being activated diverges the current revision from the activated
    one: the clone must copy the body that was actually ACTIVE, not whatever the source has
    saved since."""
    client = Fake()
    source = _record('theta')
    target = _record('iota')

    doc_v1 = await save(client, source, 'prompt_pack', 'wheel', {'v': 1})
    await activate_config(client, source, 'prompt_pack', 'wheel', doc_v1.revision)
    # A later save moves the current revision to v2 without reactivating -- the activation
    # still points at v1.
    await save(client, source, 'prompt_pack', 'wheel', {'v': 2}, expected=doc_v1.revision)

    await _apply_clone(
        target_record=target, source=source, axes=['activations'], target_activations=None
    )

    cloned = memory_config.current().stored('prompt_pack', 'wheel', 'iota')
    assert cloned is not None
    assert cloned.revisions[cloned.current]['body'] == {'v': 1}


@pytest.mark.asyncio
async def test_a_clone_records_where_it_came_from() -> None:
    client = Fake()
    source, target = _record('kappa'), _record('lambda')
    pack = await save(client, source, 'prompt_pack', 'wheel', {'v': 1})
    await activate_config(client, source, 'prompt_pack', 'wheel', pack.revision)

    await _apply_clone(
        target_record=target, source=source, axes=['activations'], target_activations=None
    )

    with bind_project(target):
        from src.services.config_store.documents import get_revision

        cloned = await get_revision(kind='prompt_pack', name='wheel', revision=1)
    assert cloned is not None
    assert cloned.cloned_from == 'kappa:wheel@1'


@pytest.mark.asyncio
async def test_clone_activations_is_a_noop_when_source_has_none() -> None:
    source = _record('gamma')
    target = _record('delta')

    # No panic, no write, when the source axis was never activated.
    await _apply_clone(
        target_record=target, source=source, axes=['activations'], target_activations=None
    )

    assert await active_ref(target, 'prompt_pack') is None
    assert await active_ref(target, 'detection_profile') is None


@pytest.mark.asyncio
async def test_clone_activations_refuses_a_target_that_already_has_one_up_front(
    monkeypatch,
) -> None:
    """M5: cloning 'activations' into a target that already has its own
    active pack/profile must 409 target_not_empty BEFORE anything is
    written -- not a bare RevisionConflictError/ActiveConflictError."""
    from src.core.exceptions import RequestRejectedError

    client = Fake()
    source = _record('epsilon')
    target = _record('zeta')

    doc = await save(client, source, 'prompt_pack', 'wheel', {})
    await activate_config(client, source, 'prompt_pack', 'wheel', doc.revision)
    # Target already has its OWN active pack (a different one).
    tdoc = await save(client, target, 'prompt_pack', 'other', {})
    await activate_config(client, target, 'prompt_pack', 'other', tdoc.revision)

    patch_lifecycle(monkeypatch, source, target)
    with pytest.raises(RequestRejectedError) as exc_info:
        await clone_into(client, source, target)
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail['error'] == 'target_not_empty'

    # Nothing else got written -- the check ran before any write.
    assert (await active_ref(target, 'prompt_pack'))['name'] == 'other'  # unchanged


@pytest.mark.asyncio
async def test_clone_refuses_a_profile_whose_detector_is_private_to_the_source_project(
    monkeypatch,
) -> None:
    import src.services.training.model_classes as mc
    import src.services.training.promoted_models as pm
    from src.services.config_store.profile_validation import validate_profile
    from src.services.triton_control import TritonControlService

    owner = {'alpha__wheel_det': 'alpha'}
    monkeypatch.setattr(mc, 'model_owner_project', owner.get)
    monkeypatch.setattr(pm, 'model_owner_project', owner.get)
    monkeypatch.setattr(mc, 'is_model_shared', lambda _n: False)
    monkeypatch.setattr(pm, 'is_model_shared', lambda _n: False)

    # Triton must report the detector READY: the clone's activation gate also checks
    # that, and without it the clone would fail on `detector_model_not_found` before the
    # ownership check (`detector_model_not_shared`) ever runs, passing for the wrong reason.
    async def _repo():
        return [{'name': 'alpha__wheel_det', 'state': 'READY', 'version': '1'}]

    monkeypatch.setattr(TritonControlService, 'get_repository_index', lambda _s: _repo())

    client = Fake()
    source, target = _record('alpha'), _record('beta')
    body = {'detector_model': 'alpha__wheel_det', 'text_reader': 'none'}
    await save(client, source, 'region_profile', 'private_rp', body)
    await activate_config(client, source, 'detection_profile', 'private_rp', 1)

    # What the gate says in the TARGET's own context.
    with bind_project(target):
        report = await validate_profile(
            None,
            body,
            for_activation=True,
            project_slug='beta',
            get_repository_index=_repo,
        )
    codes = sorted(e.code for e in report.errors)
    print('target-context gate errors:', codes)
    assert 'detector_model_not_shared' in codes

    patch_lifecycle(monkeypatch, source, target)
    clone_codes: list[str] = []
    try:
        await clone_into(client, source, target)
        outcome = 'ok'
    except Exception as exc:
        outcome = repr(exc)[:200]
        cause = exc.__cause__
        detail = getattr(cause, 'detail', cause)
        if isinstance(detail, dict) and 'report' in detail:
            clone_codes = sorted(e['code'] for e in detail['report']['errors'])
    act = await active_ref(target, 'detection_profile')
    print(
        'clone ->',
        outcome,
        clone_codes,
        '| target detection_profile activation:',
        act and act.get('name'),
    )
    assert not (act and act.get('name') == 'private_rp'), (
        "clone activated a profile using another project's non-shared detector in the target"
    )
    # The clone must fail on the ownership check specifically, not on Triton not being set up.
    assert clone_codes == ['detector_model_not_shared'], clone_codes
