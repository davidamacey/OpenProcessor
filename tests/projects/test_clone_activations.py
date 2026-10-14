"""CLONEABLE_AXES includes ``activations``: cloning copies the source project's active
prompt-pack/region-profile config bodies plus the activation itself into the target
project's own (isolated) configs index, and refuses a profile whose detector is private to
the source project."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from curation._fake_config_opensearch import FakeConfigOpenSearch

from projects.clone_flow_fakes import (  # noqa: F401 - isolated_state is an autouse fixture
    Fake,
    activate_config,
    clone_into,
    isolated_state,
    patch_lifecycle,
    save,
)
from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import CLONEABLE_AXES, ProjectRecord, resources_for_new
from src.services.config_store.index import get_activation
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
    client = FakeConfigOpenSearch()
    source = _record('alpha')
    target = _record('beta')

    from src.services.config_store.index import activate, save_config

    with bind_project(source):
        from src.config import get_curation_config

        idx = get_curation_config().configs_index
        pack_doc = await save_config(
            client,
            idx,
            kind='prompt_pack',
            name='wheel',
            body={'class_system': 'wheel-pack'},
            expected_revision=None,
        )
        await activate(
            client,
            idx,
            axis='prompt_pack',
            name='wheel',
            revision=pack_doc['revision'],
            expected_active=None,
        )
        profile_doc = await save_config(
            client,
            idx,
            kind='region_profile',
            name='wheel_profile',
            body={'detector_model': 'wheel_v1'},
            expected_revision=None,
        )
        await activate(
            client,
            idx,
            axis='detection_profile',
            name='wheel_profile',
            revision=profile_doc['revision'],
            expected_active=None,
        )

    await _apply_clone(
        client, target_record=target, source=source, axes=['activations'], target_activations=None
    )

    with bind_project(target):
        from src.config import get_curation_config

        target_idx = get_curation_config().configs_index
        pack_activation = await get_activation(client, target_idx, 'prompt_pack')
        profile_activation = await get_activation(client, target_idx, 'detection_profile')

    assert pack_activation is not None
    assert pack_activation['name'] == 'wheel'
    assert profile_activation is not None
    assert profile_activation['name'] == 'wheel_profile'

    # Source untouched (still its own activation, never overwritten).
    with bind_project(source):
        from src.config import get_curation_config

        source_idx = get_curation_config().configs_index
        source_activation = await get_activation(client, source_idx, 'prompt_pack')
    assert source_activation is not None
    assert source_activation['name'] == 'wheel'


@pytest.mark.asyncio
async def test_clone_activations_copies_the_activated_revision_not_the_current_body() -> None:
    """a pack saved again after being activated
    diverges ``pack:<name>`` (current) from ``pack:<name>@<rev>`` (the
    activated revision's immutable copy) -- the clone must copy the body
    that was actually ACTIVE, not whatever the source has saved since."""
    client = FakeConfigOpenSearch()
    source = _record('theta')
    target = _record('iota')

    from src.services.config_store.index import activate, config_doc_id, save_config

    with bind_project(source):
        from src.config import get_curation_config

        idx = get_curation_config().configs_index
        doc_v1 = await save_config(
            client, idx, kind='prompt_pack', name='wheel', body={'v': 1}, expected_revision=None
        )
        await activate(
            client,
            idx,
            axis='prompt_pack',
            name='wheel',
            revision=doc_v1['revision'],
            expected_active=None,
        )
        # A later save moves `pack:wheel` (current) to v2 without
        # reactivating -- the activation still points at v1.
        await save_config(
            client,
            idx,
            kind='prompt_pack',
            name='wheel',
            body={'v': 2},
            expected_revision=doc_v1['revision'],
        )

    await _apply_clone(
        client, target_record=target, source=source, axes=['activations'], target_activations=None
    )

    with bind_project(target):
        from src.config import get_curation_config

        target_idx = get_curation_config().configs_index
        cloned = await client.get(index=target_idx, id=config_doc_id('prompt_pack', 'wheel'))
    assert cloned['_source']['body'] == {'v': 1}


@pytest.mark.asyncio
async def test_clone_activations_is_a_noop_when_source_has_none() -> None:
    client = FakeConfigOpenSearch()
    source = _record('gamma')
    target = _record('delta')

    # No panic, no write, when the source axis was never activated.
    await _apply_clone(
        client, target_record=target, source=source, axes=['activations'], target_activations=None
    )

    with bind_project(target):
        from src.config import get_curation_config

        idx = get_curation_config().configs_index
        assert await get_activation(client, idx, 'prompt_pack') is None
        assert await get_activation(client, idx, 'detection_profile') is None


@pytest.mark.asyncio
async def test_clone_activations_refuses_a_target_that_already_has_one_up_front() -> None:
    """M5: cloning 'activations' into a target that already has its own
    active pack/profile must 409 target_not_empty BEFORE anything is
    written -- not a bare RevisionConflictError/ActiveConflictError."""
    from src.core.exceptions import RequestRejectedError
    from src.services.config_store.index import activate, save_config
    from src.services.projects.clone import _apply_clone

    client = FakeConfigOpenSearch()
    source = _record('epsilon')
    target = _record('zeta')

    with bind_project(source):
        from src.config import get_curation_config

        src_idx = get_curation_config().configs_index
        doc = await save_config(
            client, src_idx, kind='prompt_pack', name='wheel', body={}, expected_revision=None
        )
        await activate(
            client,
            src_idx,
            axis='prompt_pack',
            name='wheel',
            revision=doc['revision'],
            expected_active=None,
        )

    # Target already has its OWN active pack (a different one).
    with bind_project(target):
        from src.config import get_curation_config

        tgt_idx = get_curation_config().configs_index
        tdoc = await save_config(
            client, tgt_idx, kind='prompt_pack', name='other', body={}, expected_revision=None
        )
        await activate(
            client,
            tgt_idx,
            axis='prompt_pack',
            name='other',
            revision=tdoc['revision'],
            expected_active=None,
        )

    with pytest.raises(RequestRejectedError) as exc_info:
        await _apply_clone(
            client,
            target_record=target,
            source=source,
            axes=['activations'],
            target_activations=None,
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail['error'] == 'target_not_empty'

    # Nothing else got written -- the check ran before any write.
    from src.services.config_store.index import get_activation

    with bind_project(target):
        activation = await get_activation(client, tgt_idx, 'prompt_pack')
    assert activation is not None
    assert activation['name'] == 'other'  # unchanged


@pytest.mark.asyncio
async def test_clone_refuses_a_profile_whose_detector_is_private_to_the_source_project(
    monkeypatch,
) -> None:
    import src.services.training.model_classes as mc
    import src.services.training.promoted_models as pm
    from src.config import get_curation_config
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
    with bind_project(target):
        act = await get_activation(client, get_curation_config().configs_index, 'detection_profile')
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
