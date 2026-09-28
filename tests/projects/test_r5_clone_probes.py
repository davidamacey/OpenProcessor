"""Round-5 reviewer probe: project clone activates a source-private detector
profile in the target without the target-context activation gate."""

from __future__ import annotations

import pytest

from projects.test_r3_clone_probes import (  # noqa: F401 - reuse fixtures/helpers
    Fake,
    _activate,
    _env,
    _record,
    _save,
)
from projects.test_r4_clone_probes import _into, _patch_lifecycle
from src.config.project_context import bind_project


pytestmark = pytest.mark.unbound


@pytest.mark.asyncio
async def test_r5_clone_activates_source_private_detector_in_target(tmp_path, monkeypatch):
    import src.services.training.model_classes as mc
    import src.services.training.promoted_models as pm
    from src.config import get_curation_config
    from src.services.config_store.index import get_activation
    from src.services.config_store.profile_validation import validate_profile

    owner = {'alpha__wheel_det': 'alpha'}
    monkeypatch.setattr(mc, 'model_owner_project', lambda n: owner.get(n))
    monkeypatch.setattr(pm, 'model_owner_project', lambda n: owner.get(n))
    monkeypatch.setattr(mc, 'is_model_shared', lambda _n: False)
    monkeypatch.setattr(pm, 'is_model_shared', lambda _n: False)

    # R6-m3 fix (W3/W4 round-6 review): the real clone call below
    # (`_into`) goes through `check_activation_pair_in_target_context` ->
    # `run_activation_gate` -> `validate_profile`'s OWN, unpatched
    # `TritonControlService.get_repository_index` -- unreachable in a
    # test, so the clone 422s on `detector_model_not_found` before the
    # ownership check (`detector_model_not_shared`) ever runs. With
    # ownership disabled entirely (`project_slug=None`), the test stayed
    # green for the wrong reason: it could not tell "ownership check
    # works" from "Triton just wasn't set up." Patch Triton READY here
    # too (matching `test_r6_clone_cause.py`'s pattern) so the real clone
    # path genuinely exercises the ownership check.
    from src.services.triton_control import TritonControlService

    async def _repo():
        return [{'name': 'alpha__wheel_det', 'state': 'READY', 'version': '1'}]

    monkeypatch.setattr(TritonControlService, 'get_repository_index', lambda _s: _repo())

    client = Fake()
    source, target = _record('alpha', tmp_path), _record('beta', tmp_path)
    body = {'detector_model': 'alpha__wheel_det', 'text_reader': 'none'}
    await _save(client, source, 'region_profile', 'private_rp', body)
    await _activate(client, source, 'detection_profile', 'private_rp', 1)

    # What the gate says in the TARGET's own context.
    async def _no_triton():
        return [{'name': 'alpha__wheel_det', 'state': 'READY', 'version': '1'}]

    with bind_project(target):
        report = await validate_profile(
            None,
            body,
            for_activation=True,
            project_slug='beta',
            get_repository_index=_no_triton,
        )
    codes = sorted(e.code for e in report.errors)
    print('target-context gate errors:', codes)
    assert 'detector_model_not_shared' in codes

    _patch_lifecycle(monkeypatch, source, target)
    clone_codes: list[str] = []
    try:
        await _into(client, source, target)
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
    # R6-m3 fix: with Triton actually READY, the clone must fail on the
    # ownership check specifically -- not "Triton wasn't set up"
    # (`detector_model_not_found`), which would pass this test for the
    # wrong reason (mutation-proven: disabling the ownership check
    # entirely still left the OLD version of this test green).
    assert clone_codes == ['detector_model_not_shared'], clone_codes
