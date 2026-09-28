"""Round-6 probe: WHY does the landed R5-3 clone test's clone fail?"""

from __future__ import annotations

import pytest

from projects.test_r3_clone_probes import Fake, _activate, _record, _save
from projects.test_r4_clone_probes import _into, _patch_lifecycle


pytestmark = pytest.mark.unbound


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.asyncio
async def test_r6_clone_failure_cause(tmp_path, monkeypatch, shared):
    import src.services.training.model_classes as mc
    import src.services.training.promoted_models as pm

    owner = {'alpha__wheel_det': 'alpha'}
    from src.services.triton_control import TritonControlService

    async def _repo():
        return [{'name': 'alpha__wheel_det', 'state': 'READY', 'version': '1'}]

    monkeypatch.setattr(TritonControlService, 'get_repository_index', lambda _s: _repo())
    monkeypatch.setattr(mc, 'model_owner_project', lambda n: owner.get(n))
    monkeypatch.setattr(pm, 'model_owner_project', lambda n: owner.get(n))
    monkeypatch.setattr(mc, 'is_model_shared', lambda _n: shared)
    monkeypatch.setattr(pm, 'is_model_shared', lambda _n: shared)
    client = Fake()
    source, target = _record('alpha', tmp_path), _record('beta', tmp_path)
    body = {'detector_model': 'alpha__wheel_det', 'text_reader': 'none'}
    await _save(client, source, 'region_profile', 'private_rp', body)
    await _activate(client, source, 'detection_profile', 'private_rp', 1)
    _patch_lifecycle(monkeypatch, source, target)
    try:
        await _into(client, source, target)
        print('clone ok', shared)
    except Exception as exc:
        cause = exc.__cause__
        d = getattr(cause, 'detail', cause)
        print(
            'clone failed:',
            shared,
            [e['code'] for e in d['report']['errors']]
            if isinstance(d, dict) and 'report' in d
            else d,
        )
