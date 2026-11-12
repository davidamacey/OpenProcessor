"""Live: a finished training run is promoted and the promoted model is served.

The fake trainer (``docker/test/fake_trainer.py``) writes a real tiny detector ONNX
(``docker/test/fake_onnx.py``: the NMS-free ``[N, 300, 6]`` head). The promote then runs for real in
the API: head check, uint8 serving copy, ``config.pbtxt`` and ``labels.txt``, Triton load and
warm-up inference. Triton is ``fake_triton.py``, which lists a promoted model once loaded.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from .conftest import API_PREFIX, wait_for_state
from .test_live_models_promote import finished_run


pytestmark = pytest.mark.live

BASE = f'{API_PREFIX}/projects/default/train'


def test_train_finishes_then_promote_succeeds_and_the_model_is_listed(api: Any) -> None:
    job_id = finished_run(api)
    name = f'promoted_{uuid.uuid4().hex[:6]}'

    # The fake run's eval is below the promote score gate, so the success path is the forced one.
    promoted = api.post(
        f'{BASE}/promote/{job_id}', params={'force': True}, json={'triton_name': name}
    )
    assert promoted.status_code == 202, promoted.text
    promote_id = promoted.json()['promote_id']
    final = wait_for_state(
        lambda: api.get(f'{BASE}/promote/{job_id}/jobs/{promote_id}').json(),
        lambda s: s.get('status') in ('done', 'failed'),
        timeout=180,
    )
    assert final['status'] == 'done', final

    manifest = api.get(f'{BASE}/manifest/{job_id}').json()
    assert manifest.get('promoted_to'), manifest

    listed = api.get('/models')
    assert listed.status_code == 200, listed.text
    by_name = {m['name']: m for m in listed.json()['models']}
    assert name in by_name, sorted(by_name)
    assert by_name[name]['status'] == 'READY', by_name[name]
