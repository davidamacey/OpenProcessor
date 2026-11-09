"""``python -m scripts.db.vlm_pause <action>``: the three fixed pause actions of ``vlm use``.

``openprocessor vlm use`` swaps the VLM model under the running stack. It runs this inside the API
container (the database is reachable there, the host has no credentials):

``lock-check``    exit 0 when a train, probe or bake-off job holds cards (a swap must wait).
``pause-create``  idle the VLM workers for the swap; prints ``created``, or ``present`` when
                  another swap already holds the pause (that one owns it).
``pause-remove``  end the swap's pause.

The pause is the ``vlm_switch`` section of ``deployment_settings('vlm_paused_for_training')``
(:mod:`src.domain.gpu_arbiter_state`); a swap that crashes stops counting after
``SWITCH_TTL_S``.
"""

from __future__ import annotations

import sys
import time

from sqlalchemy.orm import Session

from src.db.engine import get_sync_engine
from src.db.repos import gpu_arbiter as arbiter_repo
from src.db.session import sync_deployment_tx
from src.domain import gpu_arbiter_state as state


def run(action: str) -> int:
    """Perform ``action`` and return the process exit code."""
    with Session(get_sync_engine()) as session, sync_deployment_tx(session, f'vlm use {action}'):
        value = arbiter_repo.read_value(session)
        if action == 'lock-check':
            return 0 if state.arbiter_record(value).claims else 1
        if action == 'pause-create':
            if state.switch_active(value):
                print('present')
                return 0
            arbiter_repo.set_section(session, state.VLM_SWITCH, {'at': time.time()})
            print('created')
            return 0
        if action == 'pause-remove':
            arbiter_repo.drop_section(session, state.VLM_SWITCH)
            return 0
    return 2


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1]) if len(sys.argv) == 2 else 2)
