"""Fake trainer for the live verification harness: a real ``gpu_train`` consumer, no GPU.

It is the production worker shape (``src/tasks/runtime``: claim, heartbeat, fenced progress,
terminal state) with the Ultralytics fit replaced by a few fake epochs, so the training write
endpoints can be exercised against a real counterparty instead of a mock::

    celery -A fake_trainer:app worker -Q gpu_train --pool=prefork --concurrency=1

Per run: ``running`` for ``FAKE_TRAINER_EPOCHS`` epochs (one per ``FAKE_TRAINER_EPOCH_S``
seconds), then ``completed`` with a manifest stored in ``training_runs`` and a run log under the
run's artifact prefix. A cancel request is seen at the next epoch boundary. It never touches a GPU,
never loads a model and never reads OpenSearch.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from src.clients.storage.roots import build_rooted_store
from src.config.storage import storage_root_paths
from src.db.repos import training_runs as runs_repo
from src.domain.training_keys import RUN_LOG_NAME, run_object_key
from src.tasks.runtime.gpu_app import make_app
from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext

TOTAL_EPOCHS = int(os.environ.get('FAKE_TRAINER_EPOCHS', '3'))
EPOCH_S = float(os.environ.get('FAKE_TRAINER_EPOCH_S', '1'))
EVAL = {'map50': 0.5, 'map50_95': 0.25}

app = make_app(include=['fake_trainer'])


async def fake_train(ctx: JobContext) -> dict[str, Any]:
    async with ctx.session() as session:
        run = await session.run_sync(lambda sync: runs_repo.get(sync, ctx.job_id))
    if run is None:
        raise LookupError(f'job {ctx.job_id} has no training_runs row')
    store = build_rooted_store(storage_root_paths())
    log_key = run_object_key(run.artifact_prefix, RUN_LOG_NAME)
    log = [f'{ctx.job_id} fake trainer started']
    epoch = 0
    while epoch < TOTAL_EPOCHS and not ctx.cancelled():
        await asyncio.sleep(EPOCH_S)
        epoch += 1
        log.append(f'{ctx.job_id} epoch {epoch}/{TOTAL_EPOCHS}')
        await ctx.progress(epoch, TOTAL_EPOCHS, 'running')
    cancelled = epoch < TOTAL_EPOCHS
    log.append(f'{ctx.job_id} {"cancelled" if cancelled else "finished"}')
    store.write_atomic(log_key, ('\n'.join(log) + '\n').encode())
    manifest: dict[str, Any] = {
        'kind': 'train',
        'job_id': str(ctx.job_id),
        'trainer': 'fake',
        'lineage': {'dataset_sha': None, 'class_remap': None},
        'results': {
            'final_state': 'cancelled' if cancelled else 'finished',
            'eval': None if cancelled else EVAL,
            'checkpoint_path': None if cancelled else '/runs/fake-run/weights/best.pt',
        },
        'promoted_to': None,
    }
    async with ctx.session() as session:
        await session.run_sync(
            lambda sync: runs_repo.set_manifest(sync, ctx.job_id, manifest, None)
        )
    return {'state': 'cancelled' if cancelled else 'finished'}


register_job('train', fake_train)
