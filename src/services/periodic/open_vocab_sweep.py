"""``op.periodic.open_vocab_sweep``: finish the ingest-time open-vocabulary pass after an outage.

Ingest stamps new images ``open_vocab_status: pending`` and starts the project's
``open_vocab_drain`` job. A drain that stopped early (segmenter down) leaves stamps behind; this
tick starts the next drain for a project that opts in on ingest and still has a pending image. The
job is exclusive per project, so a start while one is queued or running adds nothing.
"""

from __future__ import annotations

from src.config.advisory_locks import OPEN_VOCAB_SWEEP
from src.config.curation_accessors import images_index
from src.services.curation.job_binding import opensearch_client
from src.services.curation.open_vocab_ingest import start_drain
from src.services.curation.reprocess_open_vocab import current_active_set
from src.services.periodic.runner import TickProject, TickReport, for_each_project


_PENDING = {'query': {'term': {'open_vocab_status': 'pending'}}}


async def tick() -> TickReport:
    async def one(_project: TickProject) -> None:
        active = await current_active_set()
        if active is None or not active[0].run_on_ingest:
            return
        client = await opensearch_client()
        try:
            pending = int((await client.count(index=images_index(), body=_PENDING))['count'])
        finally:
            await client.close()
        if pending:
            await start_drain()

    return await for_each_project(OPEN_VOCAB_SWEEP, one, task='op.periodic.open_vocab_sweep')
