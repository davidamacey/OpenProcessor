"""The class registry against the harness PostgreSQL, which serialises writers on the registry row.

The harness runs the API with one worker (some cancel endpoints and the SSE hub keep in-process
state), so the cross-process race runs at the store level: 8 spawned processes call ``add_class``
with distinct names and then ``ensure_class`` with one shared name, straight against the
harness database. PostgreSQL is not published to the host, so the race runs inside the ``api``
container (``docker compose exec``), which has the database settings and secrets of the stack; the
test reads the result back from that run. Then one HTTP sequence checks the routes read the same
registry.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from tests.live._compose import compose
from tests.live.conftest import free_hotkey_letter


pytestmark = pytest.mark.live

REPEATS = 5
PROCESSES = 8

# Runs in the api container. A spawned child re-imports __main__ from its file, so the script is
# written to /tmp first; the worker is a module next to it for the same reason.
_RACE = r'''
import asyncio, json, multiprocessing as mp, pathlib, sys, tempfile

WORKER = """
import asyncio
from src.services.classes.registry import registry_for


def work(prefix, number, barrier):
    async def run():
        store = registry_for('default')
        barrier.wait()
        await store.add_class(f'{prefix}_{number}')
        await store.ensure_class(f'{prefix}_shared')

    asyncio.run(run())
"""


async def snapshot():
    from src.services.classes.registry import registry_for

    return await registry_for('default').snapshot()


if __name__ == '__main__':
    prefix, processes = sys.argv[1], int(sys.argv[2])
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / 'race_worker.py').write_text(WORKER)
    sys.path.insert(0, str(tmp))
    import race_worker

    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(processes)
    procs = [
        ctx.Process(target=race_worker.work, args=(prefix, i, barrier)) for i in range(processes)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    doc = asyncio.run(snapshot())
    print(json.dumps({
        'exitcodes': [p.exitcode for p in procs],
        'classes': [[c.class_id, c.class_name] for c in doc.classes],
    }))
'''


def _race(prefix: str) -> dict[str, Any]:
    proc = compose(
        'exec', '-T', 'api', 'sh', '-c', 'cat > /tmp/race.py && PYTHONPATH=/app python /tmp/race.py "$0" "$1"',
        prefix, str(PROCESSES),
        stdin=_RACE, check=False, timeout=240,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize('repeat', range(REPEATS))
def test_eight_processes_add_eight_distinct_classes_and_one_shared(repeat: int) -> None:
    prefix = f'live_race_{uuid.uuid4().hex[:8]}'

    result = _race(prefix)

    assert all(code == 0 for code in result['exitcodes']), result['exitcodes']
    classes = result['classes']
    mine = [cid for cid, name in classes if name.startswith(prefix)]
    assert len(mine) == PROCESSES + 1, classes
    assert len(set(mine)) == PROCESSES + 1
    assert len({cid for cid, _ in classes}) == len(classes), 'an id was handed out twice'


def test_the_routes_read_and_write_the_same_registry(api_client: Any) -> None:
    name = f'live_doc_probe_{uuid.uuid4().hex[:8]}'
    created = api_client.post('/classes', json={'name': name, 'group': 'harness'})
    assert created.status_code == 201, created.text
    class_id = created.json()['class_id']

    # The hotkey is released again so a rerun against the same registry finds it free.
    letter = free_hotkey_letter(api_client)
    updated = api_client.put(
        f'/classes/{class_id}', json={'hotkey_letter': letter, 'group': 'moved'}
    )
    assert updated.status_code == 200, updated.text
    try:
        listed = {c['class_id']: c for c in api_client.get('/classes').json()['classes']}
        assert listed[class_id]['hotkey_letter'] == letter
        assert listed[class_id]['group'] == 'moved'
        assert listed[class_id]['class_name'] == name
    finally:
        api_client.put(f'/classes/{class_id}', json={'hotkey_letter': ''}).raise_for_status()
