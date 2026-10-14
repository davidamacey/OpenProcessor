"""Run arms one at a time, each in a fresh container of its own compose project.

    orchestrate.py --dataset DIR --size 100000 --arms os_shipped,milvus_hnsw --out docs/benchmarks/vectordb

For every arm: ``docker compose -p vecdb up`` for the engine's service, wait until it answers,
run the arm (``run.py``), then ``down -v`` so the next arm starts from an empty volume and cold
caches. Only the ``vecdb`` project is ever touched.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.vectordb import dataset
from scripts.bench.vectordb.arms import ARMS
from scripts.bench.vectordb.run import add_common_args, write_arm


COMPOSE = [
    'docker',
    'compose',
    '-p',
    'vecdb',
    '-f',
    str(_REPO_ROOT / 'docker/bench/vectordb/compose.yml'),
]
SERVICES = {
    'opensearch': ('opensearch', '127.0.0.1:4701', 'http://127.0.0.1:4701/'),
    'milvus': ('milvus', '127.0.0.1:4702', 'http://127.0.0.1:4705/healthz'),
    'qdrant': ('qdrant', '127.0.0.1:4703', 'http://127.0.0.1:4703/readyz'),
    'pgvector': ('pgvector', '127.0.0.1:4704', ''),
}


def compose(*args: str) -> None:
    subprocess.run([*COMPOSE, *args], check=True, capture_output=True, text=True, timeout=900)  # nosec B603  fixed argv list, no shell


def wait_ready(store: str, url: str, endpoint: str) -> None:
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            if url:
                urllib.request.urlopen(url, timeout=5).read()  # nosec B310  URL is a loopback health endpoint from the SERVICES table
            else:
                import psycopg

                host, _, port = endpoint.partition(':')
                psycopg.connect(
                    host=host,
                    port=int(port),
                    user='postgres',
                    password='bench',
                    dbname='bench',  # nosec B106  credential of the throwaway benchmark container
                ).close()
            time.sleep(5)
            return
        except Exception:
            time.sleep(3)
    msg = f'{store} did not become ready'
    raise RuntimeError(msg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument('--scratch', type=Path, default=Path('artifacts_local/bench/vectordb/lance'))
    args = ap.parse_args()
    ds = dataset.load(args.dataset, args.size)
    for arm_name in args.arms.split(','):
        store = ARMS[arm_name]['store']
        if store in SERVICES:
            service, endpoint, url = SERVICES[store]
            compose('--profile', service, 'up', '-d', service)
            wait_ready(store, url, endpoint)
        else:
            endpoint = str(args.scratch / arm_name)
            Path(endpoint).mkdir(parents=True, exist_ok=True)
        args.endpoint = endpoint
        try:
            write_arm(arm_name, ds, args)
        finally:
            if store in SERVICES:
                compose('--profile', service, 'down', '-v')


if __name__ == '__main__':
    main()
