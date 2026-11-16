#!/usr/bin/env python3
"""What the Python client reads for each photo of a list, for the Rust client to be held to
(plan 18a, R1e, "owed to the sweep").

Sends every photo of ``--list`` to the engine at ``--url`` with the plan ``ingest_engine_plan``
builds (the default policy: detect, the frame vector, a vector and a JPEG for every box), through
the API's own client (``src.clients.frame_engine.process``), and writes one JSON object per line
to ``--out``: a header with the plan text, then ``{"path": ..., "expected": ...}`` where
``expected`` is the shape of ``native/op_engine_client/tests/goldens/answers.json``
(``frame_engine_answer_goldens.summarise``). The ignored test ``native_engine`` of
``op_engine_client`` replays the list through the Rust client and compares field by field.

Run it in a GPU hold, against the engine the Rust client is run against, one photo in flight so
that the model server's batch composition does not move a vector:
``CUDA_VISIBLE_DEVICES`` is the engine's business, not this script's. The photo list is one path
per line (``#`` starts a comment), as ``op-fe-loadgen`` reads it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _paths(list_path: Path) -> list[Path]:
    lines = (line.strip() for line in list_path.read_text(encoding='utf-8').splitlines())
    return [Path(line) for line in lines if line and not line.startswith('#')]


async def run(args: argparse.Namespace) -> int:
    for path in (ROOT, ROOT / 'scripts' / 'codegen'):
        sys.path.insert(0, str(path))
    os.environ['OP_FRAME_ENGINE_URL'] = args.url
    from frame_engine_answer_goldens import summarise

    from src.clients import frame_engine
    from src.config import DetectionProfile
    from src.domain import frame_engine_wire as wire
    from src.services.curation import ingest_gpu_decode
    from src.services.curation.ingest_policy import IngestPolicy

    profile = DetectionProfile(
        name='sweep',
        detector_model=args.detector,
        input_size=args.input_size,
        letterbox_fill=(args.fill,) * 3,
    )
    # the plan builder reads only the profile and the policy of the service
    service: Any = SimpleNamespace(profile=profile, policy=IngestPolicy())
    plan = ingest_gpu_decode.ingest_engine_plan(service)
    pool = await frame_engine.open_pool(pool_size=1, max_concurrent=1)
    failed = 0
    with args.out.open('w', encoding='utf-8') as out:
        header: dict[str, Any] = {
            'plan': plan.to_json(),
            'detector': args.detector,
            'input_size': args.input_size,
            'fill': args.fill,
        }
        out.write(json.dumps(header) + '\n')
        for path in _paths(args.list):
            try:
                answer = await frame_engine.process(pool, path.read_bytes(), plan)
            except frame_engine.EngineCallError as exc:
                failed += 1
                out.write(json.dumps({'path': str(path), 'error': str(exc)}) + '\n')
                continue
            out.write(json.dumps({'path': str(path), 'expected': summarise(answer, wire)}) + '\n')
    await pool.close()
    sys.stdout.write(f'{failed} photos without an answer; wrote {args.out}\n')
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='the engine, host:port')
    parser.add_argument('--list', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--detector', required=True, help="the detector's model name")
    parser.add_argument('--input-size', type=int, default=640)
    parser.add_argument('--fill', type=int, default=114)
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == '__main__':
    raise SystemExit(main())
