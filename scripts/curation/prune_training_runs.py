#!/usr/bin/env python3
"""Keep-last retention for finished training runs + bake-off output (ST-4).

Two independent prunes, both dry-run by default:

* training-run job state under ``OP_TRAIN_JOBS_DIR`` -- never a
  non-terminal run, a promoted run, or one a queued/running bake-off job
  still references (see ``src.services.training.run_retention``);
* bake-off comparison output under ``OP_BAKEOFF_OUT_DIR`` -- pure
  keep-last, no pin logic (scored output, not a training artifact).

Neither touches MLflow runs.

    # See what would be removed.
    python3 scripts/curation/prune_training_runs.py

    # Actually remove.
    python3 scripts/curation/prune_training_runs.py --apply

    # Different retention windows.
    python3 scripts/curation/prune_training_runs.py --keep-last 10 --bakeoff-out-keep-last 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.config import get_curation_config, get_gpu_arbiter_config
from src.services.training.run_retention import (
    apply_bakeoff_out_prune,
    apply_run_prune,
    plan_bakeoff_out_prune,
    plan_run_prune,
)
from src.services.training.triton_promote import resolve_triton_models_dir


def run(args: argparse.Namespace) -> int:
    jobs_dir = Path(os.environ.get('OP_TRAIN_JOBS_DIR', '/jobs'))
    model_repo = resolve_triton_models_dir()
    bakeoff_jobs_dir = Path(get_gpu_arbiter_config().bakeoff_jobs_dir)
    bakeoff_out_dir = Path(
        os.environ.get('OP_BAKEOFF_OUT_DIR', str(get_curation_config().state_dir / 'bakeoff_out'))
    )

    run_plan = plan_run_prune(
        jobs_dir=jobs_dir,
        model_repo=model_repo,
        bakeoff_jobs_dir=bakeoff_jobs_dir,
        keep_last=args.keep_last,
    )
    out_plan = plan_bakeoff_out_prune(bakeoff_out_dir, args.bakeoff_out_keep_last)

    print(f'{len(run_plan)} training run(s) planned for removal (keep_last={args.keep_last}):')
    for job_id in run_plan:
        print(f'  {job_id}')
    print(
        f'{len(out_plan)} bake-off output dir(s) planned for removal '
        f'(keep_last={args.bakeoff_out_keep_last}):'
    )
    for path in out_plan:
        print(f'  {path}')

    if not args.apply:
        print('Dry-run only. Pass --apply to remove.')
        return 0

    run_result = apply_run_prune(run_plan, jobs_dir=jobs_dir)
    out_result = apply_bakeoff_out_prune(out_plan)
    print(
        f'removed_runs={run_result["removed_runs"]} removed_run_files={run_result["removed_files"]} '
        f'removed_bakeoff_out={out_result["removed"]}'
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--keep-last', type=int, default=20)
    p.add_argument('--bakeoff-out-keep-last', type=int, default=20)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', dest='apply', action='store_false', default=False)
    g.add_argument('--apply', dest='apply', action='store_true')
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
