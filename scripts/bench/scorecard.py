"""The one envelope every cluster/search scorer writes, plus the run metadata capture.

    {schema_version, kind, level, dataset, run_meta, metrics, strata, acceptance}

Quality needs real data: a scorecard whose ``dataset.synthetic`` is true may carry runtime,
memory and latency fields only, and ``build`` refuses anything else (and refuses strata and an
acceptance block, which are quality statements). An acceptance block is only valid on the
``heldout`` half of a real set, and a dataset that lists its own scoring split among the splits
it was tuned on is refused.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMA_VERSION = 'scorecard/1'
SPLITS = ('calibration', 'heldout', 'full')
NOISY_LOAD = 16.0
# Fields a synthetic dataset may carry: cost, never quality. Matched as a name prefix.
SYNTHETIC_OK = (
    'wall',
    'runtime',
    'producer_',
    'scoring_s',
    'peak_',
    'rss',
    'vram',
    'memory',
    'latency',
    'p50',
    'p95',
    'p99',
    'qps',
    'items',
    'rows',
    'dim',
)


class ScorecardError(ValueError):
    """The scorecard would state something the data cannot support."""


def acceptance(
    metric: str, bar: float, value: float | None, *, higher_is_better: bool = True
) -> dict[str, Any]:
    """The bar, the measured value and whether it passes (inclusive); no value never passes."""
    if value is None:
        passed = False
    else:
        passed = value >= bar if higher_is_better else value <= bar
    return {'metric': metric, 'bar': bar, 'value': value, 'pass': passed}


def _check_dataset(dataset: Mapping[str, Any]) -> None:
    for key in ('name', 'synthetic'):
        if key not in dataset:
            raise ScorecardError(f'dataset needs {key!r} (synthetic must be stated, never implied)')
    if not isinstance(dataset['synthetic'], bool):
        raise ScorecardError('dataset.synthetic must be true or false')
    if dataset['synthetic']:
        return
    if not dataset.get('manifest_sha256'):
        raise ScorecardError('a real dataset needs manifest_sha256')
    if dataset.get('split') not in SPLITS:
        raise ScorecardError(f'a real dataset needs split in {SPLITS}')
    for tuned in dataset.get('tuning', []):
        same_manifest = tuned.get('manifest_sha256') == dataset['manifest_sha256']
        halves = {tuned.get('split'), dataset['split']}
        if same_manifest and (len(halves) == 1 or 'full' in halves):
            raise ScorecardError(
                f'dataset was tuned on {tuned.get("split")!r} of the manifest it is scored on'
            )


def _check_synthetic(metrics: Mapping[str, Any], strata: Any, acceptance_block: Any) -> None:
    bad = sorted(k for k in metrics if not k.startswith(SYNTHETIC_OK))
    if bad:
        raise ScorecardError(
            f'synthetic dataset cannot carry quality fields {bad}; only runtime, memory, latency'
        )
    if strata or acceptance_block:
        raise ScorecardError('synthetic dataset cannot carry strata or an acceptance block')


def build(
    kind: str,
    level: str,
    dataset: Mapping[str, Any],
    run_meta: Mapping[str, Any],
    metrics: Mapping[str, Any],
    strata: Mapping[str, Any] | None = None,
    acceptance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _check_dataset(dataset)
    for key in ('git_sha', 'seed'):
        if run_meta.get(key) is None:
            raise ScorecardError(f'run_meta needs {key!r}')
    if dataset['synthetic']:
        _check_synthetic(metrics, strata, acceptance)
    elif acceptance is not None and dataset['split'] != 'heldout':
        raise ScorecardError(
            f'acceptance is only valid on the heldout half, not {dataset["split"]!r}'
        )
    return {
        'schema_version': SCHEMA_VERSION,
        'kind': kind,
        'level': level,
        'dataset': dict(dataset),
        'run_meta': dict(run_meta),
        'metrics': dict(metrics),
        'strata': dict(strata) if strata else None,
        'acceptance': dict(acceptance) if acceptance else None,
    }


def write(path: Path, card: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2) + '\n')


def _nvidia_smi() -> str | None:
    exe = shutil.which('nvidia-smi')
    if exe is None:
        return None
    done = subprocess.run(  # nosec B603  fixed argv list, no shell
        [exe, '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv'],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return done.stdout.strip() if done.returncode == 0 else None


def current_git_sha() -> str:
    root = Path(__file__).resolve().parents[2]
    done = subprocess.run(  # nosec B603 B607  fixed argv list, git from PATH
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise ScorecardError('cannot read the git sha; pass git_sha explicitly')
    return done.stdout.strip()


class RunMeter:
    """Context manager that fills ``meta`` with the run metadata the protocol asks for."""

    def __init__(
        self,
        *,
        seed: int,
        gpu_id: int | None = None,
        embedding_field: str | None = None,
        models: Mapping[str, str] | None = None,
        git_sha: str | None = None,
    ) -> None:
        self.meta: dict[str, Any] = {
            'git_sha': git_sha or current_git_sha(),
            'seed': seed,
            'gpu_id': gpu_id,
            'embedding_field': embedding_field,
            'models': dict(models) if models else None,
        }

    def __enter__(self) -> RunMeter:
        self.meta['nvidia_smi'] = _nvidia_smi()
        self.meta['load_avg_start'] = os.getloadavg()[0]
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.meta['wall_s'] = time.perf_counter() - self._t0
        self.meta['load_avg_end'] = os.getloadavg()[0]
        self.meta['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        self.meta['noisy'] = (
            max(self.meta['load_avg_start'], self.meta['load_avg_end']) > NOISY_LOAD
        )
