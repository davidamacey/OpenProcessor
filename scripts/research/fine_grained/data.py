"""Loading extracted embeddings with their set C labels and the registered folds."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.fine_grained import setc


SPLIT_SEED = 20261105


@dataclass
class Universe:
    """Rows of the car key: set C (labelled, first) then distractors (unlabelled, never scored)."""

    meta: list[dict[str, Any]]
    half: np.ndarray  # 'calibration' | 'heldout' | '' (distractor)
    make: np.ndarray
    model: np.ndarray
    generation: np.ndarray
    new_model: np.ndarray  # True where the model is in the class-disjoint held-out set

    @property
    def labelled(self) -> np.ndarray:
        return np.asarray([m['source'] == 'setc' for m in self.meta])

    def pool(self, which: str) -> np.ndarray:
        return np.asarray(np.nonzero(self.half == which)[0])

    def fold(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """``(calibration_rows, test_rows)``. Fold 1 tests on the held-out half, fold 2 on the other."""
        cal, held = self.pool('calibration'), self.pool('heldout')
        return (cal, held) if n == 1 else (held, cal)


def restrict(u: Universe, vec: np.ndarray, distractors: int) -> tuple[Universe, np.ndarray]:
    """Set C plus the first ``distractors`` distractor rows (they come sorted by crop id)."""
    lab = np.nonzero(u.labelled)[0]
    keep = np.concatenate([lab, np.nonzero(~u.labelled)[0][:distractors]])
    sub = Universe(
        [u.meta[i] for i in keep],
        u.half[keep],
        u.make[keep],
        u.model[keep],
        u.generation[keep],
        u.new_model[keep],
    )
    return sub, vec[keep]


def env_path(name: str) -> Path:
    """A required directory or file path from the environment; the tree holds no machine paths."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f'set {name} (see the module docstring of the script you ran)')
    return Path(value)


def git_sha() -> str:
    return subprocess.run(  # nosec B603 B607  fixed argv list, no shell, git from PATH
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True
    ).stdout.strip()


def load_universe(emb_dir: Path, subset: str = 'all') -> Universe:
    meta = [json.loads(x) for x in (emb_dir / f'meta_{subset}.jsonl').read_text().splitlines() if x]
    items = setc.load_items(env_path('FG_SETC_ROOT'))
    halves = setc.split_halves(items, seed=SPLIT_SEED)
    held = setc.split_new_models(items, seed=SPLIT_SEED)
    half = np.array([halves.get(m['id'], '') if m['source'] == 'setc' else '' for m in meta])

    def col(key: str) -> np.ndarray:
        return np.array([m[key] or '' for m in meta])

    model = col('model')
    return Universe(meta, half, col('make'), model, col('generation'), np.isin(model, sorted(held)))


def load_vectors(emb_dir: Path, model: str, variant: str, subset: str = 'all') -> np.ndarray:
    return np.asarray(np.load(emb_dir / f'{model}__{variant}__{subset}.npy'))


def run_info() -> dict[str, Any]:
    return {'git_sha': git_sha(), 'load_avg': os.getloadavg()}
