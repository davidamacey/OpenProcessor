"""Give one finished harness training run illustrative evaluation results.

The harness trainer (``docker/test/fake_trainer.py``) walks a run to
``finished`` and writes a bare manifest. The docs screenshot of a run's
results needs a per-class table and a confusion matrix, so this fills them in
on the run's manifest (``training_runs.manifest``) and stores the matrix as
``confusion_matrix.png`` under the run's artifact prefix, which the artifact
route then serves.

**The numbers are illustrative, not measured:** no model ran. They come from a
fixed seed so a re-run writes the same file. Run it inside the API container
(``docker exec <api> python scripts/curation/seed_docs_run_results.py ...``).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clients.storage.roots import build_rooted_store  # noqa: E402  # import follows the sys.path setup above
from src.config.project_context import current_project_id  # noqa: E402  # import follows the sys.path setup above
from src.config.storage import storage_root_paths  # noqa: E402  # import follows the sys.path setup above
from src.db.engine import run_with_engine  # noqa: E402  # import follows the sys.path setup above
from src.db.repos import training_runs as runs_repo  # noqa: E402  # import follows the sys.path setup above
from src.db.session import async_session_factory, project_tx  # noqa: E402  # import follows the sys.path setup above
from src.domain.training_keys import run_object_key  # noqa: E402  # import follows the sys.path setup above
from src.services.projects.script_binding import abind_script_project, add_project_argument  # noqa: E402  # import follows the sys.path setup above


def confusion_counts(supports: list[int], accuracy: float = 0.9) -> list[list[int]]:
    """Square matrix, rows = true class: ``accuracy`` of each row on the
    diagonal, the rest spread over the other classes in proportion to their
    column position (deterministic: no randomness)."""
    n = len(supports)
    matrix = [[0] * n for _ in range(n)]
    for i, support in enumerate(supports):
        right = round(support * accuracy)
        matrix[i][i] = right
        others = [j for j in range(n) if j != i]
        wrong = support - right
        for k, j in enumerate(others):
            share = wrong * (len(others) - k) // (len(others) * (len(others) + 1) // 2)
            matrix[i][j] = share
        matrix[i][others[0]] += wrong - sum(matrix[i][j] for j in others)
    return matrix


def per_class_rows(names: list[str], matrix: list[list[int]]) -> list[dict[str, Any]]:
    rows = []
    for i, name in enumerate(names):
        tp = matrix[i][i]
        predicted = sum(matrix[r][i] for r in range(len(names)))
        support = sum(matrix[i])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                'class_id': i,
                'name': name,
                'precision': round(precision, 3),
                'recall': round(recall, 3),
                'f1': round(f1, 3),
                'ap50': round(min(0.99, recall * 0.97 + 0.03), 3),
                'support': support,
            }
        )
    return rows


def render_matrix(names: list[str], matrix: list[list[int]]) -> bytes:
    import matplotlib as mpl

    mpl.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(len(names)), names, rotation=45, ha='right')
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion matrix')
    top = max(max(row) for row in matrix) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value:
                ax.text(
                    j,
                    i,
                    str(value),
                    ha='center',
                    va='center',
                    color='white' if value > top / 2 else 'black',
                    fontsize=8,
                )
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    plt.close(fig)
    return buffer.getvalue()


async def apply(job_id: uuid.UUID, names: list[str], supports: list[int]) -> None:
    """Add the illustrative eval to the run's manifest and store the matrix image."""
    matrix = confusion_counts(supports)
    rows = per_class_rows(names, matrix)

    def mean(key: str) -> float:
        return round(sum(float(rows[i][key]) for i in range(len(rows))) / len(rows), 3)

    async with async_session_factory()() as session, project_tx(session, current_project_id()):
        run = await session.run_sync(lambda sync: runs_repo.get(sync, job_id))
        if run is None or run.manifest is None:
            raise SystemExit(f'run {job_id} has no manifest yet; wait for it to finish')
        store = build_rooted_store(storage_root_paths())
        matrix_key = run_object_key(run.artifact_prefix, 'confusion_matrix.png')
        store.write_atomic(matrix_key, render_matrix(names, matrix))
        results = {
            **(run.manifest.get('results') or {}),
            'eval': {
                'split': 'test',
                'map50': mean('ap50'),
                'map50_95': round(mean('ap50') * 0.62, 3),
                'precision': mean('precision'),
                'recall': mean('recall'),
                'per_class': rows,
                'confusion_matrix_path': matrix_key,
            },
        }
        manifest = {**run.manifest, 'results': results}
        await session.run_sync(
            lambda sync: runs_repo.set_manifest(sync, job_id, manifest, run.mlflow_run_id)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_project_argument(parser)
    parser.add_argument('--job-id', type=uuid.UUID, required=True)
    parser.add_argument(
        '--class', dest='classes', action='append', required=True, help='NAME=SUPPORT'
    )
    return parser


async def run(args: argparse.Namespace) -> int:
    await abind_script_project(args.project)
    names, supports = [], []
    for spec in args.classes:
        name, _, support = spec.partition('=')
        names.append(name)
        supports.append(int(support))
    await apply(args.job_id, names, supports)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(run_with_engine(run(args), 'op-seed-docs-run', one_shot=True))


if __name__ == '__main__':
    sys.exit(main())
