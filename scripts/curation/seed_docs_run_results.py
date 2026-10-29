"""Give one finished harness training run illustrative evaluation results.

The harness trainer (``docker/test/fake_trainer.sh``) walks a run to
``finished`` and writes a bare manifest. The docs screenshot of a run's
results needs a per-class table and a confusion matrix, so this fills them in
on the run's status file and writes the matrix as ``confusion_matrix.png``
into the run directory, which the artifact route then serves.

**The numbers are illustrative, not measured:** no model ran. They come from a
fixed seed so a re-run writes the same file. Run it inside the API container
(``docker exec <api> python scripts/curation/seed_docs_run_results.py ...``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.services.projects.script_binding import add_project_argument, bind_script_project  # noqa: E402  # import follows the sys.path setup above


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


def render_matrix(names: list[str], matrix: list[list[int]], out: Path) -> None:
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
    fig.savefig(out)
    plt.close(fig)


def apply(jobs_dir: Path, job_id: str, names: list[str], supports: list[int]) -> None:
    status_path = jobs_dir / f'{job_id}.status.json'
    status = json.loads(status_path.read_text())
    run_dir = jobs_dir / 'runs' / job_id
    (run_dir / 'weights').mkdir(parents=True, exist_ok=True)
    matrix = confusion_counts(supports)
    render_matrix(names, matrix, run_dir / 'confusion_matrix.png')
    rows = per_class_rows(names, matrix)

    def mean(key: str) -> float:
        return round(sum(float(rows[i][key]) for i in range(len(rows))) / len(rows), 3)

    status['checkpoint_path'] = str(run_dir / 'weights' / 'best.pt')
    status['eval'] = {
        'split': 'test',
        'map50': mean('ap50'),
        'map50_95': round(mean('ap50') * 0.62, 3),
        'precision': mean('precision'),
        'recall': mean('recall'),
        'per_class': rows,
        'confusion_matrix_path': str(run_dir / 'confusion_matrix.png'),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--opensearch-url', default=os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')
    )
    add_project_argument(parser)
    parser.add_argument('--jobs-dir', type=Path, required=True, help="the project's jobs directory")
    parser.add_argument('--job-id', required=True)
    parser.add_argument(
        '--class', dest='classes', action='append', required=True, help='NAME=SUPPORT'
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bind_script_project(args.project, opensearch_url=args.opensearch_url)
    names, supports = [], []
    for spec in args.classes:
        name, _, support = spec.partition('=')
        names.append(name)
        supports.append(int(support))
    apply(args.jobs_dir, args.job_id, names, supports)
    return 0


if __name__ == '__main__':
    sys.exit(main())
