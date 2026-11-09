"""The integer cluster-id namespace (4F-6) is being deleted from ``src/``, and no file may grow it.

The overloaded integer ``cluster_id`` (an IVF centroid, a class id and a ``RESIDUAL_CLUSTER_ID_OFFSET``
residual all in one number), its AHC ``cluster_subid`` and the ``excluded_prior_cluster_*`` pair are
replaced by the served cluster of the serving partition (``cluster_uid`` and the rest of the
``cluster_*`` partition fields, one nested ``cluster`` object on the wire). The legacy item pipeline
(the old orchestrator, the cluster cards, auto-promote, exclusion, the move route, the loop's
``ClusterFieldPartition`` reader) still spells the names; each file is pinned in
``tests/cluster_id_namespace_baseline.txt`` to the number of occurrences it has today.

A file may only go DOWN: a new occurrence (or a new file) fails, and a file that got cleaner must
have its pin lowered (or removed at zero) in the same change, so the list only shrinks and the day
it is empty this test becomes "no ``cluster_id`` in ``src/``".

The core visual-search clusters (faces, images; ``src/routers/clusters.py``, ``src/clients/opensearch``,
``src/services/visual_search``) are a different namespace with their own integer ids and are pinned
the same way until their own rebuild.

Counted: a name, attribute, parameter, keyword, import alias, function name or whole string literal
equal to one of the retired names.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).with_name('cluster_id_namespace_baseline.txt')
RETIRED = frozenset(
    {
        'RESIDUAL_CLUSTER_ID_OFFSET',
        'cluster_id',
        'cluster_subid',
        'excluded_prior_cluster_id',
        'excluded_prior_cluster_subid',
    }
)


def _hits(tree: ast.AST) -> int:
    n = 0
    for node in ast.walk(tree):
        if (
            (isinstance(node, ast.Name) and node.id in RETIRED)
            or (isinstance(node, ast.Attribute) and node.attr in RETIRED)
            or (isinstance(node, ast.arg) and node.arg in RETIRED)
            or (isinstance(node, ast.keyword) and node.arg in RETIRED)
            or (isinstance(node, ast.alias) and node.name in RETIRED)
            or (isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in RETIRED)
            or (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in RETIRED
            )
        ):
            n += 1
    return n


def occurrences(root: Path = ROOT) -> Counter[str]:
    found: Counter[str] = Counter()
    for path in sorted((root / 'src').rglob('*.py')):
        count = _hits(ast.parse(path.read_text(encoding='utf-8')))
        if count:
            found[str(path.relative_to(root))] = count
    return found


def _pinned() -> dict[str, int]:
    pins: dict[str, int] = {}
    for line in BASELINE.read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.startswith('#'):
            name, _, count = line.partition('\t')
            pins[name] = int(count)
    return pins


def test_no_file_grows_the_integer_cluster_id_namespace() -> None:
    pins = _pinned()
    grew = {
        name: (pins.get(name, 0), count)
        for name, count in occurrences().items()
        if count > pins.get(name, 0)
    }
    assert not grew, (
        'the integer cluster-id namespace is being deleted, not extended: use cluster_uid and the '
        f'cluster_* partition fields (file: (pinned, now)) {grew}'
    )


def test_the_pins_only_shrink() -> None:
    now = occurrences()
    stale = {
        name: (pin, now.get(name, 0)) for name, pin in _pinned().items() if now.get(name, 0) < pin
    }
    assert not stale, (
        f'lower or remove these pins in tests/cluster_id_namespace_baseline.txt {stale}'
    )


def test_the_deleted_namespace_stays_deleted_where_it_was_removed() -> None:
    """Files this lane cleaned stay clean (the box element, the region matcher, the partition code)."""
    now = occurrences()
    for clean in (
        'src/domain/region_boxes.py',
        'src/services/clustering/region_fp.py',
        'src/services/clustering/job.py',
        'src/services/clustering/run.py',
        'src/services/clustering/partitions.py',
        'src/routers/curation/regions_fp.py',
        'src/routers/curation/regions.py',
        'src/services/curation/region_box_edits.py',
    ):
        assert clean not in now, clean
