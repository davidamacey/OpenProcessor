"""Ratchet: Postgres rows store storage keys built by ``make_key``, never absolute paths.

An AST scan of ``src/db/repos/`` for any assignment (attribute, name, annotated name or keyword
argument such as ``values(x_storage_key=...)``) to a name ending in ``_storage_key``, ``_prefix``
or ``_key`` whose value is not a call to ``make_key`` or ``crop_key`` (``None`` clears a column and
is allowed). Names in :data:`NOT_STORAGE_KEYS` are other kinds of key (lock, idempotency, ...); add
a name there only when its column holds no storage path. The directory does not exist until the
repository lanes create it: a missing or empty directory passes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPOS_DIR = Path(__file__).resolve().parents[1] / 'src' / 'db' / 'repos'
SUFFIXES = ('_storage_key', '_prefix', '_key')
KEY_BUILDERS = frozenset({'make_key', 'crop_key'})
NOT_STORAGE_KEYS = frozenset(
    {
        'name_key',
        'space_key',
        'lock_key',
        'idempotency_key',
        'dedupe_key',
        'item_key',
        'model_key',
        'source_key',
        'index_prefix',
        # a Triton model-name prefix (``alpha__``), never a path
        'model_prefix',
    }
)


def _target_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _builds_key(value: ast.expr | None) -> bool:
    if isinstance(value, ast.Constant) and value.value is None:
        return True
    if isinstance(value, ast.Call):
        return _target_name(value.func) in KEY_BUILDERS
    if isinstance(value, ast.IfExp):
        return _builds_key(value.body) and _builds_key(value.orelse)
    return False


def _is_key_column(name: str | None) -> bool:
    return bool(name) and name not in NOT_STORAGE_KEYS and name.endswith(SUFFIXES)  # type: ignore[union-attr]


def find_violations(source: str, filename: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        pairs: list[tuple[str | None, ast.expr | None]] = []
        if isinstance(node, ast.Assign):
            pairs = [(_target_name(t), node.value) for t in node.targets]
        elif isinstance(node, ast.AnnAssign):
            # ``name: type`` with no value declares a field and assigns nothing
            pairs = [(_target_name(node.target), node.value)] if node.value is not None else []
        elif isinstance(node, ast.Call):
            pairs = [(kw.arg, kw.value) for kw in node.keywords]
        found.extend(
            f'{filename}:{node.lineno}: {name} is not built by make_key'  # type: ignore[attr-defined]
            for name, value in pairs
            if _is_key_column(name) and not _builds_key(value)
        )
    return found


def scan_directory(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    out: list[str] = []
    for path in sorted(directory.rglob('*.py')):
        out.extend(find_violations(path.read_text(encoding='utf-8'), str(path)))
    return out


def test_repos_write_keys_only_through_make_key() -> None:
    assert scan_directory(REPOS_DIR) == []


def test_missing_directory_passes(tmp_path: Path) -> None:
    assert scan_directory(tmp_path / 'absent') == []


@pytest.mark.parametrize(
    'line',
    [
        "row.artifact_storage_key = '/abs/path/model.pt'",
        'row.artifact_storage_key = path_arg',
        'stmt = insert(T).values(snapshot_storage_key=f"/data/{x}")',
        "cache_prefix = '/dev/shm/x'",
        'row.results_key: str = str(path)',
        "row.artifact_storage_key = make_key('projects', a) if a else path_arg",
    ],
)
def test_planted_violation_fails(tmp_path: Path, line: str) -> None:
    (tmp_path / 'bad.py').write_text(line + '\n', encoding='utf-8')
    assert len(scan_directory(tmp_path)) == 1


@pytest.mark.parametrize(
    'line',
    [
        "row.artifact_storage_key = make_key('projects', slug, 'training')",
        'row.crop_key = keys.crop_key(slug, item, box, 0)',
        'row.artifact_storage_key = None',
        'row.artifact_storage_key = make_key(*k.split("/")) if k is not None else None',
        'artifact_storage_key: str | None',
        'row.idempotency_key = request_id',
        'stmt = insert(T).values(lock_key=k, name=n)',
    ],
)
def test_builders_none_and_other_keys_pass(tmp_path: Path, line: str) -> None:
    (tmp_path / 'ok.py').write_text(line + '\n', encoding='utf-8')
    assert scan_directory(tmp_path) == []
