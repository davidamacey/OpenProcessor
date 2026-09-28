"""W10.0/W10.17 gate: there is exactly one class-label writer.

Before this wave, ``label_import.py`` wrote a second, parallel
``class_validated: True`` body (raw ``opensearch.bulk`` on the items
index, no OCC, no restorable snapshot — see any_domain_plan.md W10.1
I2/I3/I4). Every human AND dataset-import class-label write now goes
through ``src/services/curation/class_label.py``
(``class_label_fields`` / ``class_label_update``).

Scope note (deviation from the literal any_domain_plan.md W10.17
wording, which reads "no dict literal sets class_validated: True
outside class_label.py" with no carve-outs): one pre-existing,
out-of-scope automated validator — ``clustering/auto_promote.py``
(majority-agreement auto-validation, not a human or imported label) —
is allowlisted below. It predates W10, is not part of the W10.0
human/import label abstraction table, and rewriting it is out of this
wave's scope. See the report for the full reasoning.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

# The one legitimate writer, plus one pre-existing out-of-scope automated
# validator (see module docstring).
_ALLOWED_CLASS_VALIDATED_TRUE_FILES = {
    'src/services/curation/class_label.py',
    'src/services/curation/clustering/auto_promote.py',
}

_SCAN_ROOTS = ('src', 'scripts')


def _iter_py_files() -> list[Path]:
    files: list[Path] = []
    for root in _SCAN_ROOTS:
        files.extend((REPO_ROOT / root).rglob('*.py'))
    return files


def _dict_sets_class_validated_true(node: ast.Dict) -> bool:
    for key, value in zip(node.keys, node.values, strict=False):
        if (
            isinstance(key, ast.Constant)
            and key.value == 'class_validated'
            and isinstance(value, ast.Constant)
            and value.value is True
        ):
            return True
    return False


def test_no_second_class_validated_writer() -> None:
    """No dict literal outside class_label.py (+ the documented
    allowlist) sets ``class_validated: True`` — the "second write body"
    shape ``label_import.py`` used before this wave."""
    violations: list[str] = []
    for path in _iter_py_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in _ALLOWED_CLASS_VALIDATED_TRUE_FILES:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=rel)
        violations.extend(
            f'{rel}:{node.lineno}'
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict) and _dict_sets_class_validated_true(node)
        )
    assert not violations, (
        'dict literal(s) set class_validated: True outside class_label.py: ' + ', '.join(violations)
    )


def test_label_import_module_removed() -> None:
    """``label_import.py`` — the parallel raw-bulk item writer (W10.1
    I2/I3/I4) — is deleted outright, not kept as a second path."""
    assert not (REPO_ROOT / 'src/services/curation/label_import.py').exists()


@pytest.mark.parametrize('symbol', ['import_yolo_labels', 'import_labels_batch', '_parse_yolo_txt'])
def test_label_import_functions_have_no_importers(symbol: str) -> None:
    """The deleted per-image importer functions have no remaining callers."""
    for path in _iter_py_files():
        if path.name == 'test_class_label_single_writer.py':
            continue
        text = path.read_text(encoding='utf-8')
        assert symbol not in text, f'{path} still references {symbol}'


def test_is_human_owned_class_symbol_removed() -> None:
    """``is_human_owned_class`` is renamed to ``is_locked_class`` (W10.10)
    — no symbol of that name remains anywhere (docstrings/comments only
    reference it as history)."""
    for path in _iter_py_files():
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == 'is_human_owned_class':
                raise AssertionError(f'{path}:{node.lineno} still references is_human_owned_class')
            if isinstance(node, ast.alias) and node.name == 'is_human_owned_class':
                raise AssertionError(f'{path}:{node.lineno} still imports is_human_owned_class')
