"""Pins for :class:`RegionStatus` (Wave 5 — §0.3's overturned classification).

Ported from the reference's ``PlateStatus`` enum (a bare ``str, Enum``
state machine with zero domain logic — see
``docs/design/curation_design_rationale.md``). On-disk string values are
kept byte-identical to what earlier ported code already writes; only the
Python symbol is new.
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.config import PENDING_STATUSES, TERMINAL_STATUSES, RegionStatus


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENUM_MODULE = (_REPO_ROOT / 'src' / 'config' / 'region_state.py').resolve()


def test_every_member_round_trips_through_its_value() -> None:
    for member in RegionStatus:
        assert RegionStatus(member.value) is member


def test_expected_values_are_byte_identical_to_prior_literals() -> None:
    # These are the exact strings already written to OpenSearch by
    # pre-Wave-5 code (see the worker + routers this wave migrated) —
    # changing any of them would be a silent data-migration bug.
    assert RegionStatus.PENDING_DETECTION.value == 'pending_detection'
    assert RegionStatus.PENDING_VERIFICATION.value == 'pending_verification'
    assert RegionStatus.DETECTED.value == 'detected'
    assert RegionStatus.VERIFY_REJECTED.value == 'verify_rejected'
    assert RegionStatus.NO_PLATE_BOX.value == 'no_plate_box'
    assert RegionStatus.NO_PLATE_VISIBLE.value == 'no_plate_visible'
    assert RegionStatus.DETECTION_FAILED.value == 'detection_failed'
    assert RegionStatus.FALSE_POSITIVE.value == 'false_positive'


def test_terminal_and_pending_partition_the_whole_enum() -> None:
    union = TERMINAL_STATUSES | PENDING_STATUSES
    assert union == set(RegionStatus)
    # And they don't overlap — a status can't be both terminal and pending.
    assert set() == TERMINAL_STATUSES & PENDING_STATUSES


def _string_constants(tree: ast.AST) -> list[ast.Constant]:
    """Every string-literal AST node in ``tree``."""
    return [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _is_docstring(node: ast.Constant, docstring_ids: set[int]) -> bool:
    return id(node) in docstring_ids


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _dict_key_ids(tree: ast.AST) -> set[int]:
    """Dict-literal keys are wire/response field names, not RegionStatus
    reads/writes, even when they happen to share a status's spelling
    (e.g. a JSON response key ``'pending_detection'`` alongside the
    legacy alias key ``'pending'`` in the same dict) — exempt them, same
    scope carve-out ``RegionFields``' own guard uses for wire-response
    dict keys."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if k is not None:
                    ids.add(id(k))
    return ids


def test_no_bare_status_literal_outside_the_enum_module() -> None:
    """Real guard: no ``.py`` under ``src/`` or ``scripts/`` still hardcodes
    one of RegionStatus's string values as a bare literal, other than the
    enum module itself. Catches a future PR reintroducing the drift this
    wave just cleaned up."""
    values = {m.value for m in RegionStatus}
    violations: list[str] = []

    for base in ('src', 'scripts'):
        for path in (_REPO_ROOT / base).rglob('*.py'):
            if path.resolve() == _ENUM_MODULE:
                continue
            try:
                tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            except SyntaxError:
                continue
            docstring_ids = _docstring_ids(tree)
            dict_key_ids = _dict_key_ids(tree)
            for node in _string_constants(tree):
                if node.value not in values:
                    continue
                if _is_docstring(node, docstring_ids) or id(node) in dict_key_ids:
                    continue
                violations.append(f'{path.relative_to(_REPO_ROOT)}:{node.lineno}: {node.value!r}')

    assert violations == []
