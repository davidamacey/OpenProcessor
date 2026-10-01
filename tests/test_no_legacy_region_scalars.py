"""No code may read or write a region's per-box data off the ITEM.

W8 moved every per-box attribute -- geometry, score, detector, source,
verdict, text, candidate box, vector, cluster placement -- into the
``region_boxes`` list. The item-level ``RegionFields`` storage names that
used to hold them are gone, and so are their mappings and wire keys. This
guard keeps them gone, across ``src/`` and ``scripts/``:

1. :class:`~src.config.region_fields.RegionFields` has no retired attribute
   (nothing can be read through the indirection any more);
2. no module accesses a retired attribute on a ``RegionFields`` value --
   ``F.<attr>`` / ``_F`` / ``fields`` / ``storage`` / ``self.fields``, a local
   bound from ``get_region_fields()`` / ``RegionFields(...)``, a parameter
   annotated ``RegionFields``, or the inline ``get_region_fields().<attr>``;
3. no module spells a retired wire name as a string literal
   (``'region_bbox_norm'``), which would bypass the indirection.

Allowlist (the whole of it, each entry justified):

- ``src/services/labeling/region_overlay.py``: ``region_bbox_correct`` /
  ``region_confidence`` / ``region_text`` are the keys of the VLM's JSON
  *reply* -- a fixed protocol the prompt packs ask for -- not storage names.
  It defines them once (``REPLY_*_KEY``); every other module imports them.
- ``src/services/curation/review_sorts.py``: ``region_score`` is the served
  review-sort *id* of ``GET /review/{tab}?sort=``, not a storage name (the
  sort reads ``region_max_score``).

``region_fields.py`` itself is excluded (it defines the names).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.config.region_fields import RegionFields


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ('src', 'scripts')
DEFINING_MODULE = 'src/config/region_fields.py'

# RegionFields attribute -> the wire/storage name it used to carry.
RETIRED: dict[str, str] = {
    'bbox_norm': 'region_bbox_norm',
    'bbox_frame': 'region_bbox_frame',
    'bbox_correct': 'region_bbox_correct',
    'score': 'region_score',
    'confidence': 'region_confidence',
    'text': 'region_text',
    'text_raw': 'region_text_raw',
    'text_confidence': 'region_text_confidence',
    'text_source': 'region_text_source',
    'text_engine_version': 'region_text_engine_version',
    'text_vlm': 'region_text_vlm',
    'text_ocr': 'region_text_ocr',
    'text_disagreement': 'region_text_disagreement',
    'text_choice': 'region_text_choice',
    'text_vlm_invalid': 'region_text_vlm_invalid',
    'detector': 'region_detector',
    'detector_version': 'region_detector_version',
    'source': 'region_source',
    'candidate_bbox_norm': 'region_candidate_bbox_norm',
    'candidate_score': 'region_candidate_score',
    'candidate_detector': 'region_candidate_detector',
    'candidate_detector_version': 'region_candidate_detector_version',
    'candidate_source': 'region_candidate_source',
    'embedding': 'region_embedding',
    'cluster_id': 'region_cluster_id',
    'cluster_subid': 'region_cluster_subid',
    'cluster_distance': 'region_cluster_distance',
    'bbox_norm_legacy': 'region_bbox_norm_legacy',
    'score_legacy': 'region_score_legacy',
}
RETIRED_WIRE_NAMES = frozenset(RETIRED.values())

# Names that always denote a RegionFields value, plus the receivers of
# ``self.<name>``.
_ALIASES = frozenset({'F', '_F', 'fields', 'storage'})
_SELF_RECEIVERS = frozenset({'fields', '_fields', 'storage', 'region_fields'})
_FIELD_FACTORIES = frozenset({'get_region_fields', 'RegionFields'})

# path (relative, posix) -> wire literals allowed in that file.
LITERAL_ALLOWLIST: dict[str, frozenset[str]] = {
    'src/services/labeling/region_overlay.py': frozenset(
        {'region_bbox_correct', 'region_confidence', 'region_text'}
    ),
    'src/services/curation/review_sorts.py': frozenset({'region_score'}),
}


def _is_fields_factory_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _FIELD_FACTORIES
    if isinstance(func, ast.Attribute):
        # the classmethod constructor ``RegionFields.from_env``
        return func.attr == 'from_env' and _name_of(func.value) == 'RegionFields'
    return False


def _name_of(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _annotation_mentions_region_fields(annotation: ast.AST | None) -> bool:
    if annotation is None:
        return False
    return any(
        (isinstance(n, ast.Name) and n.id == 'RegionFields')
        or (isinstance(n, ast.Constant) and isinstance(n.value, str) and 'RegionFields' in n.value)
        for n in ast.walk(annotation)
    )


def _docstring_constants(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _bound_field_names(tree: ast.AST) -> set[str]:
    """Local names bound to a RegionFields value: assigned from a factory
    (or ``self.fields``) or annotated ``RegionFields``."""
    names: set[str] = set(_ALIASES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if _is_fields_factory_call(node.value) or _is_self_fields(node.value):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and (
                _annotation_mentions_region_fields(node.annotation)
                or (node.value is not None and _is_fields_factory_call(node.value))
            ):
                names.add(node.target.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            names.update(a.arg for a in args if _annotation_mentions_region_fields(a.annotation))
    return names


def _is_self_fields(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == 'self'
        and node.attr in _SELF_RECEIVERS
    )


def scan_source(text: str, rel_path: str = '<memory>') -> list[str]:
    """Every retired-scalar access or wire literal in ``text``."""
    tree = ast.parse(text)
    bound = _bound_field_names(tree)
    docstrings = _docstring_constants(tree)
    allowed_literals = LITERAL_ALLOWLIST.get(rel_path, frozenset())
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in RETIRED:
            receiver = node.value
            if (
                (isinstance(receiver, ast.Name) and receiver.id in bound)
                or _is_fields_factory_call(receiver)
                or _is_self_fields(receiver)
            ):
                found.append(f'{rel_path}:{node.lineno}: <RegionFields>.{node.attr}')
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in RETIRED_WIRE_NAMES
            and node.value not in allowed_literals
            and id(node) not in docstrings
        ):
            found.append(f'{rel_path}:{node.lineno}: {node.value!r}')
    return found


def _source_files() -> list[Path]:
    return sorted(
        p
        for root in SCAN_ROOTS
        for p in (REPO_ROOT / root).rglob('*.py')
        if p.relative_to(REPO_ROOT).as_posix() != DEFINING_MODULE
    )


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------


def test_region_fields_has_no_retired_attribute() -> None:
    attrs = set(RegionFields.__dataclass_fields__)
    assert not attrs & set(RETIRED), sorted(attrs & set(RETIRED))


def test_no_module_reads_or_spells_a_retired_region_scalar() -> None:
    files = _source_files()
    assert len(files) > 300, 'sanity: the scan found no source tree'
    violations: list[str] = []
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        violations.extend(scan_source(path.read_text(encoding='utf-8'), rel))
    assert not violations, 'retired region scalar(s) still referenced:\n' + '\n'.join(violations)


def test_the_allowlist_names_files_that_exist_and_still_need_it() -> None:
    for rel, literals in LITERAL_ALLOWLIST.items():
        path = REPO_ROOT / rel
        assert path.is_file(), f'{rel} missing or moved'
        text = path.read_text(encoding='utf-8')
        for literal in literals:
            assert f"'{literal}'" in text, f'{rel} no longer uses {literal!r}: drop the entry'


# ---------------------------------------------------------------------------
# the scanner itself: it must fire on every access shape, and only on those
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'snippet',
    [
        'x = F.bbox_norm\n',
        'x = _F.score\n',
        'def f(fields):\n    return fields.cluster_id\n',
        'def f(storage):\n    return storage.candidate_bbox_norm\n',
        'from x import get_region_fields\nf = get_region_fields()\ny = f.detector\n',
        'from x import RegionFields\nr = RegionFields()\ny = r.text\n',
        'from x import RegionFields\nr = RegionFields.from_env()\ny = r.embedding\n',
        'from x import get_region_fields\ny = get_region_fields().cluster_subid\n',
        'class A:\n    def m(self):\n        return self.fields.source\n',
        "def f(cfg: 'RegionFields | None'):\n    return cfg.confidence\n",
        'def f(cfg: RegionFields):\n    return cfg.text_source\n',
        "x = src.get('region_bbox_norm')\n",
        'x = {"region_cluster_id": 1}\n',
    ],
)
def test_the_scanner_catches_every_access_shape(snippet: str) -> None:
    assert scan_source(snippet), snippet


@pytest.mark.parametrize(
    'snippet',
    [
        'x = box.bbox_norm\n',
        'x = b.score\n',
        "x = doc['bbox_norm']\n",
        'f = open(p)\ny = f.text\n',
        'x = F.status\ny = F.boxes\nz = F.max_score\n',
        'x = F.rejection_reason\n',
        "x = 'region_status'\n",
        'def f():\n    """Mentions region_bbox_norm in prose."""\n',
    ],
)
def test_the_scanner_ignores_per_box_element_access_and_live_names(snippet: str) -> None:
    assert scan_source(snippet) == [], snippet


def test_the_allowlist_exempts_exactly_its_literals() -> None:
    overlay = 'src/services/labeling/region_overlay.py'
    assert scan_source("K = 'region_text'\n", overlay) == []
    assert scan_source("K = 'region_bbox_norm'\n", overlay) != []
    assert scan_source("K = 'region_text'\n", 'src/other.py') != []
