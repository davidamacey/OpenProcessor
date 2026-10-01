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
3. no module spells a retired wire name in a string -- a whole literal
   (``'region_bbox_norm'``), a name embedded in a longer string (a painless
   script, ``"ctx._source.region_score = 1"``), or one built from constant
   parts (``'region_' + 'score'``, an f-string, ``''.join``, ``'%s' % x``) --
   which would bypass the indirection;
4. nothing reaches a ``RegionFields`` value reflectively (``getattr`` /
   ``vars`` / ``__dict__`` / ``asdict``), and an alias of one (``g = F``,
   tuple unpacking, ``for f in (F,)``, an imported factory under another
   name, ``mod.get_region_fields()``, ``self.F``) is tracked like the value.

Not caught, by design: an untyped parameter (``def f(cfg): cfg.score``) and a
fully dynamic key (``src.get(prefix + name)``). Both fail loudly at runtime on
the frozen dataclass or read ``None`` from a key no writer sets; the guard is
a lint, not a seal.

Allowlist (the whole of it, each entry justified, and pinned to the exact
number of occurrences so a new one in the same file fails):

- ``src/services/labeling/region_overlay.py``: ``region_bbox_correct`` /
  ``region_confidence`` / ``region_text`` are the keys of the VLM's JSON
  *reply* -- a fixed protocol the prompt packs ask for -- not storage names.
  It defines them once (``REPLY_*_KEY``); every other module imports them.
- ``src/services/curation/review_sorts.py``: ``region_score`` is the served
  review-sort *id* of ``GET /review/{tab}?sort=``, not a storage name (the
  sort reads ``region_max_score``); the id is spelled in the sort's
  definition and in the default-sort map.
- ``src/services/labeling/vlm_prompts.py``: the default prompt packs spell
  the same three VLM reply keys in their instruction text (any number of
  times -- it is prompt prose); only those three names are exempt there.
- ``src/services/curation/wire.py``: one ``getattr`` that maps a live
  ``RegionFields`` attribute (from ``REGION_WIRE_ATTRS``) to its wire key.

``region_fields.py`` itself is excluded (it defines the names).
"""

from __future__ import annotations

import ast
import re
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
_SELF_RECEIVERS = frozenset({'F', '_F', 'fields', '_fields', 'storage', 'region_fields'})
_FIELD_FACTORIES = frozenset({'get_region_fields', 'RegionFields'})
_REFLECTION = frozenset({'getattr', 'vars', 'asdict'})
_WIRE_NAME_RE = re.compile(
    r'(?<![A-Za-z0-9_])('
    + '|'.join(sorted(RETIRED_WIRE_NAMES, key=len, reverse=True))
    + r')(?![A-Za-z0-9_])'
)

# path (relative, posix) -> {wire name: the exact number of times it is
# spelled in that file}.
LITERAL_ALLOWLIST: dict[str, dict[str, int]] = {
    'src/services/labeling/region_overlay.py': {
        'region_bbox_correct': 1,
        'region_confidence': 1,
        'region_text': 1,
    },
    'src/services/curation/review_sorts.py': {'region_score': 2},
    # The model-choices row id for the profile's detector model (a wire id of
    # `GET /models/choices`, not the retired item-level scalar).
    'src/services/curation/model_choices.py': {'region_detector': 1},
}
# Files whose prompt prose may repeat the VLM reply keys without a count.
UNCOUNTED_ALLOWLIST: dict[str, frozenset[str]] = {
    'src/services/labeling/vlm_prompts.py': frozenset(
        {'region_bbox_correct', 'region_confidence', 'region_text'}
    ),
}
# path -> the number of reflective reads of a RegionFields value allowed.
REFLECTION_ALLOWLIST: dict[str, int] = {'src/services/curation/wire.py': 1}


def _factory_names(tree: ast.AST) -> frozenset[str]:
    """The factory names in scope: the canonical ones plus any ``import ... as``
    alias of one."""
    names = set(_FIELD_FACTORIES)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.asname for a in node.names if a.asname and a.name in _FIELD_FACTORIES)
    return frozenset(names)


def _is_fields_factory_call(node: ast.AST, factories: frozenset[str] = _FIELD_FACTORIES) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in factories
    if isinstance(func, ast.Attribute):
        # ``mod.get_region_fields()`` and the constructor ``RegionFields.from_env``
        return func.attr in _FIELD_FACTORIES or (
            func.attr == 'from_env' and _name_of(func.value) in factories
        )
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


def _is_self_fields(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == 'self'
        and node.attr in _SELF_RECEIVERS
    )


def _pairs(target: ast.AST, value: ast.AST) -> list[tuple[ast.AST, ast.AST]]:
    """``(target, value)`` pairs of an assignment, unpacking matching tuples."""
    if (
        isinstance(target, ast.Tuple | ast.List)
        and isinstance(value, ast.Tuple | ast.List)
        and len(target.elts) == len(value.elts)
    ):
        return [pair for t, v in zip(target.elts, value.elts, strict=True) for pair in _pairs(t, v)]
    return [(target, value)]


def _bound_field_names(tree: ast.AST, factories: frozenset[str]) -> set[str]:
    """Local names bound to a RegionFields value: assigned from a factory,
    ``self.fields`` or another bound name (aliases, tuple unpacking, a loop
    over a tuple holding one), or annotated ``RegionFields``."""
    names: set[str] = set(_ALIASES)

    def is_fields_value(node: ast.AST) -> bool:
        return (
            (isinstance(node, ast.Name) and node.id in names)
            or _is_fields_factory_call(node, factories)
            or _is_self_fields(node)
        )

    changed = True
    while changed:
        before = len(names)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    for t, v in _pairs(target, node.value):
                        if isinstance(t, ast.Name) and is_fields_value(v):
                            names.add(t.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and (
                    _annotation_mentions_region_fields(node.annotation)
                    or (node.value is not None and is_fields_value(node.value))
                ):
                    names.add(node.target.id)
            elif isinstance(node, ast.For):
                if (
                    isinstance(node.target, ast.Name)
                    and isinstance(node.iter, ast.Tuple | ast.List)
                    and any(is_fields_value(e) for e in node.iter.elts)
                ):
                    names.add(node.target.id)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                names.update(
                    a.arg for a in args if _annotation_mentions_region_fields(a.annotation)
                )
        changed = len(names) != before
    return names


def _fold_joined(node: ast.JoinedStr) -> str | None:
    parts: list[str] = []
    for part in node.values:
        if isinstance(part, ast.Constant):
            parts.append(str(part.value))
        elif isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Constant):
            parts.append(str(part.value.value))
        else:
            return None
    return ''.join(parts)


def _fold_binop(node: ast.BinOp) -> str | None:
    left = _folded_str(node.left)
    if left is None:
        return None
    if isinstance(node.op, ast.Add):
        right = _folded_str(node.right)
        return None if right is None else left + right
    if isinstance(node.op, ast.Mod):
        operands = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
        if all(isinstance(o, ast.Constant) for o in operands):
            try:
                return left % tuple(o.value for o in operands)  # type: ignore[attr-defined]
            except (TypeError, ValueError):
                return None
    return None


def _fold_call(node: ast.Call) -> str | None:
    if not isinstance(node.func, ast.Attribute):
        return None
    receiver = _folded_str(node.func.value)
    if receiver is None:
        return None
    if node.func.attr == 'join' and len(node.args) == 1:
        seq = node.args[0]
        if isinstance(seq, ast.List | ast.Tuple):
            items = [_folded_str(e) for e in seq.elts]
            return None if None in items else receiver.join(items)  # type: ignore[arg-type]
    if node.func.attr == 'format' and all(isinstance(a, ast.Constant) for a in node.args):
        try:
            return receiver.format(*(a.value for a in node.args))  # type: ignore[attr-defined]
        except (IndexError, KeyError, ValueError):
            return None
    return None


def _folded_str(node: ast.AST) -> str | None:
    """The string ``node`` evaluates to when it is built from constants only
    (a literal, ``'a' + 'b'``, an f-string of constants, ``''.join([...])``,
    ``'%s' % 'x'``, ``'{}'.format('x')``); ``None`` otherwise."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        return _fold_joined(node)
    if isinstance(node, ast.BinOp):
        return _fold_binop(node)
    if isinstance(node, ast.Call):
        return _fold_call(node)
    return None


def _string_findings(tree: ast.AST) -> list[tuple[int, str]]:
    """``(line, wire name)`` for every retired wire name spelled in a string
    the module builds or holds (docstrings excluded)."""
    docstrings = _docstring_constants(tree)
    found: list[tuple[int, str]] = []

    def visit(node: ast.AST) -> None:
        if id(node) in docstrings:
            return
        folded = (
            _folded_str(node)
            if isinstance(node, ast.Constant | ast.JoinedStr | ast.BinOp | ast.Call)
            else None
        )
        if folded is not None:
            found.extend((node.lineno, m.group(1)) for m in _WIRE_NAME_RE.finditer(folded))  # type: ignore[attr-defined]
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return found


def wire_name_counts(text: str) -> dict[str, int]:
    """How many times each retired wire name is spelled in ``text``."""
    counts: dict[str, int] = {}
    for _line, name in _string_findings(ast.parse(text)):
        counts[name] = counts.get(name, 0) + 1
    return counts


def scan_source(text: str, rel_path: str = '<memory>') -> list[str]:
    """Every retired-scalar access, reflective read or wire-name string in ``text``."""
    tree = ast.parse(text)
    factories = _factory_names(tree)
    bound = _bound_field_names(tree, factories)

    def is_fields_value(node: ast.AST) -> bool:
        return (
            (isinstance(node, ast.Name) and node.id in bound)
            or _is_fields_factory_call(node, factories)
            or _is_self_fields(node)
        )

    found: list[str] = []
    reflective_allowed = REFLECTION_ALLOWLIST.get(rel_path, 0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and is_fields_value(node.value):
            if node.attr in RETIRED:
                found.append(f'{rel_path}:{node.lineno}: <RegionFields>.{node.attr}')
            elif node.attr == '__dict__':
                found.append(f'{rel_path}:{node.lineno}: <RegionFields>.__dict__')
        elif isinstance(node, ast.Call) and node.args and is_fields_value(node.args[0]):
            callee = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else getattr(node.func, 'attr', None)
            )
            if callee in _REFLECTION:
                if reflective_allowed > 0:
                    reflective_allowed -= 1
                else:
                    found.append(f'{rel_path}:{node.lineno}: {callee}(<RegionFields>)')
    allowed = dict(LITERAL_ALLOWLIST.get(rel_path, {}))
    uncounted = UNCOUNTED_ALLOWLIST.get(rel_path, frozenset())
    for line, name in _string_findings(tree):
        if name in uncounted:
            continue
        if allowed.get(name, 0) > 0:
            allowed[name] -= 1
            continue
        found.append(f'{rel_path}:{line}: {name!r}')
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


def test_the_allowlist_pins_the_exact_number_of_occurrences() -> None:
    for rel, expected in LITERAL_ALLOWLIST.items():
        path = REPO_ROOT / rel
        assert path.is_file(), f'{rel} missing or moved'
        assert wire_name_counts(path.read_text(encoding='utf-8')) == expected, rel


def test_the_uncounted_and_reflection_allowlists_still_need_their_entries() -> None:
    for rel, names in UNCOUNTED_ALLOWLIST.items():
        counts = wire_name_counts((REPO_ROOT / rel).read_text(encoding='utf-8'))
        assert set(counts) == set(names), f'{rel}: drop an entry that is no longer used'
    for rel, n in REFLECTION_ALLOWLIST.items():
        assert not scan_source((REPO_ROOT / rel).read_text(encoding='utf-8'), rel), rel
        text = (REPO_ROOT / rel).read_text(encoding='utf-8')
        assert len(scan_source(text, 'src/x.py')) == n, f'{rel}: reflective read count changed'


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
        # aliases of a RegionFields value
        'g = F\ny = g.bbox_norm\n',
        'a, b = F, 1\ny = a.score\n',
        'for f in (F,):\n    y = f.score\n',
        'g = F\nh = g\ny = h.detector\n',
        'from src.config import get_region_fields as grf\ny = grf().bbox_norm\n',
        'import src.config as c\ny = c.get_region_fields().score\n',
        'class A:\n    def m(self):\n        return self.F.score\n',
        'class A:\n    def m(self):\n        return self._F.detector\n',
        # reflective reads
        "y = getattr(F, 'bbox_norm')\n",
        "y = getattr(get_region_fields(), 'score')\n",
        "k = 'bbox_norm'\ny = vars(F)[k]\n",
        'y = F.__dict__\n',
        'from dataclasses import asdict\ny = asdict(F)\n',
        # a wire name built from constant parts, or embedded in a longer string
        "y = src.get('region_' + 'bbox_norm')\n",
        'y = src.get(f\'region_{"bbox_norm"}\')\n',
        "y = src.get(''.join(['region_', 'score']))\n",
        "y = src.get('region_%s' % 'bbox_norm')\n",
        "y = src.get('region_{}'.format('score'))\n",
        'script = "ctx._source.region_score = 1"\n',
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
        "x = 'region_text_disabled'\ny = 'region_detector_chain'\n",
        "y = getattr(box, 'score')\n",
        'y = vars(box)\n',
        "y = src.get('region_' + name)\n",
    ],
)
def test_the_scanner_ignores_per_box_element_access_and_live_names(snippet: str) -> None:
    assert scan_source(snippet) == [], snippet


def test_the_allowlist_exempts_exactly_its_literals() -> None:
    overlay = 'src/services/labeling/region_overlay.py'
    assert scan_source("K = 'region_text'\n", overlay) == []
    assert scan_source("K = 'region_bbox_norm'\n", overlay) != []
    assert scan_source("K = 'region_text'\n", 'src/other.py') != []


def test_the_allowlist_exempts_only_its_pinned_count() -> None:
    sorts = 'src/services/curation/review_sorts.py'
    two = "a = 'region_score'\nb = 'region_score'\n"
    assert scan_source(two, sorts) == []
    assert len(scan_source(two + "c = 'region_score'\n", sorts)) == 1
    # an embedded spelling counts the same as a whole literal
    assert len(scan_source(two + "c = 'ctx._source.region_score = 1'\n", sorts)) == 1
    # and a name the file is not allowed is flagged however often it is spelled
    assert scan_source("a = 'region_bbox_norm'\n", sorts) != []
