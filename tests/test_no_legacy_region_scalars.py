"""W8-cleanup Item 3 (scoped): a regression guard for the legacy per-box
scalar reads Item 1/2 of the W8-cleanup pass actually removed.

**Scope, honestly stated.** This is NOT the full "no file outside
``region_fields.py`` may reference a legacy scalar" guard the W8-cleanup
plan describes as the end state -- that guard can only be written once
``PUT /crops/{id}/region`` / ``PUT /crops/batch_region`` and
``region_writes.py``'s single-box write chain (``region_box_write``,
``region_box_doc``, ``region_confirm_doc``, ``same_box``,
``candidate_promotion``, ``human_status_fields``) are deleted -- they are
still live this pass (see the W8-cleanup handback report: full PUT-route
deletion needs ~5 test files' PUT-region coverage migrated to the W8a box
routes first, deliberately not rushed). ``wire.py``'s ``region_to_wire``
also still reads every legacy scalar by design (an additive, documented
wire-compat mirror -- see its module docstring) and
``edit_history.py``/``cascade_detect.py`` legitimately support that live
chain.

What this guard DOES enforce, for real: the eight production files this
pass ported off the retired item-level scalars (because the current
box-list worker never writes them, so reading them was silently wrong)
must never regress back to reading them. Each entry names exactly the
``RegionFields`` attributes removed from that file -- add a file here
only in the same commit that finishes porting it, per the same
ratchet-allowlist convention ``scripts/codegen/
check_no_literal_region_fields.py`` already uses for its own domain-
literal guard.
"""

from __future__ import annotations

import re
from dataclasses import fields as dc_fields
from pathlib import Path

from src.config.region_fields import RegionFields


REPO_ROOT = Path(__file__).resolve().parents[1]

# Wire (OpenSearch) storage name for each RegionFields attribute, e.g.
# 'bbox_norm' -> 'region_bbox_norm' -- used to also catch a bare string
# literal referencing the field, bypassing the `F.<attr>` indirection
# entirely.
_WIRE_NAME: dict[str, str] = {
    f.name: f.default for f in dc_fields(RegionFields) if isinstance(f.default, str)
}

# Per-file extra local aliases for `self.fields` / `get_region_fields()`,
# beyond the default `F` / `_F` / `fields` -- e.g.
# `export_single_class_rows.py`'s `f = self.fields` was a real, previously
# undetected regression vector (m4).
_ALIAS_OVERRIDES: dict[str, tuple[str, ...]] = {
    'src/services/curation/export_single_class_rows.py': ('F', '_F', 'fields', 'f'),
}
_DEFAULT_ALIASES: tuple[str, ...] = ('F', '_F', 'fields')

# (file, attr) pairs where the bare wire-name literal is a legitimate,
# unrelated string in that file -- not a bypass of the RegionFields
# indirection. `regions_edit.py`'s wire model's own field is literally
# named `region_rejection_reason` (Pydantic `model_fields_set` checks
# compare against that name directly); that is not an OpenSearch
# document-key reference and must not trip the literal check.
_LITERAL_CHECK_EXCLUDED: frozenset[tuple[str, str]] = frozenset(
    {('src/routers/curation/regions_edit.py', 'rejection_reason')}
)

# path (relative to repo root) -> RegionFields attribute names that file
# must never access as `F.<attr>` / `fields.<attr>` / `_F.<attr>` again.
# Deliberately narrow to what THIS pass actually removed from each file --
# not a guess at every attribute that could theoretically regress.
PORTED_FILES: dict[str, tuple[str, ...]] = {
    'scripts/curation/backfill_region_embeddings.py': ('bbox_norm',),
    'src/services/curation/region_eval.py': ('bbox_norm', 'bbox_frame', 'detector', 'score'),
    'src/services/curation/review_queries.py': ('bbox_norm', 'candidate_bbox_norm', 'text'),
    'src/routers/curation/regions_fp.py': ('bbox_norm',),
    'src/services/curation/export_single_class_rows.py': ('bbox_norm',),
    'src/routers/curation/stats.py': ('bbox_norm', 'detector'),
    'src/routers/curation/regions.py': ('bbox_norm', 'score', 'detector', 'text', 'detected_at'),
    # `bbox_norm` legitimately stays here: PUT /crops/{id}/region (not yet
    # removed) still reads/writes it via region_writes.py's single-box
    # chain. Only PATCH region_meta's now-removed unconditional item-level
    # rejection_reason write is guarded.
    'src/routers/curation/regions_edit.py': ('rejection_reason',),
}


def _attr_patterns(
    attr: str, aliases: tuple[str, ...], *, include_literal: bool = True
) -> list[re.Pattern[str]]:
    """Every way a regressed line can reference the retired ``attr``:

    - ``F.<attr>`` / ``fields.<attr>`` / ``_F.<attr>`` (+ any per-file
      alias, e.g. ``f.<attr>`` in ``export_single_class_rows.py``) --
      never a box-element attribute access (``b.<attr>`` / ``box.<attr>``),
      the per-box ``RegionBox`` dataclass legitimately has fields with the
      same names, read off actual box objects, not the item-level
      ``RegionFields`` storage-name indirection this guard polices.
    - ``get_region_fields().<attr>`` -- the indirection called inline
      instead of bound to a name first.
    - the bare wire-string literal itself (``'region_bbox_norm'`` /
      ``"region_bbox_norm"``), bypassing the indirection entirely.
    """
    alias_group = '|'.join(re.escape(a) for a in aliases)
    patterns = [
        re.compile(rf'\b(?:{alias_group})\.{re.escape(attr)}\b'),
        re.compile(rf'get_region_fields\(\)\.{re.escape(attr)}\b'),
    ]
    wire_name = _WIRE_NAME.get(attr)
    if wire_name and include_literal:
        patterns.append(re.compile(rf"""['"]{re.escape(wire_name)}['"]"""))
    return patterns


def test_ported_files_still_exist() -> None:
    """Catches a silent rename/move that would make this guard a no-op."""
    for rel_path in PORTED_FILES:
        assert (REPO_ROOT / rel_path).is_file(), f'{rel_path} missing or moved'


def test_ported_files_never_regress_to_the_removed_legacy_scalars() -> None:
    violations: list[str] = []
    for rel_path, attrs in PORTED_FILES.items():
        aliases = _ALIAS_OVERRIDES.get(rel_path, _DEFAULT_ALIASES)
        text = (REPO_ROOT / rel_path).read_text()
        for attr in attrs:
            include_literal = (rel_path, attr) not in _LITERAL_CHECK_EXCLUDED
            patterns = _attr_patterns(attr, aliases, include_literal=include_literal)
            for lineno, line in enumerate(text.splitlines(), start=1):
                if any(p.search(line) for p in patterns):
                    violations.append(f'{rel_path}:{lineno}: {line.strip()!r} (attr={attr!r})')
    assert not violations, 'legacy region scalar regression(s):\n' + '\n'.join(violations)


def _matches(attr: str, line: str, aliases: tuple[str, ...] = _DEFAULT_ALIASES) -> bool:
    return any(p.search(line) for p in _attr_patterns(attr, aliases))


def test_guard_pattern_catches_a_real_regression() -> None:
    """The guard must actually fire, not just pass by construction --
    exercises `_attr_patterns` directly against a synthetic regression."""
    assert _matches('bbox_norm', "filt.append({'exists': {'field': F.bbox_norm}})")
    assert _matches('bbox_norm', 'box = src.get(fields.bbox_norm)')


def test_guard_pattern_catches_the_f_alias() -> None:
    """m4: `export_single_class_rows.py`'s `f = self.fields` local alias
    used to be a silent guard no-op (`src.get(f.bbox_norm)` stayed green
    even after a mutation reintroduced it)."""
    aliases = _ALIAS_OVERRIDES['src/services/curation/export_single_class_rows.py']
    assert _matches('bbox_norm', 'box = src.get(f.bbox_norm)', aliases)
    # The default alias set (no per-file override) must NOT treat a bare
    # `f` as the indirection -- it's too common a name (loop vars, file
    # handles) elsewhere to blanket-match.
    assert not _matches('bbox_norm', 'box = src.get(f.bbox_norm)')


def test_guard_pattern_catches_inline_get_region_fields_call() -> None:
    """m4: `get_region_fields().bbox_norm` (the indirection called inline,
    never bound to a name) used to bypass the guard entirely."""
    assert _matches('bbox_norm', 'box = src.get(get_region_fields().bbox_norm)')


def test_guard_pattern_catches_the_bare_wire_string_literal() -> None:
    """m4: a literal `'region_bbox_norm'` bypasses the `F.<attr>`
    indirection entirely and used to slip past the guard."""
    assert _matches('bbox_norm', "src.get('region_bbox_norm')")
    assert _matches('bbox_norm', 'src.get("region_bbox_norm")')


def test_guard_pattern_does_not_false_positive_on_box_element_access() -> None:
    """A `RegionBox` instance's own `.bbox_norm` (the per-box element,
    not the item-level `RegionFields` storage name) must never trip this
    guard -- distinct concepts that happen to share an attribute name."""
    assert not _matches('bbox_norm', 'box.bbox_norm')
    assert not _matches('bbox_norm', 'b.bbox_norm')
    assert not _matches('bbox_norm', "doc['bbox_norm']")
