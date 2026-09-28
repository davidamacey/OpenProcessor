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
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

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


def _attr_pattern(attr: str) -> re.Pattern[str]:
    """Matches ``F.<attr>`` / ``fields.<attr>`` / ``_F.<attr>``, never a
    box-element attribute access (``b.<attr>`` / ``box.<attr>``) -- the
    per-box ``RegionBox`` dataclass legitimately has fields with the same
    names (``detector``, ``score``, ...), read off actual box objects,
    not the item-level ``RegionFields`` storage-name indirection this
    guard polices."""
    return re.compile(rf'\b(?:F|_F|fields)\.{re.escape(attr)}\b')


def test_ported_files_still_exist() -> None:
    """Catches a silent rename/move that would make this guard a no-op."""
    for rel_path in PORTED_FILES:
        assert (REPO_ROOT / rel_path).is_file(), f'{rel_path} missing or moved'


def test_ported_files_never_regress_to_the_removed_legacy_scalars() -> None:
    violations: list[str] = []
    for rel_path, attrs in PORTED_FILES.items():
        text = (REPO_ROOT / rel_path).read_text()
        for attr in attrs:
            for lineno, line in enumerate(text.splitlines(), start=1):
                if _attr_pattern(attr).search(line):
                    violations.append(f'{rel_path}:{lineno}: {line.strip()!r} (attr={attr!r})')
    assert not violations, 'legacy region scalar regression(s):\n' + '\n'.join(violations)


def test_guard_pattern_catches_a_real_regression() -> None:
    """The guard must actually fire, not just pass by construction --
    exercises `_attr_pattern` directly against a synthetic regression."""
    pattern = _attr_pattern('bbox_norm')
    assert pattern.search("filt.append({'exists': {'field': F.bbox_norm}})")
    assert pattern.search('box = src.get(fields.bbox_norm)')


def test_guard_pattern_does_not_false_positive_on_box_element_access() -> None:
    """A `RegionBox` instance's own `.bbox_norm` (the per-box element,
    not the item-level `RegionFields` storage name) must never trip this
    guard -- distinct concepts that happen to share an attribute name."""
    pattern = _attr_pattern('bbox_norm')
    assert not pattern.search('box.bbox_norm')
    assert not pattern.search('b.bbox_norm')
    assert not pattern.search("doc['bbox_norm']")
