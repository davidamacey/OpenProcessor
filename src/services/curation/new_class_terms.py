"""New-class proposals: the one selection, and the served term rules.

``GET /review/new_class_proposals`` (the queue),
``GET /review/new_class_proposals/summary`` (the proposed names) and
``POST /review/new_class_proposals/resolve`` (bulk-resolve a name) all
select through :func:`proposal_query`, so the summary's total equals the
queue's total and a term's count equals what a resolve for it matches
(the summary used to count only ``vlm_new_class_pending`` rows).

A proposed name is not always a class worth creating. :func:`classify_term`
flags three kinds, from rules that are served back to the client
(:meth:`ProposalTermRules.to_wire`) and never hardcode a vocabulary:

* ``existing_class`` — the name already is an active registry class (map
  to it instead of creating a duplicate);
* ``generic_parent`` — the whole name is a configured generic term
  (``OP_NEW_CLASS_GENERIC_TERMS``) or the name of a registry group (or
  one ``-``-separated part of one), i.e. a parent of existing classes;
* ``non_object`` — the name, or one ``_``-separated token of it, is a
  configured non-object term (``OP_NEW_CLASS_NON_OBJECT_TERMS``: blur,
  empty scene, ...).

Generic terms match whole names only, so a specific kind of a generic
thing (``sports_car`` under a generic ``car``) stays a proposal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from src.services.curation.class_sources import VLM_NEW_CLASS_PENDING_CLASS_SOURCE
from src.services.curation.review_queries import build_tab_query


PROPOSAL_TAB = 'new_class_proposals'
PROPOSED_NAME_FIELD = 'vlm_proposed_class'
GENERIC_TERMS_ENV = 'OP_NEW_CLASS_GENERIC_TERMS'
NON_OBJECT_TERMS_ENV = 'OP_NEW_CLASS_NON_OBJECT_TERMS'

FLAG_EXISTING_CLASS = 'existing_class'
FLAG_GENERIC_PARENT = 'generic_parent'
FLAG_NON_OBJECT = 'non_object'


def proposal_query(*, include_test: bool = False) -> dict[str, Any]:
    """Every item the new-class queue serves (same must / must_not)."""
    must, must_not, _reason = build_tab_query(
        PROPOSAL_TAB, include_test=include_test, text=None, max_rank=None
    )
    return {'bool': {'must': must, 'must_not': must_not}}


def proposal_term_query(label: str, *, include_test: bool = False) -> dict[str, Any]:
    """The queue's items proposing exactly ``label``."""
    return {
        'bool': {
            'filter': [
                proposal_query(include_test=include_test),
                {'term': {PROPOSED_NAME_FIELD: label}},
            ]
        }
    }


def is_open_proposal(doc: dict[str, Any], label: str) -> bool:
    """Python twin of :func:`proposal_term_query` (``include_test=False``),
    re-checked against the fresh doc at write time."""
    if doc.get(PROPOSED_NAME_FIELD) != label:
        return False
    if doc.get('class_validated') or doc.get('class_excluded') or doc.get('test_holdout'):
        return False
    if doc.get('review_dismissed_at') is not None:
        return False
    # A resolved class_id means this item no longer needs a new class,
    # whatever a stale needs_new_class flag says (mirrors the
    # must_not-exists-class_id guard in review_queries.build_tab_query).
    if doc.get('class_id') is not None:
        return False
    return bool(doc.get('needs_new_class')) or (
        doc.get('class_source') == VLM_NEW_CLASS_PENDING_CLASS_SOURCE
    )


def normalize_term(term: str) -> str:
    """Lowercase, trimmed, spaces and hyphens folded to ``_``."""
    return '_'.join(term.strip().lower().replace('-', ' ').replace('_', ' ').split())


def _terms(raw: str) -> frozenset[str]:
    return frozenset(t for t in (normalize_term(p) for p in raw.split(',')) if t)


@dataclass(frozen=True)
class ProposalTermRules:
    generic_terms: frozenset[str]
    non_object_terms: frozenset[str]
    # Normalized registry group names and their '-'-separated parts.
    registry_groups: frozenset[str]
    # Normalized active class name -> class_id.
    existing_classes: dict[str, int]

    def to_wire(self) -> dict[str, Any]:
        return {
            'generic_terms': sorted(self.generic_terms),
            'non_object_terms': sorted(self.non_object_terms),
            'registry_groups_are_generic': True,
            'existing_classes_flagged': True,
            'generic_terms_env': GENERIC_TERMS_ENV,
            'non_object_terms_env': NON_OBJECT_TERMS_ENV,
        }


def load_term_rules(registry: Any) -> ProposalTermRules:
    """Rules from the env (read per call) and the live class registry."""
    groups: set[str] = set()
    existing: dict[str, int] = {}
    for entry in registry.load().classes:
        group = str(getattr(entry, 'group', '') or '')
        if group:
            groups.add(normalize_term(group))
            groups.update(normalize_term(part) for part in group.split('-') if part.strip())
        if not getattr(entry, 'deprecated', False):
            existing[normalize_term(entry.class_name)] = int(entry.class_id)
    return ProposalTermRules(
        generic_terms=_terms(os.environ.get('OP_NEW_CLASS_GENERIC_TERMS', '')),
        non_object_terms=_terms(os.environ.get('OP_NEW_CLASS_NON_OBJECT_TERMS', '')),
        registry_groups=frozenset(g for g in groups if g),
        existing_classes=existing,
    )


def _matches_non_object_pattern(term: str, patterns: frozenset[str]) -> bool:
    """True when ``term`` (or one of its ``_``-separated tokens) matches a
    configured non-object rule.

    A plain rule (e.g. ``blur``) matches the whole term or one whole
    token, exactly as before. A rule ending in ``*`` (``unidentifiable_*``)
    also matches a term/token it PREFIXES; a rule starting with ``*``
    (``*_scene``) also matches a term/token it SUFFIXES -- so a single
    configured rule covers a family of terms (``unidentifiable_object``,
    ``dark_scene``, ``blurred_object``) instead of needing every literal
    variant enumerated.
    """
    candidates = (term, *term.split('_'))
    for pattern in patterns:
        is_prefix_rule = pattern.endswith('*') and not pattern.startswith('*')
        is_suffix_rule = pattern.startswith('*') and not pattern.endswith('*')
        if is_prefix_rule:
            stem = pattern[:-1].rstrip('_')
            if stem and any(c.startswith(stem) for c in candidates):
                return True
        elif is_suffix_rule:
            stem = pattern[1:].lstrip('_')
            if stem and any(c.endswith(stem) for c in candidates):
                return True
        elif pattern in candidates:
            return True
    return False


def classify_term(label: str, rules: ProposalTermRules) -> tuple[str | None, int | None]:
    """``(flag, class_id)``: ``flag`` is ``None`` for a term worth offering
    as a new class; ``class_id`` is set only for ``existing_class``."""
    term = normalize_term(label)
    if term in rules.existing_classes:
        return FLAG_EXISTING_CLASS, rules.existing_classes[term]
    if term in rules.generic_terms or term in rules.registry_groups:
        return FLAG_GENERIC_PARENT, None
    if _matches_non_object_pattern(term, rules.non_object_terms):
        return FLAG_NON_OBJECT, None
    return None, None


__all__ = [
    'FLAG_EXISTING_CLASS',
    'FLAG_GENERIC_PARENT',
    'FLAG_NON_OBJECT',
    'GENERIC_TERMS_ENV',
    'NON_OBJECT_TERMS_ENV',
    'PROPOSED_NAME_FIELD',
    'ProposalTermRules',
    'classify_term',
    'is_open_proposal',
    'load_term_rules',
    'normalize_term',
    'proposal_query',
    'proposal_term_query',
]
