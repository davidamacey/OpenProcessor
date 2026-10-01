"""The document and query forms of the reprocess lock predicates agree, so
a dry run's ``locked_skipped`` (a query) is exactly what an apply skips (a
per-document check)."""

from __future__ import annotations

from typing import Any

import pytest

from curation.query_fakes import matches
from src.config.region_fields import get_region_fields
from src.services.curation.reprocess_locks import (
    class_locked,
    class_locked_clause,
    region_locked_clause,
    region_set_locked,
)


F = get_region_fields()

CLASS_DOCS: dict[str, dict[str, Any]] = {
    'human': {'class_source': 'human'},
    'human_move': {'class_source': 'human_move'},
    'vlm_human_confirmed': {'class_source': 'vlm_human_confirmed'},
    'import_validated': {'class_source': 'external_label', 'class_validated': True},
    'import_suggestion': {'class_source': 'external_label', 'class_validated': False},
    'vlm': {'class_source': 'vlm', 'class_validated': True},
    'holdout_machine': {'class_source': 'vlm', 'test_holdout': True},
    'proposal': {'class_source': 'unlabeled_proposal'},
    'bare': {},
}


@pytest.mark.parametrize('name', sorted(CLASS_DOCS))
def test_class_lock_document_and_query_forms_agree(name: str) -> None:
    doc = CLASS_DOCS[name]
    assert class_locked(doc) == matches(doc, class_locked_clause()), name


@pytest.mark.parametrize(
    'doc',
    [{}, {F.validated: False}, {F.validated: True}, {F.validated: True, F.verifier: 'human'}],
)
def test_region_lock_document_and_query_forms_agree(doc: dict[str, Any]) -> None:
    assert region_set_locked(doc) == matches(doc, region_locked_clause())


def test_the_lock_table_has_both_outcomes() -> None:
    outcomes = {class_locked(d) for d in CLASS_DOCS.values()}
    assert outcomes == {True, False}
