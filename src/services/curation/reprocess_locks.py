"""The lock rule, as reprocess applies it: one place, two forms.

Reprocess only regenerates machine output; human and imported labels and
boxes are locked. Every scope decides "is this locked" here, never inline:

* the *document* form (``*_locked(src)``) decides per item, and is what an
  OCC merger re-checks against the fresh doc at write time;
* the *query* form (``*_locked_clause()``) selects the same set in
  OpenSearch, so a dry run counts ``locked_skipped`` exactly without
  paging every document.

``tests/curation/test_reprocess_locks.py`` runs both forms over one table
of documents and asserts they agree, so they cannot drift apart.

The document forms delegate to :mod:`src.clients.occ_locks`; the region
form is the item's region *set* only (a human-validated region set,
including a human "no region visible" verdict, has no machine output to
regenerate), because a human class label does not stop a region redetect.
"""

from __future__ import annotations

from typing import Any

from src.clients.occ_locks import is_locked_class, is_locked_item
from src.config.region_fields import RegionFields, get_region_fields
from src.services.curation.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE


def region_set_locked(src: dict[str, Any], F: RegionFields | None = None) -> bool:
    """The item's region set is validated: a human (or an import) set it."""
    return bool(src.get((F or get_region_fields()).validated))


def class_locked(src: dict[str, Any]) -> bool:
    return is_locked_class(src)


def item_locked(src: dict[str, Any], F: RegionFields | None = None) -> bool:
    """Any locked part at all (class, box, validated region verdict): the
    test ``detect`` uses before it would delete or replace an item."""
    return is_locked_item(src, F)


def region_locked_clause(F: RegionFields | None = None) -> dict[str, Any]:
    return {'term': {(F or get_region_fields()).validated: True}}


def class_locked_clause() -> dict[str, Any]:
    """Query form of :func:`~src.clients.occ_locks.is_locked_class`."""
    return {
        'bool': {
            'should': [
                {'wildcard': {'class_source': '*human*'}},
                {
                    'bool': {
                        'filter': [
                            {'term': {'class_source': LABEL_IMPORT_CLASS_SOURCE}},
                            {'term': {'class_validated': True}},
                        ]
                    }
                },
                {'term': {'test_holdout': True}},
            ],
            'minimum_should_match': 1,
        }
    }


__all__ = [
    'class_locked',
    'class_locked_clause',
    'item_locked',
    'region_locked_clause',
    'region_set_locked',
]
