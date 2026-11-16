"""Where a subject's rows live: erasure's catalogue of (table, key column) (15a section 8).

Erasure is the one path that deletes claims (R3). It reaches every row of a photo through this data:
the root delete, the tables that follow by ``ON DELETE CASCADE`` (listed so a test can prove the
chain), the tables that hold a key with no foreign key by design (deleted explicitly, by subject
key), the per-space vector tables (found through ``embedding_spaces``), and the files. The schema test
``tests/storage/test_erasure_catalogue.py`` fails when a table with a subject-key column is missing
here, and when a listed cascade is not one in the models.

This module is data and two string builders: it opens no session. The job that runs it (storage step 3)
sets the transaction-local guard ``op.erasure_job`` and deletes in chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.db.vector_ddl import qualified


Reach = Literal['root', 'cascade', 'set_null', 'explicit']


@dataclass(frozen=True)
class Entry:
    table: str
    column: str
    reach: Reach
    via: str = 'photos'
    """The table the cascade or the delete is reached from (a photo, an item, a face)."""


def _entries(reach: Reach, via: str, column: str, *tables: str) -> tuple[Entry, ...]:
    return tuple(Entry(t, column, reach, via) for t in tables)


CATALOGUE: tuple[Entry, ...] = (
    Entry('photos', 'photo_key', 'root', 'photos'),
    # photo_key, ON DELETE CASCADE
    *_entries(
        'cascade',
        'photos',
        'photo_key',
        'photo_provenance',
        'analysis_ledger',
        'photo_capture',
        'items',
        'detections',
        'text_lines',
        'faces',
        'claims',
        'photo_state',
        'photo_files',
        'photo_derivatives',
        'photo_signals',
        'photo_places',
    ),
    # item_key, ON DELETE CASCADE from items
    *_entries(
        'cascade',
        'items',
        'item_key',
        'class_opinions',
        'item_region_results',
        'region_boxes',
        'probe_predictions',
        'item_scores',
        'item_projections',
        'assist_answers',
        'item_notes',
        'item_state',
        'box_state',
    ),
    Entry('box_texts', 'box_key', 'cascade', 'region_boxes'),
    Entry('face_state', 'face_key', 'cascade', 'faces'),
    # a person: its entity row and what cascades from it; the photos stay
    Entry('entities', 'entity_key', 'root', 'entities'),
    *_entries(
        'cascade', 'entities', 'entity_key', 'entity_state', 'entity_edges', 'membership_state'
    ),
    Entry('claims', 'subject_entity_key', 'cascade', 'entities'),
    # the job that erases keeps its row and its counts; its subject columns are nulled at done
    Entry('erasure_jobs', 'entity_key', 'explicit'),
    # ON DELETE SET NULL: the row stays, the key goes
    Entry('entity_state', 'cover_photo_key', 'set_null', 'photos'),
    # no foreign key by design (the key is a text id, or a table that must not cascade): deleted by key
    Entry('vector_gaps', 'subject_key', 'explicit'),
    Entry('entity_edges', 'subject_key', 'explicit'),
    Entry('membership_state', 'subject_key', 'explicit'),
    Entry('cluster_memberships', 'item_key', 'explicit', 'items'),
    Entry('near_dup_members', 'member_id', 'explicit'),
    Entry('near_dup_edges', 'a_id', 'explicit'),
    Entry('near_dup_edges', 'b_id', 'explicit'),
    Entry('holdout_near_dup_moves', 'item_id', 'explicit', 'items'),
    Entry('near_dup_decisions', 'a_id', 'explicit'),
    Entry('near_dup_decisions', 'b_id', 'explicit'),
    Entry('near_dup_pending', 'member_id', 'explicit'),
    Entry('dataset_import_images', 'image_id', 'explicit'),
    Entry('ingest_source_files', 'image_id', 'explicit'),
    Entry('image_contents', 'image_id', 'explicit'),
    Entry('job_items', 'item_id', 'explicit'),
    # a claim that names a photo only through a wire id (a duplicate verdict's second member)
    Entry('claims', 'subject_pair_id', 'explicit'),
)

# Files a subject owns that are not rows (15a section 8); the job lists them and deletes each.
FILES = (
    'the original and bundle members when the project owns them (uploads); a source root read in place is never written',
    'crop and thumbnail cache entries',
    'exports (dataset_exports)',
    'the holdout snapshot (holdout_freezes.snapshot_storage_key), rewritten without the photo',
    'training data copies under each run artifact_prefix',
)

# Deliberately kept (plan 15 section 13, open for the owner): a model trained on an erased photo keeps what it learned.
NOT_ERASED = ('models', 'model_engines', 'run checkpoints')

# Rows that name wire ids inside a document: deleted by entry, not by key column.
DOCUMENT_REFERENCES = ('events.payload', 'jobs.params', 'label_operations.params')


def tables() -> frozenset[str]:
    return frozenset(e.table for e in CATALOGUE)


# A cascade reached from a photo goes through the photo, its items, its faces or its region boxes.
_PHOTO_CHAIN = frozenset({'photos', 'items', 'faces', 'region_boxes'})


def cascade_tables() -> frozenset[str]:
    """The tables a delete of a photo row removes by cascade, the photo itself included."""
    return frozenset(
        e.table for e in CATALOGUE if e.reach in ('root', 'cascade') and e.via in _PHOTO_CHAIN
    )


def vector_delete_statement(lock_key: int, space_key: str) -> str:
    """The delete of a subject's vectors, run before the photo delete: a vector table has no foreign
    key to its subject (the template cannot know which table the key points to)."""
    return f'DELETE FROM {qualified(lock_key, space_key)} WHERE subject_key = ANY(:keys)'  # nosec B608  validated names
