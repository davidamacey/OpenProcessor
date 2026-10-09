"""The single writer of a validated class label on an item.

Every human class-label write (``label_crop``, ``batch_label_crops``
(``src/routers/curation/crops.py``), ``move_crops``, and
``POST /review/new_class_proposals/resolve``
(``src/routers/curation/review_resolve.py``)) and every dataset-import
class-label write (W10, ``dataset_import/job.py``) go through this module.
There is no second dict literal anywhere else that sets
``class_validated: True`` together with a human/import ``class_source`` —
``tests/test_class_label_single_writer.py`` is the gate.

:class:`ItemLabel` is the one label shape both writers build: ``.human(...)``
for a human write, ``.imported(...)`` for a dataset import. Both flow
through :func:`class_label_fields` (a brand-new item, no snapshot) or
:func:`class_label_update` (an existing item, with a restorable snapshot).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from src.services.curation.audit_math import audit_outcome_fields
from src.services.curation.ingest_class_sources import (
    HUMAN_CLASS_SOURCE,
    LABEL_IMPORT_CLASS_SOURCE,
    LABEL_SOURCE_IMPORT,
)
from src.services.detection.cascade_detect import class_provenance


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def human_class_provenance() -> dict[str, Any]:
    """Class provenance for every human class write."""
    from src.services.detection.profile_registry import region_profile_or_neutral

    human = region_profile_or_neutral()
    return class_provenance(
        detector=human.human_detector_name,
        detector_version=human.human_detector_version,
        labeler='human',
    )


def import_class_provenance(import_id: str, *, now: str | None = None) -> dict[str, Any]:
    """Class provenance for a dataset-import class write (W10.10)."""
    return class_provenance(
        detector='import', detector_version=import_id, labeler='import', labeled_at=now
    )


@dataclass(frozen=True)
class ItemLabel:
    """The one label shape every class-writer builds.

    ``source`` decides ``class_source``/``label_source``/provenance:
    ``"human"`` -> ``class_source=HUMAN_CLASS_SOURCE``; ``"import"`` ->
    ``class_source=LABEL_IMPORT_CLASS_SOURCE`` (``external_label``),
    ``label_source=LABEL_SOURCE_IMPORT``. ``validated=False`` is only
    valid for ``source="import"`` (``options.label_trust: "suggestion"``,
    W10.10): the machine pipeline may still relabel it, so it must not be
    locked (:func:`src.clients.occ.is_locked_class`).
    """

    class_id: int
    class_name: str
    source: Literal['human', 'import']
    writer: str
    label_source: str | None = None
    validated: bool = True
    import_id: str | None = None
    labeled_at: str | None = None

    @classmethod
    def human(cls, *, class_id: int, class_name: str, label_source: str, writer: str) -> ItemLabel:
        return cls(
            class_id=class_id,
            class_name=class_name,
            source='human',
            writer=writer,
            label_source=label_source,
        )

    @classmethod
    def imported(
        cls,
        *,
        import_id: str,
        class_id: int,
        class_name: str,
        trust: Literal['validated', 'suggestion'] = 'validated',
        now: str | None = None,
        writer: str | None = None,
    ) -> ItemLabel:
        return cls(
            class_id=class_id,
            class_name=class_name,
            source='import',
            writer=writer or f'import:{import_id}',
            validated=trust == 'validated',
            import_id=import_id,
            labeled_at=now,
        )


def class_label_fields(label: ItemLabel) -> dict[str, Any]:
    """The class fields for a BRAND NEW item (no prior state to snapshot).

    Used by ``build_item_doc`` (``item_doc.py``) when ``DetectedItem.label``
    is set — a labeled object from a dataset import, or (in principle) any
    future first-write labeler.
    """
    if label.source == 'human':
        fields: dict[str, Any] = {
            'class_id': label.class_id,
            'class_name': label.class_name,
            'class_source': HUMAN_CLASS_SOURCE,
            'class_validated': True,
            'label_source': label.label_source or 'human',
            'cluster_id': label.class_id,
            'cluster_subid': None,
            **human_class_provenance(),
        }
    else:
        fields = {
            'class_id': label.class_id,
            'class_name': label.class_name,
            'class_source': LABEL_IMPORT_CLASS_SOURCE,
            'class_validated': label.validated,
            'label_source': label.label_source or LABEL_SOURCE_IMPORT,
            **import_class_provenance(label.import_id or '', now=label.labeled_at),
        }
        if label.validated:
            fields['cluster_id'] = label.class_id
            fields['cluster_subid'] = None
    return fields


def class_label_update(current: dict[str, Any], label: ItemLabel) -> dict[str, Any]:
    """Merge body for a class-label write on an EXISTING item: a
    restorable pre-write snapshot plus :func:`class_label_fields`.

    This is the single function every ``human_label_update``-shaped
    human writer calls — the one class-label writer the AST gate
    (``tests/test_class_label_single_writer.py``) checks for.

    W10 fix-pass correction (Opus review 2026-09-28, minor m10): dataset
    import's `import_dataset()` (`dataset_import/job.py`) does NOT call
    this on its update path — it goes through `occ_upsert_bulk`'s
    generic `_merge_preserving_human` guard instead (now with an
    explicit `is_locked_item` pre-filter, see the W10 fix-pass CHANGELOG
    entry), which has no restorable snapshot. Routing dataset import's
    update path through this function for a real snapshot is future
    work, not done this pass.
    """
    from src.services.curation.history import record_class_snapshot

    history = record_class_snapshot(current, writer=label.writer, restorable=True)
    fields = class_label_fields(label)
    fields['class_id_history'] = history
    fields['updated_at'] = _now_iso()
    if label.source == 'human' and label.validated:
        fields.update(audit_outcome_fields(current, label.class_name))
    return fields


def human_label_update(
    current: dict[str, Any],
    *,
    class_id: int,
    class_name: str,
    label_source: str,
    writer: str,
) -> dict[str, Any]:
    """Back-compat call shape for the existing human-write call sites
    (``label_crop``, ``batch_label_crops``) — builds an :class:`ItemLabel`
    and delegates to :func:`class_label_update`."""
    return class_label_update(
        current,
        ItemLabel.human(
            class_id=class_id, class_name=class_name, label_source=label_source, writer=writer
        ),
    )


def human_move_class_update(
    current: dict[str, Any], *, class_id: int, class_name: str, now: str
) -> dict[str, Any]:
    """Merge body for ``move_crops`` into a CLASS cluster: relabel via
    ``class_source='human_move'`` (crops.py's ``move_crops`` docstring:
    "move this to the X cluster" means "this is an X")."""
    from src.services.curation.history import record_class_snapshot

    history = record_class_snapshot(current, writer='human:move_crops', restorable=True)
    return {
        'cluster_id': class_id,
        'class_id': class_id,
        'class_name': class_name,
        'class_source': 'human_move',
        'class_validated': True,
        'label_source': 'human',
        'class_id_history': history,
        'cluster_subid': None,
        **human_class_provenance(),
        **audit_outcome_fields(current, class_name),
        'updated_at': now,
    }


def candidate_move_update(current: dict[str, Any], *, cluster_id: int, now: str) -> dict[str, Any]:
    """Merge body for ``move_crops`` into a candidate cluster: placement only.

    The item joins the group unvalidated. A locked (human- or
    import-owned, validated) class is cleared -- the human just said the
    item isn't that class -- while a machine suggestion is kept. The
    pre-write state is snapshotted so undo restores it exactly.
    """
    from src.clients.occ import is_locked_class
    from src.services.curation.history import CLASS_STATE_FIELDS, record_class_snapshot

    update: dict[str, Any] = {
        'class_id_history': record_class_snapshot(
            current, writer='human:move_crops', restorable=True
        ),
    }
    if is_locked_class(current):
        update.update(dict.fromkeys(CLASS_STATE_FIELDS))
    update.update(
        {
            'cluster_id': cluster_id,
            'cluster_subid': None,
            'class_validated': False,
            'updated_at': now,
        }
    )
    return update


__all__ = [
    'ItemLabel',
    'candidate_move_update',
    'class_label_fields',
    'class_label_update',
    'human_class_provenance',
    'human_label_update',
    'human_move_class_update',
    'import_class_provenance',
]
