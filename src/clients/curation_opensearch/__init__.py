"""Generic curation OpenSearch client.

Defines the OpenSearch indexes used by the generic curation / labeling
subsystem plus the ``ClassRegistry`` helper backed by an on-disk
``class_registry.json``. This package has no public re-exports: import each
name from the module that holds it.

Modules:

- ``base``: config singletons, logger, k-NN and plain index settings.
- ``bodies_core`` / ``bodies_other`` / ``items_extra``: per-role index bodies
  and the mapping fragments they share.
- ``lifecycle``: ``INDEX_BODIES`` and idempotent index creation.
- ``ensure_fields`` / ``ensure_overlay_fields`` / ``items_extra``: additive
  ``PUT _mapping`` migrations for indexes created before a field existed.
- ``settings_doc``: the shared curation-settings document.
- ``crops``: batched ``_mget`` for the items index.
- ``registry``: ``ClassRegistry`` and its on-disk file models.

Indexes (logical roles resolved via :func:`src.config.index_name`
against a :class:`~src.config.CurationConfig` instance; the actual
index *names* are deployment data, not hardcoded here):

- ``images`` (``op_prj_<project>__images``): one document per source image
  (with a global embedding).
- ``items`` (``op_prj_<project>__items``): one document per detected item
  crop (with embedding, class label, region-of-interest sub-bbox,
  holdout flag).
- ``labels_confirmed`` (``op_prj_<project>__labels_confirmed``): provenance
  ledger of imported YOLO-style ground-truth labels (no embedding),
  written only by label import. NOT the export source: export and
  training select ``class_validated=true`` items from ``items``, which
  every labeling path (human label/move, auto-promote, label import)
  sets; see ``tests/curation/test_labels_export_roundtrip.py``.
- ``classes`` (``op_prj_<project>__classes``): read-projection of
  ``class_registry.json`` for fast term filters / dashboards. The JSON
  file is the canonical source; this index is rebuilt from it via
  :py:meth:`ClassRegistry.sync_to_opensearch`.

Every OpenSearch field reference for the per-item "region of interest"
sub-annotation (e.g. a defect region on an item crop) is routed
through the module-level :class:`~src.config.RegionFields` instance
(``F``) rather than hardcoded; see ``src/config/region_fields.py``
for the full design rationale. Everything else in the item schema
(``crop_id``, ``class_id``, clustering/scoring/probe fields, ...) is
already deployment-agnostic and is not indirected.

The two ``class_registry.json`` file models live here, not in ``registry.py``:
FastAPI derives their OpenAPI schema names from the defining module, so moving
them would rename schemas in the generated contracts.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class RegistryClassEntry(BaseModel):
    """Single class entry in ``class_registry.json``.

    The on-disk file uses ``id`` / ``name`` keys for backward
    compatibility with older label-file schemas. Internally we expose
    ``class_id`` / ``class_name`` to avoid shadowing Python builtins.
    Pydantic ``AliasChoices`` handles the bridge in both directions.

    Named ``RegistryClassEntry`` (not ``ClassEntry``) to avoid
    colliding with the distinct HTTP-response ``ClassEntry`` model in
    ``src.routers.curation._common``.
    """

    model_config = ConfigDict(populate_by_name=True)

    class_id: int = Field(validation_alias=AliasChoices('class_id', 'id'))
    class_name: str = Field(validation_alias=AliasChoices('class_name', 'name'))
    group: str = 'unknown'
    sample_count: int = 0
    validated_count: int = 0
    added_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        validation_alias=AliasChoices('added_at', 'added'),
    )
    deprecated: bool = False
    notes: str = ''
    merged_into: int | None = None  # populated when this class is merged into another
    # Optional single-character keyboard shortcut for fast assignment in the
    # labeler. Persisted in class_registry.json across sessions and devices.
    hotkey_letter: str | None = None


class ClassRegistryFile(BaseModel):
    """On-disk schema for ``class_registry.json``."""

    version: int = 1
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    classes: list[RegistryClassEntry] = Field(default_factory=list)


# =============================================================================
# ClassRegistry — append-only registry with snapshots + atomic writes
# =============================================================================
