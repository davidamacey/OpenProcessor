"""Generic curation OpenSearch client.

Defines the OpenSearch indexes used by the generic curation / labeling
subsystem. This package has no public re-exports: import each name from the
module that holds it.

Modules:

- ``base``: config singletons, logger, k-NN index settings.
- ``bodies_core`` / ``items_extra``: per-role index bodies
  and the mapping fragments they share.
- ``lifecycle``: ``INDEX_BODIES`` and idempotent index creation.
- ``ensure_fields`` / ``ensure_overlay_fields`` / ``items_extra``: additive
  ``PUT _mapping`` migrations for indexes created before a field existed.
- ``crops``: batched ``_mget`` for the items index.

Indexes (logical roles resolved via :func:`src.config.index_name`
against a :class:`~src.config.CurationConfig` instance; the actual
index *names* are deployment data, not hardcoded here):

- ``images`` (``op_prj_<project>__images``): one document per source image
  (with a global embedding).
- ``items`` (``op_prj_<project>__items``): one document per detected item
  crop (with embedding, class label, region-of-interest sub-bbox,
  holdout flag).

Every other control record is in PostgreSQL.

Every OpenSearch field reference for the per-item "region of interest"
sub-annotation (e.g. a defect region on an item crop) is routed
through the module-level :class:`~src.config.RegionFields` instance
(``F``) rather than hardcoded; see ``src/config/region_fields.py``
for the full design rationale. Everything else in the item schema
(``crop_id``, ``class_id``, clustering/scoring/probe fields, ...) is
already deployment-agnostic and is not indirected.

The registry's two wire models live here, not with the PostgreSQL service:
FastAPI derives their OpenAPI schema names from the defining module, so moving
them would rename schemas in the generated contracts.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class RegistryClassEntry(BaseModel):
    """Single class entry of the registry.

    ``id`` / ``name`` are accepted as aliases (the export snapshot's
    older label-file schema). Internally we expose
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
    added_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        validation_alias=AliasChoices('added_at', 'added'),
    )
    deprecated: bool = False
    notes: str = ''
    merged_into: int | None = None  # populated when this class is merged into another
    # Optional single-character keyboard shortcut for fast assignment in the
    # labeler. Persisted with the class across sessions and devices.
    hotkey_letter: str | None = None


class ClassRegistryFile(BaseModel):
    """The registry as the export snapshot (``class_registry.json``) holds it."""

    version: int = 1
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    classes: list[RegistryClassEntry] = Field(default_factory=list)

    def get(self, class_id: int) -> RegistryClassEntry | None:
        return next((c for c in self.classes if c.class_id == class_id), None)

    def validate_id(self, class_id: int) -> bool:
        """True iff a non-deprecated class with this id exists."""
        return any(c.class_id == class_id and not c.deprecated for c in self.classes)
