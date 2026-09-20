"""OpenSearch field-name indirection for the per-item "region of interest".

This is the settled design documented in
``docs/design/curation_design_rationale.md`` §4: an
existing deployment's live OpenSearch field names (e.g. ``plate_status``,
``plate_bbox_norm``, …) are NOT renamed — there is zero data migration and
zero reindex risk. Instead, code stops hardcoding those literal strings and
routes every OpenSearch document-field reference through a
:class:`RegionFields` instance. The generic OSS defaults use ``region_*``
names; a deployment with existing data under other names (e.g. a future
overlay for that deployment) constructs its own instance with those
names — a rename becomes a config flip, not a code change.

**Scope — what this module does and does NOT govern** (§3.2 scope table):

- OpenSearch query bodies, ``_source`` lists, bulk update docs, painless
  scripts, and index mapping bodies — YES, governed by ``RegionFields``.
- Pydantic attribute names on HTTP wire models (the generic curation
  API's JSON contract) — NO, frozen independently. See
  ``docs/design/curation_api_contract.md``.
- Enum *values* in ``src/config/region_state.py`` — NO, values not field
  names, untouched in Phase 2.
- Metric names — NO, deferred to a later phase.

f-string composition of a legacy suffix (``f'{F.status}_legacy'``) is
banned — use the explicit ``*_legacy`` attributes below, so
``scripts/codegen/check_no_literal_region_fields.py`` can stay a simple
literal scan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RegionFields:
    """OpenSearch field names for the per-item "region of interest"
    sub-annotation (e.g. the license plate on a vehicle crop).

    Defaults are the generic OSS names. A deployment that already has
    data under different names (e.g. ``plate_status``,
    ``plate_bbox_norm``, etc.) constructs an instance with the existing
    names instead — no reindex required.
    """

    prefix: str = 'region'

    bbox_norm: str = 'region_bbox_norm'
    bbox_frame: str = 'region_bbox_frame'
    bbox_correct: str = 'region_bbox_correct'
    status: str = 'region_status'
    score: str = 'region_score'
    confidence: str = 'region_confidence'
    reason: str = 'region_reason'
    rejection_reason: str = 'region_rejection_reason'

    text: str = 'region_text'
    text_raw: str = 'region_text_raw'
    text_confidence: str = 'region_text_confidence'
    text_source: str = 'region_text_source'
    text_engine_version: str = 'region_text_engine_version'

    validated: str = 'region_validated'
    verified: str = 'region_verified'
    verified_at: str = 'region_verified_at'
    verifier: str = 'region_verifier'
    verifier_version: str = 'region_verifier_version'
    visible: str = 'region_visible'

    detector: str = 'region_detector'
    detector_version: str = 'region_detector_version'
    detector_chain: str = 'region_detector_chain'
    detected_at: str = 'region_detected_at'

    embedding: str = 'region_embedding'
    cluster_id: str = 'region_cluster_id'
    cluster_subid: str = 'region_cluster_subid'
    cluster_distance: str = 'region_cluster_distance'

    class_id: str = 'region_class_id'
    label_source: str = 'region_label_source'
    source: str = 'region_source'
    pairing: str = 'region_pairing'
    # Internal cascade flag: set when a detector's confidence was high
    # enough to skip the VLM verify round-trip entirely (see
    # DetectionProfile / the curation worker's fast-path). Not part of
    # the original attribute count in the reference audit -- added while
    # porting Chunk 8's worker, per the standing instruction: "if you hit
    # a literal with no matching attribute, add the attribute." (See
    # docs/design/curation_design_rationale.md §4.)
    skip_verify: str = 'region_skip_verify'

    # Legacy-suffixed columns kept for rollback (e.g. plate_*_legacy).
    bbox_norm_legacy: str = 'region_bbox_norm_legacy'
    score_legacy: str = 'region_score_legacy'
    status_legacy: str = 'region_status_legacy'

    @classmethod
    def from_env(cls, env_prefix: str = 'OP_REGION_FIELD_') -> RegionFields:
        """Per-field env override: ``OP_REGION_FIELD_STATUS=plate_status``, …

        Unset env vars fall back to the dataclass default for that field.
        """
        defaults = cls()
        overrides = {}
        for f in fields(defaults):
            env_name = f'{env_prefix}{f.name.upper()}'
            value = os.environ.get(env_name)
            if value is not None:
                overrides[f.name] = value
        return cls(**overrides)


_default_region_fields: RegionFields | None = None


def get_region_fields() -> RegionFields:
    """Module-level default ``RegionFields`` instance.

    Built via :meth:`RegionFields.from_env` so the ``OP_REGION_FIELD_*``
    env vars documented on that classmethod actually take effect for the
    process-wide default — this was previously constructing a bare
    ``RegionFields()`` and silently ignoring every ``OP_REGION_FIELD_*``
    override.

    Callers that need a deployment-specific instance (e.g. a future
    overlay for an existing deployment) should construct and inject
    their own rather than relying on this default — mirrors
    :func:`src.config.curation.get_curation_config`.
    """
    global _default_region_fields  # noqa: PLW0603 - lazily-built module singleton
    if _default_region_fields is None:
        _default_region_fields = RegionFields.from_env()
    return _default_region_fields
