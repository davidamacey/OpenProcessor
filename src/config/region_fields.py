"""OpenSearch field-name indirection for the per-item "region of interest".

This is the settled design documented in
``docs/design/curation_design_rationale.md`` §4: an
existing deployment's live OpenSearch field names (e.g. ``roi_status``,
``roi_bbox_norm``, …) are NOT renamed — there is zero data migration and
zero reindex risk. Instead, code stops hardcoding those literal strings and
routes every OpenSearch document-field reference through a
:class:`RegionFields` instance. The generic OSS defaults use ``region_*``
names; a deployment with existing data under other names (e.g. a future
overlay for that deployment) constructs its own instance with those
names — a rename becomes a config flip, not a code change.

**Scope — what this module does and does NOT govern:**

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
    sub-annotation (e.g. a defect region on an item crop).

    Defaults are the generic OSS names. A deployment that already has
    data under different names (e.g. ``roi_status``,
    ``roi_bbox_norm``, etc.) constructs an instance with the existing
    names instead — no reindex required.

    Item-level fields only. A region's per-box data (geometry, score,
    detector, state, text, cluster placement, ...) is an element of the
    ``boxes`` list and is addressed by the fixed element keys of
    :class:`~src.services.curation.region_boxes.RegionBox`, never through
    this indirection.
    """

    prefix: str = 'region'

    status: str = 'region_status'
    reason: str = 'region_reason'
    rejection_reason: str = 'region_rejection_reason'

    # Human validation only: a human confirmed (or drew / rejected) the
    # region. Machine verdicts never set it.
    validated: str = 'region_validated'
    # The worker's auto-confirm policy accepted the box (detector and
    # verifier agreed strongly enough to accept it without a human). An
    # accepted-but-unreviewed region: it stays in the human review queue.
    auto_confirmed: str = 'region_auto_confirmed'
    verified: str = 'region_verified'
    verified_at: str = 'region_verified_at'
    verifier: str = 'region_verifier'
    verifier_version: str = 'region_verifier_version'
    visible: str = 'region_visible'

    detector_chain: str = 'region_detector_chain'
    detected_at: str = 'region_detected_at'

    # Config-store provenance (W2): which activated region profile
    # (name) + revision produced this write. Item-level fields, in
    # ``_items_body()`` from the start -- not folded into ``kind ==
    # 'config'`` docs.
    profile: str = 'region_profile'
    profile_revision: str = 'region_profile_revision'

    # W8 multi-box list. ``boxes`` is the indirected storage name of the
    # list itself; the element keys inside each list entry are FIXED
    # strings (see docs/design/openprocessor_internal/any_domain_plan.md
    # W8.2) since the list is new and no deployment has legacy names for
    # them. ``boxes_state`` below is only used to build nested queries
    # (``box_query``) against the fixed element key ``state``.
    boxes: str = 'region_boxes'
    boxes_state: str = 'state'
    box_embeddings: str = 'region_box_embeddings'
    count: str = 'region_count'
    rejected_count: str = 'region_rejected_count'
    max_score: str = 'region_max_score'
    set_complete: str = 'region_set_complete'
    revision: str = 'region_revision'
    box_seq: str = 'region_box_seq'

    class_id: str = 'region_class_id'
    label_source: str = 'region_label_source'
    pairing: str = 'region_pairing'
    # Internal cascade flag: set when a detector's confidence was high
    # enough to skip the VLM verify round-trip entirely (see
    # DetectionProfile / the curation worker's fast-path). Added while
    # porting the curation worker, per the standing instruction: "if you hit
    # a literal with no matching attribute, add the attribute." (See
    # docs/design/curation_design_rationale.md §4.)
    skip_verify: str = 'region_skip_verify'

    # Legacy-suffixed columns kept for rollback (e.g. roi_*_legacy).
    status_legacy: str = 'region_status_legacy'

    @classmethod
    def from_env(cls, env_prefix: str = 'OP_REGION_FIELD_') -> RegionFields:
        """Per-field env override: ``OP_REGION_FIELD_STATUS=roi_status``, …

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
