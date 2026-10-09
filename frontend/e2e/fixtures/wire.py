"""Wire-payload fixture builder for the stubbed e2e suite.

Fills every key from the vendored OpenProcessor contract snapshot
(``contracts/json/item_wire.json``) instead of a hand-copied
shape, so a backend field rename/removal makes ``make_item`` itself go
stale in an obvious way (missing key in ``item_wire.json`` -> KeyError at
import time, since ``DEFAULTS`` is built from ``ITEM_KEYS``).

Mirrors ``src/lib/test/makeItem.ts``'s intent (P0-3) for the Python side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / "contracts" / "json" / "item_wire.json"
)
_CONTRACT: dict[str, Any] = json.loads(_CONTRACT_PATH.read_text())

ITEM_KEYS: list[str] = list(_CONTRACT["item_keys"])
REGION_KEYS: list[str] = list(_CONTRACT["region_keys"])

# Region status values a stub may need (contracts/ts/regionStatus.ts).
REGION_STATUS_VALUES = [
    "pending_detection",
    "pending_verification",
    "detected",
    "verify_rejected",
    "no_region_box",
    "no_region_visible",
    "detection_failed",
    "false_positive",
]

_BBOX = [0.1, 0.2, 0.6, 0.8]

# The served region profile (`{API_PREFIX}/health` `region_profile`,
# OpenProcessor naming-w2) of the neutral fixture domain: widgets carrying
# a tag region (docs/design/domain-neutral-audit-2026-09-24.md §4.4).
# conftest.py serves it by default; a test that needs a deployment with no
# region profile overrides `/health` with `region_profile: None`.
REGION_PROFILE: dict[str, Any] = {
    "name": "widget_tag",
    "display_name": "Widget tags",
    "display_name_singular": "Widget tag",
    "region_class_name": "widget_tag",
    "text_reader": "ocr",
    "reads_text": True,
    "text_hint_enabled": False,
    "limits": {"max_boxes_per_write": 500},
}
# What the app derives from it: the bound class, the region tab's `?tab=`
# id (the backend's own `regions` tab id) and its label (the served
# display name; conftest's default `/review/tabs` serves the same label).
REGION_CLASS = REGION_PROFILE["region_class_name"]
REGION_TAB_URL_ID = "regions"
REGION_TAB_LABEL = REGION_PROFILE["display_name"]
REGION_SINGULAR_LABEL = REGION_PROFILE["display_name_singular"]

# P1 projects cutover (docs/design/
# any-domain-rev3-and-projects-contract-review-2026-09-26.md): the
# GLOBAL `GET {api_prefix}/projects` response every test's root-layout
# bootstrap reads before anything scoped fires. Every scoped call in the
# app is then built from this project's own served `prefix` — never
# assembled client-side — so a stubbed test never has to know the
# `/projects/{slug}` shape itself beyond this fixture.
DEFAULT_PROJECT_SLUG = "default"


def project(api_prefix: str, slug: str, **over: Any) -> dict[str, Any]:
    """One served `ProjectSummary`, `prefix` built the way the server builds
    it. Every non-slug value is overridable (status, writable, selectable,
    deletable, revision, ...)."""
    out: dict[str, Any] = {
        "slug": slug,
        "display_name": slug.capitalize(),
        "description": "",
        "prefix": f"{api_prefix}/projects/{slug}",
        "status": "active",
        "writable": True,
        "selectable": True,
        "is_default": False,
        "deletable": True,
        # The server's own rule (ARCHIVABLE/UNARCHIVABLE_STATUSES); a test
        # overrides either via **over.
        "archivable": over.get("status", "active") == "active",
        "unarchivable": over.get("status") == "archived",
        "revision": 1,
        "paused": False,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "counts": {"images": 0, "items": 0, "validated": 0},
        "origin": None,
    }
    out.update(over)
    return out


def default_project(api_prefix: str) -> dict[str, Any]:
    return project(api_prefix, DEFAULT_PROJECT_SLUG, is_default=True, deletable=False)


PROJECT_STATUS_LABELS = {
    "active": "Active",
    "archived": "Archived",
    "building": "Building",
    "failed": "Failed",
    "deleting": "Deleting",
    "deleted": "Deleted",
}


def projects_response(
    api_prefix: str,
    projects: list[dict[str, Any]] | None = None,
    capacity_status: str = "ok",
) -> dict[str, Any]:
    return {
        "default_slug": DEFAULT_PROJECT_SLUG,
        "projects": projects if projects is not None else [default_project(api_prefix)],
        "capacity": {
            "status": capacity_status,
            "active_shards": 6,
            "per_project_shards": 6,
            "soft_limit": 40,
            "hard_limit": 1000,
            "heap_max_bytes": 2147483648,
            "max_shards_per_node": 1000,
            "data_nodes": 1,
            "projects_until_soft_limit": 5,
            "message": f"Served capacity message ({capacity_status}).",
            "labels": {
                "ok": "Room for more projects",
                "warn": "Near the recommended shard budget",
                "blocked": "No room for another project",
            },
        },
        "limits": {
            "slug_pattern": "^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$",
            "slug_min": 2,
            "slug_max": 32,
            "reserved_slugs": ["all", "combine", "global", "health", "new", "none", "projects", "settings", "vlm"],
            "retired_slugs": [],
            "cloneable_axes": ["settings_defaults", "classes"],
        },
        "labels": {"status": PROJECT_STATUS_LABELS},
        "include_archived": False,
    }

# Every non-default value below is distinct on purpose (same rationale as
# makeItem.ts): a mapping bug that drops a field to a hardcoded default is
# visible via a plain equality check on the field, not just "the UI looked
# fine".
_EXPLICIT: dict[str, Any] = {
    "id": "crop-fixture-001",
    "crop_id": "crop-fixture-001",
    "image_id": "image-fixture-777",
    "image_path": "/fixtures/img-001.jpg",
    "source_image_path": "/fixtures/img-001-src.jpg",
    "bbox_norm": _BBOX,
    "class_id": 42,
    "class_name": "sedan",
    "class_source": "human",
    "confidence": 0.73,
    "classifier_raw_confidence": 0.61,
    "label_source": "human_review",
    "label_validated": True,
    "class_validated": True,
    "class_detector": "model",
    "class_detector_version": "6.2.1",
    "class_labeled_at": "2026-01-02T03:04:05Z",
    "class_labeler": "labeler@example.com",
    "vlm_confidence": "high",
    # Backend main c011721 (VLM class-answer fix): when did the VLM last
    # attempt a class answer, and why it came back empty if it did.
    "vlm_class_attempted_at": "2026-01-02T03:03:00Z",
    "vlm_class_empty_reason": "no_visible_vehicle",
    # Backend main 63d57d8 (dq-queues): a class confidence that matches
    # class_source (DQ-M8's served-side half) and the VLM's raw,
    # pre-registry-match class string. Not adopted by any UI in this
    # batch (per instructions, a separate pass wires these up) -- just
    # present so make_item()'s fail-closed contract-key check passes.
    "class_confidence": 0.73,
    "class_confidence_source": "model",
    "vlm_raw_class": "pickup truck",
    "vlm_proposed_class_id": 99,
    "vlm_proposed_class_name": "pickup_truck",
    "proposed_class_id": 101,
    "proposed_class_name": "suv",
    "needs_new_class": False,
    "needs_new_class_note": None,
    "cluster_id": 17,
    "cluster_kind": "class",
    "cluster_distance": 0.33,
    "cluster_similarity": 0.81,
    "cluster_is_core": True,
    "cluster_subid": "47a",
    "cluster_nearest_id": 18,
    "class_excluded": False,
    "excluded_reason": None,
    "excluded_at": None,
    "review_dismissed_at": None,
    "source": "tag_holdout_sample",
    "test_holdout": False,
    "crop_rank_in_image": 2,
    "crop_area_norm": 0.19,
    "blur_lap_ratio": 4.5,
    "proposal_name": "car",
    "detector_class_name": "sedan",
    "detector_class_id": 8,
    "detector_confidence": 0.72,
    "probe_pred_class": "sedan",
    "probe_pred_class_id": 42,
    "probe_pred_entropy": 0.12,
    "probe_disagreement": True,
    "probe_in_scope": True,
    "probe_actionable": True,
    "probe_model_version": "probe-v1",
    "mistakenness_score": 0.27,
    "mistakenness_method": "entropy",
    "mistakenness_version": "v3",
    "mistakenness_scored_at": "2026-02-03T04:05:06Z",
    "uniqueness_score": 0.55,
    "dup_group_id": None,
    "dup_group_size": 1,
    "dup_is_representative": True,
    "updated_at": "2026-03-04T05:06:07Z",
    "thumbnail_url": "/curation/crops/crop-fixture-001/thumbnail",
    "region_thumbnail_url": "/curation/crops/crop-fixture-001/region_thumbnail",
    "region_bbox_norm": _BBOX,
    "region_bbox_in_parent": _BBOX,
    "region_bbox_frame": "source",
    "region_bbox_correct": True,
    "region_status": "detected",
    "region_score": 0.87,
    "region_confidence": 0.91,
    "region_reason": "uncertainty",
    "region_rejection_reason": None,
    "region_text": "TAG-001",
    "region_text_raw": "TAG 001",
    "region_text_confidence": 0.66,
    "region_text_source": "vlm",
    "region_text_engine_version": "tag_reader-1",
    "region_text_vlm": "TAG-001",
    "region_text_ocr": "TAG-008",
    "region_text_disagreement": True,
    "region_validated": True,
    "region_verified": True,
    "region_verified_at": "2026-04-05T06:07:08Z",
    "region_verifier": "tag_verifier",
    "region_verifier_version": "1.0.0",
    "region_visible": True,
    "region_detector": "tag_detector_v1",
    "region_detector_version": "11.0.2",
    "region_detector_chain": ["tag_detector_v1:miss", "tag_segmenter:hit", "tag_segmenter:vlm_verify_ok"],
    "region_detected_at": "2026-05-06T07:08:09Z",
    "region_cluster_id": 23,
    "region_cluster_subid": "9b",
    "region_cluster_distance": 0.41,
    "region_class_id": 42,
    "region_label_source": "human",
    "region_source": "tag_holdout_sample",
    "region_pairing": "paired",
    "region_skip_verify": False,
    "item_text_lines": [],
    # W8/W10/W3/W4 item keys (backend f582aa05). `region_boxes` is the real
    # per-box list; a test passes its own boxes via make_item(region_boxes=...).
    "vlm_prompt_pack": "tag_pack",
    "vlm_endpoint": "vlm-main",
    "vlm_model": "tag-vlm-1",
    "region_profile": "widget_tag",
    "region_profile_revision": 2,
    "region_boxes": [],
    "region_count": 0,
    "region_rejected_count": 0,
    "region_max_score": None,
    "region_set_complete": None,
    "region_revision": 0,
    "label_locked": False,
    "import_ids": [],
    "dataset_split": None,
    "imported_at": None,
    "proposed_by_import": None,
    "on_negative_frame": False,
    "import_standalone_region": False,
    "proposal_chain": [],
    "origin_project": None,
    "origin_item_id": None,
    "origin_image_id": None,
    "origin_split": None,
    "combine_conflict": False,
    "combine_conflict_origins": [],
    "combine_merged_origins": [],
    # v0.4.0 item keys (backend fce17771): an ordinary embedded item that no
    # open-vocabulary pass found and no region gate skipped.
    "embedding_state": "embedded",
    "source_prompt": None,
    "open_vocab_set": None,
    "open_vocab_revision": None,
    "mask_polygon": None,
    "region_gate_skip": None,
    # Backend main 22a3e65 (dq-region), adopted on the frontend by
    # readSlot (SlotData.text.choice/invalidReason,
    # SlotData.subBox.candidate, SlotData.lifecycle.validated/
    # autoConfirmed) and /review's candidate-box confirm flow. See
    # test_region_verify_rejected_confirm.py / test_region_gallery_status_filter.py.
    "region_text_choice": "vlm_preferred",  # one of region_text.TEXT_CHOICES
    "region_text_vlm_invalid": None,
    "region_auto_confirmed": True,
    "region_candidate_bbox_norm": _BBOX,
    "region_candidate_score": 0.42,
    "region_candidate_detector": "tag_segmenter",
    "region_candidate_detector_version": "3.0.0",
    "region_candidate_source": "segmenter",
    "region_candidate_bbox_in_parent": _BBOX,
}

# Fail loudly at import time (not silently at test time) if the contract
# snapshot grew or lost a key this fixture doesn't know about yet.
_missing = [k for k in ITEM_KEYS if k not in _EXPLICIT]
if _missing:
    raise RuntimeError(
        f"e2e/fixtures/wire.py's make_item() is missing defaults for contract "
        f"keys: {_missing}. Add them (see contracts/json/item_wire.json)."
    )

DEFAULT_ITEM: dict[str, Any] = {k: _EXPLICIT[k] for k in ITEM_KEYS}



def make_item(**overrides: Any) -> dict[str, Any]:
    """Return a full item-wire payload (every key in ``item_wire.json``), with overrides applied.

    Raises on an unknown override key, the same "unknown key is an error"
    guarantee ``makeItem.ts`` gives on the TS side.
    """
    unknown = set(overrides) - set(DEFAULT_ITEM)
    if unknown:
        raise KeyError(f"make_item() got unknown key(s) not in the item wire contract: {unknown}")
    item = dict(DEFAULT_ITEM)
    item.update(overrides)
    return item


# `GET {API_PREFIX}/review/tabs` (ReviewTabsResponse). Every filter-bar
# param /review knows; a stubbed tab honours all of them unless a test
# narrows `filters`.
REVIEW_FILTER_PARAMS = [
    "class_name",
    "exclude_class_name",
    "min_area",
    "max_area",
    "origin",
    "embedding_state",
    "review_status",
    "source",
    "conf_min",
    "conf_max",
    "text",
    "max_rank",
    "min_blur_ratio",
]
REVIEW_EMPTY_STATE = {"has_probe_predictions": True, "has_item_scores": True}


def review_tab(tab_id: str, label: str, **over: Any) -> dict[str, Any]:
    """One served `ReviewTab`, every required field present."""
    return {
        "id": tab_id,
        "label": label,
        "description": "",
        "filters": list(REVIEW_FILTER_PARAMS),
        "filter_defaults": {},
        "filter_specs": [],
        **over,
    }


def review_tabs(*tabs: dict[str, Any], empty_state: dict[str, bool] | None = None) -> dict[str, Any]:
    """A full `ReviewTabsResponse` body."""
    return {"tabs": list(tabs), "empty_state": empty_state or dict(REVIEW_EMPTY_STATE)}


def make_box(box_id: str = "b1", **over: Any) -> dict[str, Any]:
    """One served `region_boxes[]` element (the vendored `RegionTestCandidate`
    keys). Every value is overridable; `thumbnail_url` follows the served
    per-box thumbnail route."""
    box: dict[str, Any] = {
        "box_id": box_id,
        "state": "proposed",
        "bbox_norm": [0.1, 0.1, 0.3, 0.3],
        "bbox_in_parent": [0.1, 0.1, 0.3, 0.3],
        "score": 0.91,
        "detector": "tag_detector_v1",
        "detector_version": "1",
        "source": "detector",
        "bbox_correct": None,
        "confidence": None,
        "rejection_reason": None,
        "text": None,
        "text_raw": None,
        "text_confidence": None,
        "text_source": None,
        "text_engine_version": None,
        "text_vlm": None,
        "text_ocr": None,
        "text_disagreement": None,
        "text_choice": None,
        "text_vlm_invalid": None,
        "cluster_id": None,
        "cluster_subid": None,
        "cluster_distance": None,
        "detected_at": "2026-05-06T07:08:09Z",
        "thumbnail_url": f"/curation/crops/crop-fixture-001/region_thumbnail?box_id={box_id}",
    }
    box.update(over)
    return box
