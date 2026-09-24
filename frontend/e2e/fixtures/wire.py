"""Wire-payload fixture builder for the stubbed e2e suite.

Fills every key from the vendored OpenProcessor contract snapshot
(``contracts/openprocessor/json/item_wire.json``) instead of a hand-copied
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
    Path(__file__).resolve().parents[2] / "contracts" / "openprocessor" / "json" / "item_wire.json"
)
_CONTRACT: dict[str, Any] = json.loads(_CONTRACT_PATH.read_text())

ITEM_KEYS: list[str] = list(_CONTRACT["item_keys"])
REGION_KEYS: list[str] = list(_CONTRACT["region_keys"])

# Region status values a stub may need (contracts/openprocessor/ts/regionStatus.ts).
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
    "class_detector": "v6_model",
    "class_detector_version": "6.2.1",
    "class_labeled_at": "2026-01-02T03:04:05Z",
    "class_labeler": "user@example.com",
    "vlm_confidence": "high",
    # Backend main 6a1f350 (VLM class-answer fix): when did the VLM last
    # attempt a class answer, and why it came back empty if it did.
    "vlm_class_attempted_at": "2026-01-02T03:03:00Z",
    "vlm_class_empty_reason": "no_visible_vehicle",
    # Backend main 7254ec4 (dq-queues): a class confidence that matches
    # class_source (DQ-M8's served-side half) and the VLM's raw,
    # pre-registry-match class string. Not adopted by any UI in this
    # batch (per instructions, a separate pass wires these up) -- just
    # present so make_item()'s fail-closed contract-key check passes.
    "class_confidence": 0.73,
    "class_confidence_source": "v6_model",
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
    "source": "lpr_frozen_test_sample",
    "test_holdout": False,
    "crop_rank_in_image": 2,
    "crop_area_norm": 0.19,
    "blur_lap_ratio": 4.5,
    "proposal_name": "car",
    "probe_pred_class": "sedan",
    "probe_pred_class_id": 42,
    "probe_pred_entropy": 0.12,
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
    "region_text": "ABC123",
    "region_text_raw": "ABC-123",
    "region_text_confidence": 0.66,
    "region_text_source": "vlm",
    "region_text_engine_version": "gemma-4-e4b",
    "region_text_vlm": "ABC123",
    "region_text_ocr": "ABC128",
    "region_text_disagreement": True,
    "region_validated": True,
    "region_verified": True,
    "region_verified_at": "2026-04-05T06:07:08Z",
    "region_verifier": "gemma-4-e4b",
    "region_verifier_version": "1.0.0",
    "region_visible": True,
    "region_detector": "lpr_nanov11_640",
    "region_detector_version": "11.0.2",
    "region_detector_chain": ["lpr_nanov11_640:miss", "sam3:hit", "sam3:vlm_verify_ok"],
    "region_detected_at": "2026-05-06T07:08:09Z",
    "region_cluster_id": 23,
    "region_cluster_subid": "9b",
    "region_cluster_distance": 0.41,
    "region_class_id": 42,
    "region_label_source": "human",
    "region_source": "lpr_frozen_test_sample",
    "region_pairing": "paired",
    "region_skip_verify": False,
    "item_text_lines": [],
}

# Fail loudly at import time (not silently at test time) if the contract
# snapshot grew or lost a key this fixture doesn't know about yet.
_missing = [k for k in ITEM_KEYS if k not in _EXPLICIT]
if _missing:
    raise RuntimeError(
        f"e2e/fixtures/wire.py's make_item() is missing defaults for contract "
        f"keys: {_missing}. Add them (see contracts/openprocessor/json/item_wire.json)."
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
