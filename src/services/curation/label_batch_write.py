"""The item write one VLM class prediction becomes (``POST /vlm/label_batch``).

Shared by the route and the pack test-on-crop preview (W5): the update for
one prediction (:func:`label_batch_update`) and the merge onto the item's
current doc (:func:`label_batch_merge`: cluster placement, unmatched-class
clearing and the class-history snapshot), so a preview is the write the
route would land.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.class_sources import VLM_UNMATCHED_CLASS_SOURCE, unmatched_class_clear
from src.services.curation.clustering.id_normalize import class_cluster_placement
from src.services.curation.vlm_class_attempt import prediction_class_update, with_class_snapshot


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.services.labeling.vlm_models import VlmClassPrediction


def label_batch_update(
    pred: VlmClassPrediction,
    *,
    name_to_id: dict[str, int],
    resolve: Callable[..., str | None],
    now: str,
    pack_stamp: str,
    vlm_endpoint: str,
    vlm_model: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """``(update, new_class_proposal)`` for one prediction; ``update`` is
    ``None`` when the call never completed. The update carries who answered
    and with which prompt pack."""
    provenance = {
        'class_detector': 'vlm',
        'class_detector_version': '1',
        'class_labeler': 'vlm',
        'class_labeled_at': now,
    }
    update, proposal = prediction_class_update(
        pred, name_to_id=name_to_id, resolve=resolve, now=now, provenance=provenance
    )
    if update is None:
        return None, None
    update['vlm_prompt_pack'] = pack_stamp
    update['vlm_endpoint'] = vlm_endpoint
    update['vlm_model'] = vlm_model
    return update, proposal


def label_batch_merge(update: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """``update`` as written onto ``current``: a registry class moves the item
    into its class cluster (as the worker's combined call does), an unmatched
    answer does not keep the class it just contradicted, and the previous
    class is snapshotted for undo."""
    merged = dict(update)
    merged.update(class_cluster_placement(merged, current))
    if merged.get('class_source') == VLM_UNMATCHED_CLASS_SOURCE:
        merged.update(unmatched_class_clear(current))
    return with_class_snapshot(merged, current, writer='vlm_label_batch')


__all__ = ['label_batch_merge', 'label_batch_update']
