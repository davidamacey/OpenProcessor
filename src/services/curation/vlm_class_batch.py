"""The VLM class answer of a batch of items, from crops to written updates.

Shared by the auto-label VLM stage and the assist runs so there is one class
write path: crops for a chunk of items (:func:`chunk_crops`), the item update
each prediction becomes (:func:`updates_from_predictions`), and the OCC bulk
write under which a concurrent human edit always wins (:func:`write_updates`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NamedTuple

from opensearchpy.exceptions import OpenSearchException
from PIL import Image

from src.config import get_region_fields
from src.config.curation_accessors import items_index
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.services.curation.event_hub import publish_crop_classified
from src.services.labeling.run_labeler import vlm_stamp


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.class_write_guard import ClassWriteGuard


logger = get_logger(__name__)


class ChunkResult(NamedTuple):
    """One chunk's outcome. ``failed``: crops whose VLM request or reply failed;
    ``unreadable``: selected crops whose image could not be read, so the VLM never
    saw them."""

    predicted: int
    updated: int
    proposals: list[dict[str, Any]]
    failed: int = 0
    unreadable: int = 0


@dataclass
class VlmContext:
    """The labeler, the class vocabulary and the write guard one VLM run uses."""

    labeler: Any
    guard: ClassWriteGuard
    class_names: list[str]
    name_to_id: dict[str, int]
    class_catalog: Any
    pack_stamp: str
    force_fit_bypass: dict[str, int]

    def resolve_class_name(self, raw: str, *, confidence: str | None = None) -> str | None:
        """Map a VLM reply to a registry class name (or None). Skips fuzzy/synonym
        resolution when ``confidence='low'`` -- see
        :func:`src.services.labeling.vlm_class_names.resolve_class_name`."""
        from src.services.labeling.vlm_class_names import resolve_class_name

        if confidence == 'low':
            self.force_fit_bypass['low_conf_skipped'] += 1
        else:
            self.force_fit_bypass['attempted'] += 1
        return resolve_class_name(raw, self.name_to_id, confidence=confidence)  # type: ignore[arg-type]


def class_context(labeler: Any, guard: ClassWriteGuard, *, revision: int | None) -> VlmContext:
    """The class vocabulary and prompt catalog of the bound project's registry, with
    ``labeler`` (called directly: no HTTP hop) and the write ``guard``."""
    from src.clients.curation_opensearch.registry import get_class_registry
    from src.services.curation.region_class import item_classes
    from src.services.labeling.vlm_class_names import format_class_catalog
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    labelable = item_classes(get_class_registry().load().classes)
    # Render the registry as a grouped+described catalog so the VLM's prompt tells it what
    # each cryptic slug means visually -- a big quality lift over the bare CSV.
    class_dicts = [
        {'class_name': c.class_name, 'group': getattr(c, 'group', None)} for c in labelable
    ]
    return VlmContext(
        labeler=labeler,
        guard=guard,
        class_names=[c.class_name for c in labelable],
        name_to_id={c.class_name: c.class_id for c in labelable},
        class_catalog=format_class_catalog(class_dicts, labeler._pack),
        pack_stamp=prompt_pack_stamp(labeler._pack, revision=revision),
        # How many crops bypass the synonym/fuzzy force-fit because the VLM's confidence is
        # low: those route straight to the raw-label cluster pipeline.
        force_fit_bypass={'low_conf_skipped': 0, 'attempted': 0},
    )


def chunk_crops(docs: dict[str, Any], ctx: VlmContext) -> list[Any]:
    from src.services.curation.image_serving import (
        THUMBNAIL_CACHE,
        resolve_crop_root,
        resolve_safe_path,
    )
    from src.services.labeling.detector_hint import hinted_crop

    crops = []
    for crop_id, doc in docs.items():
        src = doc.get('_source') or {}
        image_path = src.get('image_path', '')
        bbox = src.get('bbox_norm')
        if not image_path or not bbox or len(bbox) != 4:
            continue
        try:
            safe = resolve_safe_path(image_path, resolve_crop_root(image_path))
            jpeg = THUMBNAIL_CACHE.get_or_compute(safe, tuple(bbox), size=224)
        except (RequestRejectedError, OSError, ValueError, Image.DecompressionBombError) as exc:
            logger.warning('pipeline_thumb_failed', crop_id=crop_id, error=str(exc))
            continue
        crops.append(hinted_crop(crop_id, jpeg, src, ctx.labeler._pack))
    return crops


def updates_from_predictions(
    preds: list[Any], ctx: VlmContext, *, set_cluster: bool = True
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """The item update per prediction, and the new-class proposals. ``set_cluster``
    moves a resolved item into its class cluster (the auto-label stage); an assist run
    leaves the item where it is so the scope it was asked about stays whole."""
    from src.services.curation.vlm_class_attempt import prediction_class_update
    from src.services.detection.cascade_detect.sanity import class_provenance

    updates_by_id: dict[str, dict[str, Any]] = {}
    proposals: list[dict[str, Any]] = []
    now = datetime.now(UTC).isoformat()
    provenance = class_provenance(
        detector=ctx.labeler.identity.model,
        detector_version='1',
        labeler=ctx.labeler.identity.model,
        labeled_at=now,
    )
    for p in preds:
        # Always write the VLM's region_visible when reported, regardless of which
        # class-resolution path fires.
        extras: dict[str, Any] = {}
        if p.region_visible is not None:
            extras[get_region_fields().visible] = p.region_visible
        update, proposal = prediction_class_update(
            p,
            name_to_id=ctx.name_to_id,
            resolve=ctx.resolve_class_name,
            now=now,
            provenance=provenance,
            extras=extras,
            set_cluster=set_cluster,
        )
        if update is None:
            continue
        update['vlm_prompt_pack'] = ctx.pack_stamp
        update.update(vlm_stamp(ctx.labeler))
        if proposal is not None:
            proposals.append(proposal)
        updates_by_id[p.img_id] = update
    return updates_by_id, proposals


def publish_classified(updates_by_id: dict[str, dict[str, Any]]) -> None:
    """Live UI updates per written crop."""
    for raw_id, doc in updates_by_id.items():
        if 'class_source' not in doc:
            continue  # empty answer: the class did not change
        try:
            publish_crop_classified(
                str(raw_id),
                class_id=doc.get('class_id'),
                class_name=doc.get('class_name'),
                class_source=doc.get('class_source', ''),
            )
        except (OSError, RuntimeError, ValueError) as exc:  # nosec B112 — advisory
            logger.debug('event_publish_failed', error=str(exc))


async def write_updates(
    opensearch: AsyncOpenSearch,
    ctx: VlmContext,
    updates_by_id: dict[str, dict[str, Any]],
    *,
    writer_id: str,
) -> None:
    """Worker-context bulk write via OCC: a concurrent human edit always wins, and
    ``class_id_history`` snapshots the prior assignment when a write changes the class."""
    from src.clients.occ import occ_skip_on_conflict_bulk
    from src.services.curation.vlm_class_attempt import with_class_snapshot

    def merge(doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        # Only onto the class state the sweep selected on: a human write (or validation)
        # since the scroll wins.
        if not ctx.guard.allows(doc_id, current):
            return {}
        return with_class_snapshot(dict(updates_by_id[doc_id]), current, writer=writer_id)

    try:
        await occ_skip_on_conflict_bulk(
            opensearch,
            doc_ids=list(updates_by_id.keys()),
            merger=merge,
            index=items_index(),
            refresh=False,
            writer_id=writer_id,
        )
    except OpenSearchException as exc:
        logger.warning('pipeline_bulk_failed', error=str(exc))
    else:
        publish_classified(updates_by_id)


__all__ = [
    'ChunkResult',
    'VlmContext',
    'chunk_crops',
    'class_context',
    'publish_classified',
    'updates_from_predictions',
    'write_updates',
]
