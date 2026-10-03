"""What the ``embed`` scope embeds: which items, which parts, only the missing.

Resolved once at plan time and persisted with a job, so a dry run, an
in-request run and a background job act on the same ids. ``crop_ids`` is
``None`` when the target was whole images (every item on them); a crop-id or
filter target embeds exactly the items it named.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from src.config import get_curation_config
from src.config.region_fields import get_region_fields
from src.services.curation.embedding_state import not_embedded_clause
from src.services.curation.region_box_embeddings import embeddable, missing_boxes
from src.services.curation.region_boxes import read_boxes
from src.services.curation.reprocess_embed import ALL_PARTS, EmbedTarget, target_has_work
from src.services.curation.reprocess_targets import items_by_terms, scan_items


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.reprocess_models import EmbedOptions

# Frame vectors are written by ingest and never go missing for an item, so an
# only-missing run leaves them alone unless asked.
_MISSING_PARTS = frozenset({'crop', 'region'})
_ID_CHUNK = 1000


class EmbedSpec(BaseModel):
    crop_ids: list[str] | None = None
    only_missing: bool = False
    parts: list[str]

    @classmethod
    def build(cls, options: EmbedOptions, crop_ids: list[str] | None) -> EmbedSpec:
        default = _MISSING_PARTS if options.only_missing else ALL_PARTS
        return cls(
            crop_ids=crop_ids,
            only_missing=options.only_missing,
            parts=sorted(set(options.parts) if options.parts is not None else default),
        )

    @property
    def part_set(self) -> frozenset[str]:
        return frozenset(self.parts)


async def ids_without_vector(opensearch: AsyncOpenSearch, crop_ids: list[str]) -> set[str]:
    """The subset of ``crop_ids`` whose item has no crop vector (the
    authoritative ``exists`` test, not the recorded state)."""
    missing: set[str] = set()
    index = get_curation_config().items_index
    for start in range(0, len(crop_ids), _ID_CHUNK):
        query = {
            'bool': {
                'filter': [
                    {'terms': {'crop_id': crop_ids[start : start + _ID_CHUNK]}},
                    not_embedded_clause(),
                ]
            }
        }
        missing.update(
            cid for cid, _ in await scan_items(opensearch, query, index=index, includes=['crop_id'])
        )
    return missing


async def embed_scope_items(
    opensearch: AsyncOpenSearch, image_ids: list[str], spec: EmbedSpec, includes: list[str]
) -> list[tuple[str, dict[str, Any]]]:
    """Items on ``image_ids`` the spec covers (all of them, or the named ones)."""
    items = await items_by_terms(
        opensearch,
        'image_id',
        image_ids,
        index=get_curation_config().items_index,
        includes=includes,
    )
    if spec.crop_ids is None:
        return items
    wanted = set(spec.crop_ids)
    return [(cid, src) for cid, src in items if cid in wanted]


async def embed_targets(
    opensearch: AsyncOpenSearch, docs: dict[str, dict[str, Any]], spec: EmbedSpec
) -> list[EmbedTarget]:
    """The targets the ``embed`` scope runs over (``docs`` maps image id to its
    images doc): the one selection behind the applied run and the dry run."""
    F = get_region_fields()
    items = await embed_scope_items(
        opensearch,
        list(docs),
        spec,
        [
            'image_id',
            'bbox_norm',
            F.boxes,
            F.box_embeddings,
            'class_id',
            'cluster_id',
            'crop_rank_in_image',
            'blur_lap_ratio',
        ],
    )
    skip: frozenset[str] = frozenset()
    if spec.only_missing and items:
        ids = [cid for cid, _ in items]
        skip = frozenset(ids) - await ids_without_vector(opensearch, ids)
    by_image: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for crop_id, src in items:
        by_image.setdefault(src.get('image_id') or '', []).append((crop_id, src))
    return [
        EmbedTarget(
            image_id=image_id,
            image_path=doc.get('image_path') or '',
            items=by_image.get(image_id, []),
            skip_crop=skip,
        )
        for image_id, doc in docs.items()
    ]


async def plan_embed_counts(
    opensearch: AsyncOpenSearch, image_ids: list[str], spec: EmbedSpec
) -> dict[str, int]:
    """Dry-run counters from the same targets the applied run uses:
    ``to_embed`` equals ``crop_written`` and ``images_to_embed`` equals
    ``queued`` when every image is readable."""
    targets = await embed_targets(opensearch, {i: {} for i in image_ids}, spec)
    items = [pair for t in targets for pair in t.items]
    skipped = sum(len([1 for cid, _ in t.items if cid in t.skip_crop]) for t in targets)
    if spec.only_missing:
        without = len(items) - skipped
    else:
        without = len(await ids_without_vector(opensearch, [cid for cid, _ in items]))
    crops = sum(
        1 for t in targets for cid, s in t.items if s.get('bbox_norm') and cid not in t.skip_crop
    )
    to_embed = crops if 'crop' in spec.part_set else 0
    F = get_region_fields()
    cfg = get_curation_config()
    return {
        'items': len(items),
        'without_vector': without,
        'to_embed': to_embed,
        'estimated_vector_kb': to_embed * cfg.encoder_embedding_dim * 4 // 1000,
        'images_to_embed': sum(
            target_has_work(t, spec.part_set, only_missing=spec.only_missing) for t in targets
        ),
        'region_boxes_to_embed': sum(
            len(embeddable(read_boxes(src, F)) if not spec.only_missing else missing_boxes(src, F))
            for _, src in items
        )
        if 'region' in spec.part_set
        else 0,
    }


__all__ = [
    'EmbedSpec',
    'embed_scope_items',
    'embed_targets',
    'ids_without_vector',
    'plan_embed_counts',
]
