"""The encoder side of the re-index job: one decode per image through the reprocess decode path.

``encode`` resolves each document to its image file, groups the documents by image, and calls the
reprocess embed path (``reprocess_embed._target_vectors``: servable-path check, GPU decode for
large photos, CPU decode otherwise, then the shared ``PEEncoder`` on Triton), so there is no
second Triton client and no second decode. Only the model the shared ``PEEncoder`` serves can be
built here; a space of any other model is refused (fail closed) until its encoder is registered.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from src.clients.pe_encoder import PE_IMAGE_MODEL
from src.config import get_curation_config
from src.db.repos import embedding_spaces as repo
from src.db.session import async_session_factory, project_tx
from src.domain.embedding_spaces import EmbeddingSpaceError, Target
from src.services.curation.reprocess_embed import _target_vectors, new_image_gate
from src.services.curation.reprocess_targets import existing_images
from src.services.embedding_spaces.reindex import DocRef, Encoder
from src.services.embedding_spaces.reindex_opensearch import split_doc_id
from src.utils.vectors import float_list


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.services.curation.reprocess_gpu_embed import GpuEmbedRoute


class ReindexEncoder(Encoder):
    """``opensearch`` resolves documents to images; ``pe`` is the shared ``PEEncoder``."""

    def __init__(
        self,
        opensearch: Any,
        pe: Any,
        gpu: GpuEmbedRoute | None = None,
        *,
        owned: Sequence[Any] = (),
    ) -> None:
        self._owned = list(owned)
        self._os = opensearch
        self._pe = pe
        self._gpu = gpu
        self._gate = new_image_gate()
        self._checked: set[Any] = set()

    async def aclose(self) -> None:
        """Close what the factory handed over (the job wrapper closes the client it built)."""
        for resource in self._owned:
            closer = getattr(resource, 'aclose', None) or resource.close
            await closer()

    async def _check_model(self, space: repo.SpaceRow) -> None:
        if space.model_version_id in self._checked:
            return
        async with async_session_factory()() as session, project_tx(session, space.project_id):
            version = await repo.get_model_version_by_id(session, space.model_version_id)
        if version is None or version.triton_image_model != PE_IMAGE_MODEL:
            name = version.triton_image_model if version else None
            raise EmbeddingSpaceError(
                'embedding_model_unavailable',
                f'{space.space_key} needs {name!r}; this worker encodes {PE_IMAGE_MODEL!r} only',
            )
        self._checked.add(space.model_version_id)

    async def _image_ids(self, space: repo.SpaceRow, docs: Sequence[DocRef]) -> dict[str, str]:
        """``{document id: image id}``; a frame document is its own image."""
        ids = list(dict.fromkeys(split_doc_id(d.doc_id)[0] for d in docs))
        if space.target is Target.FRAME:
            return {i: i for i in ids}
        resp = await self._os.mget(
            index=get_curation_config().items_index,
            body={'ids': ids},
            _source_includes=['image_id'],
        )
        return {
            d['_id']: d['_source']['image_id']
            for d in resp['docs']
            if d.get('found') and (d.get('_source') or {}).get('image_id')
        }

    async def encode(
        self, space: repo.SpaceRow, docs: Sequence[DocRef]
    ) -> dict[str, Sequence[float]]:
        await self._check_model(space)
        image_of = await self._image_ids(space, docs)
        images = await existing_images(
            self._os,
            list(dict.fromkeys(image_of.values())),
            index=get_curation_config().images_index,
        )
        by_image: dict[str, list[DocRef]] = defaultdict(list)
        for doc in docs:
            image_id = image_of.get(split_doc_id(doc.doc_id)[0])
            if image_id is not None and images.get(image_id, {}).get('image_path'):
                by_image[image_id].append(doc)
        found: dict[str, Sequence[float]] = {}

        async def one(image_id: str, group: list[DocRef]) -> None:
            boxes: list[Sequence[float]] = [d.bbox_norm for d in group if d.bbox_norm is not None]
            frame = space.target is Target.FRAME
            if not frame and len(boxes) != len(group):
                return
            async with self._gate:
                vectors = await _target_vectors(
                    self._pe,
                    self._gpu,
                    images[image_id]['image_path'],
                    boxes if space.target is Target.ITEM else [],
                    boxes if space.target is Target.BOX else [],
                    frame=frame,
                )
            if vectors is None:
                return
            if frame:
                if vectors.frame is not None:
                    found[group[0].doc_id] = float_list(vectors.frame)
                return
            rows = vectors.crops if space.target is Target.ITEM else vectors.regions
            for doc, row in zip(group, rows, strict=True):
                found[doc.doc_id] = float_list(row)

        async with asyncio.TaskGroup() as tg:
            for image_id, group in by_image.items():
                tg.create_task(one(image_id, group))
        return found


__all__ = ['ReindexEncoder']
