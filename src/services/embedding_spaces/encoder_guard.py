"""The check every vector writer makes: the bound project's space is the encoder's space.

``item`` and ``box`` vectors are made by the crop encoder, ``frame`` vectors by the frame encoder
(:func:`~src.clients.pe_encoder.image_model_for`). A writer calls :func:`require_encoder_for` with
the targets it writes before it encodes, so a project of another encoder (one created before the
crop encoder) is refused rather than given vectors of a space it does not have. The comparison is
:func:`~src.domain.embedding_spaces.require_encoder`, the same one the re-index job and search by
picture make.
"""

from __future__ import annotations

from src.clients.pe_encoder import image_model_for
from src.config.project_context import current_project
from src.domain.embedding_spaces import Target, require_encoder
from src.services.embedding_spaces.resolve import embedding_space


async def require_encoder_for(*targets: Target) -> None:
    """Raise ``EmbeddingSpaceError`` unless each target's primary space of the bound project is
    registered for the model that encodes it."""
    project = current_project().record.slug
    for target in targets:
        space = await embedding_space(target)
        require_encoder(
            project=project,
            space_key=space.space_key,
            target=target,
            space_model=space.triton_image_model,
            served_model=image_model_for(target),
        )


__all__ = ['require_encoder_for']
