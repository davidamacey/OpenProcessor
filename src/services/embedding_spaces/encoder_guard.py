"""The check every vector writer makes: the bound project's space is the encoder's space.

``item`` and ``box`` vectors are made by the crop encoder, ``frame`` vectors by the frame encoder
(:func:`~src.clients.pe_encoder.served_encoder_for`: the model and its input size). A writer calls
:func:`require_encoder_for` with the targets it writes before it encodes, so a project of another
encoder (one created before the crop encoder, or at the other crop input size) is refused rather
than given vectors of a space it does not have. The comparison is
:func:`~src.domain.embedding_spaces.encoder_mismatch_message`, the same one the re-index job and
search by picture make; :func:`encoder_mismatch_of` serves the same finding for a project record.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.clients.pe_encoder import served_encoder_for
from src.config.project_context import current_project
from src.domain.embedding_spaces import (
    EmbeddingSpaceError,
    Target,
    encoder_mismatch_message,
    require_encoder,
)
from src.services.embedding_spaces.resolve import embedding_space, primary_spaces


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
            space_build=space.encoder_build,
            served_build=served_encoder_for(target),
        )


@dataclass(frozen=True)
class SpaceMismatch:
    target: str
    space_key: str
    registered_model: str | None
    encoder_model: str
    message: str


@dataclass(frozen=True)
class EncoderMismatch:
    """What a project's record serves: its item or box spaces are of another model than the
    deployment's encoders, so its writers refuse (``message`` is theirs, joined)."""

    message: str
    spaces: tuple[SpaceMismatch, ...]


async def encoder_mismatch_of(slug: str) -> EncoderMismatch | None:
    """The item and box spaces of project ``slug`` that are not the encoders' space, or ``None``
    when they all are (nothing is stored: read from the registry at call time). A project with no
    registered spaces (being created, deleted) has nothing to compare."""
    try:
        spaces = await primary_spaces(slug)
    except EmbeddingSpaceError:
        return None
    found: list[SpaceMismatch] = []
    for space in spaces:
        if space.target not in (Target.ITEM, Target.BOX):
            continue
        served = served_encoder_for(space.target)
        message = encoder_mismatch_message(
            project=slug,
            space_key=space.space_key,
            target=space.target,
            space_build=space.encoder_build,
            served_build=served,
        )
        if message is not None:
            found.append(
                SpaceMismatch(
                    space.target.value,
                    space.space_key,
                    space.triton_image_model,
                    served.model,
                    message,
                )
            )
    if not found:
        return None
    return EncoderMismatch(' '.join(s.message for s in found), tuple(found))


__all__ = [
    'EncoderMismatch',
    'SpaceMismatch',
    'encoder_mismatch_of',
    'require_encoder_for',
]
