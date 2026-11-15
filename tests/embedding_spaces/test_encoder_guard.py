"""One check: the encoder that makes a vector must be the model its space is registered for.

A project created before the crop encoder has item and box spaces of the old encoder
(`pe_core_l14_336`); vectors of `pe_crop_encoder` written into them would sit beside vectors they
cannot be compared with. Every writer asks :func:`require_encoder_for` before it writes.
"""

from __future__ import annotations

import pytest
from _embedding_registry_fake import new_space

from src.clients.pe_encoder import PE_CROP_MODEL, PE_IMAGE_MODEL
from src.domain.embedding_spaces import EmbeddingSpaceError, Target, require_encoder
from src.services.embedding_spaces.encoder_guard import require_encoder_for


def test_a_space_of_the_serving_encoder_passes() -> None:
    require_encoder(
        project='p',
        space_key='k',
        target=Target.ITEM,
        space_model=PE_CROP_MODEL,
        served_model=PE_CROP_MODEL,
    )


def test_the_refusal_names_project_space_and_both_models_and_the_remedy() -> None:
    with pytest.raises(EmbeddingSpaceError) as err:
        require_encoder(
            project='old-cars',
            space_key='pe_core_l14_336_v1',
            target=Target.BOX,
            space_model=PE_IMAGE_MODEL,
            served_model=PE_CROP_MODEL,
        )
    assert err.value.code == 'embedding_model_unavailable'
    text = err.value.detail
    for part in ('old-cars', 'pe_core_l14_336_v1', PE_IMAGE_MODEL, PE_CROP_MODEL, 're-ingest'):
        assert part in text, part
    assert 'new project' in text


def test_a_space_with_no_registered_model_is_refused() -> None:
    with pytest.raises(EmbeddingSpaceError):
        require_encoder(
            project='p',
            space_key='k',
            target=Target.ITEM,
            space_model=None,
            served_model=PE_CROP_MODEL,
        )


@pytest.mark.asyncio
async def test_the_default_spaces_of_a_new_project_pass_for_every_target() -> None:
    await require_encoder_for(Target.ITEM, Target.BOX, Target.FRAME)


@pytest.mark.asyncio
@pytest.mark.parametrize('target', [Target.ITEM, Target.BOX])
async def test_a_primary_space_of_another_model_is_refused(
    embedding_registry, target: Target
) -> None:
    embedding_registry.add(new_space(target, 'older_encoder'), primary=True)
    with pytest.raises(EmbeddingSpaceError) as err:
        await require_encoder_for(target)
    assert 'older_encoder_image' in err.value.detail
    assert 'default' in err.value.detail  # the bound project's name


@pytest.mark.asyncio
async def test_only_the_asked_targets_are_checked(embedding_registry) -> None:
    embedding_registry.add(new_space(Target.BOX, 'older_encoder'), primary=True)
    await require_encoder_for(Target.ITEM)
    with pytest.raises(EmbeddingSpaceError):
        await require_encoder_for(Target.ITEM, Target.BOX)
