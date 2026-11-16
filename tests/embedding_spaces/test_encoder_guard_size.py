"""Two builds of one set of weights at two input sizes make vectors that cannot be compared, so
the encoder guard compares the model AND its input size: a project whose item and box spaces are
the 224 crop space refuses a deployment running the 336 build, and the reverse. The same one rule
serves every writer, the re-index encoder, search by picture and the project record."""

from __future__ import annotations

import dataclasses

import pytest
from _embedding_registry_fake import default_space

from src.clients import pe_encoder
from src.domain.embedding_spaces import EmbeddingSpaceError, Target
from src.services.embedding_spaces.encoder_guard import encoder_mismatch_of, require_encoder_for


def _crop_spaces_at(registry, size: int) -> None:
    """The project's item and box spaces are the crop spaces built at ``size``."""
    for target in (Target.ITEM, Target.BOX):
        registry.add(
            dataclasses.replace(
                default_space(target, f'pe_core_l14_crop{size}_v1'), input_size=size
            ),
            primary=True,
        )


def _deployment_runs(monkeypatch: pytest.MonkeyPatch, size: int) -> None:
    monkeypatch.setattr(pe_encoder, 'PE_CROP_SIZE', size)


@pytest.mark.asyncio
@pytest.mark.parametrize('target', [Target.ITEM, Target.BOX])
async def test_a_224_project_refuses_a_336_deployment(
    embedding_registry, monkeypatch: pytest.MonkeyPatch, target: Target
) -> None:
    _crop_spaces_at(embedding_registry, 224)
    _deployment_runs(monkeypatch, 336)
    with pytest.raises(EmbeddingSpaceError) as err:
        await require_encoder_for(target)
    assert err.value.code == 'embedding_model_unavailable'
    for part in ('224', '336', 'pe_crop_encoder', 'new project'):
        assert part in err.value.detail, part


@pytest.mark.asyncio
@pytest.mark.parametrize('target', [Target.ITEM, Target.BOX])
async def test_a_336_project_refuses_a_224_deployment(
    embedding_registry, monkeypatch: pytest.MonkeyPatch, target: Target
) -> None:
    _crop_spaces_at(embedding_registry, 336)
    _deployment_runs(monkeypatch, 224)
    with pytest.raises(EmbeddingSpaceError) as err:
        await require_encoder_for(target)
    assert '336' in err.value.detail
    assert '224' in err.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize('size', [224, 336])
async def test_a_project_of_the_deployments_size_passes(
    embedding_registry, monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    _crop_spaces_at(embedding_registry, size)
    _deployment_runs(monkeypatch, size)
    await require_encoder_for(Target.ITEM, Target.BOX)
    assert await encoder_mismatch_of('default') is None


@pytest.mark.asyncio
async def test_the_frame_space_does_not_depend_on_the_crop_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deployment_runs(monkeypatch, 336)
    await require_encoder_for(Target.FRAME)


@pytest.mark.asyncio
async def test_the_project_record_reports_the_size_mismatch_with_the_writers_text(
    embedding_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _crop_spaces_at(embedding_registry, 224)
    _deployment_runs(monkeypatch, 336)
    found = await encoder_mismatch_of('default')
    assert found is not None
    assert sorted(s.target for s in found.spaces) == ['box', 'item']
    for target in (Target.ITEM, Target.BOX):
        with pytest.raises(EmbeddingSpaceError) as err:
            await require_encoder_for(target)
        assert err.value.detail in found.message


@pytest.mark.parametrize(
    ('spec', 'size'),
    [
        ({'image': {'size': [224, 224]}}, 224),
        ({'image': {'size': [336, 336]}}, 336),
        ({'image': {'size': [224, 336]}}, None),
        ({'image': {'size': [{'min': 1, 'max': 9}, 224]}}, None),
        ({'text': {'context_length': 32}}, None),
    ],
)
def test_the_registered_size_is_a_fixed_square_or_unknown(spec: dict, size: int | None) -> None:
    from src.domain.embedding_spaces import image_input_size

    assert image_input_size(spec) == size


def test_a_space_of_unknown_size_is_refused() -> None:
    from src.domain.embedding_spaces import EncoderBuild, require_encoder

    with pytest.raises(EmbeddingSpaceError):
        require_encoder(
            project='p',
            space_key='k',
            target=Target.ITEM,
            space_build=EncoderBuild('pe_crop_encoder', None),
            served_build=EncoderBuild('pe_crop_encoder', 224),
        )
