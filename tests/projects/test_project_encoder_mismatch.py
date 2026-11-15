"""A project whose item or box space is of another model than the deployment's encoders says so on
its record (``encoder_mismatch``), derived at read time from the registered spaces and the configured
encoders; it is the same rule, with the same text, the writers refuse with."""

from __future__ import annotations

import pytest
from _embedding_registry_fake import new_space

from src.domain.embedding_spaces import EmbeddingSpaceError, Target
from src.routers.curation._project_views import summarize
from src.schemas.curation.projects import ProjectCounts
from src.services.embedding_spaces.encoder_guard import encoder_mismatch_of, require_encoder_for
from tests.projects.test_project_counts import _record


@pytest.mark.asyncio
async def test_a_project_on_the_deployments_encoders_has_no_mismatch() -> None:
    assert await encoder_mismatch_of('default') is None


@pytest.mark.asyncio
async def test_a_project_on_an_old_space_names_it_with_the_refusal_text(embedding_registry) -> None:
    old = embedding_registry.add(new_space(Target.ITEM, 'older_encoder'), primary=True)
    found = await encoder_mismatch_of('default')
    assert found is not None
    assert [s.target for s in found.spaces] == ['item']
    [space] = found.spaces
    assert space.space_key == old.space_key
    assert space.registered_model == 'older_encoder_image'
    assert space.encoder_model == 'pe_crop_encoder'
    for part in ('default', old.space_key, 'older_encoder_image', 'pe_crop_encoder', 'new project'):
        assert part in found.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'old_targets', [(), (Target.ITEM,), (Target.BOX,), (Target.ITEM, Target.BOX)]
)
async def test_the_record_and_the_writers_apply_one_rule(
    embedding_registry, old_targets: tuple[Target, ...]
) -> None:
    for target in old_targets:
        embedding_registry.add(new_space(target, 'older_encoder'), primary=True)
    found = await encoder_mismatch_of('default')
    for target in (Target.ITEM, Target.BOX):
        try:
            await require_encoder_for(target)
            refused = None
        except EmbeddingSpaceError as exc:
            refused = exc.detail
        listed = [s for s in (found.spaces if found else ()) if s.target == target.value]
        assert (refused is None) == (not listed), target
        if refused is not None:
            assert listed[0].message == refused


def test_the_project_record_serves_the_mismatch_and_nothing_for_a_match() -> None:
    from src.services.embedding_spaces.encoder_guard import EncoderMismatch, SpaceMismatch

    mismatch = EncoderMismatch(
        message='m',
        spaces=(SpaceMismatch('box', 'k', 'older_image', 'pe_crop_encoder', 'm'),),
    )
    old = summarize(_record('alpha'), ProjectCounts(), mismatch).model_dump()
    assert old['encoder_mismatch'] == {
        'message': 'm',
        'spaces': [
            {
                'target': 'box',
                'space_key': 'k',
                'registered_model': 'older_image',
                'encoder_model': 'pe_crop_encoder',
                'message': 'm',
            }
        ],
    }
    assert (
        summarize(_record('alpha'), ProjectCounts(), None).model_dump()['encoder_mismatch'] is None
    )
