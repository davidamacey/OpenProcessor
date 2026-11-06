"""The near-duplicate profile refuses a body that would group silently wrong."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.curation.near_duplicates import NearDupProfileBody


def _body(**image: object) -> dict[str, object]:
    return {'version': 1, 'image': image}


def test_the_defaults_are_todays_cuts() -> None:
    body = NearDupProfileBody()
    assert (body.image.dedup_cosine, body.image.leak_cosine, body.image.k) == (0.98, 0.98, 64)
    assert (body.item.dedup_cosine, body.item.leak_cosine) == (0.98, 0.98)
    assert NearDupProfileBody.model_validate(body.model_dump(mode='json')) == body


@pytest.mark.parametrize(
    'image',
    [
        {'dedup_cosine': 0.9, 'leak_cosine': 0.95},  # leak stricter than dedup
        {'k': 0},
        {'k': 7},
        {'k': 257},
        {'dedup_cosine': 1.01},
        {'leak_cosine': 0.4, 'dedup_cosine': 0.9},
        {'max_group_size': 1},
        {'unknown_option': 1},
    ],
)
def test_a_bad_image_section_refuses(image: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        NearDupProfileBody.model_validate(_body(**image))


def test_leak_may_be_looser_than_dedup() -> None:
    body = NearDupProfileBody.model_validate(_body(dedup_cosine=0.99, leak_cosine=0.9))
    assert body.image.leak_cosine == 0.9


def test_an_unknown_version_refuses() -> None:
    with pytest.raises(ValidationError):
        NearDupProfileBody.model_validate({'version': 2})
