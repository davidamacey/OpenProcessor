"""Validation of the ingest policy shapes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.services.curation.ingest_policy import (
    DetectFilter,
    EmbeddingPolicy,
    IngestPolicy,
    IngestPolicyBody,
    unknown_names,
)


def test_defaults_store_and_embed_everything() -> None:
    policy = IngestPolicy()
    assert policy.revision == 0
    assert policy.embedding.mode == 'all'
    assert policy.detect == DetectFilter()
    assert policy.detect.classes is None


def test_selected_without_a_criterion_is_refused() -> None:
    with pytest.raises(ValidationError, match='at least one criterion'):
        EmbeddingPolicy(mode='selected')


@pytest.mark.parametrize(
    'criterion',
    [
        {'classes': ['car']},
        {'min_confidence': 0.5},
        {'min_box_area_frac': 0.01},
        {'max_per_image': 3},
    ],
)
def test_selected_accepts_any_single_criterion(criterion: dict) -> None:
    assert EmbeddingPolicy(mode='selected', **criterion).mode == 'selected'


@pytest.mark.parametrize(
    'bad',
    [{'min_confidence': 1.5}, {'min_box_area_frac': -0.1}, {'max_per_image': 0}, {'bogus': 1}],
)
def test_ranges_and_unknown_keys_are_refused(bad: dict) -> None:
    with pytest.raises(ValidationError):
        DetectFilter(**bad)


def test_unknown_names_are_reported_not_refused() -> None:
    body = IngestPolicyBody(
        detect=DetectFilter(classes=['Traffic Light', 'unicorn'], exclude_classes=['dragon']),
        embedding=EmbeddingPolicy(mode='selected', classes=['person']),
    )
    assert unknown_names(body, {'traffic_light', 'person'}) == ['dragon', 'unicorn']
