"""Which detections the embedding policy embeds, and the state the rest get."""

from __future__ import annotations

from src.services.curation.ingest_policy import Candidate, EmbeddingPolicy, embedding_states


def _c(
    score: float = 0.9,
    area: float = 0.1,
    *,
    proposal: str | None = 'car',
    name: str | None = None,
    labeled: bool = False,
) -> Candidate:
    return Candidate(score, area, name, proposal, labeled)


def test_all_embeds_everything() -> None:
    assert embedding_states([_c(), _c()], EmbeddingPolicy()) == ['embedded', 'embedded']


def test_lazy_defers_everything_but_labeled_items() -> None:
    states = embedding_states([_c(), _c(labeled=True)], EmbeddingPolicy(mode='lazy'))
    assert states == ['deferred', 'embedded']


def test_selected_matches_class_by_normalized_name_on_proposal_or_class_name() -> None:
    policy = EmbeddingPolicy(mode='selected', classes=['Traffic-Light', 'widget'])
    cands = [_c(proposal='traffic light'), _c(proposal=None, name='widget'), _c(proposal='dog')]
    assert embedding_states(cands, policy) == ['embedded', 'embedded', 'not_selected']


def test_selected_criteria_combine_with_and() -> None:
    policy = EmbeddingPolicy(
        mode='selected', classes=['car'], min_confidence=0.5, min_box_area_frac=0.05
    )
    cands = [
        _c(0.9, 0.1),  # all pass
        _c(0.4, 0.1),  # low confidence
        _c(0.9, 0.01),  # too small
        _c(0.9, 0.1, proposal='dog'),  # wrong class
    ]
    assert embedding_states(cands, policy) == [
        'embedded',
        'not_selected',
        'not_selected',
        'not_selected',
    ]


def test_max_per_image_keeps_best_score_then_larger_box() -> None:
    policy = EmbeddingPolicy(mode='selected', max_per_image=2)
    cands = [_c(0.5, 0.5), _c(0.9, 0.1), _c(0.9, 0.3), _c(0.7, 0.9)]
    assert embedding_states(cands, policy) == [
        'not_selected',
        'embedded',
        'embedded',
        'not_selected',
    ]


def test_cap_counts_only_items_that_pass_the_other_criteria() -> None:
    policy = EmbeddingPolicy(mode='selected', classes=['car'], max_per_image=1)
    cands = [_c(0.99, proposal='dog'), _c(0.6), _c(0.8)]
    assert embedding_states(cands, policy) == ['not_selected', 'not_selected', 'embedded']


def test_a_labeled_item_always_embeds_even_when_nothing_matches() -> None:
    policy = EmbeddingPolicy(mode='selected', classes=['person'])
    assert embedding_states([_c(labeled=True)], policy) == ['embedded']
