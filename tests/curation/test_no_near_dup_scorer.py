"""The bucketed near-duplicate scorer and its threshold are gone (4A-6); the other scorers stay."""

from __future__ import annotations

import importlib

import pytest

from src.config.retired_env import RETIRED_ENV
from src.services.curation.item_scores import SCORER_METADATA, available_scorers, get_scorer


def test_near_dup_is_not_a_scorer() -> None:
    assert 'near_dup' not in available_scorers()
    assert 'near_dup' not in SCORER_METADATA
    with pytest.raises(ValueError, match='unknown crop scorer'):
        get_scorer('near_dup')
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('src.services.curation.item_scores.near_dup')


def test_the_other_scorers_still_resolve() -> None:
    assert available_scorers() == ['mistakenness', 'uniqueness']
    for name in available_scorers():
        assert get_scorer(name).name == name


def test_the_threshold_is_retired_naming_the_replacement() -> None:
    assert RETIRED_ENV['OP_CROP_DUP_THRESHOLD'].startswith('removed:')
    assert 'near_dup_profile' in RETIRED_ENV['OP_CROP_DUP_THRESHOLD']
