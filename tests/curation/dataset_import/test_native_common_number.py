"""finite_number: the one numeric parse both JSON box readers (Label Studio, CreateML) use."""

from __future__ import annotations

import pytest

from src.services.curation.dataset_import.native_common import finite_number


def test_parses_numbers_and_numeric_strings() -> None:
    assert finite_number(3) == 3.0
    assert finite_number('0.25') == 0.25


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), '-inf', 'nan'])
def test_rejects_non_finite(bad: object) -> None:
    with pytest.raises(ValueError, match=r'nan|inf'):
        finite_number(bad)


def test_rejects_non_numeric() -> None:
    with pytest.raises((ValueError, TypeError)):
        finite_number(None)
