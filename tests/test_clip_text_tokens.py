"""Found live: the MobileCLIP text tower was fed HuggingFace-padded tokens
(pad id 49407), giving embeddings unrelated to the image tower's -- every
image/text cosine was ~0.06, so text search could not rank anything."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.utils import cache


if TYPE_CHECKING:
    import pytest


SOT, EOT = 49406, 49407


class _HfStyleTokenizer:
    """Returns ids the way ``CLIPTokenizer(text, truncation=True,
    max_length=n)`` does without ``padding``: SOT .. EOT, cut to ``n`` with
    the EOT kept last."""

    def __init__(self, word_ids: list[int]) -> None:
        self._word_ids = word_ids

    def __call__(self, _text: str, *, truncation: bool, max_length: int) -> dict[str, Any]:
        assert truncation
        ids = [SOT, *self._word_ids, EOT]
        if len(ids) > max_length:
            ids = [*ids[: max_length - 1], EOT]
        return {'input_ids': ids}


def test_padding_after_the_end_token_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache, 'get_clip_tokenizer', lambda: _HfStyleTokenizer([320, 1125]))

    tokens = cache.clip_text_tokens('a photo')

    assert tokens.shape == (1, 77)
    assert tokens.dtype == np.int64
    assert tokens[0, :4].tolist() == [SOT, 320, 1125, EOT]
    assert not tokens[0, 4:].any()


def test_long_text_is_truncated_with_the_end_token_last(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache, 'get_clip_tokenizer', lambda: _HfStyleTokenizer([320] * 200))

    tokens = cache.clip_text_tokens('long')

    assert tokens.shape == (1, 77)
    assert tokens[0, 0] == SOT
    assert tokens[0, -1] == EOT
