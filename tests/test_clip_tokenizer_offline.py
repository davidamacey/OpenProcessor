"""The CLIP tokenizer loads from local files only: /embed/text must work with no network."""

from __future__ import annotations

import json
import socket
from typing import TYPE_CHECKING

import pytest

from src.utils import cache


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*_a: object, **_kw: object) -> None:
        raise OSError('network disabled for this test')

    monkeypatch.setattr(socket, 'getaddrinfo', _refuse)
    monkeypatch.setattr(socket.socket, 'connect', _refuse)
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    monkeypatch.setattr(cache, '_clip_tokenizer', None)


def _write_tiny_tokenizer(path: Path) -> None:
    path.mkdir()
    vocab = {'<|startoftext|>': 0, '<|endoftext|>': 1, 'a</w>': 2, 'b</w>': 3}
    (path / 'vocab.json').write_text(json.dumps(vocab))
    (path / 'merges.txt').write_text('#version: 0.2\n')


def test_the_tokenizer_loads_from_the_local_directory_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_network: None
) -> None:
    tok_dir = tmp_path / 'clip-tokenizer'
    _write_tiny_tokenizer(tok_dir)
    monkeypatch.setattr(cache, 'CLIP_TOKENIZER_DIR', tok_dir)
    tokenizer = cache.get_clip_tokenizer()
    assert tokenizer('a b')['input_ids'] == [0, 2, 3, 1]


def test_a_missing_tokenizer_directory_fails_closed_with_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_network: None
) -> None:
    monkeypatch.setattr(cache, 'CLIP_TOKENIZER_DIR', tmp_path / 'absent')
    with pytest.raises(RuntimeError, match='CLIP tokenizer files'):
        cache.get_clip_tokenizer()
