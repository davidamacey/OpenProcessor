"""The VLM gate pairs a pack/profile with the project's classes. An unreadable
class registry must stop the gate; it must not pass as "no classes"."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.clients.curation_opensearch.registry import ClassRegistry
from src.services.curation.class_registry_names import registry_class_names


if TYPE_CHECKING:
    from pathlib import Path


def _registry_at(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(
        'src.clients.curation_opensearch.registry.get_class_registry',
        lambda: ClassRegistry(path=path),
    )


def test_a_corrupt_registry_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'class_registry.json'
    path.write_text('{not json')
    _registry_at(monkeypatch, path)

    with pytest.raises(ValueError, match='Expecting'):
        registry_class_names()


def test_a_missing_registry_is_no_classes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _registry_at(monkeypatch, tmp_path / 'absent.json')

    assert registry_class_names() == frozenset()
