"""The images index sets ``index.knn``, a final setting: a place vector added later is then a
mapping addition and not a reindex (2F-7)."""

from __future__ import annotations

from src.clients.curation_opensearch.bodies_core import _images_body, _items_body


def test_the_images_body_enables_knn() -> None:
    assert _images_body()['settings']['index']['knn'] is True


def test_the_items_body_enables_knn() -> None:
    assert _items_body()['settings']['index']['knn'] is True
