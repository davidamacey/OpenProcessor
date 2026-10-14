"""Clustering steps whose OpenSearch call failed report the failure; they no longer
answer ``0`` (nothing parked, no false positives), which the run treats as a result."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.curation.clustering.region_box_clustering import count_false_positive_boxes
from src.services.curation.clustering.residual_gate import _park_gated_residuals
from src.services.curation.probe_predictions import _registry_class_ids


class _Down:
    async def search(self, **_: Any) -> dict[str, Any]:
        raise RuntimeError('opensearch is down')

    async def update_by_query(self, **_: Any) -> dict[str, Any]:
        raise RuntimeError('opensearch is down')


@pytest.mark.asyncio
async def test_a_failed_park_is_an_error_not_zero_parked() -> None:
    with pytest.raises(RuntimeError, match='down'):
        await _park_gated_residuals(_Down(), max_rank=3, min_blur_ratio=0.5)


@pytest.mark.asyncio
async def test_a_failed_false_positive_count_is_an_error_not_zero() -> None:
    with pytest.raises(RuntimeError, match='down'):
        await count_false_positive_boxes(_Down())


def test_a_corrupt_class_registry_stops_the_probe(tmp_path) -> None:
    from src.config.curation import CurationConfig

    path = tmp_path / 'class_registry.json'
    path.write_text('{not json')
    cfg = CurationConfig(class_registry_path=path)

    with pytest.raises(ValueError, match='Expecting'):
        _registry_class_ids(cfg)


def test_an_absent_class_registry_is_no_classes(tmp_path) -> None:
    from src.config.curation import CurationConfig

    cfg = CurationConfig(class_registry_path=tmp_path / 'absent.json')

    assert _registry_class_ids(cfg) == {}
