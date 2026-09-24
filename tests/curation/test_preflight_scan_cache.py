"""The label-scan cache must key on the class filter, not just the export.

A preflight for one ``include_classes`` subset used to be cached per
(export_dir, manifest) and then returned for every later subset on the same
export -- so a UI that preflighted while a user was still picking classes
reported most images as empty for the final selection, and different API
workers disagreed depending on which subset each had cached first.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.services.training import preflight_scan
from src.services.training.preflight_scan import scan_export_labels


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    # Registry ids 10/20/30 map to dense export ids 0/1/2; one image per class.
    (tmp_path / 'class_registry.json').write_text(
        json.dumps({'export_id_map': {'10': 0, '20': 1, '30': 2}})
    )
    (tmp_path / 'manifest.json').write_text(json.dumps({'dataset_sha': 'x'}))
    labels = tmp_path / 'labels' / 'train'
    labels.mkdir(parents=True)
    for dense in (0, 1, 2):
        (labels / f'img{dense}.txt').write_text(f'{dense} 0.5 0.5 0.2 0.2\n')
    preflight_scan._scan_cache.clear()
    return tmp_path


def test_a_different_class_filter_is_not_served_from_the_cache(export_dir: Path) -> None:
    one = scan_export_labels(export_dir, include_classes=[10])
    assert one.empty_label_images == 2

    two = scan_export_labels(export_dir, include_classes=[10, 20])
    assert two.empty_label_images == 1

    everything = scan_export_labels(export_dir)
    assert everything.empty_label_images == 0


def test_the_same_filter_in_any_order_reuses_the_cached_scan(export_dir: Path) -> None:
    first = scan_export_labels(export_dir, include_classes=[20, 10])
    assert scan_export_labels(export_dir, include_classes=[10, 20]) is first
