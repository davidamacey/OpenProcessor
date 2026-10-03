"""Detector label vocabulary and the name-based registry seed plan."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config import DetectionProfile
from src.routers.curation._class_models import CLASS_NAME_PATTERN
from src.services.curation.detector_vocabulary import (
    DetectorLabel,
    SeedPlan,
    detector_labels,
    plan_seed,
)
from src.utils.class_names import normalize_class_name


REPO_ROOT = Path(__file__).resolve().parents[2]
COCO_LABELS = REPO_ROOT / 'models' / 'yolov11_small_trt_end2end' / 'labels.txt'


def _labels(*names: str) -> list[DetectorLabel]:
    return [DetectorLabel(i, n, normalize_class_name(n)) for i, n in enumerate(names)]


@pytest.mark.parametrize(
    ('raw', 'slug'),
    [
        ('traffic light', 'traffic_light'),
        ('  Hot  Dog ', 'hot_dog'),
        ('t-shirt', 't_shirt'),
        ('car', 'car'),
        ('__x__', 'x'),
        ('', ''),
    ],
)
def test_normalize_class_name(raw: str, slug: str) -> None:
    assert normalize_class_name(raw) == slug


def test_coco_vocabulary_slugifies_all_fifteen_spaced_labels() -> None:
    profile = DetectionProfile(name='item', detector_model='m', labels_path=str(COCO_LABELS))
    labels = detector_labels(profile)
    assert len(labels) == 80
    assert sum(' ' in label.name for label in labels) == 15
    slugs = [label.slug for label in labels]
    assert len(set(slugs)) == 80
    assert all(re.fullmatch(CLASS_NAME_PATTERN, s) for s in slugs)
    by_name = {label.name: label.slug for label in labels}
    assert by_name['traffic light'] == 'traffic_light'
    assert by_name['dining table'] == 'dining_table'


def test_labels_fall_back_to_the_models_own_labels_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'src.services.curation.detector_vocabulary.get_class_names',
        lambda model: {0: 'a b', 2: 'c'} if model == 'm' else {},
    )
    got = detector_labels(DetectionProfile(name='item', detector_model='m'))
    assert [(g.class_id, g.slug) for g in got] == [(0, 'a_b'), (1, ''), (2, 'c')]


def test_no_labels_is_empty() -> None:
    assert detector_labels(DetectionProfile(name='item', detector_model='nope')) == []


def test_plan_creates_everything_new_and_skips_existing_by_name() -> None:
    plan = plan_seed(_labels('person', 'traffic light', 'car'), {'car': False}, None)
    assert [e.slug for e in plan.create] == ['person', 'traffic_light']
    assert [(e.label.slug, e.reason) for e in plan.skipped] == [('car', 'exists')]
    assert plan.conflicts == []
    assert plan.unknown == []


def test_plan_does_not_resurrect_a_deprecated_class() -> None:
    plan = plan_seed(_labels('car'), {'car': True}, None)
    assert plan.create == []
    assert [e.reason for e in plan.skipped] == ['deprecated']


def test_plan_reports_slug_collisions_and_blank_labels() -> None:
    plan = plan_seed(_labels('hot dog', 'hot-dog', ''), {}, None)
    assert [e.slug for e in plan.create] == ['hot_dog']
    assert [(c.label.name, c.reason) for c in plan.conflicts] == [
        ('hot-dog', 'duplicate_slug'),
        ('', 'unnamed_label'),
    ]


def test_plan_names_subset_matches_by_normalized_name() -> None:
    plan = plan_seed(_labels('person', 'traffic light', 'car'), {}, ['Traffic Light', 'nope'])
    assert [e.slug for e in plan.create] == ['traffic_light']
    assert plan.unknown == ['nope']


def test_plan_is_a_plain_dataclass_of_lists() -> None:
    assert SeedPlan([], [], [], []).create == []
