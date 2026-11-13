"""The promote form starts from a readable name (project and run time), never the run's UUID."""

from __future__ import annotations

import datetime as dt
import re

from src.services.training.suggested_name import suggested_triton_name


WHEN = dt.datetime(2026, 11, 13, 9, 5, 59, tzinfo=dt.UTC)


def test_name_is_project_and_run_time() -> None:
    assert suggested_triton_name('coco-e2e', WHEN) == 'coco-e2e_20261113_0905'


def test_name_is_always_a_valid_promote_name() -> None:
    name = suggested_triton_name('Weird Project!! ' + 'x' * 100, WHEN)
    assert re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name)
    assert name.endswith('_20261113_0905')


def test_name_without_a_usable_slug_still_names_the_run() -> None:
    assert suggested_triton_name('!!!', WHEN) == 'model_20261113_0905'
