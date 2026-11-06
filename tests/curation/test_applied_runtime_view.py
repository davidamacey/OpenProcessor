"""A worker's report is the epoch it applied; ``applied[]`` shows the refs that epoch holds.

Found live: every ``GET .../{prompt_packs,region_profiles}/active`` answered 500 once a detection
worker had written its report because the view read the worker's refs as a mapping.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import replace

import pytest

from src.db.repos.config_base import RuntimeReport
from src.services.config_store.activation_view import applied_runtime
from src.services.config_store.snapshot import _EMPTY_SNAPSHOT


def _report(epoch: int) -> RuntimeReport:
    return RuntimeReport(uuid.uuid4(), 'detection_worker', 'host-1', epoch, dt.datetime.now(dt.UTC))


def _snapshot(epoch: int):
    return replace(
        _EMPTY_SNAPSHOT,
        config_revision=epoch,
        active_profile=('wheel_profile', 2),
        active_pack=('generic_item_v1', 1),
        active_vlm=('env', None),
    )


def test_a_current_worker_runs_the_current_activations() -> None:
    runtime = applied_runtime(_report(5), _snapshot(5))

    assert runtime.host == 'host-1'
    assert runtime.applied_at
    assert (runtime.profile.name, runtime.profile.revision) == ('wheel_profile', 2)
    assert (runtime.pack.name, runtime.pack.revision) == ('generic_item_v1', 1)
    assert runtime.vlm is not None
    assert (runtime.vlm.name, runtime.vlm.revision) == ('env', None)
    assert runtime.lagging is False


def test_a_worker_behind_the_epoch_is_lagging_and_its_refs_are_unknown() -> None:
    runtime = applied_runtime(_report(3), _snapshot(5))

    assert runtime.lagging is True
    assert runtime.applied_config_revision == 3
    assert (runtime.profile.name, runtime.pack.name) == (None, None)


@pytest.mark.parametrize('axis_ref', ['off', None])
def test_an_axis_that_is_off_or_never_activated_serves_an_empty_ref_for_vlm(axis_ref) -> None:
    runtime = applied_runtime(_report(5), replace(_snapshot(5), active_vlm=axis_ref))

    assert runtime.vlm is not None
    assert runtime.vlm.name is None
