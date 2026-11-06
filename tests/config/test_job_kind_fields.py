"""The research fields of ``JobKind`` (pausable, slice_s, gpu_affinity) and their rules."""

from __future__ import annotations

import pytest

from src.config.job_kinds.kind import VLM_REQUEST_TIMEOUT_S, JobKind
from src.config.queue import QUEUE_PROFILE, TaskQueues


def _kind(**overrides: object) -> JobKind:
    fields: dict[str, object] = {
        'name': 'probe',
        'task_name': 'op.probe.kind',
        'queue': TaskQueues.JOBS,
        'project_scope': 'project',
        'soft_limit_s': 1800,
        'hard_limit_s': 2100,
        'max_queue_wait_s': 86400,
    }
    fields.update(overrides)
    return JobKind(**fields)  # type: ignore[arg-type]


def test_the_defaults_are_safe() -> None:
    kind = _kind()
    assert kind.pausable is False
    assert kind.slice_s is None
    assert kind.gpu_affinity == 'none'


def test_a_sliced_kind_needs_a_soft_limit_above_the_slice_plus_one_vlm_timeout() -> None:
    ok = _kind(pausable=True, slice_s=600, soft_limit_s=600 + VLM_REQUEST_TIMEOUT_S + 1)
    assert ok.slice_s == 600
    with pytest.raises(ValueError, match='slice_s'):
        _kind(pausable=True, slice_s=600, soft_limit_s=600 + VLM_REQUEST_TIMEOUT_S)


def test_a_slice_without_pausable_is_refused() -> None:
    with pytest.raises(ValueError, match='pausable'):
        _kind(slice_s=600, soft_limit_s=3600, hard_limit_s=4000)


@pytest.mark.parametrize('affinity', ['any_card', 'none'])
def test_known_gpu_affinities_are_accepted(affinity: str) -> None:
    assert _kind(gpu_affinity=affinity).gpu_affinity == affinity


def test_an_unknown_gpu_affinity_is_refused() -> None:
    with pytest.raises(ValueError, match='gpu_affinity'):
        _kind(gpu_affinity='one_card')


def test_the_cluster_queue_is_declared_for_the_curation_profile() -> None:
    assert TaskQueues.CLUSTER == 'cluster'
    assert TaskQueues.CLUSTER in TaskQueues.ALL
    assert QUEUE_PROFILE[TaskQueues.CLUSTER] == 'curation'
