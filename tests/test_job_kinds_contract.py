"""``contracts/json/job_kinds.json`` carries what the Rust job runner reads from the Python tables.

The drift hook keeps the file equal to its generator; these tests pin the generator to the sources
the Python wrapper uses, so the runner cannot read a different kind table, state machine or lease
timing from the one the wrapper runs.
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from src.config.job_kinds import KINDS
from src.config.job_kinds.kind import JobKind
from src.domain import job_states, job_timing


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'codegen'))
import job_kinds_contract


COMMITTED = json.loads(
    (Path(__file__).resolve().parents[1] / 'contracts' / 'json' / 'job_kinds.json').read_text(
        encoding='utf-8'
    )
)


@pytest.fixture(scope='module')
def rendered() -> dict:
    return json.loads(job_kinds_contract.render_job_kinds_json())


def test_every_kind_is_listed_with_every_field_of_the_kind_record(rendered: dict) -> None:
    wanted = {f.name for f in fields(JobKind)} | {'runtime'}
    assert {row['name'] for row in rendered['kinds']} == set(KINDS)
    for row in rendered['kinds']:
        assert set(row) == wanted, row['name']
        kind = KINDS[row['name']]
        for field in fields(JobKind):
            assert row[field.name] == getattr(kind, field.name), (row['name'], field.name)
    assert rendered['kind_fields'] == [f.name for f in fields(JobKind)] + ['runtime']


def test_only_the_kinds_the_rust_runner_registers_say_rust(rendered: dict) -> None:
    rust = {row['name'] for row in rendered['kinds'] if row['runtime'] == 'rust'}
    assert rust == {'noop', 'sleep_noop'} == set(job_kinds_contract.RUST_RUNTIME_KINDS)


def test_a_rust_kind_that_does_not_exist_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job_kinds_contract, 'RUST_RUNTIME_KINDS', ('noop', 'no_such_kind'))
    with pytest.raises(ValueError, match='no_such_kind'):
        job_kinds_contract.render_job_kinds()


def test_the_state_machine_is_the_one_the_trigger_is_built_from(rendered: dict) -> None:
    states = rendered['job_states']
    assert states['statuses'] == list(job_states.STATUSES)
    assert set(states['terminal']) == job_states.TERMINAL
    assert set(states['active']) == job_states.ACTIVE
    assert {tuple(pair) for pair in states['transitions']} == job_states.TRANSITIONS


def test_the_lease_timing_is_the_wrappers(rendered: dict) -> None:
    assert rendered['timing'] == {
        'lease_ttl_s': job_timing.LEASE_TTL_S,
        'heartbeat_tick_s': job_timing.HEARTBEAT_TICK_S,
        'fence_margin_s': job_timing.FENCE_MARGIN_S,
        'progress_min_interval_s': job_timing.PROGRESS_MIN_INTERVAL_S,
    }
    assert (rendered['timing']['lease_ttl_s'], rendered['timing']['heartbeat_tick_s']) == (120, 15)


def test_the_committed_file_is_the_generators_output() -> None:
    # The sandboxed generator (scrubbed environment) is what the hook runs; an `OP_JOB_*` value in
    # this shell must not change what is committed, so compare only when none is set.
    import os

    if any(key.startswith('OP_JOB_') for key in os.environ):
        pytest.skip('an OP_JOB_* variable is set; the hook runs the generator with a clean env')
    assert json.loads(job_kinds_contract.render_job_kinds_json()) == COMMITTED
