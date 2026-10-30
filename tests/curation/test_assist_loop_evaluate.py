"""The node gate: the exact bound decides, dissent is limited and can be topped up once, only
a registry class passes, a new-class proposal is dissent, and validated members can veto.
Also the per-crop answers the assist chunk now reports.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from curation.assist_loop_fakes import CREATED_AT, FakeVlm, World, make_world
from src.domain.loop_stats import homogeneity_lower_bound, required_sample
from src.services.curation.assist.chunk import process_chunk
from src.services.curation.assist.proposal import ClassAnswer
from src.services.curation.assist.slot_write import RunMeta
from src.services.curation.assist_loop.evaluate import GateParams, judge
from src.services.curation.class_write_guard import ClassWriteGuard
from src.services.curation.vlm_class_batch import class_context


if TYPE_CHECKING:
    from pathlib import Path

GATE = GateParams(theta=0.9, alpha=0.05, max_dissent=1)
REGISTRY = frozenset({'car', 'truck'})
N = 2000


def _answers(**counts: int) -> list[ClassAnswer]:
    out = []
    for name, count in counts.items():
        out.extend(ClassAnswer(class_name=name) for _ in range(count))
    return out


def _judge(answers: list[ClassAnswer], n: int = N, truth: tuple[str, ...] = ()) -> object:
    return judge(answers, n_eligible=n, gate=GATE, registry=REGISTRY, truth=list(truth))


def test_the_stage_alpha_splits_alpha_over_the_dissent_stages() -> None:
    assert GATE.alpha_stage == pytest.approx(0.025)


def test_enough_unanimous_answers_pass() -> None:
    need = required_sample(N, 0.9, 0.025, 0)

    verdict = _judge(_answers(car=need))

    assert verdict.passed  # type: ignore[attr-defined]
    assert verdict.class_name == 'car'  # type: ignore[attr-defined]
    assert verdict.need == need  # type: ignore[attr-defined]
    assert verdict.bound == homogeneity_lower_bound(need, need, N, 0.025)  # type: ignore[attr-defined]


def test_one_answer_short_does_not_pass_and_may_top_up() -> None:
    need = required_sample(N, 0.9, 0.025, 0)

    verdict = _judge(_answers(car=need - 1))

    assert not verdict.passed  # type: ignore[attr-defined]
    assert verdict.may_top_up  # type: ignore[attr-defined]


def test_one_dissent_needs_the_larger_sample() -> None:
    k0 = required_sample(N, 0.9, 0.025, 0)
    k1 = required_sample(N, 0.9, 0.025, 1)

    short = _judge(_answers(car=k0 - 1, truck=1))
    enough = _judge(_answers(car=k1 - 1, truck=1))

    assert not short.passed  # type: ignore[attr-defined]
    assert short.may_top_up  # type: ignore[attr-defined]
    assert short.need == k1  # type: ignore[attr-defined]
    assert enough.passed  # type: ignore[attr-defined]
    assert enough.dissent == 1  # type: ignore[attr-defined]


def test_more_dissent_than_allowed_fails_without_a_top_up() -> None:
    verdict = _judge(_answers(car=80, truck=2))

    assert not verdict.passed  # type: ignore[attr-defined]
    assert not verdict.may_top_up  # type: ignore[attr-defined]


def test_without_allowed_dissent_there_is_no_second_stage() -> None:
    strict = GateParams(theta=0.9, alpha=0.05, max_dissent=0)
    need = required_sample(N, 0.9, 0.05, 0)

    verdict = judge(_answers(car=need - 1), n_eligible=N, gate=strict, registry=REGISTRY, truth=[])

    assert strict.alpha_stage == pytest.approx(0.05)
    assert not verdict.passed
    assert not verdict.may_top_up


def test_a_class_outside_the_registry_never_passes() -> None:
    verdict = _judge(_answers(scooter=60))

    assert not verdict.passed  # type: ignore[attr-defined]
    assert verdict.class_name == 'scooter'  # type: ignore[attr-defined]


def test_a_new_class_proposal_is_dissent_and_never_passes() -> None:
    proposals = [ClassAnswer(class_name=None, proposed='tuk_tuk') for _ in range(60)]

    alone = _judge(proposals)
    mixed = _judge([*_answers(car=60), *proposals[:2]])

    assert not alone.passed  # type: ignore[attr-defined]
    assert alone.class_name is None  # type: ignore[attr-defined]
    assert alone.new_class_proposals == [('tuk_tuk', 60)]  # type: ignore[attr-defined]
    assert mixed.dissent == 2  # type: ignore[attr-defined]
    assert not mixed.passed  # type: ignore[attr-defined]


def test_a_small_node_asked_whole_is_an_exact_answer() -> None:
    n = 12
    need = required_sample(n, 0.9, 0.025, 0)

    verdict = _judge(_answers(car=need), n=n)

    assert verdict.passed  # type: ignore[attr-defined]


def test_three_validated_members_against_the_class_veto() -> None:
    verdict = _judge(_answers(car=60), truth=('truck', 'truck', 'car'))

    assert verdict.vetoed  # type: ignore[attr-defined]
    assert not verdict.passed  # type: ignore[attr-defined]


def test_validated_members_carrying_the_class_do_not_veto() -> None:
    verdict = _judge(_answers(car=60), truth=('car',) * 10)

    assert not verdict.vetoed  # type: ignore[attr-defined]
    assert verdict.passed  # type: ignore[attr-defined]


def test_two_validated_members_are_too_few_to_veto() -> None:
    verdict = _judge(_answers(car=60), truth=('truck', 'truck'))

    assert not verdict.vetoed  # type: ignore[attr-defined]
    assert verdict.passed  # type: ignore[attr-defined]


# ---------------------------------------------------------------- per-crop answers


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    w = make_world(tmp_path, monkeypatch)
    w.add_cluster(10001, {'car': 3, 'truck': 2}, prefix='a')
    return w


@pytest.mark.usefixtures('readable_crops')
def test_process_chunk_reports_which_crop_gave_which_answer(world: World) -> None:
    ids = sorted(world.truth)
    vlm = FakeVlm(world.truth, answers={ids[0]: None})
    ctx = class_context(vlm, ClassWriteGuard('assist_run'), revision=None)
    meta = RunMeta('run1', CREATED_AT, 'fake-vlm@1', 'fake-vlm', 'generic', None)

    out = asyncio.run(
        process_chunk(world.os, ctx, ids, classify=True, attributes=False, meta=meta)  # type: ignore[arg-type]
    )

    by_crop = {a.crop_id: a.class_name for a in out.answers}
    assert by_crop == {cid: world.truth[cid] for cid in ids[1:]}
