"""Wave 6: the gate inside the full-image pass -- tier 1 registry rules, tier 2
vision-model pre-check, tier 3 hit-rate, skips counted, nothing edited."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest

from curation.open_vocab_fixtures import (
    FakeSegmenter,
    StatefulRegistry,
    cand,
    image_doc,
    ingested_world,
    make_set,
)
from curation.reprocess_fixtures import docs
from src.services.curation.open_vocab_gate import load_tracker, save_tracker
from src.services.curation.open_vocab_run import GateContext, run_open_vocab_image
from src.services.curation.reprocess_models import ReprocessScopeResult
from src.services.curation.reprocess_open_vocab import OpenVocabPass
from src.services.detection.segmenter_gate import HitRateTracker


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = StatefulRegistry()
    monkeypatch.setattr('src.services.curation.open_vocab_run.get_class_registry', lambda: registry)


def _seg() -> FakeSegmenter:
    seg = FakeSegmenter()
    seg.default = [cand()]
    return seg


class _Vlm:
    def __init__(self, answers: dict[str, bool | None]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    async def __call__(self, jpeg: bytes, prompt: str) -> bool | None:  # noqa: ARG002
        self.asked.append(prompt)
        answer = self.answers.get(prompt, True)
        if answer is None:
            raise RuntimeError('endpoint down')
        return answer


TWO = [
    {'prompt': 'cone', 'class_name': 'cone'},
    {'prompt': 'cup', 'class_name': 'cup'},
]


@pytest.mark.asyncio
async def test_tier1_parent_classes_skip_an_image_without_such_an_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, (image_id,) = await ingested_world(tmp_path, monkeypatch)
    docs(fake)['car-1'] = {
        'crop_id': 'car-1',
        'image_id': image_id,
        'bbox_norm': [0.6, 0.6, 0.9, 0.9],
        'class_source': 'primary_model',
        'class_name': 'Car',
    }
    ov = make_set(
        targets=[
            {'prompt': 'wheel', 'class_name': 'wheel', 'parent_classes': ['car']},
            {'prompt': 'leaf', 'class_name': 'leaf', 'parent_classes': ['tree']},
        ]
    )
    seg = _seg()

    result = await run_open_vocab_image(
        fake, service, image_id, image_doc(fake, image_id), ov, revision=1, segment=seg
    )

    assert seg.prompts == ['wheel']
    assert result.skipped == {'tier1_no_parent_class': 1}
    assert result.calls == 1


@pytest.mark.asyncio
async def test_a_skipped_target_never_removes_its_earlier_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, (image_id,) = await ingested_world(tmp_path, monkeypatch)
    ov = make_set(targets=TWO)
    seg = _seg()
    seg.by_prompt = {'cone': [cand((0.1, 0.1, 0.3, 0.3))], 'cup': [cand((0.5, 0.5, 0.8, 0.8))]}
    doc = image_doc(fake, image_id)
    await run_open_vocab_image(fake, service, image_id, doc, ov, revision=1, segment=seg)
    assert {d['source_prompt'] for d in docs(fake).values()} == {'cone', 'cup'}

    vlm = _Vlm({'cup': False})
    gated = make_set(targets=TWO, gating={'tier2_vlm_precheck': True})
    seg.calls.clear()
    result = await run_open_vocab_image(
        fake,
        service,
        image_id,
        doc,
        gated,
        revision=2,
        segment=seg,
        gate=GateContext(vlm_visible=vlm),
    )

    assert seg.prompts == ['cone']
    assert result.skipped == {'tier2_vlm_no': 1}
    assert {d['source_prompt'] for d in docs(fake).values()} == {'cone', 'cup'}  # cup kept


@pytest.mark.asyncio
async def test_a_target_removed_from_the_set_is_removed_once_something_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, (image_id,) = await ingested_world(tmp_path, monkeypatch)
    seg = _seg()
    seg.by_prompt = {'cone': [cand((0.1, 0.1, 0.3, 0.3))], 'cup': [cand((0.5, 0.5, 0.8, 0.8))]}
    doc = image_doc(fake, image_id)
    await run_open_vocab_image(
        fake, service, image_id, doc, make_set(targets=TWO), revision=1, segment=seg
    )

    only_cone = make_set(targets=TWO[:1])
    result = await run_open_vocab_image(
        fake, service, image_id, doc, only_cone, revision=2, segment=seg
    )

    assert {d['source_prompt'] for d in docs(fake).values()} == {'cone'}
    assert result.removed == 1


@pytest.mark.asyncio
async def test_when_every_target_is_skipped_nothing_is_written_or_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, (image_id,) = await ingested_world(tmp_path, monkeypatch)
    seg = _seg()
    seg.by_prompt = {'cone': [cand((0.1, 0.1, 0.3, 0.3))], 'cup': [cand((0.5, 0.5, 0.8, 0.8))]}
    doc = image_doc(fake, image_id)
    await run_open_vocab_image(
        fake, service, image_id, doc, make_set(targets=TWO), revision=1, segment=seg
    )
    before = copy.deepcopy(docs(fake))
    seg.calls.clear()
    # The new revision has dropped 'cup' and the gate skips the one target left:
    # nothing was looked at, so nothing (not even cup's output) is touched.
    gated = make_set(targets=TWO[:1], gating={'tier2_vlm_precheck': True})

    result = await run_open_vocab_image(
        fake,
        service,
        image_id,
        doc,
        gated,
        revision=2,
        segment=seg,
        gate=GateContext(vlm_visible=_Vlm({'cone': False})),
    )

    assert (seg.calls, result.calls, result.written, result.removed) == ([], 0, 0, 0)
    assert docs(fake) == before


@pytest.mark.asyncio
async def test_a_vision_model_error_runs_the_call_and_a_missing_model_does_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, (image_id,) = await ingested_world(tmp_path, monkeypatch)
    gated = make_set(targets=TWO[:1], gating={'tier2_vlm_precheck': True})
    doc = image_doc(fake, image_id)
    for gate in (GateContext(vlm_visible=_Vlm({'cone': None})), GateContext()):
        seg = _seg()
        result = await run_open_vocab_image(
            fake, service, image_id, doc, gated, revision=1, segment=seg, gate=gate
        )
        assert (seg.prompts, dict(result.skipped)) == (['cone'], {})


@pytest.mark.asyncio
async def test_tier2_is_off_unless_the_set_asks_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, (image_id,) = await ingested_world(tmp_path, monkeypatch)
    vlm, seg = _Vlm({'cone': False}), _seg()
    await run_open_vocab_image(
        fake,
        service,
        image_id,
        image_doc(fake, image_id),
        make_set(),
        revision=1,
        segment=seg,
        gate=GateContext(vlm_visible=vlm),
    )
    assert (vlm.asked, seg.prompts) == ([], ['traffic cone'])


@pytest.mark.asyncio
async def test_tier3_learns_from_the_pass_and_samples_a_missing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, ids = await ingested_world(tmp_path, monkeypatch, n=6)
    saved: list[HitRateTracker] = []

    async def load(_client: Any) -> HitRateTracker:
        return HitRateTracker()

    async def save(_client: Any, tracker: HitRateTracker) -> None:
        saved.append(tracker)

    monkeypatch.setattr('src.services.curation.reprocess_open_vocab.load_tracker', load)
    monkeypatch.setattr('src.services.curation.reprocess_open_vocab.save_tracker', save)
    ov = make_set(
        targets=TWO[:1],
        gating={
            'tier3_hit_rate': {
                'enabled': True,
                'window': 4,
                'miss_threshold': 3,
                'sample_floor': 0.0,
            }
        },
    )
    seg = FakeSegmenter()  # never finds anything
    run = await OpenVocabPass.start(fake, ov, 1, seg, ReprocessScopeResult(scope='open_vocab'))
    for image_id in ids:
        await run.run_image(fake, service, image_id, image_doc(fake, image_id))
    await run.finish(fake)

    # Three misses in a row trip the gate; with a zero floor nothing more runs.
    assert len(seg.calls) == 3
    assert run.result.detail['skipped_gate_tier3_hit_rate'] == 3
    statuses = [image_doc(fake, i).get('open_vocab_status') for i in ids]
    assert statuses == ['done'] * 3 + ['skipped_gate'] * 3
    assert [t.dump() for t in saved] == [{'cone|cone': [False, False, False]}]


@pytest.mark.asyncio
async def test_a_pass_without_tier3_loads_and_saves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, (image_id,) = await ingested_world(tmp_path, monkeypatch)
    touched: list[str] = []

    async def boom(*_a: Any) -> None:
        touched.append('x')

    monkeypatch.setattr('src.services.curation.reprocess_open_vocab.load_tracker', boom)
    monkeypatch.setattr('src.services.curation.reprocess_open_vocab.save_tracker', boom)
    run = await OpenVocabPass.start(
        fake, make_set(), 1, _seg(), ReprocessScopeResult(scope='open_vocab')
    )
    await run.run_image(fake, service, image_id, image_doc(fake, image_id))
    await run.finish(fake)
    assert touched == []


@pytest.mark.asyncio
async def test_the_hit_rate_windows_persist_across_passes() -> None:
    from curation._fake_config_opensearch import FakeConfigOpenSearch

    client = FakeConfigOpenSearch()
    assert (await load_tracker(client)).dump() == {}
    tracker = HitRateTracker()
    for hit in (False, False, True):
        tracker.record('cone|cone', hit=hit, window=5)
    await save_tracker(client, tracker)

    assert (await load_tracker(client)).dump() == {'cone|cone': [False, False, True]}


@pytest.mark.asyncio
async def test_a_failed_stats_write_never_fails_the_pass() -> None:
    class Broken:
        async def index(self, **_kw: Any) -> None:
            raise RuntimeError('opensearch down')

    await save_tracker(Broken(), HitRateTracker())  # logged, not raised
