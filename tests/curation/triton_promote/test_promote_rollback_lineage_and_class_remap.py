"""promote(): rollback on failure, job back-pointer and sharing carry-forward, and class-remap copying."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from curation.triton_promote._helpers import CLASS_MAP, FakeRecorder, _promoter
from src.services.training.triton_promote import PromoteResult, TritonPromoter


if TYPE_CHECKING:
    from pathlib import Path

    from src.services.training.job_models import TrainJobStatus


# =============================================================================
# test_promote_restores_previous_version_on_failure
# =============================================================================


@pytest.mark.asyncio
async def test_promote_restores_previous_version_on_failure(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the fix: a failed copy happened AFTER the existing model dir
    was already rmtree'd — the model dir would simply be gone. Now: the
    previously-serving version must remain fully intact and servable, and
    the half-written new version must not linger to confuse Triton."""
    triton_name = 'op_smoke_v1'
    model_dir = scratch_models_dir / triton_name
    v1 = model_dir / '1'
    v1.mkdir(parents=True)
    (v1 / 'model.onnx').write_bytes(b'ORIGINAL-V1-WEIGHTS')
    (model_dir / 'config.pbtxt').write_text('# original config')

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        msg = 'simulated disk failure mid-copy'
        raise OSError(msg)

    monkeypatch.setattr('src.services.training.triton_promote.add_uint8_input', _boom)
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))

    promoter = _promoter(scratch_models_dir)
    with pytest.raises(OSError, match='simulated disk failure'):
        await promoter.promote(
            status=fake_status,
            triton_name=triton_name,
            recorder=FakeRecorder(CLASS_MAP),
            overwrite=True,
        )

    # The previously-serving version was never touched — nothing to
    # "restore" because it was never at risk.
    assert (v1 / 'model.onnx').read_bytes() == b'ORIGINAL-V1-WEIGHTS'
    assert (model_dir / 'config.pbtxt').read_text() == '# original config'
    # The half-written new version must be cleaned up, not left behind.
    assert not (model_dir / '2').exists()


@pytest.mark.asyncio
async def test_promote_config_write_failure_also_rolls_back_new_version(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure writing config.pbtxt (after a successful ONNX copy) must
    also roll back the new version dir, not leave a weights-only half-model."""
    triton_name = 'op_smoke_v1'
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))

    def _boom(*_args: Any, **_kwargs: Any) -> str:
        msg = 'simulated config render failure'
        raise RuntimeError(msg)

    # triton_promote.py does `from ...yolo26_triton_config import render_config`,
    # binding the name into its own module namespace — patch it there, not on
    # the source module, or the already-imported reference won't see it.
    monkeypatch.setattr('src.services.training.triton_promote.render_config', _boom)

    promoter = _promoter(scratch_models_dir)
    with pytest.raises(RuntimeError, match='simulated config render failure'):
        await promoter.promote(
            status=fake_status,
            triton_name=triton_name,
            recorder=FakeRecorder(CLASS_MAP),
        )
    assert not (scratch_models_dir / triton_name / '1').exists()


# =============================================================================
# test_promote_writes_job_id_backpointer_and_version
# =============================================================================


@pytest.mark.asyncio
async def test_promote_records_the_version_and_writes_labels_from_the_recorded_rows(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The promoter stores the promotion (name and version) before it writes anything the
    model directory says, and renders labels.txt from what was stored, not from its input."""
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))
    promoter = _promoter(scratch_models_dir)
    triton_name = 'op_smoke_v1'
    recorder = FakeRecorder({0: 'ignored', 1: 'ignored'})
    recorder.class_id_to_name = {0: 'stored_a', 1: 'stored_b'}

    result: PromoteResult = await promoter.promote(
        status=fake_status, triton_name=triton_name, recorder=recorder
    )

    assert recorder.recorded == [(triton_name, 1)]
    assert (scratch_models_dir / triton_name / 'labels.txt').read_text().split() == [
        'stored_a',
        'stored_b',
    ]
    assert not (scratch_models_dir / triton_name / 'promote.json').exists()
    assert result.version == '1'


@pytest.mark.asyncio
async def test_a_re_promote_records_the_next_version(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))
    promoter = _promoter(scratch_models_dir)
    recorder = FakeRecorder(CLASS_MAP)
    await promoter.promote(status=fake_status, triton_name='op_smoke_v1', recorder=recorder)
    await promoter.promote(
        status=fake_status, triton_name='op_smoke_v1', recorder=recorder, overwrite=True
    )

    assert recorder.recorded == [('op_smoke_v1', 1), ('op_smoke_v1', 2)]
    assert recorder.withdrawn == 0


@pytest.mark.asyncio
async def test_a_failed_load_withdraws_the_recorded_promotion_and_the_new_version(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A promote that never served must not stay live: Triton refusing the load (4xx) withdraws
    the rows and removes the new version directory."""
    from src.services.training.promote_errors import TritonLoadError

    monkeypatch.setattr(
        TritonPromoter, '_trigger_load', AsyncMock(side_effect=TritonLoadError('refused'))
    )
    promoter = _promoter(scratch_models_dir)
    recorder = FakeRecorder(CLASS_MAP)

    with pytest.raises(TritonLoadError):
        await promoter.promote(status=fake_status, triton_name='op_smoke_v1', recorder=recorder)

    assert recorder.recorded == [('op_smoke_v1', 1)]
    assert recorder.withdrawn == 1
    assert not (scratch_models_dir / 'op_smoke_v1' / '1').exists()


@pytest.mark.asyncio
async def test_a_fail_soft_load_keeps_the_promotion(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rows record what is on disk, not whether Triton has picked it up yet: a load that
    timed out (fail-soft) still leaves the promotion and its files."""
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=False))
    promoter = _promoter(scratch_models_dir)
    recorder = FakeRecorder(CLASS_MAP)

    result = await promoter.promote(
        status=fake_status, triton_name='op_smoke_v1', recorder=recorder
    )
    assert result.triton_loaded is False
    assert recorder.withdrawn == 0
    assert (scratch_models_dir / 'op_smoke_v1' / 'labels.txt').is_file()


# =============================================================================
# test_class_remap_missing_fails_loudly
# =============================================================================


def test_class_remap_corrupt_weights_dir_file_raises_loudly(tmp_path: Path) -> None:
    """P2-7: a *present but corrupt* class_remap.json used to silently fall
    back to "full-class run" — now it's a loud, typed failure instead of a
    quiet None, since a mislabeled labels.txt is a serving-correctness bug."""
    from src.services.training.class_remap import resolve_class_remap
    from src.services.training.promote_errors import ClassRemapUnreadableError

    run_dir = tmp_path / 'job-with-bad-remap'
    run_dir.mkdir()
    (run_dir / 'class_remap.json').write_text('{not valid json')
    checkpoint_path = run_dir / 'best.pt'

    with pytest.raises(ClassRemapUnreadableError):
        resolve_class_remap(
            job_id='job-with-bad-remap', checkpoint_path=checkpoint_path, manifest=None
        )


def test_class_remap_genuinely_absent_is_not_an_error(tmp_path: Path) -> None:
    """A full-class run legitimately has no class_remap anywhere — that
    resolves to the 'none' source, not an error. The caller (promote_run)
    is responsible for refusing to serve that for an actual subset run."""
    from src.services.training.class_remap import resolve_class_remap

    run_dir = tmp_path / 'job-full-class'
    run_dir.mkdir()
    checkpoint_path = run_dir / 'best.pt'

    result = resolve_class_remap(
        job_id='job-full-class', checkpoint_path=checkpoint_path, manifest=None
    )

    assert result.source == 'none'
    assert result.mapping == {}


def test_class_remap_real_payload_shape_round_trips(tmp_path: Path) -> None:
    """subset_dataset.py's REAL payload shape (original_to_new/new_to_original/
    single_cls/names/include_classes) must parse correctly — this is the P2-7
    bug: the old parser only understood a flat {orig: new} dict or a
    {'mapping': {...}} wrapper and silently produced an empty result here."""
    from src.services.training.class_remap import resolve_class_remap

    run_dir = tmp_path / 'job-subset'
    run_dir.mkdir()
    payload = {
        'original_to_new': {'12': 0, '81': 1},
        'new_to_original': {'0': 12, '1': 81},
        'single_cls': False,
        'names': ['pickup', 'license_plate'],
        'include_classes': [12, 81],
    }
    (run_dir / 'class_remap.json').write_text(json.dumps(payload))
    checkpoint_path = run_dir / 'best.pt'

    result = resolve_class_remap(
        job_id='job-subset', checkpoint_path=checkpoint_path, manifest=None
    )

    assert result.source == 'weights_dir'
    assert result.mapping == {12: 0, 81: 1}
    assert result.names == ['pickup', 'license_plate']
    assert result.single_cls is False


def test_class_remap_single_cls_payload(tmp_path: Path) -> None:
    from src.services.training.class_remap import build_class_id_to_name, resolve_class_remap

    run_dir = tmp_path / 'job-single-cls'
    run_dir.mkdir()
    payload = {
        'original_to_new': {'12': 0, '81': 0},
        'new_to_original': {},
        'single_cls': True,
        'names': ['object'],
        'include_classes': [12, 81],
    }
    (run_dir / 'class_remap.json').write_text(json.dumps(payload))
    checkpoint_path = run_dir / 'best.pt'

    result = resolve_class_remap(
        job_id='job-single-cls', checkpoint_path=checkpoint_path, manifest=None
    )
    assert result.single_cls is True

    labels = build_class_id_to_name(remap=result, full_registry={12: 'pickup', 81: 'license_plate'})
    assert labels == {0: 'object'}


def test_class_remap_manifest_lineage_preferred_over_weights_dir(tmp_path: Path) -> None:
    """Every already-completed run has lineage.class_remap in its manifest,
    captured before the trainer's /tmp cleanup — this must resolve correctly
    even when there's no weights-dir file at all (older runs, pre-fix)."""
    from src.services.training.class_remap import resolve_class_remap

    run_dir = tmp_path / 'job-old-run'
    run_dir.mkdir()
    checkpoint_path = run_dir / 'best.pt'  # no class_remap.json next to it
    manifest = {
        'lineage': {
            'class_remap': {
                'original_to_new': {'3': 0},
                'single_cls': False,
                'names': ['sedan'],
                'include_classes': [3],
            }
        }
    }

    result = resolve_class_remap(
        job_id='job-old-run', checkpoint_path=checkpoint_path, manifest=manifest
    )
    assert result.source == 'manifest'
    assert result.mapping == {3: 0}


@pytest.mark.asyncio
async def test_promote_copies_the_stored_class_remap_into_model_dir(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored remap lands in the served model dir too (class_remap.json alongside
    config.pbtxt/labels.txt), rendered from the promotion row."""
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))
    promoter = _promoter(scratch_models_dir)
    stored = {
        'original_to_new': {'12': 0, '81': 1},
        'single_cls': False,
        'names': ['pickup', 'license_plate'],
        'include_classes': [12, 81],
        'source': 'manifest',
        'job_id': 'job-abc',
    }
    result = await promoter.promote(
        status=fake_status,
        triton_name='op_smoke_remap',
        recorder=FakeRecorder({0: 'pickup', 1: 'license_plate'}, class_remap=stored),
    )
    assert result.class_remap_source == 'manifest'
    remap_dest = json.loads(
        (scratch_models_dir / 'op_smoke_remap' / 'class_remap.json').read_text()
    )
    assert remap_dest['original_to_new'] == {'12': 0, '81': 1}
    assert remap_dest['source'] == 'manifest'
    assert remap_dest['version'] == '1'
