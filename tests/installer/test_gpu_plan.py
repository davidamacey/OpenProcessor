"""recommend_plan table tests (installer plan section 2.1, owner answers 11.1).

GPUs are given the way gpu_normalize emits them: "index total_mib used_mib".
"""

from __future__ import annotations

import os
import subprocess
from collections import Counter

import pytest
from installer_harness import REPO_ROOT, SCRIPT


LIB = REPO_ROOT / 'scripts' / 'lib' / 'vlm_catalog.sh'


def gb(n: int) -> int:
    return n * 1024


def plan(
    gpus: list[tuple[int, int, int]], tiers: str = '', *opts: str
) -> tuple[int, dict[str, str], list[str]]:
    lines = '\n'.join(f'{i} {t} {u}' for i, t, u in gpus)
    quoted = ' '.join(f"'{o}'" for o in opts)
    result = subprocess.run(
        [
            'bash',
            '-c',
            f'OP_SOURCE_ONLY=1 source "{SCRIPT}"; source "{LIB}"; '
            f'recommend_plan "$1" "$2" {quoted}',
            '_',
            lines,
            tiers,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ},
    )
    keys = [ln.split('=', 1)[0] for ln in result.stdout.splitlines() if not ln.startswith('warn=')]
    dupes = [k for k, c in Counter(keys).items() if c > 1]
    assert not dupes, f'keys emitted more than once: {dupes}'
    kv = dict(ln.split('=', 1) for ln in result.stdout.splitlines() if not ln.startswith('warn='))
    warns = [ln[5:] for ln in result.stdout.splitlines() if ln.startswith('warn=')]
    return result.returncode, kv, warns


def test_zero_gpus_refused() -> None:
    rc, kv, _ = plan([])
    assert rc == 1
    assert 'no NVIDIA GPU' in kv['refuse']


def test_7gb_refused_unless_forced() -> None:
    rc, kv, _ = plan([(0, gb(7), 0)])
    assert rc == 1
    assert '8 GB' in kv['refuse']
    rc, kv, warns = plan([(0, gb(7), 0)], '', 'force=1')
    assert rc == 0
    assert kv['GPU_PROFILE'] == 'minimal'
    assert kv['tiers'] == 'core'
    assert any('--force' in w for w in warns)


def test_10gb_is_minimal_core_without_segmenter_or_local_vlm() -> None:
    rc, kv, _ = plan([(0, gb(10), 0)])
    assert rc == 0
    assert kv['GPU_PROFILE'] == 'minimal'
    assert kv['recommended_tiers'] == 'core,curation'
    rc, kv, _ = plan([(0, gb(10), 0)], 'core curation segmenter')
    assert rc == 1
    assert '12 GB' in kv['refuse']
    rc, kv, _ = plan([(0, gb(10), 0)], 'core curation vlm')
    assert rc == 1
    assert 'no tested catalog entry' in kv['refuse']


def test_14gb_offers_one_segmenter_instance_but_does_not_recommend_it() -> None:
    rc, kv, _ = plan([(0, gb(14), 0)])
    assert 'segmenter' not in kv['recommended_tiers']
    rc, kv, _ = plan([(0, gb(14), 0)], 'core curation segmenter')
    assert rc == 0
    assert kv['SEGMENTER_INSTANCES'] == '1'


def test_24gb_single_card_gets_no_unverified_vlm_automatically() -> None:
    rc, kv, warns = plan([(0, gb(24), 0)])
    assert rc == 0
    assert kv['GPU_PROFILE'] == 'standard'
    assert kv['recommended_tiers'] == 'core,curation,segmenter'
    assert kv['VLM_CATALOG_ID'] == ''
    assert any('local VLM not offered' in w for w in warns)


def test_24gb_explicit_unverified_vlm_needs_force_and_warns() -> None:
    rc, kv, _ = plan([(0, gb(24), 0)], 'core curation vlm', 'vlm_id=qwen2.5-vl-7b-awq')
    assert rc == 1
    assert '--force' in kv['refuse']
    rc, kv, warns = plan(
        [(0, gb(24), 0)], 'core curation vlm', 'vlm_id=qwen2.5-vl-7b-awq', 'force=1'
    )
    assert rc == 0
    assert kv['VLM_CATALOG_ID'] == 'qwen2.5-vl-7b-awq'
    assert kv['vlm_status'] == 'to_verify'
    assert any('unverified/experimental' in w for w in warns)
    assert any('shares GPU 0 with Triton' in w for w in warns)


def test_unknown_vlm_id_refused() -> None:
    rc, kv, _ = plan([(0, gb(48), 0)], 'core curation vlm', 'vlm_id=nope')
    assert rc == 1
    assert 'not in the VLM catalog' in kv['refuse']


def test_36gb_has_no_trainer_by_default_and_picks_the_smaller_tested_vlm() -> None:
    rc, kv, _ = plan([(0, gb(36), 0)])
    assert rc == 0
    assert 'trainer' not in kv['recommended_tiers']
    assert kv['VLM_CATALOG_ID'] == 'qwen3-vl-4b'
    assert kv['vlm_status'] == 'tested'


def test_48gb_single_card_picks_the_tested_vlm_after_triton_and_segmenter() -> None:
    rc, kv, warns = plan([(0, 49140, 0)])
    assert rc == 0
    assert kv['VLM_CATALOG_ID'] == 'gemma-4-e4b'
    assert kv['vlm_status'] == 'tested'
    assert kv['vlm_available_gb'] == str(47 - 12 - 2 - 4)
    assert kv['VLM_GPU_MEMORY_UTILIZATION'] == '0.42'
    assert kv['recommended_tiers'] == 'core,curation,segmenter,vlm'
    assert any('shares GPU 0' in w for w in warns)


def test_48gb_trainer_warns_about_train_mode() -> None:
    rc, _, warns = plan([(0, 49140, 0)], 'core curation segmenter vlm trainer')
    assert rc == 0
    assert any('train-mode on' in w for w in warns)


def test_used_vram_is_subtracted_and_reported() -> None:
    rc, kv, warns = plan([(0, 49140, gb(20))])
    assert rc == 0
    assert kv['VLM_CATALOG_ID'] == ''
    assert any('GPU 0: 41% of its VRAM is already in use' in w for w in warns)


def test_two_gpus_12_and_48() -> None:
    rc, kv, warns = plan([(0, gb(12), 0), (1, 49140, 0)], 'core curation segmenter vlm trainer')
    assert rc == 0
    assert kv['TRITON_GPU_ID'] == kv['API_GPU_ID'] == '0'
    assert kv['VLM_GPU_ID'] == kv['SEGMENTER_GPU_ID'] == '1'
    assert kv['OP_TRAIN_GPU_ORDER'] == kv['EVALUATOR_GPU_ID'] == '1'
    assert kv['VLM_CATALOG_ID'] == 'gemma-4-e4b'
    assert any('train-mode' in w for w in warns)


def test_two_gpus_both_48() -> None:
    rc, kv, _ = plan([(0, 49140, 0), (1, 49140, 0)], 'core curation segmenter vlm trainer')
    assert rc == 0
    assert (
        kv['TRITON_GPU_ID'],
        kv['SEGMENTER_GPU_ID'],
        kv['OP_TRAIN_GPU_ORDER'],
        kv['VLM_GPU_ID'],
    ) == ('0', '0', '0', '1')
    assert kv['vlm_available_gb'] == '47'
    assert kv['GPU_PROFILE'] == 'full'


def test_three_gpus_shaped_like_this_host() -> None:
    rc, kv, _ = plan(
        [(0, 49140, 0), (1, 12288, 0), (2, 49140, 0)], 'core curation segmenter vlm trainer'
    )
    assert rc == 0
    assert kv['TRITON_GPU_ID'] == '1'
    assert kv['VLM_GPU_ID'] == '0'
    assert kv['SEGMENTER_GPU_ID'] == kv['OP_TRAIN_GPU_ORDER'] == kv['EVALUATOR_GPU_ID'] == '2'
    assert kv['GPU_PROFILE'] == 'standard'
    assert kv['OP_GPU_ALLOWED_IDS'] == '0,1,2'
    assert kv['VLM_GPU_TOTAL_MIB'] == str(48 * 1024)


def test_three_gpus_vlm_prefers_the_largest_with_more_free() -> None:
    rc, kv, _ = plan([(0, 49140, gb(30)), (1, 12288, 0), (2, 49140, 0)], 'core curation vlm')
    assert rc == 0
    assert kv['VLM_GPU_ID'] == '2'
    assert kv['SEGMENTER_GPU_ID'] == '0'


def test_gpu_plan_override_and_its_validation() -> None:
    rc, kv, _ = plan(
        [(0, 49140, 0), (1, 12288, 0), (2, 49140, 0)],
        'core curation vlm',
        'gpu_plan=triton=2,vlm=0',
    )
    assert rc == 0
    assert kv['TRITON_GPU_ID'] == '2'
    assert kv['VLM_GPU_ID'] == '0'
    rc, kv, _ = plan([(0, 49140, 0)], '', 'gpu_plan=triton=5')
    assert rc == 1
    assert 'no GPU with index 5' in kv['refuse']
    rc, kv, _ = plan([(0, 49140, 0)], '', 'gpu_plan=bogus=0')
    assert rc == 1
    assert 'unknown key' in kv['refuse']


def test_explicit_triton_on_a_small_card_is_refused_unless_forced() -> None:
    gpus = [(0, 49140, 0), (1, 12288, 0), (2, 49140, 0)]
    rc, kv, _ = plan(gpus, 'core curation', 'gpu_plan=triton=1')
    assert rc == 1
    assert 'out-of-GPU-memory' in kv['refuse']
    assert '--gpu-plan' in kv['refuse']
    assert '--force' in kv['refuse']
    rc, kv, warns = plan(gpus, 'core curation', 'gpu_plan=triton=1', 'force=1')
    assert rc == 0
    assert kv['TRITON_GPU_ID'] == kv['API_GPU_ID'] == '1'
    assert any('share GPU 1' in w for w in warns)


def test_explicit_triton_on_a_big_card_is_not_refused() -> None:
    rc, _kv, warns = plan(
        [(0, 49140, 0), (1, 12288, 0), (2, 49140, 0)], 'core curation', 'gpu_plan=triton=2'
    )
    assert rc == 0
    assert not any('engine exports' in w for w in warns)


def test_automatic_plan_on_a_small_triton_card_only_warns() -> None:
    rc, kv, warns = plan([(0, 49140, 0), (1, 12288, 0), (2, 49140, 0)], 'core curation')
    assert rc == 0
    assert kv['TRITON_GPU_ID'] == '1'
    assert any('engine exports' in w and '--gpu-plan' in w for w in warns)


def test_remote_vlm_is_never_recommended_as_local() -> None:
    rc, kv, _ = plan([(0, 49140, 0)], '', 'remote=1')
    assert rc == 0
    assert 'vlm' not in kv['recommended_tiers'].split(',')


@pytest.mark.parametrize(
    ('row', 'expected'),
    [
        ('0, NVIDIA RTX A6000, 49140, 1234, 8.6', '0 49140 1234'),
        ('3, Weird, Name, With, Commas, 12288, 10, 8.6', '3 12288 10'),
        ('1, NVIDIA GeForce RTX 3080 Ti, 12288 MiB, 5 MiB, 8.6', '1 12288 5'),
    ],
)
def test_gpu_normalize_parses_from_the_right(row: str, expected: str) -> None:
    result = subprocess.run(
        [
            'bash',
            '-c',
            f'OP_SOURCE_ONLY=1 source "{SCRIPT}"; printf "%s\\n" "$1" | gpu_normalize',
            '_',
            row,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == expected


def test_nvidia_smi_is_queried_without_units() -> None:
    assert '--format=csv,noheader,nounits' in SCRIPT.read_text()
