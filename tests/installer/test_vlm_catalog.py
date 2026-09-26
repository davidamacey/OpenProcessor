"""Unit tests for scripts/lib/vlm_catalog.sh (installer plan section 2.2)."""

from __future__ import annotations


LIB = 'scripts/lib/vlm_catalog.sh'


def _run(bash, extra: str):
    return bash(f'set -euo pipefail; source {LIB}; {extra}')


def test_pick_vlm_48gb_picks_tested_gemma(bash) -> None:
    result = _run(bash, 'pick_vlm 48')
    assert result.returncode == 0
    assert result.stdout.strip() == 'gemma-4-e4b\ttested'


def test_pick_vlm_24gb_shared_card_still_fits(bash) -> None:
    result = _run(bash, 'pick_vlm 24')
    assert result.returncode == 0
    assert result.stdout.strip().startswith('gemma-4-e4b')


def test_pick_vlm_below_floor_refuses(bash) -> None:
    result = _run(bash, 'pick_vlm 12')
    assert result.returncode != 0
    assert result.stdout.strip() == ''


def test_pick_vlm_prefers_tested_over_higher_rank_unverified(bash) -> None:
    # 20 GB fits gemma-4-e4b (23GB, tested)? No -- 23 > 20, so it must fall
    # through to a to_verify row. This proves "tested wins over any rank
    # to_verify that also fits" doesn't wrongly promote a non-fitting tested
    # row.
    result = _run(bash, 'pick_vlm 20')
    assert result.returncode == 0
    stdout = result.stdout.strip()
    assert 'to_verify' in stdout


def test_vlm_catalog_floor_gb(bash) -> None:
    result = _run(bash, 'vlm_catalog_floor_gb')
    assert result.stdout.strip() == '15'


def test_vlm_gpu_memory_utilization_clamped_low(bash) -> None:
    result = _run(bash, 'vlm_gpu_memory_utilization 5 48')
    assert float(result.stdout.strip()) == 0.2


def test_vlm_gpu_memory_utilization_clamped_high(bash) -> None:
    result = _run(bash, 'vlm_gpu_memory_utilization 45 12')
    assert float(result.stdout.strip()) == 0.9


def test_vlm_gpu_memory_utilization_gemma_on_48gb(bash) -> None:
    result = _run(bash, 'vlm_gpu_memory_utilization 23 48')
    assert float(result.stdout.strip()) == 0.42


def test_vlm_catalog_field_hf_repo(bash) -> None:
    result = _run(bash, 'vlm_catalog_field gemma-4-e4b hf_repo')
    assert result.stdout.strip() == 'google/gemma-4-E4B-it'


def test_vlm_catalog_field_unknown_id_fails(bash) -> None:
    result = _run(bash, 'vlm_catalog_field nonexistent-id hf_repo')
    assert result.returncode != 0


def test_vlm_catalog_get_returns_full_row(bash) -> None:
    result = _run(bash, 'vlm_catalog_get gemma-4-e4b')
    assert result.returncode == 0
    assert result.stdout.startswith('gemma-4-e4b\t')


def test_no_status_other_than_tested_or_to_verify_is_ever_returned(bash) -> None:
    # Regression guard for owner answer 11.1 #6: an "unverified" (or any
    # other) status must never be auto-picked, even if it would rank
    # highest and fit. Simulate by adding one to a scratch copy of the
    # catalog with an "unverified" status.
    result = bash(
        'set -euo pipefail; '
        f'source {LIB}; '
        'tmp=$(mktemp); '
        f'cp {LIB.replace("scripts/lib/vlm_catalog.sh", "examples/vlm/catalog.tsv")} "$tmp"; '
        'printf "zzz-fake\\tfake/repo\\tApache-2.0\\t1\\t8192\\t8192\\t8\\tunverified\\t999\\tfalse\\tvlm_generic\\n" >> "$tmp"; '
        'pick_vlm 48 "$tmp"'
    )
    assert result.returncode == 0
    assert 'zzz-fake' not in result.stdout
