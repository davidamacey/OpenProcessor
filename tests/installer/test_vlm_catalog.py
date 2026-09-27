"""Unit tests for scripts/lib/vlm_catalog.sh (installer plan section 2.2)."""

from __future__ import annotations


LIB = 'scripts/lib/vlm_catalog.sh'


def _run(bash, extra: str):
    return bash(f'set -euo pipefail; source {LIB}; {extra}')


def test_pick_vlm_48gb_picks_tested_gemma(bash) -> None:
    result = _run(bash, 'pick_vlm 48')
    assert result.returncode == 0
    assert result.stdout.strip() == 'gemma-4-e4b\ttested'


def test_pick_vlm_is_a_pure_function_of_the_available_gb(bash) -> None:
    # pick_vlm sees VRAM already net of Triton and the segmenter; the
    # subtraction is recommend_plan's job (tests/installer/test_gpu_plan.py).
    result = _run(bash, 'pick_vlm 24')
    assert result.returncode == 0
    assert result.stdout.strip() == 'gemma-4-e4b\ttested'


def test_pick_vlm_below_floor_refuses(bash) -> None:
    result = _run(bash, 'pick_vlm 12')
    assert result.returncode != 0
    assert result.stdout.strip() == ''


def test_pick_vlm_never_auto_picks_a_to_verify_row(bash) -> None:
    # Owner answer 11.1 #6: 20 GB fits four to_verify rows but no tested
    # one, so nothing is picked.
    result = _run(bash, 'pick_vlm 20')
    assert result.returncode == 1
    assert result.stdout.strip() == ''


def test_pick_vlm_candidates_lists_fitting_rows_with_status(bash) -> None:
    result = _run(bash, 'pick_vlm_candidates 20')
    rows = [ln.split('\t') for ln in result.stdout.strip().splitlines()]
    assert [r[0] for r in rows] == [
        'qwen3-vl-8b-fp8',
        'gemma-4-e2b',
        'qwen3-vl-4b',
        'qwen2.5-vl-7b-awq',
    ]
    assert all(r[3] == 'to_verify' for r in rows)
    first = _run(bash, 'pick_vlm_candidates 48').stdout.splitlines()[0]
    assert first.startswith('gemma-4-e4b\t')
    assert first.endswith('\ttested')


def test_vlm_catalog_floor_gb_counts_tested_rows_only(bash) -> None:
    result = _run(bash, 'vlm_catalog_floor_gb')
    assert result.stdout.strip() == '23'


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


def test_a_high_rank_tiny_to_verify_row_is_never_auto_picked(bash, tmp_path) -> None:
    # Regression guard for owner answer 11.1 #6 using the catalog's real
    # unverified status: a to_verify row that fits anything and outranks
    # every tested row is still never returned by pick_vlm.
    catalog = tmp_path / 'catalog.tsv'
    shipped = bash('cat examples/vlm/catalog.tsv').stdout
    catalog.write_text(
        shipped
        + 'zzz-fake\tfake/repo\tApache-2.0\t1\t8192\t8192\t8\tto_verify\t999\tfalse\tvlm_generic\n'
    )
    big = _run(bash, f'pick_vlm 48 "{catalog}"')
    assert big.returncode == 0
    assert big.stdout.strip() == 'gemma-4-e4b\ttested'
    small = _run(bash, f'pick_vlm 5 "{catalog}"')
    assert small.returncode == 1
    assert small.stdout.strip() == ''
