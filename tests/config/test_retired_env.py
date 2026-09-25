"""``src/config/retired_env.py`` — startup guard for retired env-var names.

A clean break, no aliases. A
retired name being *set* must fail loudly at startup, naming its
replacement, rather than being silently ignored (which would look like
a working config that quietly does nothing -- e.g. a stale ``SAM3_URL``
turning the segmenter off with no error).
"""

from __future__ import annotations

import pytest

from src.config.retired_env import RETIRED_ENV, reject_retired_env


def test_retired_env_table_covers_every_section_3_name() -> None:
    expected_old_names = {
        'VLM_URL',
        'GEMMA_URL',
        'OPENWEBUI_BASE_URL',
        'OPENWEBUI_MODEL',
        'OPENWEBUI_API_KEY',
        'VLM_IMAGES_PER_CALL',
        'GEMMA_IMAGES_PER_CALL',
        'VLM_HTTPX_MAX_CONNECTIONS',
        'GEMMA_HTTPX_MAX_CONNECTIONS',
        'VLM_HTTPX_KEEPALIVE',
        'GEMMA_HTTPX_KEEPALIVE',
        'SAM3_URL',
        'SAM3_URLS',
        'SAM3_HTTPX_MAX_CONNECTIONS',
        'SAM3_HTTPX_KEEPALIVE',
        'SAM3_SKIP_VLM_VERIFY_SCORE',
        'SAM3_SKIP_GEMMA_VERIFY_SCORE',
        'SAM_WORKER_VLM_CONCURRENCY',
        'SAM_WORKER_GEMMA_CONCURRENCY',
        'SAM_WORKER_VLM_VISIBLE_CONCURRENCY',
        'SAM_WORKER_GEMMA_VISIBLE_CONCURRENCY',
        'SAM_WORKER_METRICS_PORT',
        'OP_REGION_DETECTION_SAM_TEXT_PROMPT',
        'GEMMA_CROP_CACHE_DIR',
        # Bake-offs score every class.
        'OP_BAKEOFF_PROFILE_TARGET_CLASS_ID',
        'OP_BAKEOFF_PROFILE_TARGET_CLASS_NAME',
    }
    assert expected_old_names <= set(RETIRED_ENV)


@pytest.mark.parametrize('old_name', sorted(RETIRED_ENV))
def test_each_retired_name_raises_naming_its_replacement(
    old_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(old_name, 'x')
    with pytest.raises(RuntimeError, match=RETIRED_ENV[old_name]):
        reject_retired_env()


def test_no_retired_names_set_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in RETIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    reject_retired_env()  # must not raise


def test_error_names_both_old_and_new(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SAM3_URL', 'http://old-sam3:8000')
    with pytest.raises(RuntimeError) as exc_info:
        reject_retired_env()
    msg = str(exc_info.value)
    assert 'SAM3_URL' in msg
    assert 'OP_SEGMENTER_URL' in msg


def test_multiple_retired_names_set_reports_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SAM3_URL', 'x')
    monkeypatch.setenv('GEMMA_URL', 'y')
    with pytest.raises(RuntimeError) as exc_info:
        reject_retired_env()
    msg = str(exc_info.value)
    assert 'SAM3_URL' in msg
    assert 'GEMMA_URL' in msg


def test_removed_bakeoff_target_class_env_says_what_to_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_TARGET_CLASS_NAME', 'object')
    with pytest.raises(RuntimeError) as exc_info:
        reject_retired_env()
    msg = str(exc_info.value)
    assert 'OP_BAKEOFF_PROFILE_TARGET_CLASS_NAME was removed' in msg
    assert 'OP_BAKEOFF_PROFILE_CLASS_FILTER' in msg
    assert 'renamed' not in msg


def test_bakeoff_evaluator_entry_point_rejects_retired_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import sys

    from scripts.curation.bakeoff import bakeoff_runner

    monkeypatch.setenv('OP_BAKEOFF_PROFILE_TARGET_CLASS_ID', '0')
    monkeypatch.setattr(sys, 'argv', ['bakeoff_runner', '--job', str(tmp_path / 'none.json')])
    with pytest.raises(RuntimeError, match='OP_BAKEOFF_PROFILE_TARGET_CLASS_ID'):
        bakeoff_runner.main()
