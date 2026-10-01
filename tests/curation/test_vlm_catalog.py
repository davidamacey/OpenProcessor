"""The local VLM catalog (W9.7): one TSV, two readers. Bash picks (installer
and CLI), Python only displays and computes ``fits``; this pins that both see
the same rows and that the catalog contains what the docs promise."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.services.labeling.vlm_catalog import (
    COLUMNS,
    DEFAULT_CATALOG_PATH,
    catalog_entry,
    catalog_entry_for_root,
    desired_command,
    fits,
    gpu_total_gb_from_env,
    load_catalog,
    local_vlm_status,
)


ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / 'scripts' / 'lib' / 'vlm_catalog.sh'


def _bash(script: str) -> str:
    result = subprocess.run(
        ['bash', '-c', f'set -euo pipefail; source {LIB}; {script}'],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return result.stdout


def test_the_default_catalog_is_the_shipped_file() -> None:
    assert DEFAULT_CATALOG_PATH == ROOT / 'examples' / 'vlm' / 'catalog.tsv'
    assert load_catalog()


def test_every_column_is_read() -> None:
    header = next(
        line for line in DEFAULT_CATALOG_PATH.read_text().splitlines() if not line.startswith('#')
    )
    assert tuple(header.split('\t')) == COLUMNS


def test_bash_and_python_see_the_same_ids_in_the_same_order() -> None:
    bash_ids = _bash('vlm_catalog_ids').split()
    assert bash_ids == [e.id for e in load_catalog()]


@pytest.mark.parametrize(
    ('field', 'attr'),
    [
        ('hf_repo', 'hf_repo'),
        ('vram_gb', 'vram_gb'),
        ('served_context', 'max_model_len'),
        ('max_context', 'context_max'),
        ('max_images', 'max_images'),
        ('status', 'status'),
        ('rank', 'rank'),
        ('vllm_image_key', 'vllm_image_key'),
        ('reasoning_parser', 'reasoning_parser'),
        ('chat_template', 'chat_template'),
        ('extra_args', 'extra_args'),
    ],
)
def test_every_field_bash_reads_matches_what_python_reads(field: str, attr: str) -> None:
    for entry in load_catalog():
        from_bash = _bash(f'vlm_catalog_field {entry.id} {field}').strip()
        value = getattr(entry, attr)
        if isinstance(value, float):
            assert float(from_bash) == value, (entry.id, field)
        else:
            assert from_bash == str(value), (entry.id, field)


def test_licence_and_gating_agree() -> None:
    for entry in load_catalog():
        assert _bash(f'vlm_catalog_field {entry.id} licence').strip().lower() == entry.license
        assert (_bash(f'vlm_catalog_field {entry.id} gated').strip() == 'true') is entry.gated


def test_only_permissive_ungated_models_are_listed_and_exactly_one_is_tested() -> None:
    entries = load_catalog()
    assert all(e.license == 'apache-2.0' for e in entries)
    assert not any(e.gated for e in entries)
    assert [e.id for e in entries if e.status == 'tested'] == ['gemma-4-e4b']
    assert {e.status for e in entries} <= {'tested', 'to_verify'}
    # the default served model carries the parser and template the compose
    # file used to hardcode
    default = catalog_entry('gemma-4-e4b')
    assert default is not None
    assert default.reasoning_parser == 'gemma4'
    assert default.chat_template.endswith('tool_chat_template_gemma4.jinja')
    assert default.text_reading_verified is True


def test_a_catalog_missing_a_column_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / 'catalog.tsv'
    bad.write_text('id\thf_repo\nx\ty\n')
    with pytest.raises(ValueError, match='missing column'):
        load_catalog(bad)


def test_a_reordered_catalog_is_read_by_name_not_position(tmp_path: Path) -> None:
    lines = [ln for ln in DEFAULT_CATALOG_PATH.read_text().splitlines() if not ln.startswith('#')]
    header = lines[0].split('\t')
    order = list(reversed(range(len(header))))
    shuffled = ['\t'.join(row.split('\t')[i] for i in order) for row in lines]
    path = tmp_path / 'catalog.tsv'
    path.write_text('\n'.join(shuffled) + '\n')
    assert [e.id for e in load_catalog(path)] == [e.id for e in load_catalog()]


def test_lookup_by_id_and_by_served_root() -> None:
    gemma = catalog_entry('gemma-4-e4b')
    assert gemma is not None
    assert catalog_entry_for_root(gemma.hf_repo) == gemma
    assert catalog_entry('nope') is None
    assert catalog_entry(None) is None
    assert catalog_entry_for_root(None) is None
    assert catalog_entry_for_root('someone/else') is None


def test_fits_compares_vram_with_the_card() -> None:
    gemma = catalog_entry('gemma-4-e4b')
    assert gemma is not None
    assert fits(gemma, 48.0) is True
    assert fits(gemma, gemma.vram_gb) is True
    assert fits(gemma, gemma.vram_gb - 0.5) is False
    assert fits(gemma, None) is None


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        (None, None),
        ('', None),
        ('  ', None),
        ('abc', None),
        ('0', None),
        ('-5', None),
        ('49140', 49140 / 1024),
    ],
)
def test_the_card_size_env_is_read_in_mib(raw: str | None, expected: float | None) -> None:
    assert gpu_total_gb_from_env(raw) == expected


def test_a_restart_is_required_until_the_probe_reports_the_desired_root() -> None:
    qwen = catalog_entry('qwen3-vl-4b')
    gemma = catalog_entry('gemma-4-e4b')
    assert qwen is not None
    assert gemma is not None

    def status(root: str | None, desired: str | None) -> dict:
        return local_vlm_status(
            endpoint_name='env',
            served_model='local-vlm',
            served_root=root,
            served_max_model_len=8192,
            desired={'catalog_id': desired, 'requested_at': 't'} if desired else None,
            gpu_total_gb=48.0,
        )

    waiting = status(gemma.hf_repo, qwen.id)
    assert waiting['restart_required'] is True
    assert waiting['poll_after_s'] == 10
    assert waiting['desired']['command'] == desired_command(qwen.id)
    assert waiting['served']['catalog_id'] == gemma.id
    assert waiting['can_restart_from_api'] is False

    applied = status(qwen.hf_repo, qwen.id)
    assert applied['restart_required'] is False
    assert applied['poll_after_s'] is None
    assert applied['served']['catalog_id'] == qwen.id

    # never probed: a desired model is still "not applied"
    assert status(None, qwen.id)['restart_required'] is True
    assert status(gemma.hf_repo, None)['restart_required'] is False


def test_a_deployment_without_the_in_compose_vlm_is_not_configured() -> None:
    result = local_vlm_status(
        endpoint_name='',
        served_model=None,
        served_root=None,
        served_max_model_len=None,
        desired=None,
        gpu_total_gb=None,
    )
    assert result['configured'] is False
    assert result['served'] is None
    assert result['endpoint'] is None
