"""The written .env and summary, tier selection, flag validation and exit codes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _setup_dry_run_helpers import configure, dry, env_file
from installer_harness import GPU_HOST, PROJECT, RELEASE


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed


# --- .env content ------------------------------------------------------------------


def _meminfo(shimmed: Shimmed, ram_gib: int) -> str:
    f = shimmed.root / f'meminfo_{ram_gib}'
    f.write_text(f'MemTotal:       {ram_gib * 1024 * 1024} kB\nMemFree:  1 kB\n')
    return str(f)


@pytest.mark.parametrize(
    ('ram_gib', 'heap'),
    [(2, '2g'), (16, '8g'), (32, '16g'), (64, '30g'), (256, '30g')],
)
def test_opensearch_heap_is_half_of_ram_clamped(shimmed: Shimmed, ram_gib: int, heap: str) -> None:
    result = configure(shimmed, OP_MEMINFO_PATH=_meminfo(shimmed, ram_gib))
    assert result.returncode == 0, result.stderr
    assert env_file(shimmed)['OPENSEARCH_HEAP'] == heap


def test_summary_prints_heap_and_soft_shard_budget(shimmed: Shimmed) -> None:
    result = configure(shimmed, OP_MEMINFO_PATH=_meminfo(shimmed, 16))
    assert result.returncode == 0, result.stderr
    assert 'OpenSearch  : heap 8g, soft shard budget 160 (20 shards per heap GB)' in result.stdout


def test_summary_budget_uses_the_user_heap_and_shard_knob(shimmed: Shimmed) -> None:
    mem = _meminfo(shimmed, 16)
    assert configure(shimmed, OP_MEMINFO_PATH=mem).returncode == 0
    env_path = shimmed.root / 'inst' / '.env'
    text = env_path.read_text().replace('OPENSEARCH_HEAP=8g', 'OPENSEARCH_HEAP=3g', 1)
    env_path.write_text(text + 'OP_SHARDS_PER_HEAP_GB=25\n')
    result = configure(shimmed, OP_MEMINFO_PATH=mem)
    assert result.returncode == 0, result.stderr
    assert env_file(shimmed)['OPENSEARCH_HEAP'] == '3g'
    assert 'OpenSearch  : heap 3g, soft shard budget 75 (25 shards per heap GB)' in result.stdout


def test_rerun_is_idempotent_and_keeps_user_values(shimmed: Shimmed) -> None:
    assert configure(shimmed).returncode == 0
    env_path = shimmed.root / 'inst' / '.env'
    first = env_path.read_text()
    assert configure(shimmed).returncode == 0
    assert env_path.read_text() == first, 'a re-run changed .env'
    assert list((shimmed.root / 'inst' / 'backups').iterdir()), 'a re-run must back up first'

    edited = first.replace('OPENSEARCH_HEAP=', 'OPENSEARCH_HEAP=3g\n#was ', 1)
    env_path.write_text(edited)
    result = configure(shimmed)
    assert result.returncode == 0, result.stderr
    assert env_file(shimmed)['OPENSEARCH_HEAP'] == '3g'
    assert 'keeping your OPENSEARCH_HEAP=3g' in result.stdout


def test_vlm_tier_writes_the_catalog_keys(shimmed: Shimmed) -> None:
    result = configure(shimmed, tiers='vlm')
    assert result.returncode == 0, result.stderr
    env = env_file(shimmed)
    lock = dict(
        ln.split('=', 1)
        for ln in (shimmed.root / 'inst' / 'images.lock').read_text().splitlines()
        if ln and not ln.startswith('#')
    )
    assert env['VLM_SERVED_MODEL_NAME'] == 'local-vlm'
    assert env['OP_VLM_MODEL'] == 'local-vlm'
    assert env['OP_VLM_URL'] == 'http://vlm:8000/v1'
    assert env['VLM_IMAGE'] == lock['vlm_gemma4']
    assert env['VLM_GPU_MEMORY_UTILIZATION'] == '0.42'
    assert env['VLM_LIMIT_MM_IMAGES'] == env['OP_VLM_MAX_IMAGES_PER_CALL'] == '8'
    assert 'vlm' in env['COMPOSE_PROFILES'].split(',')


def test_vlm_tier_refused_where_no_tested_entry_fits(shimmed: Shimmed) -> None:
    shimmed.gpus('0, NVIDIA GeForce RTX 4090, 24564, 0, 8.9\n')
    result = dry(shimmed, tiers='vlm')
    assert result.returncode == 4
    assert 'no tested catalog entry fits' in result.stderr


def test_unverified_vlm_only_by_explicit_id_and_force(shimmed: Shimmed) -> None:
    shimmed.gpus('0, NVIDIA GeForce RTX 4090, 24564, 0, 8.9\n')
    assert dry(shimmed, '--vlm-model-id', 'qwen2.5-vl-7b-awq', tiers='vlm').returncode == 4
    result = configure(shimmed, '--vlm-model-id', 'qwen2.5-vl-7b-awq', '--force', tiers='vlm')
    assert result.returncode == 0, result.stderr
    assert 'unverified/experimental' in result.stderr
    assert env_file(shimmed)['VLM_CATALOG_ID'] == 'qwen2.5-vl-7b-awq'


def test_host_shaped_three_gpus_write_the_placement(shimmed: Shimmed, tmp_path: Path) -> None:
    shimmed.gpus(GPU_HOST)
    tok = tmp_path / 'tok'
    tok.write_text('hf_TESTSECRET1234567890\n')
    tok.chmod(0o600)
    result = configure(shimmed, tiers='segmenter,vlm,trainer', HF_TOKEN_FILE=str(tok))
    assert result.returncode == 0, result.stderr
    env = env_file(shimmed)
    assert env['TRITON_GPU_ID'] == env['API_GPU_ID'] == '1'
    assert env['VLM_GPU_ID'] == '0'
    assert env['SEGMENTER_GPU_ID'] == env['OP_TRAIN_GPU_ORDER'] == env['EVALUATOR_GPU_ID'] == '2'
    assert env['OP_GPU_ALLOWED_IDS'] == '0,1,2'
    assert env['OP_GPU_LABELS'].startswith('0=NVIDIA RTX A6000,1=NVIDIA GeForce RTX 3080 Ti')


def test_segmenter_tier_sets_api_segmenter_url(shimmed: Shimmed, tmp_path: Path) -> None:
    # /models/status builds the sam3 row from OP_SEGMENTER_URL only (#113).
    shimmed.gpus(GPU_HOST)
    tok = tmp_path / 'tok'
    tok.write_text('hf_TESTSECRET1234567890\n')
    tok.chmod(0o600)
    result = configure(shimmed, tiers='segmenter', HF_TOKEN_FILE=str(tok))
    assert result.returncode == 0, result.stderr
    assert env_file(shimmed)['OP_SEGMENTER_URL'] == 'http://segmenter:8000'


def test_no_segmenter_tier_leaves_segmenter_url_unset(shimmed: Shimmed) -> None:
    result = configure(shimmed)
    assert result.returncode == 0, result.stderr
    assert not env_file(shimmed).get('OP_SEGMENTER_URL')


def _map_count(shimmed: Shimmed, value: int) -> str:
    f = shimmed.root / f'map_count_{value}'
    f.write_text(f'{value}\n')
    return str(f)


def test_low_vm_max_map_count_warns_with_the_fix(shimmed: Shimmed) -> None:
    result = configure(shimmed, OP_MAX_MAP_COUNT_PATH=_map_count(shimmed, 65530))
    assert result.returncode == 0, result.stderr
    assert 'vm.max_map_count is 65530' in result.stderr
    assert 'sysctl -w vm.max_map_count=262144' in result.stderr


def test_sufficient_vm_max_map_count_is_silent(shimmed: Shimmed) -> None:
    result = configure(shimmed, OP_MAX_MAP_COUNT_PATH=_map_count(shimmed, 262144))
    assert result.returncode == 0, result.stderr
    assert 'vm.max_map_count' not in result.stderr


# --- HF token ----------------------------------------------------------------------


def test_gated_tier_without_token_exits_6(shimmed: Shimmed) -> None:
    result = dry(shimmed, tiers='segmenter')
    assert result.returncode == 6
    assert 'HF_TOKEN_FILE' in result.stderr


def test_rejected_token_exits_6_with_licence_hint(shimmed: Shimmed, tmp_path: Path) -> None:
    tok = tmp_path / 'tok'
    tok.write_text('hf_TESTSECRET1234567890\n')
    tok.chmod(0o600)
    shimmed.flag('hf_code', '403')
    result = dry(shimmed, tiers='segmenter', HF_TOKEN_FILE=str(tok))
    assert result.returncode == 6
    assert 'accept the licence at https://huggingface.co/facebook/sam3' in result.stderr


def test_core_never_asks_for_a_token(shimmed: Shimmed) -> None:
    result = dry(shimmed)
    assert result.returncode == 0
    assert 'no HuggingFace token needed' in result.stdout
    assert not [ln for ln in shimmed.log_lines('curl') if 'huggingface.co' in ln]


# --- flags -------------------------------------------------------------------------


def test_unknown_flag_exits_2(shimmed: Shimmed) -> None:
    assert shimmed.run(['--not-a-real-flag']).returncode == 2


def test_missing_flag_value_exits_2_with_a_message(shimmed: Shimmed) -> None:
    result = shimmed.run(['--dir'])
    assert result.returncode == 2
    assert '--dir needs a value' in result.stderr
    assert 'unbound variable' not in result.stderr


def test_invalid_values_exit_2(shimmed: Shimmed) -> None:
    for args in (
        ['--project', 'Bad Name'],
        ['--port-base', 'abc'],
        ['--profile', 'huge'],
        ['--bind', 'example.com'],
        ['--bind', '::1'],
        ['--image-tag', 'latest'],
        ['--purge-data'],
    ):
        assert shimmed.run(['--dry-run', *args]).returncode == 2, args


def test_unattended_truthy_values(shimmed: Shimmed) -> None:
    for value in ('true', 'yes', '1'):
        result = shimmed.run(
            [
                '--dry-run',
                '--dir',
                'inst',
                '--project',
                PROJECT,
                '--version',
                RELEASE,
                '--tiers',
                'core',
                '--bind',
                '0.0.0.0',
            ],
            OP_UNATTENDED=value,
        )
        assert 'OP_ALLOW_PUBLIC_BIND=1' in result.stderr, value


def test_cpu_exits_4_with_the_honest_message(shimmed: Shimmed) -> None:
    result = dry(shimmed, '--cpu')
    assert result.returncode == 4
    assert 'no CPU inference path' in result.stderr


def test_no_gpu_exits_4(shimmed: Shimmed) -> None:
    shimmed.gpus(None)
    assert dry(shimmed).returncode == 4


def test_control_plane_only_starts_api_and_opensearch_only(shimmed: Shimmed) -> None:
    shimmed.gpus(None)
    result = dry(shimmed, '--cpu', '--control-plane-only')
    assert result.returncode == 0, result.stderr
    dry_compose = [ln for ln in result.stdout.splitlines() if ln.startswith('DRY: docker compose')]
    assert any(ln.endswith(' up -d --remove-orphans opensearch') for ln in dry_compose)
    assert any(ln.endswith(' up -d --no-deps api') for ln in dry_compose)
    assert not any('triton-server' in ln for ln in dry_compose)
    assert all('docker-compose.cpu.yml' in ln for ln in dry_compose)
    assert configure(shimmed, '--cpu', '--control-plane-only').returncode == 0
    overlay = (shimmed.root / 'inst' / 'docker-compose.cpu.yml').read_text()
    assert '!reset' in overlay
    # `./openprocessor start` runs a plain `up -d`: Triton must stay off and the
    # API must not wait for it or be marked unhealthy by its own /health 503.
    assert 'triton-server' in overlay
    assert 'depends_on: !override' in overlay
    assert 'localhost:8000/health' not in overlay
    assert 'NOT a functional inference install' in result.stdout + result.stderr


# --- messages a first-time user reads -----------------------------------------------


def test_cpu_alone_says_so_instead_of_claiming_no_gpu(shimmed: Shimmed) -> None:
    result = dry(shimmed, '--cpu')
    assert result.returncode == 4
    assert '--cpu on its own' in result.stderr
    assert 'No usable NVIDIA GPU' not in result.stderr


def test_bad_version_is_refused_before_any_preflight(shimmed: Shimmed) -> None:
    result = dry(shimmed, '--version', '1.2')
    assert result.returncode == 2
    assert 'Preflight' not in result.stdout


def test_control_plane_only_install_does_not_mention_gpu_or_triton_downloads(
    shimmed: Shimmed,
) -> None:
    shimmed.gpus(None)
    result = dry(shimmed, '--cpu', '--control-plane-only')
    assert result.returncode == 0, result.stderr
    assert 'nvidia runtime' not in result.stdout + result.stderr
    assert 'Triton image alone' not in result.stdout


def test_control_plane_only_summary_points_at_the_liveness_url_not_the_503(
    shimmed: Shimmed,
) -> None:
    shimmed.gpus(None)
    result = configure(shimmed, '--cpu', '--control-plane-only')
    assert result.returncode == 0, result.stderr
    urls = [ln for ln in result.stdout.splitlines() if ln.startswith('    API ')]
    health = [ln for ln in urls if '/health' in ln]
    assert all('503' in ln for ln in health), urls
    assert any(ln.rstrip().split()[2].endswith('/') for ln in urls if 'live' in ln), urls


def test_a_default_install_summary_keeps_the_plain_health_url(shimmed: Shimmed) -> None:
    result = configure(shimmed)
    assert result.returncode == 0, result.stderr
    health = [ln for ln in result.stdout.splitlines() if ln.startswith('    API health')]
    assert health
    assert '503' not in health[0]


def test_no_start_hint_skips_model_install_for_control_plane_only(shimmed: Shimmed) -> None:
    shimmed.gpus(None)
    result = configure(shimmed, '--cpu', '--control-plane-only')
    assert result.returncode == 0, result.stderr
    assert 'run ./openprocessor start' in result.stdout
    assert 'models install' not in result.stdout


def test_low_disk_error_names_the_install_dir_even_in_a_dry_run(shimmed: Shimmed) -> None:
    shimmed.flag('low_disk')
    result = dry(shimmed)
    assert result.returncode != 0
    assert str(shimmed.root / 'inst') in result.stderr
    assert 'have 4 GB' in result.stderr


def test_a_dry_run_checks_free_disk_on_the_target_filesystem(shimmed: Shimmed) -> None:
    # The dry run works on a scratch copy under TMPDIR; the free space that
    # matters is the install dir's own filesystem (#363).
    import shutil
    import tempfile

    scratch_parent = tempfile.mkdtemp(prefix='op-dry-tmp-')
    try:
        (shimmed.state / 'low_disk_path').write_text(str(shimmed.root))
        result = dry(shimmed, TMPDIR=scratch_parent)
    finally:
        shutil.rmtree(scratch_parent, ignore_errors=True)
    assert result.returncode != 0
    assert str(shimmed.root / 'inst') in result.stderr
    assert 'have 4 GB' in result.stderr


def test_install_creates_every_bind_mount_source_as_the_user(shimmed: Shimmed) -> None:
    # Docker creates a missing bind source root-owned on the first `up`, which
    # later blocks `openprocessor vlm key set` and removing the install dir.
    assert configure(shimmed).returncode == 0
    inst = shimmed.root / 'inst'
    assert (inst / 'secrets' / 'vlm').is_dir()
    assert oct((inst / 'secrets').stat().st_mode & 0o777) == '0o700'
    assert (inst / 'data' / 'source').is_dir()
