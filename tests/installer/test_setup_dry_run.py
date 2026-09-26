"""Dry-run / security-invariant tests for setup-openprocessor.sh (installer
plan section 9.2). No Docker, no network, no GPU: --dry-run makes every
mutating docker call print with a DRY: prefix instead of executing."""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / 'setup-openprocessor.sh'


def run_installer(
    args: list[str], *, env_extra: dict[str, str], tmp_path: Path
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(env_extra)
    env.setdefault('OP_UNATTENDED', '1')
    env.setdefault('OP_DRY_RUN', '1')
    return subprocess.run(
        ['bash', str(SCRIPT), *args],
        check=False,
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_dry_run_never_calls_a_mutating_docker_subcommand(tmp_path: Path) -> None:
    result = run_installer(
        ['--dir', 'inst', '--project', 'opinst-unittest', '--tiers', 'core'],
        env_extra={},
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    for banned in (' run ', ' up ', ' pull ', ' down ', ' rm ', ' restart '):
        # every mutating compose call must be prefixed DRY: and not
        # actually executed -- the dry-run dc()/dc_cw() print instead of
        # exec'ing, so there should be no literal container-affecting
        # side effect; the DRY: prefix marks that it was intercepted.
        for line in result.stdout.splitlines():
            if banned.strip() in line and 'docker compose' in line:
                assert line.startswith('DRY:'), f'non-dry-run compose call: {line}'


def test_every_dry_docker_compose_line_has_dash_p(tmp_path: Path) -> None:
    result = run_installer(
        ['--dir', 'inst', '--project', 'opinst-unittest', '--tiers', 'core', '--uninstall'],
        env_extra={},
        tmp_path=tmp_path,
    )
    dry_lines = [ln for ln in result.stdout.splitlines() if ln.startswith('DRY: docker compose')]
    assert dry_lines, 'expected at least one DRY: docker compose line'
    for line in dry_lines:
        assert ' -p ' in line, f'missing -p in: {line}'


def test_unknown_flag_exits_2(tmp_path: Path) -> None:
    result = run_installer(['--not-a-real-flag'], env_extra={}, tmp_path=tmp_path)
    assert result.returncode == 2


def test_cpu_without_control_plane_only_exits_4(tmp_path: Path) -> None:
    result = run_installer(
        ['--dir', 'inst', '--project', 'opinst-unittest', '--cpu'], env_extra={}, tmp_path=tmp_path
    )
    assert result.returncode == 4


def test_cpu_with_control_plane_only_proceeds(tmp_path: Path) -> None:
    result = run_installer(
        ['--dir', 'inst', '--project', 'opinst-unittest', '--cpu', '--control-plane-only'],
        env_extra={},
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert (
        'control-plane-only' in result.stdout.lower()
        or 'NOT a functional inference install' in result.stdout
    )


def test_bind_public_without_env_var_fails_unattended(tmp_path: Path) -> None:
    result = run_installer(
        ['--dir', 'inst', '--project', 'opinst-unittest', '--bind', '0.0.0.0', '--tiers', 'core'],
        env_extra={},
        tmp_path=tmp_path,
    )
    assert result.returncode != 0
    assert 'OP_ALLOW_PUBLIC_BIND' in (result.stdout + result.stderr)


def test_bind_public_with_env_var_succeeds(tmp_path: Path) -> None:
    result = run_installer(
        ['--dir', 'inst', '--project', 'opinst-unittest', '--bind', '0.0.0.0', '--tiers', 'core'],
        env_extra={'OP_ALLOW_PUBLIC_BIND': '1'},
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_unknown_tier_rejected(tmp_path: Path) -> None:
    result = run_installer(
        ['--dir', 'inst', '--project', 'opinst-unittest', '--tiers', 'not-a-real-tier'],
        env_extra={},
        tmp_path=tmp_path,
    )
    assert result.returncode != 0


def test_tier_dependency_closure_pulls_in_core(tmp_path: Path) -> None:
    result = run_installer(
        ['--dir', 'inst', '--project', 'opinst-unittest', '--tiers', 'vlm'],
        env_extra={},
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert 'tiers: core curation vlm' in result.stdout


def test_project_collision_exits_3(tmp_path: Path) -> None:
    (tmp_path / 'other_dir').mkdir()
    result = run_installer(
        ['--dir', 'inst', '--project', 'openprocessor', '--tiers', 'core'],
        env_extra={
            'OP_TEST_COMPOSE_PROJECTS': str((tmp_path / 'other_dir').resolve()),
        },
        tmp_path=tmp_path,
    )
    assert result.returncode == 3


def test_uninstall_purge_requires_confirm_purge_env_unattended(tmp_path: Path) -> None:
    result = run_installer(
        ['--dir', 'inst', '--project', 'opinst-unittest', '--uninstall', '--purge-data'],
        env_extra={},
        tmp_path=tmp_path,
    )
    assert result.returncode != 0
    assert 'OP_CONFIRM_PURGE' in (result.stdout + result.stderr)


def test_uninstall_purge_data_never_lists_source_root_outside_install_dir(tmp_path: Path) -> None:
    outside = tmp_path / 'outside_source'
    outside.mkdir()
    result = run_installer(
        ['--dir', 'inst', '--project', 'opinst-unittest', '--uninstall', '--purge-data'],
        env_extra={
            'OP_CONFIRM_PURGE': 'opinst-unittest',
            'OP_SOURCE_ROOT_HOST': str(outside.resolve()),
        },
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert str(outside.resolve()) not in result.stdout


def test_hf_token_file_permissions_enforced(tmp_path: Path) -> None:
    token_file = tmp_path / 'hf_token'
    token_file.write_text('hf_TESTSECRET123456\n')
    token_file.chmod(0o644)  # too open
    result = subprocess.run(
        ['bash', '-c', f'source {SCRIPT}; read_hf_token_unattended'],
        check=False,
        env={**os.environ, 'HF_TOKEN_FILE': str(token_file)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'hf_TESTSECRET123456' not in result.stdout
    assert 'hf_TESTSECRET123456' not in result.stderr


def test_hf_token_file_ok_permissions_reads_token(tmp_path: Path) -> None:
    token_file = tmp_path / 'hf_token'
    token_file.write_text('hf_TESTSECRET123456\n')
    token_file.chmod(0o600)
    result = subprocess.run(
        ['bash', '-c', f'source {SCRIPT}; read_hf_token_unattended'],
        check=False,
        env={**os.environ, 'HF_TOKEN_FILE': str(token_file)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == 'hf_TESTSECRET123456'


def test_hf_token_never_appears_in_process_argv_during_store(tmp_path: Path) -> None:
    """store_hf_token must never hand the token to an external process as a
    literal argv element (e.g. `sed -i "s|...|$tok|"`). Proven here by
    running under `bash -x` (which echoes every command about to run,
    including argv) and asserting the token never appears in that trace
    except as a bash-internal variable expansion inside upsert_env_var's
    own `printf` (which writes it to the file, not to another process's
    argv)."""
    env_file = tmp_path / '.env'
    env_file.write_text('FOO=bar\n')
    secret = 'hf_TESTSECRET123456'  # gitleaks:allow
    result = subprocess.run(
        ['bash', '-x', '-c', f'source {SCRIPT}; store_hf_token "{env_file}" "{secret}"'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert secret in env_file.read_text()
    # The trace may show `printf '%s=%s\n' "$key" "$value"` (unresolved
    # var names) but must never show the literal secret being passed to
    # an *external* command like sed/curl/echo as a bare argument.
    for line in result.stderr.splitlines():
        if secret not in line:
            continue
        # The bash -x trace legitimately shows our own `store_hf_token`
        # call and the internal `printf`/`mv` that write the file. The one
        # thing that must never appear is an *external* command (sed,
        # curl, echo, ...) receiving the secret as a literal argument.
        assert not re.match(r'^\+\s*(sed|curl|echo|awk)\b', line.strip()), (
            f"secret leaked into an external command's argv trace: {line}"
        )
    assert env_file.stat().st_mode & 0o777 == 0o600


def test_env_file_permissions_are_600_after_upsert(tmp_path: Path) -> None:
    env_file = tmp_path / '.env'
    env_file.write_text('FOO=bar\n')
    env_file.chmod(0o644)
    result = subprocess.run(
        ['bash', '-c', f'source {SCRIPT}; upsert_env_var "{env_file}" "HF_TOKEN" "hf_x"'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    mode = stat.S_IMODE(env_file.stat().st_mode)
    assert mode == 0o600


def test_upsert_env_var_replaces_existing_key_without_duplicating(tmp_path: Path) -> None:
    env_file = tmp_path / '.env'
    env_file.write_text('FOO=old\nBAR=baz\n')
    subprocess.run(
        ['bash', '-c', f'source {SCRIPT}; upsert_env_var "{env_file}" "FOO" "new"'],
        capture_output=True,
        text=True,
        check=True,
    )
    text = env_file.read_text()
    assert text.count('FOO=') == 1
    assert 'FOO=new' in text
    assert 'BAR=baz' in text


def test_default_bind_address_is_loopback() -> None:
    text = SCRIPT.read_text()
    assert 'OP_BIND_ADDRESS="${OP_BIND_ADDRESS:-127.0.0.1}"' in text


def test_images_lock_rejects_latest_and_requires_digest() -> None:
    lock = (REPO_ROOT / 'images.lock').read_text()
    for raw_line in lock.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        assert ':latest' not in line, f'images.lock has a :latest entry: {line}'
        assert '@sha256:' in line, f'images.lock entry has no digest: {line}'
