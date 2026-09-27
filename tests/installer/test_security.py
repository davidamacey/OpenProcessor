"""Security invariants of setup-openprocessor.sh (installer plan section 7)."""

from __future__ import annotations

import os
import stat
import subprocess
from typing import TYPE_CHECKING

import pytest
from installer_harness import PROJECT, RELEASE, SCRIPT


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed


SECRET = 'hf_TESTSECRET1234567890abc'  # gitleaks:allow
ARGV_TOOLS = [
    'sed',
    'awk',
    'grep',
    'cut',
    'mv',
    'cp',
    'chmod',
    'mktemp',
    'cat',
    'install',
    'tee',
    'env',
    'stat',
    'sha256sum',
    'tar',
    'find',
    'sort',
    'head',
    'tail',
    'tr',
    'mkdir',
    'rm',
    'realpath',
    'dirname',
    'date',
    'id',
    'sleep',
    'bash',
]


def source_and(snippet: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['bash', '-c', f'OP_SOURCE_ONLY=1 source "{SCRIPT}"; {snippet}'],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=30,
        start_new_session=True,
    )


def token_file(tmp_path: Path, content: str = SECRET + '\n', mode: int = 0o600) -> Path:
    f = tmp_path / 'hf_token'
    f.write_text(content)
    f.chmod(mode)
    return f


# --- the token never leaks ------------------------------------------------------


@pytest.mark.parametrize('how', ['file_xtrace', 'shellopts_xtrace', 'env_var', 'piped_xtrace'])
def test_token_never_in_output_trace_log_or_any_argv(
    shimmed: Shimmed, tmp_path: Path, how: str
) -> None:
    shimmed.wrap_argv_loggers(ARGV_TOOLS)
    shimmed.flag('allow_mutations')
    args = [
        '--no-start',
        '--unattended',
        '--dir',
        'inst',
        '--project',
        PROJECT,
        '--version',
        RELEASE,
        '--tiers',
        'segmenter',
    ]
    env: dict[str, str] = {}
    if how == 'env_var':
        env['HF_TOKEN'] = SECRET
    else:
        env['HF_TOKEN_FILE'] = str(token_file(tmp_path))
    if how == 'shellopts_xtrace':
        env['SHELLOPTS'] = 'xtrace'
    result = shimmed.run(
        args, xtrace=how in ('file_xtrace', 'piped_xtrace'), piped=how == 'piped_xtrace', **env
    )
    assert result.returncode == 0, result.stderr[-3000:]
    inst = shimmed.root / 'inst'
    assert f'HF_TOKEN={SECRET}' in (inst / '.env').read_text()
    assert SECRET not in result.stdout
    assert SECRET not in result.stderr, 'token visible in the xtrace/stderr stream'
    # The log/console redactor would mask a traced token; seeing its marker
    # means the token was traced, which is a leak in itself.
    assert 'REDACTED' not in result.stdout + result.stderr, (
        'token reached the trace (masked only by the redactor)'
    )
    assert SECRET not in (inst / '.install' / 'install.log').read_text()
    argv_log = shimmed.log.read_text()
    assert 'curl ' in argv_log
    assert 'sed ' in argv_log, 'argv loggers did not run'
    assert SECRET not in argv_log, 'token passed to an external process as an argument'
    hf_calls = [ln for ln in shimmed.log_lines('curl') if 'huggingface.co' in ln]
    assert hf_calls
    assert all('-H @' in ln for ln in hf_calls)


def test_install_log_is_redacted(bash) -> None:
    result = bash(
        f'OP_SOURCE_ONLY=1 source "{SCRIPT}"; printf "%s\\n" "tok {SECRET}" "Authorization: Bearer abc.def" '
        '"export OPENAI_API_KEY=sk-123" "HF_TOKEN=hf_x" "DB_PASSWORD=p4ss" "fine=value" | _op_redact'
    )
    out = result.stdout
    assert SECRET not in out
    assert 'abc.def' not in out
    assert 'sk-123' not in out
    assert 'p4ss' not in out
    assert 'fine=value' in out


# --- HF_TOKEN_FILE permissions ------------------------------------------------------


@pytest.mark.parametrize('mode', [0o640, 0o460, 0o077, 0o644, 0o604, 0o620])
def test_hf_token_file_with_group_or_world_bits_is_rejected(tmp_path: Path, mode: int) -> None:
    f = token_file(tmp_path, mode=mode)
    result = source_and(
        'read_hf_token_unattended && printf "%s" "$_OP_HF_TOKEN"', HF_TOKEN_FILE=str(f)
    )
    f.chmod(0o600)
    assert result.returncode != 0
    assert SECRET not in result.stdout + result.stderr
    assert 'too open' in result.stderr


@pytest.mark.parametrize('mode', [0o600, 0o400])
def test_hf_token_file_owner_only_is_accepted(tmp_path: Path, mode: int) -> None:
    f = token_file(tmp_path, mode=mode)
    result = source_and(
        'read_hf_token_unattended && printf "%s" "$_OP_HF_TOKEN"', HF_TOKEN_FILE=str(f)
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == SECRET


def test_empty_hf_token_file_is_rejected(tmp_path: Path) -> None:
    f = token_file(tmp_path, content='')
    assert source_and('read_hf_token_unattended', HF_TOKEN_FILE=str(f)).returncode != 0


def test_pasted_prefix_and_whitespace_are_stripped(tmp_path: Path) -> None:
    f = token_file(tmp_path, content=f'HF_TOKEN={SECRET}  \n')
    result = source_and(
        'read_hf_token_unattended && printf "%s" "$_OP_HF_TOKEN"', HF_TOKEN_FILE=str(f)
    )
    assert result.stdout == SECRET


# --- external VLM URL classification ----------------------------------------------


@pytest.mark.parametrize(
    'url',
    [
        'http://127.0.0.1@evil.example/v1',
        'http://127.0.0.1:8000@evil.example/v1',
        'http://10.evil.example/v1',
        'http://192.168.attacker.net/v1',
        'http://127.0.0.1.evil.example/v1',
        'http://172.32.0.1/v1',
        'http://172.15.0.1/v1',
        'https://api.example.invalid/v1',
        'http://[::1]:8000/v1',
        'http://999.1.1.1/v1',
        'ftp://127.0.0.1/v1',
    ],
)
def test_url_is_external(url: str) -> None:
    result = source_and(f'is_private_url "{url}"')
    assert result.returncode != 0, url


@pytest.mark.parametrize(
    'url',
    [
        'http://127.0.0.1:8000/v1',
        'http://localhost:11434/v1',
        'http://host.docker.internal:1234/v1',
        'http://10.1.2.3/v1',
        'http://192.168.1.5:8000/v1',
        'http://172.16.0.1/v1',
        'http://172.31.255.1/v1',
        'HTTP://LOCALHOST:1/v1',
    ],
)
def test_url_is_private(url: str) -> None:
    assert source_and(f'is_private_url "{url}"').returncode == 0, url


def test_userinfo_url_needs_consent_end_to_end(shimmed: Shimmed) -> None:
    result = shimmed.run(
        [
            '--dry-run',
            '--unattended',
            '--dir',
            'inst',
            '--project',
            PROJECT,
            '--version',
            RELEASE,
            '--tiers',
            'core',
            '--vlm-remote',
            'http://127.0.0.1@evil.example/v1',
            '--vlm-model',
            'm',
        ]
    )
    assert result.returncode == 9
    assert 'OP_ALLOW_EXTERNAL_VLM=1' in result.stderr


# --- .env writes ----------------------------------------------------------------------


def test_upsert_rejects_a_newline_injection(tmp_path: Path) -> None:
    env = tmp_path / '.env'
    env.write_text('A=1\n')
    result = source_and(f'upsert_env_var "{env}" OP_VLM_MODEL $\'m\\nOP_BIND_ADDRESS=0.0.0.0\'')
    assert result.returncode != 0
    assert env.read_text() == 'A=1\n'
    assert list(tmp_path.iterdir()) == [env], 'temp file left behind'


def test_upsert_rejects_a_carriage_return(tmp_path: Path) -> None:
    env = tmp_path / '.env'
    env.write_text('A=1\n')
    assert source_and(f'upsert_env_var "{env}" K $\'v\\r\'').returncode != 0


def test_upsert_is_atomic_600_and_replaces_in_place(tmp_path: Path) -> None:
    env = tmp_path / '.env'
    env.write_text('FOO=old\nBAR=baz\nFOO=dup\n')
    env.chmod(0o644)
    result = source_and(f'upsert_env_var "{env}" FOO new')
    assert result.returncode == 0, result.stderr
    assert env.read_text() == 'FOO=new\nBAR=baz\n'
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_read_env_var_matches_the_key_exactly(tmp_path: Path) -> None:
    env = tmp_path / '.env'
    env.write_text('API_PORTX=1\nAPI.PORT=2\nAPI_PORT=4603\n')
    assert source_and(f'read_env_var "{env}" API_PORT').stdout.strip() == '4603'
    assert source_and(f'read_env_var "{env}" "API.PORT"').stdout.strip() == '2'
    assert source_and(f'read_env_var "{env}" "API_PORT.*"').returncode != 0


def test_vlm_model_with_a_newline_is_rejected(shimmed: Shimmed) -> None:
    result = shimmed.run(
        [
            '--dry-run',
            '--vlm-remote',
            'http://127.0.0.1:9/v1',
            '--vlm-model',
            'm\nOP_BIND_ADDRESS=0.0.0.0',
        ]
    )
    assert result.returncode == 2


# --- test seams are gone ---------------------------------------------------------------


def test_no_env_test_seam_can_disable_a_safety_check() -> None:
    text = SCRIPT.read_text()
    for seam in (
        'OP_TEST_GPU_CSV',
        'OP_TEST_PORTS_IN_USE',
        'OP_TEST_COMPOSE_PROJECTS',
        'OP_TEST_LATEST_REF',
        'OP_TEST_BRANCH_SHA',
    ):
        assert seam not in text
