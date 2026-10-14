"""Bind address consent, remote VLM keys and the Cropwright bind."""

from __future__ import annotations

import stat
from typing import TYPE_CHECKING

from _setup_dry_run_helpers import configure, dry, env_file
from installer_harness import PROJECT, fake_digest


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed


# --- bind / external consent ---------------------------------------------------------


def test_public_bind_unattended_needs_explicit_consent(shimmed: Shimmed) -> None:
    result = dry(shimmed, '--bind', '0.0.0.0')
    assert result.returncode == 9
    assert 'OP_ALLOW_PUBLIC_BIND=1' in result.stderr
    ok = configure(shimmed, '--bind', '0.0.0.0', OP_ALLOW_PUBLIC_BIND='1')
    assert ok.returncode == 0, ok.stderr
    assert env_file(shimmed)['OP_BIND_ADDRESS'] == '0.0.0.0'


def test_localhost_bind_is_normalised_to_an_ip(shimmed: Shimmed) -> None:
    assert configure(shimmed, '--bind', 'localhost').returncode == 0
    assert env_file(shimmed)['OP_BIND_ADDRESS'] == '127.0.0.1'


def test_external_vlm_needs_consent(shimmed: Shimmed) -> None:
    for url in (
        'https://api.example.invalid/v1',
        'http://127.0.0.1@evil.example/v1',
        'http://10.evil.example/v1',
    ):
        result = dry(shimmed, '--vlm-remote', url, '--vlm-model', 'some-model')
        assert result.returncode == 9, url
        assert 'OP_ALLOW_EXTERNAL_VLM=1' in result.stderr, url
    ok = dry(shimmed, '--vlm-remote', 'http://192.168.1.20:8000/v1', '--vlm-model', 'm')
    assert ok.returncode == 0, ok.stderr


def test_remote_vlm_key_lands_only_in_secrets(shimmed: Shimmed, tmp_path: Path) -> None:
    key = 'sk-TESTVLMKEY0123456789'  # gitleaks:allow
    key_file = tmp_path / 'vlm.key'
    key_file.write_text(key)
    key_file.chmod(0o600)
    result = configure(
        shimmed,
        '--vlm-remote',
        'https://api.example.invalid/v1',
        '--vlm-model',
        'm',
        '--vlm-key-file',
        str(key_file),
        OP_ALLOW_EXTERNAL_VLM='1',
    )
    assert result.returncode == 0, result.stderr
    inst = shimmed.root / 'inst'
    secret = inst / 'secrets' / 'vlm' / 'env'
    assert secret.read_text() == key
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    for text in (
        (inst / '.env').read_text(),
        result.stdout,
        result.stderr,
        (inst / '.install' / 'install.log').read_text(),
        shimmed.log.read_text(),
    ):
        assert key not in text


def test_remote_and_local_vlm_are_exclusive(shimmed: Shimmed) -> None:
    result = dry(shimmed, '--vlm-remote', 'http://127.0.0.1:9/v1', '--vlm-model', 'm', tiers='vlm')
    assert result.returncode == 2


# --- Cropwright ----------------------------------------------------------------------


def _cw_env(shimmed: Shimmed) -> str:
    return (shimmed.root / 'inst' / '.env').read_text()


def test_cropwright_is_reachable_from_the_lan_by_default_with_a_warning(shimmed: Shimmed) -> None:
    # Owner decision (plan 7 addendum): the UI serves this computer AND the LAN.
    result = configure(shimmed, tiers='cropwright')
    assert result.returncode == 0, result.stderr
    cw_env = _cw_env(shimmed)
    assert 'CROPWRIGHT_BIND_ADDRESS=0.0.0.0' in cw_env
    assert f'CROPWRIGHT_IMAGE=davidamacey/cropwright@{fake_digest("cropwright")}' in cw_env
    assert 'cropwright' in env_file(shimmed)['COMPOSE_PROFILES'].split(',')
    assert stat.S_IMODE((shimmed.root / 'inst' / '.env').stat().st_mode) == 0o600
    assert not (shimmed.root / 'inst' / 'cropwright').exists()
    # The backend stays on loopback; only Cropwright is on the LAN.
    assert env_file(shimmed)['OP_BIND_ADDRESS'] == '127.0.0.1'
    assert 'reachable from your LAN and has NO login' in result.stderr
    assert 'never port-forward' in result.stderr
    dry_run = dry(shimmed, tiers='cropwright')
    assert dry_run.returncode == 0, dry_run.stderr
    assert not [ln for ln in dry_run.stdout.splitlines() if f'-p {PROJECT}-cw' in ln]


def test_local_only_keeps_cropwright_on_this_computer(shimmed: Shimmed) -> None:
    result = configure(shimmed, '--local-only', tiers='cropwright')
    assert result.returncode == 0, result.stderr
    assert 'CROPWRIGHT_BIND_ADDRESS=127.0.0.1' in _cw_env(shimmed)
    assert 'reachable from your LAN' not in result.stderr


def test_a_shell_bind_cannot_override_the_chosen_cropwright_bind(shimmed: Shimmed) -> None:
    result = configure(
        shimmed, '--local-only', tiers='cropwright', CROPWRIGHT_BIND_ADDRESS='0.0.0.0'
    )
    assert result.returncode == 0, result.stderr
    assert 'CROPWRIGHT_BIND_ADDRESS=127.0.0.1' in _cw_env(shimmed)


def test_cropwright_compose_that_ignores_the_bind_is_refused(shimmed: Shimmed) -> None:
    shimmed.flag('cw_ignores_bind')
    result = configure(shimmed, '--local-only', tiers='cropwright')
    assert result.returncode == 7
    assert 'must honour CROPWRIGHT_BIND_ADDRESS' in result.stderr


def test_summary_security_line_excepts_a_lan_cropwright(shimmed: Shimmed) -> None:
    # Review s1: "every port is bound to 127.0.0.1" contradicted the LAN line.
    result = configure(shimmed, tiers='cropwright')
    assert result.returncode == 0, result.stderr
    assert 'every port is bound to' not in result.stdout
    assert (
        'Security: every OpenProcessor API port is bound to 127.0.0.1; '
        'Cropwright is the exception (your LAN, above).'
    ) in result.stdout


def test_summary_security_line_is_plain_when_everything_is_local(shimmed: Shimmed) -> None:
    result = configure(shimmed, '--local-only', tiers='cropwright')
    assert result.returncode == 0, result.stderr
    assert 'Security: every port is bound to 127.0.0.1 only.' in result.stdout
    assert 'Cropwright is the exception' not in result.stdout


def test_specific_bind_address_also_applies_to_cropwright(shimmed: Shimmed) -> None:
    # Review s2: --bind 10.10.10.20 must not leave Cropwright on 0.0.0.0.
    result = configure(
        shimmed, '--bind', '10.10.10.20', tiers='cropwright', OP_ALLOW_PUBLIC_BIND='1'
    )
    assert result.returncode == 0, result.stderr
    assert 'CROPWRIGHT_BIND_ADDRESS=10.10.10.20' in _cw_env(shimmed)
    assert 'Cropwright on your LAN: http://10.10.10.20:' in result.stdout
    assert 'reachable from your LAN and has NO login' in result.stderr


def test_rerun_without_bind_keeps_a_specific_cropwright_bind(shimmed: Shimmed) -> None:
    first = configure(
        shimmed, '--bind', '10.10.10.20', tiers='cropwright', OP_ALLOW_PUBLIC_BIND='1'
    )
    assert first.returncode == 0, first.stderr
    again = configure(shimmed, tiers='cropwright')
    assert again.returncode == 0, again.stderr
    assert 'CROPWRIGHT_BIND_ADDRESS=10.10.10.20' in _cw_env(shimmed)


def test_local_only_wins_over_a_specific_bind_for_cropwright(shimmed: Shimmed) -> None:
    result = configure(
        shimmed,
        '--bind',
        '10.10.10.20',
        '--local-only',
        tiers='cropwright',
        OP_ALLOW_PUBLIC_BIND='1',
    )
    assert result.returncode == 0, result.stderr
    assert 'CROPWRIGHT_BIND_ADDRESS=127.0.0.1' in _cw_env(shimmed)


def test_wildcard_bind_keeps_cropwright_on_every_interface(shimmed: Shimmed) -> None:
    result = configure(shimmed, '--bind', '0.0.0.0', tiers='cropwright', OP_ALLOW_PUBLIC_BIND='1')
    assert result.returncode == 0, result.stderr
    assert 'CROPWRIGHT_BIND_ADDRESS=0.0.0.0' in _cw_env(shimmed)
