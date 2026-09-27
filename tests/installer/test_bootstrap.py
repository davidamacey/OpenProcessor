"""The one-liner path: the script is read from a pipe (`curl ... | bash`).

B1: under a pipe BASH_SOURCE is empty; main must still run, and the raw
copy must only resolve, download, verify and exec the release's own
setup-openprocessor.sh (M9). A truncated download must execute nothing.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

from installer_harness import PROJECT, RELEASE, SCRIPT


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed


ARGS = ['--dry-run', '--dir', 'inst', '--project', PROJECT, '--tiers', 'core']


def test_piped_help_prints_usage_without_touching_the_network(shimmed: Shimmed) -> None:
    result = shimmed.run(['--help'], piped=True)
    assert result.returncode == 0, result.stderr
    assert 'Usage: setup-openprocessor.sh' in result.stdout
    assert 'unbound variable' not in result.stderr
    assert shimmed.log.read_text() == ''


def test_piped_dry_run_reexecs_the_verified_release_script(shimmed: Shimmed) -> None:
    result = shimmed.run([*ARGS, '--version', RELEASE], piped=True)
    assert result.returncode == 0, result.stderr
    assert f'running the verified {RELEASE} installer' in result.stdout
    curl = shimmed.log_lines('curl')
    assert any(f'assets/{RELEASE}/setup-openprocessor.sh' in ln for ln in curl)
    assert any(f'assets/{RELEASE}/SHA256SUMS' in ln for ln in curl)
    # No --unattended given: with no terminal it switched to unattended
    # instead of reading answers from the piped script.
    assert 'nothing was pulled, started or removed' in result.stdout
    assert shimmed.mutating_docker_calls() == []
    assert (shimmed.root / 'inst' / '.install' / 'state.json').exists()


def test_piped_run_without_version_installs_the_latest_release(shimmed: Shimmed) -> None:
    result = shimmed.run(ARGS, piped=True)
    assert result.returncode == 0, result.stderr
    assert any('releases/latest' in ln for ln in shimmed.log_lines('curl'))
    state = (shimmed.root / 'inst' / '.install' / 'state.json').read_text()
    assert f'"version": "{RELEASE}"' in state


def _tampered_release(shimmed: Shimmed, tmp_path: Path) -> Path:
    release = tmp_path / 'rel'
    shutil.copytree(shimmed.release, release)
    shimmed.release = release
    return release / 'assets' / RELEASE


def test_piped_bootstrap_refuses_a_tampered_release_script(
    shimmed: Shimmed, tmp_path: Path
) -> None:
    assets = _tampered_release(shimmed, tmp_path)
    script = assets / 'setup-openprocessor.sh'
    script.write_text(script.read_text().replace('main() {', 'main() {\n    echo PWNED-MARKER', 1))
    result = shimmed.run([*ARGS, '--version', RELEASE], piped=True)
    assert result.returncode == 7
    assert 'PWNED-MARKER' not in result.stdout + result.stderr
    assert 'checksum' in result.stderr
    assert not (shimmed.root / 'inst').exists()


def test_piped_bootstrap_refuses_a_missing_checksum_line(shimmed: Shimmed, tmp_path: Path) -> None:
    assets = _tampered_release(shimmed, tmp_path)
    sums = assets / 'SHA256SUMS'
    sums.write_text(
        ''.join(
            ln
            for ln in sums.read_text().splitlines(True)
            if not ln.rstrip().endswith(' setup-openprocessor.sh')
        )
    )
    result = shimmed.run([*ARGS, '--version', RELEASE], piped=True)
    assert result.returncode == 7
    assert 'no entry for setup-openprocessor.sh' in result.stderr
    assert not (shimmed.root / 'inst').exists()


def test_bootstrap_temp_dir_is_removed_by_the_child(shimmed: Shimmed) -> None:
    tmp = shimmed.root / 'tmp'
    tmp.mkdir()
    result = shimmed.run([*ARGS, '--version', RELEASE], piped=True, TMPDIR=str(tmp))
    assert result.returncode == 0, result.stderr
    assert list(tmp.iterdir()) == []


def test_truncated_piped_script_executes_nothing(shimmed: Shimmed) -> None:
    data = SCRIPT.read_bytes()
    offsets = sorted(
        {*range(0, len(data), max(1, len(data) // 60)), *range(len(data) - 120, len(data) - 1, 3)}
    )
    for n in offsets:
        subprocess.run(
            [
                'bash',
                '-s',
                '--',
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
            ],
            input=data[:n],
            check=False,
            cwd=str(shimmed.root),
            env=shimmed.env(),
            capture_output=True,
            timeout=30,
            start_new_session=True,
        )
        assert shimmed.log.read_text() == '', f'truncated at {n} bytes ran commands'
        assert not (shimmed.root / 'inst').exists(), (
            f'truncated at {n} bytes created the install dir'
        )
    # And the complete script does run, so the loop above is not vacuous.
    full = shimmed.run([*ARGS, '--version', RELEASE], piped=True)
    assert full.returncode == 0, full.stderr
