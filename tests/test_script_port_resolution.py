"""Tests for the F-66 fix: shell scripts must resolve ports from .env.

Fresh-start E2E findings (2026-09-25), F-66: ``make status`` /
``scripts/openprocessor.sh`` hardcoded ``localhost:4603/4600/4607/4605``.
On a shared host running a second, remapped-port OpenProcessor stack this
silently read ANOTHER stack's ``/health`` and reported it as this one's,
with no way to override. The Makefile already handled this via
``-include .env`` + ``API_PORT ?= 4603``; ``scripts/lib/ports.sh::env_port``
is the shell-script equivalent, now shared by ``scripts/openprocessor.sh``,
``scripts/setup.sh`` and ``scripts/export_paddleocr.sh``.

No Docker/GPU required -- these just exercise the bash functions and
grep the scripts for regressions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PORTS_LIB = REPO_ROOT / 'scripts' / 'lib' / 'ports.sh'


def _run_env_port(env_file_contents: str | None, key: str, default: str, tmp_path: Path) -> str:
    project_dir = tmp_path
    if env_file_contents is not None:
        (project_dir / '.env').write_text(env_file_contents)

    script = f"""
set -euo pipefail
PROJECT_DIR="{project_dir}"
source "{PORTS_LIB}"
env_port {key} {default}
"""
    result = subprocess.run(
        ['bash', '-c', script], capture_output=True, text=True, check=True, timeout=10
    )
    return result.stdout.strip()


class TestEnvPort:
    def test_falls_back_to_default_when_no_env_file(self, tmp_path: Path) -> None:
        assert _run_env_port(None, 'API_PORT', '4603', tmp_path) == '4603'

    def test_falls_back_to_default_when_key_absent(self, tmp_path: Path) -> None:
        assert _run_env_port('GPU_PROFILE=standard\n', 'API_PORT', '4603', tmp_path) == '4603'

    def test_reads_a_remapped_port_from_env_file(self, tmp_path: Path) -> None:
        # Second isolated stack per env.template's "Isolation" section.
        env_contents = 'COMPOSE_PROJECT_NAME=opfresh\nAPI_PORT=4853\nTRITON_HTTP_PORT=4850\n'
        assert _run_env_port(env_contents, 'API_PORT', '4603', tmp_path) == '4853'
        assert _run_env_port(env_contents, 'TRITON_HTTP_PORT', '4600', tmp_path) == '4850'

    def test_last_definition_wins_like_a_real_env_file(self, tmp_path: Path) -> None:
        env_contents = 'API_PORT=4603\n# overridden below\nAPI_PORT=4863\n'
        assert _run_env_port(env_contents, 'API_PORT', '4603', tmp_path) == '4863'


class TestNoStrayHardcodedPorts:
    """Regression guard: the fixed scripts must resolve every stack port
    through env_port()/the Makefile's ``-include .env`` pattern, not a bare
    literal in a curl call."""

    STOCK_PORTS = ('4600', '4601', '4602', '4603', '4605', '4607', '4608')

    def _bare_literal_port_lines(self, path: Path) -> list[str]:
        offenders: list[str] = []
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith('#') or 'env_port' in line or '${' in line:
                # Comments, or already resolved through a variable -- fine
                # even if the variable's *default* argument is a stock
                # port literal.
                continue
            offenders.extend(
                f'{path}:{lineno}: {stripped}'
                for port in self.STOCK_PORTS
                if f'localhost:{port}' in line or f':{port}/' in line
            )
        return offenders

    def test_openprocessor_sh_has_no_bare_localhost_ports(self) -> None:
        offenders = self._bare_literal_port_lines(REPO_ROOT / 'scripts' / 'openprocessor.sh')
        assert offenders == []

    def test_export_paddleocr_sh_has_no_bare_localhost_ports(self) -> None:
        offenders = self._bare_literal_port_lines(REPO_ROOT / 'scripts' / 'export_paddleocr.sh')
        assert offenders == []

    def test_ports_lib_exists_and_is_sourced_by_both_scripts(self) -> None:
        assert PORTS_LIB.exists()
        for name in ('openprocessor.sh', 'export_paddleocr.sh', 'setup.sh'):
            body = (REPO_ROOT / 'scripts' / name).read_text()
            assert 'lib/ports.sh' in body, f'{name} does not source ports.sh'
