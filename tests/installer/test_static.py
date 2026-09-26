"""Static/contract tests for the installer scripts (installer plan
section 9.1)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS_WITH_DC = [
    'setup-openprocessor.sh',
    'openprocessor',
    'scripts/lib/model_setup.sh',
]

ALL_SCRIPTS = [
    *SCRIPTS_WITH_DC,
    'scripts/openprocessor.sh',
    'scripts/lib/vlm_catalog.sh',
]


def _lines_outside_dc_wrapper(text: str) -> list[str]:
    """Every 'docker compose' occurrence that's not the dc()/dc_cw()
    definition line itself, and not inside a comment."""
    bad = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if 'docker compose' not in line:
            continue
        # The wrapper definitions themselves are allowed to contain the
        # literal string.
        if re.match(r'^\s*dc(_cw)?\(\)\s*\{', line) or 'docker compose -p' in line:
            continue
        bad.append(line)
    return bad


def test_docker_compose_only_inside_dc_wrapper() -> None:
    for rel in SCRIPTS_WITH_DC:
        text = (REPO_ROOT / rel).read_text()
        bad = _lines_outside_dc_wrapper(text)
        assert not bad, f"{rel}: bare 'docker compose' usage outside dc()/dc_cw(): {bad}"


def test_no_hardcoded_github_coordinates_outside_variable_block() -> None:
    text = (REPO_ROOT / 'setup-openprocessor.sh').read_text()
    # Everything after the variable block may reference OP_GH_REPO /
    # CW_GH_REPO as variables, but not hardcode "github.com/<org>/<repo>"
    # or "raw.githubusercontent.com/<org>/<repo>" with a literal org/repo.
    after_block = text.split('SCRIPT_VERSION=', 1)[1]
    hardcoded = re.findall(
        r'(?:github\.com|raw\.githubusercontent\.com)/(?!\$\{)[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',
        after_block,
    )
    # api.github.com/repos/${OP_GH_REPO}/... is fine (uses the variable);
    # filter those out explicitly since the regex can't see the brace.
    hardcoded = [h for h in hardcoded if '${' not in h]
    assert not hardcoded, f'hardcoded GitHub coordinates found: {hardcoded}'


def test_release_manifest_paths_exist() -> None:
    manifest = (REPO_ROOT / 'release-manifest.txt').read_text()
    for raw_line in manifest.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        path = line.split('\t')[0].strip()
        if '**' in path:
            continue  # optional glob, e.g. monitoring/**
        assert (REPO_ROOT / path).exists(), f'release-manifest.txt lists missing path: {path}'


def test_bash_dash_n_passes() -> None:
    for rel in ALL_SCRIPTS:
        result = subprocess.run(
            ['bash', '-n', str(REPO_ROOT / rel)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f'{rel}: bash -n failed: {result.stderr}'


def test_set_x_never_enabled() -> None:
    for rel in ALL_SCRIPTS:
        text = (REPO_ROOT / rel).read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            assert 'set -x' not in stripped, f"{rel}: 'set -x' found: {line}"
            assert not re.search(r'set\s+-\w*x\w*\b', stripped), f'{rel}: xtrace flag found: {line}'


def test_no_unsafe_token_read() -> None:
    for rel in ['setup-openprocessor.sh', 'openprocessor']:
        text = (REPO_ROOT / rel).read_text()
        for line in text.splitlines():
            if line.strip().startswith('#'):
                continue
            if re.search(r'\bread\b', line) and re.search(
                r'token|api[_ ]?key|password', line, re.IGNORECASE
            ):
                assert '-rs' in line or '-s' in line, f'{rel}: unsafe token read: {line}'


def test_shellcheck_clean_on_all_scripts() -> None:
    result = subprocess.run(
        ['shellcheck', '--severity=warning', *[str(REPO_ROOT / s) for s in ALL_SCRIPTS]],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
