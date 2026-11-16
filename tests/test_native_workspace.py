"""The native (Rust) workspace under ``native/`` is wired into every gate that must see it.

Plan: docs/design/v070_rebuild/17_native_engine_plan.md section 3.1. These checks read the
configuration files only (no cargo, no network); the Rust gates themselves run in their hooks and
in the CI job ``native``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'native'
PRE_COMMIT = yaml.safe_load((ROOT / '.pre-commit-config.yaml').read_text(encoding='utf-8'))
WORKFLOW = yaml.safe_load((ROOT / '.github' / 'workflows' / 'ci.yml').read_text(encoding='utf-8'))
DEPENDABOT = yaml.safe_load((ROOT / '.github' / 'dependabot.yml').read_text(encoding='utf-8'))
PYPROJECT = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
DENY = tomllib.loads((NATIVE / 'deny.toml').read_text(encoding='utf-8'))
CARGO = tomllib.loads((NATIVE / 'Cargo.toml').read_text(encoding='utf-8'))

CARGO_HOOKS = {
    'cargo-fmt': 'cargo fmt --manifest-path native/Cargo.toml --all -- --check',
    'cargo-clippy': (
        'cargo clippy --manifest-path native/Cargo.toml --workspace --all-targets --locked'
        ' -- -D warnings'
    ),
    'cargo-doc': 'cargo doc --manifest-path native/Cargo.toml --workspace --no-deps --locked',
    'cargo-test': 'cargo test --manifest-path native/Cargo.toml --workspace --locked',
    'cargo-deny': 'cargo deny --manifest-path native/Cargo.toml check licenses bans sources',
}


def _hooks() -> dict[str, dict[str, Any]]:
    return {h['id']: h for repo in PRE_COMMIT['repos'] for h in repo['hooks']}


@pytest.mark.parametrize(('hook_id', 'command'), sorted(CARGO_HOOKS.items()))
def test_cargo_hook_runs_its_documented_command_on_native_only(hook_id: str, command: str) -> None:
    hook = _hooks()[hook_id]
    assert command in hook['entry']
    assert hook['files'] == '^native/'
    assert hook['pass_filenames'] is False
    assert hook['language'] == 'system'
    assert 'stages' not in hook, 'the four gates run at commit time'


def test_advisories_run_before_push_on_native_only() -> None:
    hook = _hooks()['cargo-deny-advisories']
    assert 'cargo deny --manifest-path native/Cargo.toml check advisories' in hook['entry']
    assert hook['stages'] == ['pre-push']
    assert hook['files'] == '^native/'


def test_ci_native_job_runs_the_gates_once_on_native_changes() -> None:
    job = WORKFLOW['jobs']['native']
    assert 'changes' in job['needs']
    assert 'needs.changes.outputs.native' in job['if']
    run = '\n'.join(step.get('run', '') for step in job['steps'])
    for command in CARGO_HOOKS.values():
        assert command in run
    assert 'cargo deny --manifest-path native/Cargo.toml check advisories' in run
    assert 'cargo install --locked cargo-deny@0.20.2' in run
    assert 'pytest -m native tests/native/' in run


def test_ci_pre_commit_job_skips_the_hooks_the_native_job_runs() -> None:
    steps = WORKFLOW['jobs']['pre-commit']['steps']
    run_all = next(s for s in steps if s.get('name') == 'Run all hooks')
    skipped = set(run_all['env']['SKIP'].split(','))
    assert skipped == {*CARGO_HOOKS, 'cargo-deny-advisories'}


def test_dependabot_watches_the_native_workspace() -> None:
    entry = next(
        e
        for e in DEPENDABOT['updates']
        if e['package-ecosystem'] == 'cargo' and e['directory'] == '/native'
    )
    assert entry['schedule']['interval'] == 'weekly'
    assert {'applies-to': 'security-updates', 'patterns': ['*']} in entry['groups'].values()


def test_native_pytest_marker_is_registered() -> None:
    markers = '\n'.join(PYPROJECT['tool']['pytest']['ini_options']['markers'])
    assert re.search(r'^native: ', markers, re.MULTILINE)


def _offline_selectors() -> dict[str, list[str]]:
    sources = {
        'Makefile': ROOT / 'Makefile',
        'CLAUDE.md': ROOT / 'CLAUDE.md',
        'CONTRIBUTING.md': ROOT / 'CONTRIBUTING.md',
        'ci.yml': ROOT / '.github' / 'workflows' / 'ci.yml',
        'PULL_REQUEST_TEMPLATE.md': ROOT / '.github' / 'PULL_REQUEST_TEMPLATE.md',
    }
    found = {
        name: re.findall(r"pytest [^\n]*?-m '(not live[^']*)'", path.read_text(encoding='utf-8'))
        for name, path in sources.items()
    }
    addopts = PYPROJECT['tool']['pytest']['ini_options']['addopts']
    found['pyproject.toml'] = [addopts[addopts.index('-m') + 1]]
    return found


def _leaves_out(expression: str, marker: str) -> bool:
    """Whether a ``-m`` expression is a plain conjunction with ``not <marker>`` among its terms.

    The place of the term does not matter: a marker added after it (``... and not differential``)
    leaves the tier out just the same.
    """
    if ' or ' in expression or '(' in expression:
        return False
    return f'not {marker}' in expression.split(' and ')


@pytest.mark.parametrize(
    'source',
    [
        'Makefile',
        'CLAUDE.md',
        'CONTRIBUTING.md',
        'ci.yml',
        'PULL_REQUEST_TEMPLATE.md',
        'pyproject.toml',
    ],
)
def test_every_offline_selector_leaves_out_native(source: str) -> None:
    expressions = _offline_selectors()[source]
    assert expressions, f'{source} has no offline pytest selector'
    for expression in expressions:
        assert _leaves_out(expression, 'native'), (source, expression)


@pytest.mark.parametrize(
    ('expression', 'expected'),
    [
        ('not live and not native', True),
        ('not live and not native and not differential', True),
        ('not native', True),
        ('not live and not families', False),
        ('not live and not native_gpu', False),
        ('not live and native', False),
        ('not live or not native', False),
    ],
)
def test_a_selector_leaves_a_marker_out_only_as_a_term_of_its_conjunction(
    expression: str, expected: bool
) -> None:
    assert _leaves_out(expression, 'native') is expected


def test_contributing_documents_the_native_tier() -> None:
    text = (ROOT / 'CONTRIBUTING.md').read_text(encoding='utf-8')
    assert re.search(r'^\| `native` \| `native` \|', text, re.MULTILINE)
    assert '-m native' in text


def test_toolchain_is_pinned_exactly_with_the_gate_components() -> None:
    toolchain = tomllib.loads((NATIVE / 'rust-toolchain.toml').read_text(encoding='utf-8'))
    assert re.fullmatch(r'\d+\.\d+\.\d+', toolchain['toolchain']['channel'])
    assert {'rustfmt', 'clippy'} <= set(toolchain['toolchain']['components'])


def test_workspace_denies_the_lints_the_plan_names_and_every_member_inherits_them() -> None:
    assert CARGO['workspace']['lints']['rust']['unsafe_op_in_unsafe_fn'] == 'deny'
    clippy = CARGO['workspace']['lints']['clippy']
    assert clippy['undocumented_unsafe_blocks'] == 'deny'
    assert clippy['unwrap_used'] == 'deny'
    assert CARGO['workspace']['package']['license'] == PYPROJECT['project']['license']['text']
    for member in CARGO['workspace']['members']:
        manifest = tomllib.loads((NATIVE / member / 'Cargo.toml').read_text(encoding='utf-8'))
        assert manifest['lints'] == {'workspace': True}, member
    clippy_config = tomllib.loads((NATIVE / 'clippy.toml').read_text(encoding='utf-8'))
    assert clippy_config['allow-unwrap-in-tests'] is True


def test_deny_allows_only_licences_that_fit_an_agpl_repository() -> None:
    allowed = set(DENY['licenses']['allow'])
    assert not [lic for lic in allowed if re.search(r'GPL|SSPL|BUSL|Commons-Clause|CC-BY-NC', lic)]
    assert {'MIT', 'Apache-2.0', 'BSD-3-Clause'} <= allowed
    assert DENY['sources']['unknown-registry'] == 'deny'
    assert DENY['sources']['unknown-git'] == 'deny'
    assert DENY['sources']['allow-git'] == []


def test_build_output_is_not_tracked_or_sent_to_a_docker_build() -> None:
    assert '/native/target/' in (ROOT / '.gitignore').read_text(encoding='utf-8').splitlines()
    assert 'native/target/' in (ROOT / '.dockerignore').read_text(encoding='utf-8').splitlines()


@pytest.mark.parametrize(
    'path',
    [
        'contracts/json/frame_engine_v1.json',
        'contracts/json/frame_engine_plan_cases.json',
        'src/domain/frame_engine_wire.py',
        'scripts/codegen/frame_engine_contract.py',
        'scripts/codegen/frame_engine_plan_cases.py',
    ],
)
def test_the_contract_drift_hook_runs_when_a_frame_engine_contract_input_changes(path: str) -> None:
    pattern = _hooks()['api-contracts-drift']['files']
    assert re.search(pattern, path), path
