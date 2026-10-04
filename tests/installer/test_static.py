"""Static/contract tests for the installer scripts (installer plan 9.1)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SHIMS = sorted(
    str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / 'tests/installer/shims').iterdir()
)

SCRIPTS_WITH_DC = [
    'setup-openprocessor.sh',
    'openprocessor',
    'scripts/lib/model_setup.sh',
]

ALL_SCRIPTS = [
    *SCRIPTS_WITH_DC,
    'scripts/openprocessor.sh',
    'scripts/lib/vlm_catalog.sh',
    'scripts/release/build_deploy_bundle.sh',
    *SHIMS,
]


def compose_uses_outside_wrappers(text: str) -> list[str]:
    """Every non-comment `docker compose` (or legacy `docker-compose`, or a
    double-spaced variant) that is not inside the body of dc() / dc_cw().
    A wrapper body runs from its `dc() {` line to the `}` at the same indent."""
    bad: list[str] = []
    closing: str | None = None
    for line in text.splitlines():
        if closing is not None:
            if line == closing:
                closing = None
            continue
        m = re.match(r'^(\s*)dc(_cw)?\(\)\s*\{\s*$', line)
        if m:
            closing = f'{m.group(1)}}}'
            continue
        if line.strip().startswith('#'):
            continue
        code = line.split(' #', 1)[0]
        if re.search(r'docker\s+compose\b|docker-compose(\s|$)', code):
            bad.append(line)
    return bad


def test_docker_compose_only_inside_dc_wrappers() -> None:
    for rel in SCRIPTS_WITH_DC:
        bad = compose_uses_outside_wrappers((REPO_ROOT / rel).read_text())
        assert not bad, f"{rel}: 'docker compose' outside dc()/dc_cw(): {bad}"


def test_the_wrapper_check_catches_a_bare_call() -> None:
    sample = 'dc() {\n    docker compose -p "$p" "$@"\n}\nfoo() {\n    docker compose -p openprocessor down\n}\n'
    assert compose_uses_outside_wrappers(sample) == ['    docker compose -p openprocessor down']
    assert compose_uses_outside_wrappers('x=1; docker  compose ps\n')
    assert compose_uses_outside_wrappers('docker-compose up -d\n')
    assert not compose_uses_outside_wrappers('f=docker-compose.yml\n')


def test_no_hardcoded_github_coordinates_outside_variable_block() -> None:
    text = (REPO_ROOT / 'setup-openprocessor.sh').read_text()
    after_block = text.split('SCRIPT_VERSION=', 1)[1]
    hardcoded = re.findall(
        r'(?:github\.com|raw\.githubusercontent\.com)/(?!\$\{)[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',
        after_block,
    )
    assert not hardcoded, f'hardcoded GitHub coordinates found: {hardcoded}'


def test_release_manifest_paths_exist_and_include_the_installer() -> None:
    entries = []
    for raw_line in (REPO_ROOT / 'release-manifest.txt').read_text().splitlines():
        parts = raw_line.split()
        if not parts or parts[0].startswith('#'):
            continue
        entries.append(parts)
        if parts[0].endswith('**'):
            continue
        assert (REPO_ROOT / parts[0]).exists(), (
            f'release-manifest.txt lists missing path: {parts[0]}'
        )
    names = {e[0] for e in entries}
    assert {
        'setup-openprocessor.sh',
        'release-manifest.txt',
        'openprocessor',
        'images.lock',
    } <= names
    assert ['setup-openprocessor.sh', 'exec'] in entries


def test_bash_dash_n_passes() -> None:
    for rel in ALL_SCRIPTS:
        result = subprocess.run(
            ['bash', '-n', str(REPO_ROOT / rel)], check=False, capture_output=True, text=True
        )
        assert result.returncode == 0, f'{rel}: bash -n failed: {result.stderr}'


def test_xtrace_is_never_enabled() -> None:
    for rel in ALL_SCRIPTS:
        for line in (REPO_ROOT / rel).read_text().splitlines():
            code = line.split('#', 1)[0]
            assert not re.search(r'\bset\s+-[A-Za-z]*x', code), f'{rel}: xtrace enabled: {line}'
            assert not re.search(r'\bset\s+-o\s+xtrace\b', code), f'{rel}: xtrace enabled: {line}'
            assert 'SHELLOPTS' not in code, f'{rel}: touches SHELLOPTS: {line}'
            assert 'BASH_XTRACEFD=' not in code, f'{rel}: redirects xtrace: {line}'


def test_installer_main_disables_xtrace_before_anything_else() -> None:
    text = (REPO_ROOT / 'setup-openprocessor.sh').read_text()
    body = text[text.index('\nmain() {') :]
    first = [ln.strip() for ln in body.splitlines()[2:8] if not ln.strip().startswith('#')]
    assert first[:3] == ['{ set +x; } 2>/dev/null', 'set -euo pipefail', 'shopt -s inherit_errexit']


READ_RE = re.compile(r'\bread\s+((?:-[A-Za-z]+\s+|-p\s+"[^"]*"\s+)*)([A-Za-z_][A-Za-z0-9_]*)(.*)')
SECRET_VAR = re.compile(r'token|api_key|key_value|secret|password', re.IGNORECASE)


def _reads(rel: str) -> list[tuple[str, str, str, str]]:
    out = []
    for line in (REPO_ROOT / rel).read_text().splitlines():
        code = line.split(' #', 1)[0]
        if code.strip().startswith('#'):
            continue
        m = READ_RE.search(code)
        if m:
            opts = ''.join(
                tok.lstrip('-') for tok in m.group(1).split() if tok.startswith('-') and tok != '-p'
            )
            out.append((line, opts, m.group(2), m.group(3)))
    return out


def test_secret_reads_are_silent_and_from_the_terminal() -> None:
    silent = 0
    for rel in ('setup-openprocessor.sh', 'openprocessor'):
        for line, opts, var, rest in _reads(rel):
            if SECRET_VAR.search(var):
                assert 's' in opts, f'{rel}: a secret is read with echo on: {line}'
            if 's' in opts:
                silent += 1
                assert '</dev/tty' in rest.replace(' ', ''), (
                    f'{rel}: silent read not from /dev/tty: {line}'
                )
    text = (REPO_ROOT / 'setup-openprocessor.sh').read_text()
    prompt_secret = text[text.index('prompt_secret() {') : text.index('confirm_yes() {')]
    assert 'IFS= read -rs __reply </dev/tty' in prompt_secret
    assert silent >= 2


def test_every_installer_prompt_reads_the_terminal() -> None:
    for line, _, var, rest in _reads('setup-openprocessor.sh'):
        if var.startswith('__reply'):
            assert '</dev/tty' in rest.replace(' ', ''), f'prompt reads stdin: {line}'
    assert 'read -r -p' not in (REPO_ROOT / 'setup-openprocessor.sh').read_text()


def test_installer_sources_only_verified_install_files() -> None:
    text = (REPO_ROOT / 'setup-openprocessor.sh').read_text()
    sources = [ln.strip() for ln in text.splitlines() if re.match(r'^\s*(source|\.)\s', ln)]
    assert set(sources) == {
        'source "${OP_DIR}/scripts/lib/vlm_catalog.sh"',
        'source "${OP_DIR}/scripts/lib/model_setup.sh"',
        'source "${OP_DIR}/scripts/lib/image_keys.sh"',
        'source "${OP_DIR}/scripts/lib/opensearch_heap.sh"',
    }
    body = text[text.index('\ndo_install() {') :]
    assert body.index('install_staged "${OP_DIR}/.install/staging"') < body.index(
        'source "${OP_DIR}/scripts/lib/vlm_catalog.sh"'
    )
    assert 'BASH_SOURCE' not in text.split('__op_define && {', 1)[0].replace('BASH_SOURCE[0]:-', '')


def test_script_is_one_function_plus_a_single_call_line() -> None:
    lines = (REPO_ROOT / 'setup-openprocessor.sh').read_text().rstrip('\n').splitlines()
    code = [ln for ln in lines if ln.strip() and not ln.startswith('#')]
    assert code[0] == '__op_define() {'
    assert code[-2] == '}'
    assert code[-1].startswith('__op_define && {')
    assert code[-1].endswith('main "$@"; }')


def test_compose_opensearch_heap_is_driven_by_opensearch_heap() -> None:
    compose = (REPO_ROOT / 'docker-compose.yml').read_text()
    assert '"OPENSEARCH_JAVA_OPTS=-Xms${OPENSEARCH_HEAP:-2g} -Xmx${OPENSEARCH_HEAP:-2g}"' in compose
    assert re.search(r'^OPENSEARCH_HEAP=', (REPO_ROOT / 'env.template').read_text(), re.MULTILINE)


def _compose_services() -> dict:
    import yaml

    return yaml.safe_load((REPO_ROOT / 'docker-compose.yml').read_text())['services']


def test_every_compose_service_rotates_its_logs() -> None:
    for name, svc in _compose_services().items():
        opts = svc.get('logging', {}).get('options', {})
        assert svc.get('logging', {}).get('driver') == 'json-file', f'{name}: no log rotation'
        assert opts.get('max-size'), f'{name}: no max-size'
        assert opts.get('max-file'), f'{name}: no max-file'


def test_every_compose_service_has_a_memory_reservation() -> None:
    for name, svc in _compose_services().items():
        assert svc.get('mem_reservation'), f'{name}: no mem_reservation'


def test_opensearch_memory_reservation_follows_the_heap() -> None:
    assert _compose_services()['opensearch']['mem_reservation'] == '${OPENSEARCH_HEAP:-2g}'


def test_shellcheck_clean_on_all_scripts() -> None:
    result = subprocess.run(
        ['shellcheck', '--severity=warning', *[str(REPO_ROOT / s) for s in ALL_SCRIPTS]],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_committed_images_lock_has_a_digest_on_every_line_and_no_latest() -> None:
    for raw_line in (REPO_ROOT / 'images.lock').read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        assert ':latest' not in line, line
        assert '@sha256:' in line, line


_LAN_SENTENCES = (
    'Cropwright is reachable on your LAN by default, for homelab or\nsmall-business use.',
    'The API itself stays bound to 127.0.0.1.',
    'There is no login\non Cropwright',
    'Pass `--local-only` to opt out and keep\neverything on 127.0.0.1.',
)


def _flat(text: str) -> str:
    return ' '.join(text.split())


@pytest.mark.parametrize('doc', ['README.md', 'INSTALLATION.md', 'SECURITY.md'])
def test_docs_carry_the_lan_decision_wording(doc: str) -> None:
    text = _flat((REPO_ROOT / doc).read_text())
    for sentence in _LAN_SENTENCES:
        assert _flat(sentence) in text, f'{doc} lacks: {sentence!r}'
    assert 'port-forward' in text
    assert 'reverse proxy with authentication' in text


@pytest.mark.parametrize('doc', ['README.md', 'INSTALLATION.md', 'SECURITY.md'])
def test_docs_say_checksums_are_integrity_not_authenticity(doc: str) -> None:
    text = _flat((REPO_ROOT / doc).read_text())
    assert 'integrity' in text
    assert 'authenticity' in text


def test_installation_documents_every_installer_flag() -> None:
    usage = subprocess.run(
        ['bash', str(REPO_ROOT / 'setup-openprocessor.sh'), '--help'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    flags = set(re.findall(r'(?<![\w-])(--[a-z][a-z-]+)', usage)) - {'--help'}
    doc = (REPO_ROOT / 'INSTALLATION.md').read_text()
    missing = sorted(f for f in flags if f'`{f}' not in doc)
    assert not missing, missing
