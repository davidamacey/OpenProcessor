"""The docs must describe the code as it is.

Runs ``scripts/docs/check_docs_vs_code.py`` over the real docs (routes and env
vars; links and anchors are asserted in ``tests/test_doc_links.py``), and proves each check can fail by feeding it a
synthetic doc with a bogus route, a bogus env var and a dangling anchor.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    'check_docs_vs_code', REPO_ROOT / 'scripts' / 'docs' / 'check_docs_vs_code.py'
)
assert _SPEC is not None
assert _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
sys.modules['check_docs_vs_code'] = checker
_SPEC.loader.exec_module(checker)


@pytest.fixture(autouse=True)
def _reset_only():
    yield
    checker.set_only(None)


@pytest.mark.parametrize(
    'name', ['routes', 'env', 'compose', 'template', 'cli', 'make', 'screenshots', 'envfiles']
)
def test_real_docs_match_the_code(name: str) -> None:
    errors = checker.CHECKS[name]()
    assert errors == [], f'{name} check found {len(errors)} problem(s):\n' + '\n'.join(errors)


def _check_one(tmp_path: Path, body: str, name: str) -> list[str]:
    doc = tmp_path / 'sample.md'
    doc.write_text(body, encoding='utf-8')
    checker.set_only([doc])
    return checker.CHECKS[name]()


def test_route_check_accepts_real_and_rejects_bogus_routes(tmp_path: Path) -> None:
    assert _check_one(tmp_path, 'POST /curation/projects/{project}/reprocess\n', 'routes') == []
    assert _check_one(tmp_path, 'POST /detect\n', 'routes') == []
    bad = _check_one(tmp_path, 'POST /curation/projects/{project}/no_such_route\n', 'routes')
    assert len(bad) == 1
    assert 'no_such_route' in bad[0]


def test_route_check_resolves_project_relative_paths(tmp_path: Path) -> None:
    assert _check_one(tmp_path, '| GET | `/crops/{crop_id}` |\n', 'routes') == []
    assert _check_one(tmp_path, '| GET | `/crops/{crop_id}/nope` |\n', 'routes') != []


def test_route_check_rejects_wrong_method(tmp_path: Path) -> None:
    assert _check_one(tmp_path, 'DELETE /detect\n', 'routes') != []


def test_env_check_rejects_unknown_var(tmp_path: Path) -> None:
    assert _check_one(tmp_path, 'Set `OP_VLM_URL` to the endpoint.\n', 'env') == []
    bad = _check_one(tmp_path, 'Set `OP_DEFINITELY_NOT_A_REAL_VAR` here.\n', 'env')
    assert len(bad) == 1
    assert 'OP_DEFINITELY_NOT_A_REAL_VAR' in bad[0]


def test_compose_check_rejects_a_variable_missing_from_env_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / 'docker-compose.yml').write_text(
        'services:\n  a:\n    image: ${KNOWN_IMAGE:-x}\n    environment:\n'
        '      - B=${NOT_DOCUMENTED:-1}\n      - C=$${ESCAPED_BY_COMPOSE}\n'
        '      - D=${COMMENTED_ONLY}\n',
        encoding='utf-8',
    )
    (tmp_path / 'env.template').write_text('KNOWN_IMAGE=x\n# COMMENTED_ONLY=1\n', encoding='utf-8')
    monkeypatch.setattr(checker, 'REPO_ROOT', tmp_path)

    assert checker.check_compose_vars() == [
        'docker-compose.yml: ${NOT_DOCUMENTED} is not in env.template (commented is fine)'
    ]


def test_compose_check_wants_an_assignment_not_a_prose_mention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / 'docker-compose.yml').write_text(
        'services:\n  a:\n    image: ${ONLY_IN_PROSE:-x}\n', encoding='utf-8'
    )
    (tmp_path / 'env.template').write_text(
        '# Set ONLY_IN_PROSE to pin the image, see the docs.\n', encoding='utf-8'
    )
    monkeypatch.setattr(checker, 'REPO_ROOT', tmp_path)

    assert checker.check_compose_vars() == [
        'docker-compose.yml: ${ONLY_IN_PROSE} is not in env.template (commented is fine)'
    ]


def test_compose_check_skips_retired_names_the_installer_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / 'docker-compose.yml').write_text(
        'services:\n  a:\n    environment:\n      - OLD_NAME=${OLD_NAME:-}\n', encoding='utf-8'
    )
    (tmp_path / 'env.template').write_text('# NEW_NAME=1\n', encoding='utf-8')
    (tmp_path / 'scripts' / 'lib').mkdir(parents=True)
    (tmp_path / 'scripts' / 'lib' / 'retired_env.sh').write_text(
        'RETIRED_ENV_NAMES=(\n    OLD_NAME\n)\n', encoding='utf-8'
    )
    monkeypatch.setattr(checker, 'REPO_ROOT', tmp_path)

    assert checker.check_compose_vars() == []


def test_template_check_rejects_a_setting_nothing_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / 'docker-compose.yml').write_text(
        'services:\n  a:\n    image: ${READ_BY_COMPOSE:-x}\n', encoding='utf-8'
    )
    (tmp_path / 'env.template').write_text(
        'READ_BY_COMPOSE=x\n# READ_NOWHERE=info\nOP_SKIPPED=1\n', encoding='utf-8'
    )
    monkeypatch.setattr(checker, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(checker, '_CODE_ENV_ROOTS', ())

    assert checker.check_template_names() == [
        'env.template: READ_NOWHERE is set by no compose file, script or code'
    ]


def test_link_check_rejects_missing_file_and_anchor(tmp_path: Path) -> None:
    (tmp_path / 'other.md').write_text('# Real Heading\n', encoding='utf-8')
    assert _check_one(tmp_path, '[ok](other.md#real-heading)\n', 'links') == []
    assert _check_one(tmp_path, '[gone](missing.md)\n', 'links') != []
    assert _check_one(tmp_path, '[bad](other.md#no-such-heading)\n', 'links') != []
    assert _check_one(tmp_path, '[self](#nowhere)\n', 'links') != []


def test_slugify_matches_github_rules() -> None:
    assert checker.slugify('The `lock` rule (v2)') == 'the-lock-rule-v2'


def test_doc_files_skip_untracked_local_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Untracked local files (private notes, gitignored artifacts) are not published
    docs, so a bogus route in one must not fail the checks."""
    import subprocess  # nosec B404 - fixed git invocation in a temp repo

    def git(*args: str) -> None:
        subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True)  # nosec B603 B607

    git('init', '-q')
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs' / 'tracked.md').write_text('# tracked\n', encoding='utf-8')
    (tmp_path / 'docs' / 'local_note.md').write_text('POST /no_such_route\n', encoding='utf-8')
    git('add', 'docs/tracked.md')
    monkeypatch.setattr(checker, 'REPO_ROOT', tmp_path)

    names = {p.name for p in checker.doc_files()}

    assert names == {'tracked.md'}


def test_cli_check_rejects_an_unknown_subcommand_and_flag(tmp_path: Path) -> None:
    assert _check_one(tmp_path, 'Run `./openprocessor curation up --help`.\n', 'cli') == []
    bad = _check_one(tmp_path, 'Run `./openprocessor frobnicate` first.\n', 'cli')
    assert len(bad) == 1
    assert 'frobnicate' in bad[0]
    bad = _check_one(tmp_path, '`./openprocessor logs --no-such-flag`\n', 'cli')
    assert len(bad) == 1
    assert '--no-such-flag' in bad[0]


def test_cli_check_reads_a_flag_only_up_to_the_end_of_its_code_span(tmp_path: Path) -> None:
    body = '`./openprocessor upgrade`, and `./setup-openprocessor.sh --rollback`\n'
    assert _check_one(tmp_path, body, 'cli') == []


def test_make_check_rejects_an_unknown_target_in_code_only(tmp_path: Path) -> None:
    assert _check_one(tmp_path, 'Run `make up` then\n```bash\nmake status\n```\n', 'make') == []
    assert _check_one(tmp_path, 'We make the API send requests.\n', 'make') == []
    assert _check_one(tmp_path, 'Globs like `make clone-refs-*` are fine.\n', 'make') == []
    bad = _check_one(tmp_path, 'Run `make no-such-target`.\n', 'make')
    assert len(bad) == 1
    assert 'no-such-target' in bad[0]
    bad = _check_one(tmp_path, '```bash\n$ make also-missing\n```\n', 'make')
    assert len(bad) == 1


def test_screenshot_check_rejects_a_name_with_no_file(tmp_path: Path) -> None:
    assert (
        _check_one(tmp_path, '<Screenshot name="cropwright/<file>.png" alt="x" />\n', 'screenshots')
        == []
    )
    real = next((checker.REPO_ROOT / 'docs-site' / 'static' / 'img' / 'screenshots').rglob('*.png'))
    name = real.relative_to(checker.REPO_ROOT / 'docs-site' / 'static' / 'img' / 'screenshots')
    assert (
        _check_one(tmp_path, f'<Screenshot name="{name.as_posix()}" alt="x" />\n', 'screenshots')
        == []
    )
    bad = _check_one(
        tmp_path, '<Screenshot name="cropwright/no-such-1600.png" alt="x" />\n', 'screenshots'
    )
    assert len(bad) == 1
    assert 'no-such-1600.png' in bad[0]


def test_envfiles_check_rejects_an_assigned_name_nothing_reads(tmp_path: Path) -> None:
    ok = '```bash\nCOMPOSE_PROJECT_NAME=mine\n```\n'
    assert _check_one(tmp_path, ok, 'envfiles') == []
    local = '```bash\nBK=~/backup\nmkdir -p "$BK"\n```\n'
    assert _check_one(tmp_path, local, 'envfiles') == []
    bad = _check_one(tmp_path, '```bash\nNOT_A_REAL_SETTING=1\n```\n', 'envfiles')
    assert len(bad) == 1
    assert 'NOT_A_REAL_SETTING' in bad[0]
