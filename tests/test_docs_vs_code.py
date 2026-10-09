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


@pytest.mark.parametrize('name', ['routes', 'env'])
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
