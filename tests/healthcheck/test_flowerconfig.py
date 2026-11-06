"""Flower reads its basic-auth value from the mounted secret file, never from argv or env.

The `flower` service runs `celery -A src.tasks.app flower --conf=/app/scripts/flower/flowerconfig.py`
(02c section 4, review R-30); Flower executes that file as Python and takes its module globals as
options.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest


CONF = Path(__file__).resolve().parents[2] / 'scripts' / 'flower' / 'flowerconfig.py'


def _run(monkeypatch: pytest.MonkeyPatch, auth_file: Path) -> dict[str, object]:
    monkeypatch.setenv('FLOWER_AUTH_FILE', str(auth_file))
    return runpy.run_path(str(CONF))


def test_basic_auth_comes_from_the_secret_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth = tmp_path / 'auth'
    auth.write_text('admin:s3cret\n')
    options = _run(monkeypatch, auth)
    assert options['basic_auth'] == ['admin:s3cret']


@pytest.mark.parametrize('content', [None, '', '\n', 'admin:', 'no-colon'])
def test_a_missing_or_unusable_secret_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str | None
) -> None:
    auth = tmp_path / 'auth'
    if content is not None:
        auth.write_text(content)
    with pytest.raises(SystemExit, match='flower auth'):
        _run(monkeypatch, auth)
