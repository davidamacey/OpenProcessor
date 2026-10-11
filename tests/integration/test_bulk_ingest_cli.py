"""``scripts/curation/bulk_ingest.py`` against the real app (through the route
test harness): the CLI is a client of the one runner, so it is checked end to end."""

from __future__ import annotations

import contextlib
import io
import types
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from PIL import Image

from integration.ingest_fakes import (
    FakeOpenSearch,
    FakeTritonPool,
    curation_app,
    route_bulk_ingest_through_the_app,
)


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


@pytest.fixture
def served_root(tmp_path: Path) -> Path:
    root = tmp_path / 'served'
    root.mkdir()
    return root


@pytest.fixture
def client(
    served_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setenv('OP_BULK_INGEST_DIR', str(tmp_path / 'bulk'))
    with curation_app(FakeOpenSearch(), FakeTritonPool(), monkeypatch) as c:
        route_bulk_ingest_through_the_app(monkeypatch)
        from src.services.curation import image_serving

        monkeypatch.setattr(
            image_serving,
            '_configured_roots',
            lambda config=None: (served_root.resolve(),),  # noqa: ARG005
        )
        yield c


@pytest.fixture
def cli(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The CLI module with its HTTP client replaced by the in-process app."""
    from scripts.curation import bulk_ingest

    # Binding reads the project registry from OpenSearch; the app under test has its own.
    monkeypatch.setattr(bulk_ingest, 'bind_script_project', lambda _slug: None)
    monkeypatch.setattr(
        bulk_ingest,
        'httpx',
        types.SimpleNamespace(
            Client=lambda: contextlib.nullcontext(client), HTTPError=httpx.HTTPError
        ),
    )
    return bulk_ingest


def _write(root: Path, names: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        buf = io.BytesIO()
        Image.new('RGB', (64, 48), (10 + 11 * i, 40, 30)).save(buf, format='JPEG')
        (root / name).write_bytes(buf.getvalue())


def _args(*extra: str) -> list[str]:
    return ['--api-base', 'http://testserver/curation', '--poll-interval', '0.05', *extra]


def test_a_folder_run_prints_the_outcome_and_exits_zero(
    cli: Any, served_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(served_root / 'imgs', [f'{i}.jpg' for i in range(6)])

    code = cli.main(_args('--folder', str(served_root / 'imgs'), '--source', 'cli_test'))

    out = capsys.readouterr().out
    assert code == 0
    assert 'started run bulk_' in out
    assert 'completed' in out
    assert 'ingested 6' in out
    assert 'stage seconds:' in out


def test_a_run_with_unreadable_entries_exits_one_and_lists_them(
    cli: Any, served_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(served_root / 'imgs', ['a.jpg'])
    manifest = served_root / 'list.txt'
    manifest.write_text(f'{served_root / "imgs" / "a.jpg"}\n{served_root / "imgs" / "gone.jpg"}\n')

    code = cli.main(_args('--manifest', str(manifest)))

    out = capsys.readouterr().out
    assert code == 1
    assert 'FAILED' in out
    assert 'gone.jpg [decode_failed]' in out
    assert 'start the same source again to resume' in out


def test_a_refused_source_exits_two_with_the_reason(
    cli: Any, served_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(_args('--folder', str(served_root / 'missing')))
    assert code == 2
    assert 'not a directory' in capsys.readouterr().err


def test_status_and_cancel_address_a_run_by_id(
    cli: Any, served_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(served_root / 'imgs', ['a.jpg'])
    cli.main(_args('--folder', str(served_root / 'imgs'), '--no-wait'))
    run_id = capsys.readouterr().out.split()[2]
    assert cli.main(_args('--status', run_id)) in (0, 1)
    assert capsys.readouterr().out.strip()
    assert cli.main(_args('--status', 'bulk_20260101T000000_deadbeef')) == 2
    assert '404' in capsys.readouterr().err
