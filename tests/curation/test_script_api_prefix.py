"""Scripts and side-car containers call the API under the configured
``OP_API_PREFIX``, never a hardcoded ``/curation``."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
_SCANNED = [REPO_ROOT / 'scripts', REPO_ROOT / 'docker']


def _url_literals_with_prefix(path: Path) -> list[str]:
    """f-strings, and plain strings that look like URLs, embedding
    ``/curation/`` — i.e. URL construction, not help text or docstrings."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            text = ''.join(
                v.value
                for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            )
            if '/curation/' in text or text.endswith('/curation'):
                hits.append(f'{path.relative_to(REPO_ROOT)}:{node.lineno}')
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith(('http://', 'https://'))
            and '/curation' in node.value
        ):
            hits.append(f'{path.relative_to(REPO_ROOT)}:{node.lineno}')
    return hits


def test_no_hardcoded_curation_prefix_in_script_urls() -> None:
    hits: list[str] = []
    for root in _SCANNED:
        for path in sorted(root.rglob('*.py')):
            hits.extend(_url_literals_with_prefix(path))
    assert hits == []


class _RecordingClient:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def post(self, url: str, **_: Any) -> Any:
        self.urls.append(url)

        class _Resp:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {}

        return _Resp()


_ENV_READING_MODULES = ('scripts.curation.vlm_worker', 'scripts.curation.cluster_refresh_daemon')


@pytest.fixture
def custom_prefix() -> Any:
    """OP_API_PREFIX=/custom-mount for the test, then the env-reading
    modules are reloaded with the real env so no other test inherits the
    custom module-level prefix."""
    mp = pytest.MonkeyPatch()
    mp.setenv('OP_API_PREFIX', '/custom-mount')
    yield '/custom-mount'
    mp.undo()
    for name in _ENV_READING_MODULES:
        importlib.reload(importlib.import_module(name))


@pytest.mark.asyncio
async def test_vlm_worker_uses_configured_prefix(custom_prefix: str) -> None:
    mod = importlib.reload(importlib.import_module('scripts.curation.vlm_worker'))
    client = _RecordingClient()
    await mod.label_batch(client, api='http://api', crop_ids=['c1'])  # type: ignore[arg-type]
    assert client.urls == [f'http://api{custom_prefix}/vlm/label_batch']


@pytest.mark.asyncio
async def test_cluster_refresh_daemon_uses_configured_prefix(custom_prefix: str) -> None:
    mod = importlib.reload(importlib.import_module('scripts.curation.cluster_refresh_daemon'))
    client = _RecordingClient()
    await mod._trigger_auto_promote(client, 'http://api')  # type: ignore[arg-type]
    await mod._trigger_auto_label(client, 'http://api')  # type: ignore[arg-type]
    assert client.urls == [
        f'http://api{custom_prefix}/clusters/auto_promote',
        f'http://api{custom_prefix}/pipeline/auto_label',
    ]


class _Cfg:
    api_prefix = '/custom-mount'


@pytest.mark.parametrize(
    ('module', 'argv'),
    [
        ('scripts.curation.ingest_upload', ['--root', '/tmp']),
        ('scripts.curation.import_labeled_dataset', ['--dataset', '/tmp']),
    ],
)
def test_ingest_tools_default_api_base_follows_prefix(
    monkeypatch: pytest.MonkeyPatch, module: str, argv: list[str]
) -> None:
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, 'get_curation_config', lambda: _Cfg())
    args = mod.build_parser().parse_args(argv)
    assert args.api_base == 'http://localhost:4603/custom-mount'


def test_ingest_walker_default_api_base_follows_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = importlib.import_module('scripts.curation.ingest_walker')
    monkeypatch.setattr(mod, 'get_curation_config', lambda: _Cfg())
    monkeypatch.setattr('sys.argv', ['ingest_walker', '--root', '/tmp'])
    seen: dict[str, Any] = {}

    def _fake_run(**kwargs: Any) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(mod, 'run', _fake_run)
    monkeypatch.setattr(mod.asyncio, 'run', lambda _coro: None)
    mod.main()
    assert seen['api_base'] == 'http://localhost:4603/custom-mount'
