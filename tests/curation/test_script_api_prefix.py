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
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {}

        return _Resp()


@pytest.fixture
def custom_prefix(monkeypatch: pytest.MonkeyPatch) -> Any:
    """OP_API_PREFIX=/custom-mount, and a fresh env-built base config."""
    import src.config.curation as curation_config_mod

    monkeypatch.setenv('OP_API_PREFIX', '/custom-mount')
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    return '/custom-mount'


@pytest.fixture
def beta_bound() -> Any:
    """The workers' ``--project beta``: the whole process bound to beta."""
    from datetime import UTC, datetime

    from src.config.curation import base_curation_config
    from src.config.project_context import bind_process_project
    from src.config.projects import ProjectRecord, resources_for_new

    now = datetime.now(UTC).isoformat()
    record = ProjectRecord(
        slug='beta',
        display_name='Beta',
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new('beta', base_curation_config()),
    )
    bind_process_project(record)
    yield record
    bind_process_project(None)


@pytest.mark.unbound
@pytest.mark.asyncio
async def test_vlm_worker_honours_project_and_prefix(custom_prefix: str, beta_bound: Any) -> None:
    mod = importlib.import_module('scripts.curation.vlm_worker')
    client = _RecordingClient()
    await mod.label_batch(client, api='http://api', crop_ids=['c1'])  # type: ignore[arg-type]
    assert client.urls == [f'http://api{custom_prefix}/projects/beta/vlm/label_batch']
    assert mod._items_index() == 'op_prj_beta__items'


@pytest.mark.unbound
@pytest.mark.asyncio
async def test_cluster_refresh_daemon_honours_project_and_prefix(
    custom_prefix: str, beta_bound: Any
) -> None:
    mod = importlib.import_module('scripts.curation.cluster_refresh_daemon')
    client = _RecordingClient()
    await mod._trigger_auto_promote(client, 'http://api')  # type: ignore[arg-type]
    await mod._trigger_auto_label(client, 'http://api')  # type: ignore[arg-type]
    assert client.urls == [
        f'http://api{custom_prefix}/projects/beta/clusters/auto_promote',
        f'http://api{custom_prefix}/projects/beta/pipeline/auto_label',
    ]
    assert mod._items_index() == 'op_prj_beta__items'


@pytest.mark.unbound
@pytest.mark.asyncio
@pytest.mark.usefixtures('reference_region_profile')
async def test_worker_event_publisher_posts_to_the_bound_project(
    monkeypatch: pytest.MonkeyPatch, custom_prefix: str, beta_bound: Any
) -> None:
    """The detection worker's region events go to its own project's
    ``/events/publish``, never an unscoped path."""
    from scripts.curation.worker import bulk_writer

    posted: list[str] = []

    class _Client:
        async def post(self, url: str, **_: Any) -> None:
            posted.append(url)

    class _Task:
        crop_id = 'beta-item-1'
        update_doc = {bulk_writer.get_region_fields().status: 'detected'}

    monkeypatch.setattr(bulk_writer, '_EVENT_API_URL', 'http://api')
    monkeypatch.setattr(bulk_writer, '_EVENT_CLIENT', _Client())
    await bulk_writer._publish_region_events([_Task()])  # type: ignore[list-item]
    assert posted == [f'http://api{custom_prefix}/projects/beta/events/publish']


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
    monkeypatch.delenv('OP_CURATION_PROJECT', raising=False)
    seen: dict[str, Any] = {}

    def _fake_run(**kwargs: Any) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(mod, 'run', _fake_run)
    monkeypatch.setattr(mod, 'bind_script_project', lambda _slug: None)
    monkeypatch.setattr(mod.asyncio, 'run', lambda _coro: None)
    mod.main()
    assert seen['api_base'] == 'http://localhost:4603/custom-mount/projects/default'
