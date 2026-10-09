"""A running curation app for the test-on-crop routes (W5).

The real app (scoped + global routers) over an in-memory config store and an
in-memory item store, with a fake VLM and a fake segmenter behind real
``httpx`` transports. Servable source images live under ``tmp_path``. Nothing
here is the code under test: it is the boundary the routes talk to.
"""

from __future__ import annotations

import copy
import io
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
from PIL import Image

from curation.conftest import SCOPED, HybridOpenSearch, VlmApi, good_probe
from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch.registry import ClassRegistry
from src.config.curation import base_curation_config


if TYPE_CHECKING:
    from pathlib import Path

    import pytest

ITEMS = base_curation_config().items_index
VLM_HOST = 'a.vlm.test'
SEGMENTER_URL = 'http://seg.test:8000'
ITEM_BOX = [0.1, 0.1, 0.9, 0.9]


class Upstream:
    """``a.vlm.test`` (OpenAI-shaped) and ``seg.test`` (the segmenter service).

    Every request is recorded; ``vlm_content`` is the canned chat reply and
    ``segments`` the canned candidates (crop-frame, as the service serves)."""

    def __init__(self) -> None:
        self.vlm_content = '{}'
        self.vlm_reasoning: str | None = None
        self.vlm_requests: list[dict[str, Any]] = []
        self.vlm_delay: Any = None
        self.segments: list[dict[str, Any]] = []
        self.segment_requests: list[dict[str, Any]] = []
        self.segmenter_up = True
        #: A non-200 status the segmenter / the VLM answer with (an upstream 5xx).
        self.segmenter_status = 200
        self.vlm_status = 200
        #: Hosts besides ``*.vlm.test`` that answer chat completions.
        self.extra_vlm_hosts: set[str] = set()

    async def handle(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == 'seg.test':
            if not self.segmenter_up:
                raise httpx.ConnectError('segmenter down', request=request)
            if request.url.path == '/health':
                return httpx.Response(200, json={'status': 'healthy', 'loaded': True})
            payload = json.loads(request.content)
            self.segment_requests.append(payload)
            if self.segmenter_status != 200:
                return httpx.Response(self.segmenter_status, json={'detail': 'boom'})
            return httpx.Response(
                200,
                json={
                    'candidates': self.segments,
                    'elapsed_ms': 1.0,
                    'crop_size': [10, 10],
                    'prompt': payload['text_prompt'],
                },
            )
        is_vlm = host.endswith('.vlm.test') or host in self.extra_vlm_hosts
        if is_vlm and request.url.path.endswith('/chat/completions'):
            payload = json.loads(request.content)
            payload['_host'] = host
            self.vlm_requests.append(payload)
            if self.vlm_delay is not None:
                await self.vlm_delay()
            if self.vlm_status != 200:
                return httpx.Response(self.vlm_status, json={'error': 'boom'})
            message: dict[str, Any] = {'content': self.vlm_content}
            if self.vlm_reasoning is not None:
                message['reasoning_content'] = self.vlm_reasoning
            return httpx.Response(200, json={'choices': [{'message': message}]})
        raise httpx.ConnectError(f'no network to {host}', request=request)


@dataclass
class Stack:
    api: VlmApi
    items: QueryFakeOpenSearch
    upstream: Upstream
    registry: ClassRegistry
    tmp_path: Path
    image_path: str
    seeded: list[str] = field(default_factory=list)

    @property
    def client(self) -> Any:
        return self.api.client

    def seed(self, crop_id: str, **doc: Any) -> dict[str, Any]:
        item: dict[str, Any] = {
            'crop_id': crop_id,
            'image_id': f'img-{crop_id}',
            'image_path': self.image_path,
            'bbox_norm': ITEM_BOX,
            'class_id': None,
            'class_name': '',
            'class_source': '',
            'class_validated': False,
            'region_status': 'pending_detection',
            'region_boxes': [],
            'region_count': 0,
            'region_rejected_count': 0,
            'region_revision': 0,
            'region_box_seq': 0,
        }
        item.update(doc)
        self.items.store.setdefault(ITEMS, {})[crop_id] = item
        self.seeded.append(crop_id)
        return item

    def post(self, path: str, **body: Any) -> Any:
        return self.client.post(f'{SCOPED}{path}', json=body)

    def snapshot(self) -> dict[str, Any]:
        """Every document of both stores, for a before/after comparison."""
        return {
            'items': copy.deepcopy(self.items.store),
            'configs': copy.deepcopy(self.api.fake_os._docs),
        }


def build_stack(vlm_api: VlmApi, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Stack:
    from curation.conftest import ACTIVE
    from src.routers.curation import _raw_opensearch_dep, config_test as config_test_mod
    from src.services.curation import image_serving

    upstream = Upstream()

    async def serve(_transport: Any, request: httpx.Request) -> httpx.Response:
        return await upstream.handle(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', serve)
    monkeypatch.setenv('OP_SEGMENTER_URL', SEGMENTER_URL)
    vlm_api.dns['a.vlm.test'] = ['10.0.0.1']

    frame = tmp_path / 'frame.jpg'
    buf = io.BytesIO()
    Image.new('RGB', (400, 300), (90, 120, 150)).save(buf, format='JPEG')
    frame.write_bytes(buf.getvalue())
    monkeypatch.setattr(image_serving, '_configured_roots', lambda config=None: (tmp_path,))  # noqa: ARG005
    image_serving.THUMBNAIL_CACHE.clear()

    import src.config.curation as curation_config_mod

    monkeypatch.setenv('OP_CROP_CACHE_DIR', str(tmp_path / 'crop_cache'))
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)

    registry = ClassRegistry(path=tmp_path / 'class_registry.json')
    registry.add_class('widget')
    registry.add_class('gadget')
    monkeypatch.setattr(config_test_mod, 'get_class_registry', lambda: registry)

    items = QueryFakeOpenSearch({ITEMS: {}})
    hybrid = HybridOpenSearch(vlm_api.fake_os, items)
    vlm_api.client.app.dependency_overrides[_raw_opensearch_dep] = lambda: hybrid

    vlm_api.probe_record[0] = good_probe(root='org/a-root')
    vlm_api.ready('ua', base_url=f'http://{VLM_HOST}/v1', model='alias-a')
    assert (
        vlm_api.client.post(f'{ACTIVE}/ua/activate', json={'expected_active': None}).status_code
        == 200
    )
    return Stack(vlm_api, items, upstream, registry, tmp_path, str(frame))


def profile_body(**over: Any) -> dict[str, Any]:
    """A text-free, segmenter-only draft profile (no Triton leg)."""
    from dataclasses import asdict

    from src.config import DetectionProfile

    raw = asdict(DetectionProfile(name='p'))
    raw.pop('name')
    for key, value in raw.items():
        if isinstance(value, frozenset):
            raw[key] = sorted(value)
        elif isinstance(value, tuple):
            raw[key] = list(value)
    raw.update(
        detector_model='',
        text_reader='none',
        text_hint_enabled=False,
        ocr_pipeline_model='',
        segmenter_text_prompt='wheel',
        region_class_name='wheel',
        display_name='Wheels',
        display_name_singular='Wheel',
        max_regions_per_item=2,
    )
    raw.update(over)
    return raw


def namespace(**kw: Any) -> SimpleNamespace:
    return SimpleNamespace(**kw)
