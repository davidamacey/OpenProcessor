"""Fakes and the worker driver shared by the region cascade tests: a fake OpenSearch, item builders and ``_drive_worker``."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import io
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from _fake_project_registry import install_static_project_registry
from PIL import Image

from scripts.curation.worker import runner as runner_mod, stage_a as stage_a_mod
from scripts.curation.worker.__main__ import parse_args
from src.config import get_region_fields
from src.services.detection.profile_registry import get_active_region_profile, register_profile
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_labeler import VlmLabeler
from src.services.labeling.vlm_models import VlmCombinedReply


if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from src.config import DetectionProfile
    from src.services.detection.cascade_detect.candidate import RegionCandidate


CLASS_NAMES = ['sedan', 'chevycar', 'toyotacar', 'pickup']


def _profile() -> DetectionProfile:
    profile = get_active_region_profile()
    assert profile is not None
    return profile


def _chat(content: str) -> dict[str, Any]:
    return {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}


def _labeler(reply: dict[str, Any] | str) -> VlmLabeler:
    lab = VlmLabeler(base_url='http://vlm.invalid/v1')
    content = reply if isinstance(reply, str) else json.dumps(reply)
    lab._post_chat = AsyncMock(return_value=_chat(content))  # type: ignore[method-assign]
    return lab


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (64, 48), (90, 90, 90)).save(buf, format='JPEG')
    return buf.getvalue()


def _combined(**over: Any) -> dict[str, Any]:
    """A realistic combined reply, as the deployment pack asks for it.

    the box-verdict fields (``region_bbox_correct`` / ``region_text``
    / ``region_confidence``) are nested one level down, under
    ``region_boxes: [{"box": 1, ...}]`` -- pass them as flat kwargs here
    (unchanged call sites) and they land on box 1.
    """
    box_overrides = {k: over.pop(k) for k in list(over) if k in _BOX_KEYS}
    reply = {
        'class_id': 1,
        'class_confidence': 'high',
        'region_visible': True,
        'region_boxes': [
            {
                'box': 1,
                'region_bbox_correct': True,
                'region_text': 'DNV20',
                'region_confidence': 'high',
                **box_overrides,
            }
        ],
        'make': 'Chevrolet',
        'model': 'Silverado',
    }
    reply.update(over)
    return reply


_BOX_KEYS = frozenset({'region_bbox_correct', 'region_text', 'region_confidence'})


def _capture_signal_handler(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    handlers: list[Any] = []

    def _record(_self: Any, _sig: Any, cb: Any, *_a: Any) -> None:
        handlers.append(cb)

    monkeypatch.setattr(
        asyncio.get_event_loop().__class__, 'add_signal_handler', _record, raising=False
    )
    return handlers


class _FakeOpenSearch:
    """Items index with realtime ``mget`` and refresh-gated ``search``.

    ``search`` sees the last refreshed snapshot, and responds
    ``search_delay`` seconds after it took it (a search in flight while a
    write lands returns pre-write state). A bulk with
    ``refresh='wait_for'`` / ``True`` is searchable once it returns; any
    other bulk only after ``lag_searches`` further searches.
    """

    def __init__(
        self, docs: dict[str, dict[str, Any]], *, search_delay: float, lag_searches: int
    ) -> None:
        self.live = copy.deepcopy(docs)
        self.searchable = copy.deepcopy(docs)
        self.search_delay = search_delay
        self.lag_searches = lag_searches
        self._pending_refresh: int | None = None
        self.seq = dict.fromkeys(docs, 1)
        self.writes: list[tuple[str, dict[str, Any]]] = []
        self.pending_statuses = {
            'pending',
            'pending_detection',
            'pending_verify',
            'pending_verification',
        }
        # The multi-project detection worker discovers its projects
        # through this same (patchable, per-test) OpenSearch client
        # (scripts/curation/worker/runner.py `run()`) -- seed a single
        # `default` project doc so ProjectRegistry.ensure_fresh() finds
        # exactly one active project instead of silently discovering
        # none and processing nothing.
        from src.config.curation import base_curation_config
        from src.config.projects import DEFAULT_SLUG, new_project_record
        from src.services.projects.registry import REVISION_DOC_ID, projects_index, record_to_doc

        self._projects_index = projects_index()
        self._project_doc = record_to_doc(new_project_record(DEFAULT_SLUG, base_curation_config()))
        self._revision_doc_id = REVISION_DOC_ID

    async def get(self, *, index: str, **kw: Any) -> dict[str, Any]:
        doc_id = kw.get('id')
        if index == self._projects_index and doc_id == self._revision_doc_id:

            class _NotFoundError(Exception):
                status_code = 404

            raise _NotFoundError('no revision doc in the fake registry index')
        msg = (
            f'_FakeOpenSearch.get is only wired for the projects revision doc, got {index}/{doc_id}'
        )
        raise AttributeError(msg)

    async def search(self, *, index: str | None = None, **_kw: Any) -> dict[str, Any]:
        if index == self._projects_index:
            # One page, no search_after pagination needed for one doc.
            if _kw.get('body', {}).get('search_after'):
                return {'hits': {'hits': []}}
            return {'hits': {'hits': [{'_source': self._project_doc, 'sort': ['default']}]}}
        if self._pending_refresh is not None:
            self._pending_refresh -= 1
            if self._pending_refresh <= 0:
                self.searchable = copy.deepcopy(self.live)
                self._pending_refresh = None
        status_key = get_region_fields().status
        hits = [
            {'_id': cid, '_source': copy.deepcopy(src)}
            for cid, src in self.searchable.items()
            if src.get(status_key) in self.pending_statuses
        ]
        await asyncio.sleep(self.search_delay)
        return {'hits': {'hits': hits}}

    async def mget(self, *, body: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        docs = []
        for d in body['docs']:
            cid = d['_id']
            docs.append(
                {
                    '_id': cid,
                    'found': cid in self.live,
                    '_seq_no': self.seq.get(cid, 1),
                    '_primary_term': 1,
                    '_source': copy.deepcopy(self.live.get(cid, {})),
                }
            )
        return {'docs': docs}

    async def bulk(self, *, body: list[dict[str, Any]], refresh: Any = False) -> dict[str, Any]:
        items = []
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            cid = action['update']['_id']
            if action['update'].get('if_seq_no') != self.seq[cid]:
                items.append({'update': {'_id': cid, 'status': 409}})
                continue
            self.live[cid].update(doc['doc'])
            self.seq[cid] += 1
            self.writes.append((cid, copy.deepcopy(doc['doc'])))
            items.append({'update': {'_id': cid, 'status': 200, 'result': 'updated'}})
        if refresh in ('wait_for', True):
            self.searchable = copy.deepcopy(self.live)
        else:
            self._pending_refresh = self.lag_searches
        return {'errors': False, 'items': items}

    async def close(self) -> None:
        return None


def _item(status: str = 'pending_detection') -> dict[str, Any]:
    F = get_region_fields()
    return {
        'crop_id': 'c1',
        'image_path': '/nonexistent/source.jpg',
        'bbox_norm': [0.1, 0.1, 0.9, 0.9],
        F.status: status,
        'class_name': 'sedan',
        'group': 'cars',
        'class_source': 'classifier_model',
        'confidence': 0.95,
        'created_at': '2026-09-24T00:00:00+00:00',
    }


def _as_candidate_list(c: RegionCandidate | list[RegionCandidate] | None) -> list[RegionCandidate]:
    """Normalize a ``_drive_worker`` leg input to a list (a test
    can feed 2+ raw candidates per leg, not just 0 or 1)."""
    if c is None:
        return []
    if isinstance(c, list):
        return c
    return [c]


async def _drive_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake_os: _FakeOpenSearch,
    primary: RegionCandidate | list[RegionCandidate] | None,
    segmenter: RegionCandidate | list[RegionCandidate] | None,
    reply: VlmCombinedReply,
    visible: bool | None = True,
    combined_side_effect: Any = None,
    visible_side_effect: Any = None,
    until_writes: int = 1,
    on_write: Any = None,
    class_group: Any = None,
    profile_overrides: dict[str, Any] | None = None,
    region_embed_ready: bool = False,
) -> dict[str, Any]:
    """Run the streaming worker in continuous mode until the item is
    written plus several more polls, then stop it. Returns the mocks.
    ``visible=None``: the visibility pre-filter gives no verdict at all.
    ``combined_side_effect`` / ``visible_side_effect`` replace the VLM
    mocks' fixed answers (called with the crop list). The run stops once
    ``until_writes`` writes landed (or a timeout); ``on_write(n)`` is
    called as the n-th write is seen. ``class_group`` overrides the
    class -> group resolver (default: every class is group-less);
    ``profile_overrides`` re-registers the active profile with those
    dataclass fields replaced (e.g. ``secondary_shape_groups``).

    ``primary`` / ``segmenter`` each take a single :class:`RegionCandidate`,
    ``None``, OR a ``list[RegionCandidate]`` -- a list feeds N raw
    candidates through that leg's ``detect_batch_multi`` / ``segment_multi``
    in one pass, exercising the real multi-box selection/verify/write path
    instead of the single-candidate shape every pre-M8 test used.

    ``region_embed_ready`` (R-M1 re-review fix gate, 2026-09-27): every
    other test leaves this False, which is exactly the pre-existing gap
    the re-review flagged -- ``run()``'s readiness probe
    (``pool.is_model_ready(...)``) always fails against a bare
    ``MagicMock``, so the region-embed stage is silently disabled and no
    test here ever drove it. Passing True stubs ``pool.is_model_ready``
    to succeed so a caller can also patch
    ``src.clients.pe_encoder.PEEncoder`` and assert on the REAL written
    ``F.embedding`` / published event, instead of spying on an internal
    helper.
    """
    handlers = _capture_signal_handler(monkeypatch)
    monkeypatch.setenv('OP_REGION_WORKER_METRICS_PORT', '0')
    if profile_overrides:
        register_profile(dataclasses.replace(_profile(), **profile_overrides), default=True)

    pool = MagicMock(initialize=AsyncMock(), close=AsyncMock())
    if region_embed_ready:
        pool.is_model_ready = AsyncMock(return_value=True)
    monkeypatch.setattr(runner_mod, 'AsyncTritonPool', MagicMock(return_value=pool))
    monkeypatch.setattr(runner_mod, 'make_script_opensearch', MagicMock(return_value=fake_os))
    # The worker discovers its projects through the registry; keep that
    # read in-memory too, never against the unresolvable --opensearch host.
    install_static_project_registry(monkeypatch)

    primary_list = _as_candidate_list(primary)
    segmenter_list = _as_candidate_list(segmenter)

    primary_det = MagicMock()
    primary_det.confidence_floor = 0.0
    primary_det.detect_batch_multi = AsyncMock(return_value=[primary_list])
    monkeypatch.setattr(runner_mod, 'RegionDetector', MagicMock(return_value=primary_det))
    ocr = MagicMock()
    ocr.detect_regions = AsyncMock(return_value=[])
    ocr.pick_best_text_region = MagicMock(return_value=None)
    monkeypatch.setattr(runner_mod, 'PaddleOcrTextRecognizer', MagicMock(return_value=ocr))
    monkeypatch.setattr(stage_a_mod, '_crop_jpeg_for_task', lambda *_a: _jpeg())

    seg = MagicMock(aclose=AsyncMock())
    seg.segment = AsyncMock(return_value=segmenter_list[0] if segmenter_list else None)
    seg.segment_multi = AsyncMock(return_value=segmenter_list)
    monkeypatch.setattr(runner_mod, 'SegmenterClient', MagicMock(return_value=seg))

    vlm = MagicMock(aclose=AsyncMock())
    vlm.class_names = []
    vlm.label_combined_batch = AsyncMock(
        side_effect=combined_side_effect or (lambda crops, **_kw: {c.crop_id: reply for c in crops})
    )
    vlm.region_visible_batch = AsyncMock(
        side_effect=visible_side_effect
        or (lambda crops, **_kw: {} if visible is None else {c.crop_id: visible for c in crops})
    )
    vlm_cls = MagicMock(return_value=vlm)
    monkeypatch.setattr(runner_mod, 'build_vlm_labeler', vlm_cls)
    # Registry load is best-effort; keep it out of the test.
    monkeypatch.setattr(
        'src.clients.curation_opensearch.registry.ClassRegistry',
        MagicMock(side_effect=RuntimeError('no registry in test')),
    )
    # Stage A resolves the class group through the process-wide registry
    # singleton; without this the test depends on another test having
    # loaded it first.
    monkeypatch.setattr(
        'scripts.curation.worker.state._class_group', class_group or (lambda _name: None)
    )

    monkeypatch.setenv('OP_VLM_URL', 'http://vlm.invalid:8000')
    args = parse_args(
        [
            '--opensearch=http://os.invalid:9200',
            '--triton=triton.invalid:8001',
            '--segmenter-url=http://seg.invalid:8000',
            f'--pause-sentinel={tmp_path / "absent.sentinel"}',
            '--continuous',
            '--poll-interval=0.01',
            '--batch-size=4',
            '--concurrency=2',
        ]
    )

    async def _stopper() -> None:
        # This used to be a FIXED 500 x 10ms iteration budget -- a ~5s nominal wall-clock allowance assuming
        # each `asyncio.sleep(0.01)` returns close to schedule. The actual
        # cause of the flake is plain wall-clock variance against that
        # fixed ~5s budget, not individual sleeps somehow running long
        # (a slower `asyncio.sleep(0.01)` makes 500 iterations take MORE
        # wall time, not less, so that direction was never the problem).
        # Under load (e.g. full-suite xdist runs sharing CPU across many
        # parallel workers), THIS test's own actual work -- the worker's
        # producer polls plus 6 real combined-VLM round trips across 2
        # write cycles -- can simply take longer in wall-clock terms than
        # a fixed ~5s allowance, independent of how the budget is framed;
        # an observed ~1/12 failure rate standalone confirmed it (0/12
        # with a generous budget). Switched to a deadline on the event
        # loop's own monotonic clock: 18s gives ~7x headroom over the
        # measured ~2.66s single-attempt latency, so CPU contention costs
        # wall-clock slack against a generous ceiling instead of a
        # marginal iteration count.
        seen = 0
        deadline = asyncio.get_running_loop().time() + 18.0
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
            while seen < len(fake_os.writes):
                seen += 1
                if on_write is not None:
                    on_write(seen)
            if seen >= until_writes:
                break
        # Keep polling well past the write so a lagged re-fetch would show.
        await asyncio.sleep(0.6)
        handlers[0]()

    stopper = asyncio.create_task(_stopper())
    rc = await asyncio.wait_for(runner_mod.run(args), timeout=30)
    await stopper
    assert rc == 0
    return {'primary': primary_det, 'seg': seg, 'vlm': vlm, 'vlm_cls': vlm_cls}


def _accept(text: str = 'DNV20') -> VlmCombinedReply:
    return VlmCombinedReply(
        img_id='c1',
        region_visible=True,
        region_boxes=[VlmBoxVerdict(box=1, bbox_correct=True, confidence='high', text_reply=text)],
    )
