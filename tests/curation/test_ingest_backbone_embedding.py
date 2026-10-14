"""Backbone embedding from a dual-head secondary detector, and visible failures of that detector."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from curation.ingest_fakes import (
    FakeInferResult,
    FakePEEncoder,
    FakeTritonPool,
    _jpeg_bytes,
    _two_class_registry,
)
from curation.occ_fakes import FakeIngestOpenSearch
from src.config import CurationConfig, DetectionProfile
from src.services.curation.ingest import CurationIngestService


# =============================================================================
# N3 — backbone embedding from a dual-head secondary detector
# =============================================================================

_SECONDARY_MODEL = 'secondary_raw'
_FEATURE_DIM = 8
_GRID = 10  # 320 px secondary input / stride 32


class FakeDualTritonPool(FakeTritonPool):
    """Primary end2end detector plus a raw-output secondary detector that
    optionally exposes a backbone feature map (``sppf_feat``).

    Like Triton, asking for an output the model does not have is an error,
    so a caller that requests ``sppf_feat`` from a single-head model fails
    loudly here rather than silently succeeding.
    """

    def __init__(
        self,
        detections: list[tuple[float, float, float, float, float, int]],
        *,
        secondary_raw: np.ndarray,
        feature_map: np.ndarray | None,
        has_probe: bool = True,
        probe_error: bool = False,
    ) -> None:
        super().__init__(detections)
        self.secondary_raw = secondary_raw
        self.feature_map = feature_map
        self.requested: list[list[str]] = []
        self.probe_error = probe_error
        if has_probe:
            self.get_model_output_names = self._output_names

    async def _output_names(self, model_name: str) -> list[str]:
        if self.probe_error:
            raise RuntimeError('metadata unavailable')
        if model_name == _SECONDARY_MODEL:
            return ['output0'] + (['sppf_feat'] if self.feature_map is not None else [])
        return ['num_dets', 'det_boxes', 'det_scores', 'det_classes']

    async def infer(self, model_name: str, inputs: list, outputs: list) -> FakeInferResult:
        if model_name != _SECONDARY_MODEL:
            return await super().infer(model_name, inputs, outputs)
        batch = int(inputs[0].shape()[0])
        names = [o.name() for o in outputs]
        self.calls.append(model_name)
        self.batch_sizes.append(batch)
        self.requested.append(names)
        out = {'output0': np.tile(self.secondary_raw[None], (batch, 1, 1))}
        if 'sppf_feat' in names:
            if self.feature_map is None:
                raise RuntimeError("unexpected inference output 'sppf_feat'")
            out['sppf_feat'] = np.tile(self.feature_map[None], (batch, 1, 1, 1))
        return FakeInferResult(out)


def _secondary_raw() -> np.ndarray:
    """``(N, 5 + nc)`` YOLOv5 rows in 320-px letterbox space. Row 0 matches
    the primary box (letterbox 16..176) and votes class 0; the rest are
    below any confidence floor."""
    rows = np.zeros((4, 7), dtype=np.float32)
    rows[0] = [96.0, 96.0, 160.0, 160.0, 0.95, 1.0, 0.0]
    return rows


def _feature_map(channels: int = _FEATURE_DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((channels, _GRID, _GRID)).astype(np.float32)


def _make_dual_service(
    *,
    feature_map: np.ndarray | None,
    backbone_dim: int = _FEATURE_DIM,
    has_probe: bool = True,
    probe_error: bool = False,
) -> tuple[CurationIngestService, FakeIngestOpenSearch, FakeDualTritonPool]:
    os_fake = FakeIngestOpenSearch()
    triton = FakeDualTritonPool(
        [(0.05, 0.05, 0.55, 0.55, 0.9, 1)],
        secondary_raw=_secondary_raw(),
        feature_map=feature_map,
        has_probe=has_probe,
        probe_error=probe_error,
    )
    svc = CurationIngestService(
        opensearch=os_fake,
        triton_pool=triton,
        registry=_two_class_registry(),
        profile=DetectionProfile(
            name='primary',
            detector_model='primary_end2end',
            assigns_class=True,
            input_size=320,
            batch_limit=8,
        ),
        secondary_profile=DetectionProfile(
            name='secondary',
            detector_model=_SECONDARY_MODEL,
            detector_version='3',
            input_size=320,
            confidence_floor=0.5,
            batch_limit=8,
        ),
        pe_encoder=FakePEEncoder(),
        config=CurationConfig(backbone_embedding_dim=backbone_dim),
    )
    return svc, os_fake, triton


# The primary box (0.05..0.55 of the 320-px input) in the secondary's own
# letterbox space — both detectors share a 320-px input here, so it is the
# same square the primary saw. Edges sit mid-cell (stride 32) so float
# rounding can't move the pooled grid window.
_ITEM_LETTERBOX_BOX = (16.0, 16.0, 176.0, 176.0)


class TestBackboneEmbedding:
    @pytest.mark.asyncio
    async def test_dual_head_secondary_writes_pooled_backbone_embedding(self) -> None:
        from src.config import BACKBONE_EMBEDDING_FIELD
        from src.services.detection.geometry import roi_pool_sppf

        fmap = _feature_map()
        svc, os_fake, triton = _make_dual_service(feature_map=fmap)
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/dual.jpg')
        assert result.status == 'success'

        assert triton.requested == [['output0', 'sppf_feat']]
        [doc] = list(os_fake.items.values())
        expected = roi_pool_sppf(fmap, _ITEM_LETTERBOX_BOX, input_size=320, target_dim=8)
        assert len(doc[BACKBONE_EMBEDDING_FIELD]) == svc.config.backbone_embedding_dim
        assert doc[BACKBONE_EMBEDDING_FIELD] == pytest.approx(expected.tolist(), abs=1e-6)
        # Secondary override also carries its own class provenance (N1).
        assert doc['class_id'] == 0
        assert doc['class_detector'] == _SECONDARY_MODEL
        assert doc['class_detector_version'] == '3'

    @pytest.mark.asyncio
    async def test_single_head_secondary_is_called_exactly_as_before(self) -> None:
        from src.config import BACKBONE_EMBEDDING_FIELD

        svc, os_fake, triton = _make_dual_service(feature_map=None)
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/single.jpg')

        assert result.status == 'success'
        assert triton.requested == [['output0']]
        [doc] = list(os_fake.items.values())
        assert BACKBONE_EMBEDDING_FIELD not in doc
        assert doc['class_id'] == 0  # class override still applied

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('has_probe', 'probe_error'), [(False, False), (True, True)])
    async def test_unknown_outputs_never_request_the_feature_map(
        self, has_probe: bool, probe_error: bool
    ) -> None:
        """No way to learn the model's outputs (pool without a metadata
        probe, or the probe failing) must fall back to ``output0`` only."""
        from src.config import BACKBONE_EMBEDDING_FIELD

        svc, os_fake, triton = _make_dual_service(
            feature_map=_feature_map(), has_probe=has_probe, probe_error=probe_error
        )
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/noprobe.jpg')

        assert result.status == 'success'
        assert triton.requested == [['output0']]
        [doc] = list(os_fake.items.values())
        assert BACKBONE_EMBEDDING_FIELD not in doc

    @pytest.mark.asyncio
    async def test_batched_path_writes_identical_embeddings_in_one_call(self) -> None:
        from src.config import BACKBONE_EMBEDDING_FIELD

        images = [_jpeg_bytes(seed=800 + s) for s in range(3)]
        paths = [f'/tmp/bb{s}.jpg' for s in range(3)]
        fmap = _feature_map()

        svc_batch, os_batch, triton = _make_dual_service(feature_map=fmap)
        await svc_batch.ingest_batch(images, paths)
        assert triton.calls.count(_SECONDARY_MODEL) == 1
        assert triton.requested == [['output0', 'sppf_feat']]
        assert triton.batch_sizes[triton.calls.index(_SECONDARY_MODEL)] == 3

        svc_single, os_single, _ = _make_dual_service(feature_map=fmap)
        for data, path in zip(images, paths, strict=True):
            await svc_single.ingest_one(data, path, source='batch')

        volatile = {'created_at', 'updated_at', 'class_labeled_at'}

        def _comparable(store: dict) -> list[dict]:
            return sorted(
                ({k: v for k, v in d.items() if k not in volatile} for d in store.values()),
                key=lambda d: d['crop_id'],
            )

        assert all(BACKBONE_EMBEDDING_FIELD in d for d in os_batch.items.values())
        assert _comparable(os_batch.items) == _comparable(os_single.items)

    @pytest.mark.asyncio
    async def test_narrower_feature_map_is_zero_padded_to_the_mapping_dim(self) -> None:
        from src.config import BACKBONE_EMBEDDING_FIELD

        svc, os_fake, _ = _make_dual_service(feature_map=_feature_map(), backbone_dim=12)
        await svc.ingest_one(_jpeg_bytes(), '/tmp/pad.jpg')

        [doc] = list(os_fake.items.values())
        vec = doc[BACKBONE_EMBEDDING_FIELD]
        assert len(vec) == 12
        assert vec[_FEATURE_DIM:] == [0.0] * (12 - _FEATURE_DIM)
        assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.asyncio
    async def test_wider_feature_map_than_mapping_dim_is_skipped_not_truncated(self) -> None:
        """Truncating channels would write a vector that silently means
        something else; skip the field (and log) instead. Ingest still
        succeeds."""
        from src.config import BACKBONE_EMBEDDING_FIELD

        svc, os_fake, _ = _make_dual_service(feature_map=_feature_map(), backbone_dim=4)
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/wide.jpg')

        assert result.status == 'success'
        [doc] = list(os_fake.items.values())
        assert BACKBONE_EMBEDDING_FIELD not in doc
        assert doc['class_id'] == 0

    @pytest.mark.asyncio
    async def test_nms_failure_keeps_embedding_and_primary_class(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The class override and the embedding pooling fail independently."""
        from src.config import BACKBONE_EMBEDDING_FIELD
        from src.services.detection import ensemble_nms

        def _broken_nms(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError('nms unavailable')

        monkeypatch.setattr(ensemble_nms, 'apply_ensemble_nms', _broken_nms)
        svc, os_fake, _ = _make_dual_service(feature_map=_feature_map())
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/nonms.jpg')

        assert result.status == 'success'
        [doc] = list(os_fake.items.values())
        assert doc['class_id'] == 1  # primary's class, no override
        assert doc['class_detector'] == 'primary_end2end'
        assert len(doc[BACKBONE_EMBEDDING_FIELD]) == _FEATURE_DIM


# =============================================================================
# a failing secondary detector call must be visible, not swallowed
# =============================================================================


class _SecondaryFailsTritonPool(FakeDualTritonPool):
    """Secondary detector raises on every call (e.g. a gRPC
    DEADLINE_EXCEEDED); the primary detector still succeeds normally."""

    async def infer(self, model_name: str, inputs: list, outputs: list) -> FakeInferResult:
        if model_name == _SECONDARY_MODEL:
            self.calls.append(model_name)
            raise RuntimeError('StatusCode.DEADLINE_EXCEEDED')
        return await super().infer(model_name, inputs, outputs)


def _make_failing_secondary_service() -> tuple[
    CurationIngestService, FakeIngestOpenSearch, _SecondaryFailsTritonPool
]:
    os_fake = FakeIngestOpenSearch()
    triton = _SecondaryFailsTritonPool(
        [(0.05, 0.05, 0.55, 0.55, 0.9, 1)],
        secondary_raw=_secondary_raw(),
        feature_map=None,
    )
    svc = CurationIngestService(
        opensearch=os_fake,
        triton_pool=triton,
        registry=_two_class_registry(),
        profile=DetectionProfile(
            name='primary',
            detector_model='primary_end2end',
            assigns_class=True,
            input_size=320,
            batch_limit=8,
        ),
        secondary_profile=DetectionProfile(
            name='secondary',
            detector_model=_SECONDARY_MODEL,
            detector_version='3',
            input_size=320,
            confidence_floor=0.5,
            batch_limit=8,
        ),
        pe_encoder=FakePEEncoder(),
        config=CurationConfig(),
    )
    return svc, os_fake, triton


class TestSecondaryDetectorFailureVisibility:
    @pytest.mark.asyncio
    async def test_ingest_one_still_succeeds_but_reports_the_secondary_error(self) -> None:
        """The image ingests fine on the primary detector alone, but
        the caller must be able to see that the configured secondary
        detector never ran -- not just a 'warning' log line."""
        svc, os_fake, _triton = _make_failing_secondary_service()

        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/secondary_fail.jpg')

        assert result.status == 'success'
        assert result.secondary_detector_error is not None
        assert 'DEADLINE_EXCEEDED' in result.secondary_detector_error
        # Primary-only class provenance -- the secondary override never
        # landed because the secondary call itself failed.
        [doc] = list(os_fake.items.values())
        assert doc['class_detector'] == 'primary_end2end'

    @pytest.mark.asyncio
    async def test_batch_summary_counts_secondary_failures(self) -> None:
        """A batch of otherwise-successful ingests must roll up how
        many of them silently lost their secondary classifier, in the
        summary the caller actually reads (not just per-item)."""
        svc, _os_fake, _triton = _make_failing_secondary_service()
        images = [_jpeg_bytes(seed=900 + s) for s in range(3)]
        paths = [f'/tmp/secfail{s}.jpg' for s in range(3)]

        batch_result = await svc.ingest_batch(images, paths)

        assert batch_result.summary.successful == 3
        assert batch_result.summary.secondary_detector_failures == 3
        assert all(r.secondary_detector_error is not None for r in batch_result.results)

    @pytest.mark.asyncio
    async def test_router_batch_response_echoes_secondary_failures(self) -> None:
        """The wire response (what an operator/CI actually reads)
        must carry the count and the per-item error, not just the
        internal service-layer dataclass."""
        from src.routers.curation.ingest import _batch_response
        from src.services.curation.ingest_models import (
            BatchIngestResult,
            IngestResult,
            IngestSummary,
        )

        batch = BatchIngestResult(
            summary=IngestSummary(successful=1, secondary_detector_failures=1),
            results=[
                IngestResult(
                    status='success',
                    image_path='/a.jpg',
                    n_crops=1,
                    secondary_detector_error='StatusCode.DEADLINE_EXCEEDED',
                )
            ],
        )
        resp = _batch_response(batch, [])
        assert resp.summary.secondary_detector_failures == 1
        assert resp.results[0].secondary_detector_error == 'StatusCode.DEADLINE_EXCEEDED'
