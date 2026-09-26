"""``_build_yolov5_objectness_predictor`` must never treat the project class
registry's id space as the model's own dense output-index order.

The registry is append-only and gapped by design (deprecated/merged
classes keep their old id, per the class identity invariant in
``docs/design/openprocessor_internal/any_domain_plan.md``). A checkpoint's
dense output index 0..nc-1 has no relationship to those ids -- only to the
checkpoint's own embedded ``names`` metadata. These tests pin that: a
gapped registry must not affect which name a given model output index
resolves to, and a checkpoint with no embedded metadata must fail loudly
instead of guessing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest


if TYPE_CHECKING:
    from pathlib import Path

from src.services.curation import probe_models as pm


class _FakeIO:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeModelMeta:
    def __init__(self, custom_metadata_map: dict[str, str]) -> None:
        self.custom_metadata_map = custom_metadata_map


class _FakeSession:
    """Stands in for ``onnxruntime.InferenceSession``.

    ``run()`` returns a canned ``(1, num_anchors, 5 + nc)`` raw tensor with
    a single dominant anchor whose top class is ``winning_index`` -- so the
    test can assert exactly which *name* the predictor resolves for a given
    dense model-output index, independent of any registry id.
    """

    def __init__(self, names_metadata: str | None, num_classes: int, winning_index: int) -> None:
        self._names_metadata = names_metadata
        self._num_classes = num_classes
        self._winning_index = winning_index

    def get_inputs(self) -> list[_FakeIO]:
        return [_FakeIO('images')]

    def get_outputs(self) -> list[_FakeIO]:
        return [_FakeIO('output0')]

    def get_modelmeta(self) -> _FakeModelMeta:
        meta = {} if self._names_metadata is None else {'names': self._names_metadata}
        return _FakeModelMeta(meta)

    def run(self, output_names: list[str], feed: dict[str, Any]) -> list[np.ndarray]:
        del output_names, feed
        row = np.zeros(5 + self._num_classes, dtype=np.float32)
        row[4] = 1.0  # obj_conf
        row[5 + self._winning_index] = 1.0  # this class's raw sigmoid score
        return [row[None, None, :]]  # (1, 1, 5 + nc)


@pytest.fixture
def crop() -> Any:
    from PIL import Image

    return Image.new('RGB', (64, 64), (128, 128, 128))


def _patch_session(monkeypatch: pytest.MonkeyPatch, session: _FakeSession) -> None:
    import types

    fake_ort = types.ModuleType('onnxruntime')
    fake_ort.InferenceSession = lambda *_a, **_k: session  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__('sys').modules, 'onnxruntime', fake_ort)


def test_yolov5_objectness_uses_model_metadata_not_registry_id_space(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, crop: Any
) -> None:
    """Dense model-output index 2 must resolve to the checkpoint's own
    3rd name, unaffected by a registry whose class_id space has a gap."""
    # Model's own dense-order names (index -> name), embedded the way
    # Ultralytics writes them on export.
    names_metadata = "{0: 'sedan', 1: 'truck', 2: 'motorcycle'}"
    session = _FakeSession(names_metadata, num_classes=3, winning_index=2)
    _patch_session(monkeypatch, session)

    predict, _version = pm._build_yolov5_objectness_predictor(tmp_path / 'model.onnx')
    cls_name, conf, _entropy, _margin = predict(crop)

    assert cls_name == 'motorcycle'
    assert conf == pytest.approx(1.0, abs=1e-6)
    # Regression guard: this must come from the model's own metadata, not
    # a registry dict keyed by (possibly gapped) class_id -- there is no
    # registry involved in this predictor at all.
    assert predict.class_names == ('sedan', 'truck', 'motorcycle')  # type: ignore[attr-defined]


def test_yolov5_objectness_raises_loudly_without_model_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No embedded ``names`` metadata -> fail loudly, never fall back to
    guessing via the project class registry's id space."""
    session = _FakeSession(None, num_classes=3, winning_index=0)
    _patch_session(monkeypatch, session)

    with pytest.raises(pm.ProbeLabelsMissingError):
        pm._build_yolov5_objectness_predictor(tmp_path / 'model.onnx')
