"""The YOLOv5 objectness probe reads its input size from the ONNX instead of assuming 1280.

An original-YOLOv5 export (official ``ultralytics/yolov5`` ``export.py``) is 640 static by
default; the probe hard-coded 1280 and onnxruntime refused the feed ("Got 1280 Expected 640").
The synthetic graphs below run real onnxruntime on the CPU, offline; the ``families`` tests use
the official exports from the cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from src.services.curation import probe_models as pm


if TYPE_CHECKING:
    from pathlib import Path

NAMES = {0: 'aa', 1: 'bb', 2: 'cc'}


def _write_probe_onnx(
    path: Path,
    input_hw: list[int | str],
    *,
    nc: int = 3,
    names: dict[int, str] | None = NAMES,
    winner: int = 2,
) -> Path:
    """A graph with a constant ``(1, 4, 5 + nc)`` output whose strongest anchor says ``winner``."""
    out = np.zeros((1, 4, 5 + nc), dtype=np.float32)
    out[0, 1, 4] = 1.0
    out[0, 1, 5 + winner] = 1.0
    const = helper.make_node(
        'Constant', [], ['output0'], value=numpy_helper.from_array(out, name='c')
    )
    graph = helper.make_graph(
        [const],
        'probe',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, *input_hw])],
        [helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1, 4, 5 + nc])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    if names is not None:
        entry = model.metadata_props.add()
        entry.key, entry.value = 'names', repr(names)
    onnx.save(model, str(path))
    return path


@pytest.fixture
def crop() -> Image.Image:
    return Image.new('RGB', (96, 64), (128, 128, 128))


@pytest.mark.parametrize('size', [640, 320])
def test_static_input_size_comes_from_the_onnx(tmp_path: Path, crop: Image.Image, size: int):
    predict, _ = pm._build_yolov5_objectness_predictor(
        _write_probe_onnx(tmp_path / 'm.onnx', [size, size])
    )

    name, conf, _, _ = predict(crop)

    assert name == 'cc'
    assert conf == pytest.approx(1.0)


def test_dynamic_input_uses_the_named_fallback_size(tmp_path: Path, crop: Image.Image):
    predict, _ = pm._build_yolov5_objectness_predictor(
        _write_probe_onnx(tmp_path / 'm.onnx', ['height', 'width'])
    )

    assert predict(crop)[0] == 'cc'
    assert pm.YOLOV5_OBJ_DYNAMIC_INPUT_SIZE == 1280


def test_a_rectangular_static_input_is_refused(tmp_path: Path):
    with pytest.raises(pm.ProbeModelShapeError, match='square'):
        pm._build_yolov5_objectness_predictor(_write_probe_onnx(tmp_path / 'm.onnx', [640, 1280]))


def test_an_output_whose_width_does_not_match_the_names_is_refused(tmp_path: Path):
    path = _write_probe_onnx(tmp_path / 'm.onnx', [640, 640], nc=5)  # output 10 wide, 3 names

    with pytest.raises(pm.ProbeModelShapeError, match=r'5 \+ nc|names'):
        pm._build_yolov5_objectness_predictor(path)


@pytest.fixture
def street_image(tmp_path: Path) -> Image.Image:
    """The public COCO street scene (a bus, readable text) the demo targets use."""
    from scripts.datasets.fetch_test_images import STREET_IMAGE, fetch

    fetch(tmp_path)
    return Image.open(tmp_path / STREET_IMAGE).convert('RGB')


@pytest.mark.families
@pytest.mark.parametrize('export', ['yolov5n_640.onnx', 'yolov5n_dynamic.onnx'])
def test_official_yolov5_exports_predict_a_named_class(
    op_weights, street_image: Image.Image, export: str
) -> None:
    predict, _ = pm._build_yolov5_objectness_predictor(op_weights(export))

    name, conf, _, _ = predict(street_image)

    assert name in predict.class_names  # type: ignore[attr-defined]
    assert conf > 0


@pytest.mark.families
def test_ensemble_nms_returns_detections_on_the_street_image(
    op_weights, street_image: Image.Image
) -> None:
    import onnxruntime as ort

    from src.services.detection.ensemble_nms import apply_ensemble_nms

    path = op_weights('yolov5n_640.onnx')
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    chw = pm._letterbox_yolov5_objectness(street_image, 640)
    raw = session.run(None, {session.get_inputs()[0].name: chw})[0]

    detections = apply_ensemble_nms(raw, conf_thres=0.25)

    assert detections
    assert detections[0]
