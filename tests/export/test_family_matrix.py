"""The U-6.1 family matrix at the pinned Ultralytics: load, head kind, export, parity, over-batch.

Marker ``families``: official weights from ``~/.cache/op-test-weights`` (CI job ``families`` fills
it), CPU only, nothing committed. One row per family of ``src/domain/model_families.py``; each
row goes through the production function the stack uses for every head kind,
``export/ultralytics_export.py::export_fused_onnx`` with its ``inspect_fused_output`` head check;
the arguments it passes are ``model_families.export_kwargs``, the recipe the trainer's promote
export runs too (``docker/trainer/trainer_fit.py``).

Documented arguments: https://docs.ultralytics.com/modes/export/ (``nms``, ``batch``, ``dynamic``,
``simplify``). The original-YOLOv5 ONNX rows (item 6) live in ``tests/test_yolov5_onnx_probe.py``,
the original-YOLOv5 ``.pt`` refusal in ``tests/test_validate_pytorch_model_families.py``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
import pytest

from src.domain.model_families import FAMILIES, export_kwargs, head_kind_of


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

pytestmark = pytest.mark.families

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_BATCH = 8
IMGSZ = 640

# row id -> (weights file, registry family)
ROWS: dict[str, tuple[str, str]] = {
    'yolov5nu': ('yolov5nu.pt', 'yolov5u'),
    'yolov8n': ('yolov8n.pt', 'yolov8'),
    'yolov9t': ('yolov9t.pt', 'yolov9'),
    'yolov10n': ('yolov10n.pt', 'yolov10'),
    'yolo11n': ('yolo11n.pt', 'yolo11'),
    'yolo12n': ('yolo12n.pt', 'yolo12'),
    'yolo26n': ('yolo26n.pt', 'yolo26'),
    'yolo26s': ('yolo26s.pt', 'yolo26'),
}
SPECS = {spec.name: spec for spec in FAMILIES}
NMS_ROWS = [r for r, (_, fam) in ROWS.items() if SPECS[fam].head_kind == 'nms']
ONE2ONE_ROWS = [r for r, (_, fam) in ROWS.items() if SPECS[fam].head_kind == 'one2one']


@pytest.fixture(scope='module')
def ultralytics_export() -> Iterator[ModuleType]:
    """``export/ultralytics_export.py`` (a script that imports its siblings by bare name)."""
    mp = pytest.MonkeyPatch()
    mp.syspath_prepend(str(REPO_ROOT / 'export'))
    try:
        yield importlib.import_module('ultralytics_export')
    finally:
        mp.undo()


@pytest.fixture(scope='module')
def load(
    op_weights: Callable[[str], Path], tmp_path_factory: pytest.TempPathFactory
) -> Callable[[str], Any]:
    """A fresh model per call, loaded from a private copy: Ultralytics writes the export next to
    the weights, and two exports of one file would overwrite each other in the shared cache."""
    import shutil

    from ultralytics import YOLO

    def get(row: str) -> Any:
        work = tmp_path_factory.mktemp(f'{row}_pt')
        return YOLO(str(Path(shutil.copy(op_weights(ROWS[row][0]), work))))

    return get


def _graph_ops(path: Path) -> list[str]:
    return [node.op_type for node in onnx.load(str(path)).graph.node]


def _output_dims(path: Path) -> list[list[int | str]]:
    graph = onnx.load(str(path)).graph
    return [
        [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        for out in graph.output
    ]


@pytest.fixture(scope='module')
def exports(
    load: Callable[[str], Any],
    ultralytics_export: ModuleType,
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[str], Path]:
    """The served ONNX of a row, exported once per module."""
    cache: dict[str, Path] = {}

    def get(row: str) -> Path:
        if row in cache:
            return cache[row]
        ultralytics_export.MODELS_DIR = tmp_path_factory.mktemp(row)
        path, _, _ = ultralytics_export.export_fused_onnx(load(row), row, MAX_BATCH)
        cache[row] = path
        return path

    return get


def _images(count: int) -> np.ndarray:
    """``count`` letterboxed 640 RGB images from the two sample images Ultralytics ships."""
    import cv2
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ASSETS

    letterbox = LetterBox((IMGSZ, IMGSZ), auto=False)
    frames = [cv2.imread(str(ASSETS / name)) for name in ('bus.jpg', 'zidane.jpg')]
    batch = [letterbox(image=frames[i % 2])[..., ::-1] for i in range(count)]
    return np.ascontiguousarray(np.stack(batch).transpose(0, 3, 1, 2), dtype=np.float32) / 255.0


def _run(path: Path, batch: np.ndarray) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    return session.run(None, {session.get_inputs()[0].name: batch})


@pytest.mark.parametrize('row', ROWS)
def test_the_pt_loads_and_the_registry_names_its_head(
    row: str, load: Callable[[str], Any], op_weights: Callable[[str], Path]
) -> None:
    from src.services.model_export import validate_pytorch_model

    ok, msg, info = validate_pytorch_model(op_weights(ROWS[row][0]))

    assert ok, msg
    spec = SPECS[ROWS[row][1]]
    assert spec.pt_supported
    assert info['head_kind'] == spec.head_kind
    assert head_kind_of(load(row).model.model[-1]) == spec.head_kind


@pytest.mark.parametrize('row', ROWS)
def test_the_served_export_is_one_output_300_by_6(row: str, exports: Callable[[str], Path]) -> None:
    path = exports(row)
    ((batch_dim, rows, cols),) = _output_dims(path)

    assert (rows, cols) == (300, 6)
    assert batch_dim in (MAX_BATCH, 'batch')
    ops = _graph_ops(path)
    if SPECS[ROWS[row][1]].head_kind == 'nms':
        assert ops.count('NonMaxSuppression') == MAX_BATCH
    else:
        assert 'TopK' in ops


@pytest.mark.parametrize('row', ROWS)
def test_the_export_records_the_batch_it_was_exported_for(
    row: str, exports: Callable[[str], Path], ultralytics_export: ModuleType
) -> None:
    """The config writer's J7 guard reads this metadata key (``export/yolo_triton_config.py``)."""
    from yolo_triton_config import onnx_batch

    assert onnx_batch(exports(row)) == MAX_BATCH


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


@pytest.mark.parametrize('row', NMS_ROWS)
def test_the_embedded_nms_equals_a_torchvision_oracle_over_the_raw_head(
    row: str, exports: Callable[[str], Path], load: Callable[[str], Any], tmp_path: Path
) -> None:
    import torch
    import torchvision

    raw_path = Path(
        load(row).export(
            format='onnx',
            imgsz=IMGSZ,
            dynamic=True,
            simplify=True,
            device='cpu',
        )
    )
    images = _images(2)
    (served,) = _run(exports(row), images)
    (raw,) = _run(raw_path, images)

    iou, conf, max_det = 0.7, 0.25, 300
    for index in range(2):
        pred = torch.from_numpy(raw[index].T)  # [anchors, 4 + nc], xywh in input pixels
        scores, classes = pred[:, 4:].max(1)
        keep = scores > conf
        xywh, scores, classes = pred[keep, :4], scores[keep], classes[keep]
        xyxy = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
        kept = torchvision.ops.nms(xyxy, scores, iou)[:max_det]
        oracle = [(xyxy[i].numpy(), float(scores[i]), int(classes[i])) for i in kept]

        got = served[index][served[index][:, 4] > 0]
        assert oracle, 'the sample images must produce detections'
        matched = sum(
            any(
                _iou(box, row_[:4]) >= 0.95
                and int(row_[5]) == cls
                and abs(float(row_[4]) - score) < 1e-4
                for row_ in got
            )
            for box, score, cls in oracle
        )
        assert matched / len(oracle) == 1.0, (matched, len(oracle), len(got))


@pytest.mark.parametrize('row', NMS_ROWS)
def test_images_past_the_export_batch_come_back_empty(
    row: str, exports: Callable[[str], Path]
) -> None:
    """Documents that the ONNX ``batch`` is a hard cap: rows past it carry no detections."""
    (out,) = _run(exports(row), _images(MAX_BATCH + 2))

    assert out.shape[0] == MAX_BATCH + 2
    assert (out[:MAX_BATCH, :, 4] > 0).any(axis=1).all()
    assert not (out[MAX_BATCH:, :, 4] > 0).any()


@pytest.mark.parametrize('row', ['yolo26n', 'yolo26s'])
@pytest.mark.parametrize('simplify', [True, False])
def test_yolo26_gives_the_nms_free_tensor_whether_or_not_it_is_simplified(
    row: str,
    simplify: bool,
    load: Callable[[str], Any],
    ultralytics_export: ModuleType,
    tmp_path: Path,
) -> None:
    """Discrepancy rows: ``nms=False`` with ``simplify`` on and off; a raw head is refused."""
    path = Path(
        load(row).export(
            format='onnx',
            imgsz=IMGSZ,
            dynamic=True,
            simplify=simplify,
            device='cpu',
            **export_kwargs('one2one', MAX_BATCH),
        )
    )

    (out,) = _run(path, _images(2))
    assert out.shape == (2, 300, 6)
    assert 'TopK' in _graph_ops(path)
    if simplify:
        assert ultralytics_export.inspect_fused_output(path)[1] == [300, 6]
    else:
        # Without onnxslim the graph's output dims stay symbolic, so the declared-shape check
        # refuses it (fail closed) although the runtime tensor is the NMS-free one. The served
        # export always simplifies.
        with pytest.raises(RuntimeError, match='raw one-to-many head'):
            ultralytics_export.inspect_fused_output(path)


def test_a_raw_head_is_refused_never_passed(
    load: Callable[[str], Any], ultralytics_export: ModuleType, tmp_path: Path
) -> None:
    raw = Path(
        load('yolo26n').export(format='onnx', imgsz=IMGSZ, dynamic=True, device='cpu', nms=None)
    )

    with pytest.raises(RuntimeError, match='raw one-to-many head'):
        ultralytics_export.inspect_fused_output(raw)


def test_ultralytics_is_the_pinned_version() -> None:
    pin = re.search(
        r'^ultralytics==(\S+)$',
        (REPO_ROOT / 'requirements-test.txt').read_text(encoding='utf-8'),
        re.MULTILINE,
    )
    assert pin, 'requirements-test.txt pins ultralytics exactly'
    assert importlib.metadata.version('ultralytics') == pin.group(1)


def test_no_production_module_passes_end2end_to_an_export() -> None:
    """``end2end=`` is deprecated and mapped to ``nms`` (the export docs); only ``nms`` is used."""
    import ast

    offenders: list[str] = []
    for top in ('src', 'export', 'docker', 'scripts'):
        for path in sorted((REPO_ROOT / top).rglob('*.py')):
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            offenders.extend(
                f'{path.relative_to(REPO_ROOT)}:{node.value.lineno}'
                for node in ast.walk(tree)
                if isinstance(node, ast.keyword) and node.arg == 'end2end'
            )
    assert not offenders, offenders
