"""A default build must leave the tree clean: every committed Triton config the exporters write or
rewrite carries the same tensors (name, datatype, dims), platform and batch size that the
exporter's renderer produces for a default build, so an install or upgrade dirties nothing."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

# What the default FP16 detector build emits (measured on the TensorRT 11.0 Triton image):
# export_models.py's sync_output_dtypes rewrites the config to these after every build.
DEFAULT_E2E_DTYPES = {'det_boxes': 'TYPE_FP32', 'det_scores': 'TYPE_FP16'}


def signature(text: str) -> dict[str, Any]:
    """Platform, batch size and the (block, name, datatype, dims) of every tensor."""
    text = re.sub(r'#.*', '', text)
    tensors = []
    for block in ('input', 'output'):
        match = re.search(rf'^{block}\s*\[(.*?)^\]', text, re.DOTALL | re.MULTILINE)
        assert match, block
        for entry in re.finditer(
            r'name:\s*"(\w+)"\s*data_type:\s*(TYPE_\w+)\s*dims:\s*\[([^\]]*)\]', match.group(1)
        ):
            dims = tuple(int(d) for d in entry.group(3).replace(' ', '').split(','))
            tensors.append((block, entry.group(1), entry.group(2), dims))
    return {
        'platform': re.search(r'^platform:\s*"(\w+)"', text, re.MULTILINE).group(1),  # type: ignore[union-attr]
        'max_batch': int(re.search(r'^max_batch_size:\s*(\d+)', text, re.MULTILINE).group(1)),  # type: ignore[union-attr]
        'tensors': tensors,
    }


def committed(model: str) -> dict[str, Any]:
    return signature((ROOT / 'models' / model / 'config.pbtxt').read_text())


def function_from(path: Path, name: str, **namespace: Any):
    """One top-level function of an export script, without importing its torch/tensorrt imports."""
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns: dict[str, Any] = dict(namespace)
    exec(compile(ast.Module([node], []), str(path), 'exec'), ns)
    return ns[name]


def test_yolo11_end2end_config_matches_the_default_build() -> None:
    render = function_from(
        EXPORT_DIR / 'yolo_triton_config.py', 'generate_triton_config', IMG_SIZE=640
    )
    rendered = signature(
        render(
            'yolov11_small_trt_end2end',
            'trt_end2end',
            64,
            has_nms=True,
            output_dtypes=DEFAULT_E2E_DTYPES,
        )
    )
    assert committed('yolov11_small_trt_end2end') == rendered


def test_yolo26_config_matches_the_exporter(tmp_path: Path) -> None:
    write = function_from(
        EXPORT_DIR / 'export_yolo26.py',
        'write_triton_config',
        IMG_SIZE=640,
        logger=__import__('logging').getLogger('t'),
        Path=Path,
    )
    write(tmp_path, 'yolo26_small_trt', 32, 'output0', [300, 6])
    assert committed('yolo26_small_trt') == signature((tmp_path / 'config.pbtxt').read_text())


def test_pe_image_config_matches_the_renderer() -> None:
    from export_pe_image_encoder import PETritonConfig, render_config

    assert committed('pe_image_encoder') == signature(render_config(PETritonConfig()))


def test_pe_text_config_matches_the_renderer() -> None:
    from export_pe_text_encoder import PETextTritonConfig, render_config

    assert committed('pe_text_encoder') == signature(render_config(PETextTritonConfig()))


@pytest.mark.parametrize('model', ['mobileclip2_s2_image_encoder', 'mobileclip2_s2_text_encoder'])
def test_mobileclip_exporters_do_not_write_configs(model: str) -> None:
    # Their configs are hand-maintained; an exporter that starts writing one needs a renderer test.
    for script in ('export_mobileclip_image_encoder.py', 'export_mobileclip_text_encoder.py'):
        assert 'config.pbtxt' not in (EXPORT_DIR / script).read_text().replace(
            'Create config.pbtxt and restart Triton', ''
        )
    assert (ROOT / 'models' / model / 'config.pbtxt').exists()
