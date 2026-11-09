"""What the export worker runs: one command for every head kind, and the engines it produces.

A job without a head kind (queued by an older version) fails closed. The export script itself is
``export/ultralytics_export.py`` (``tests/export/test_ultralytics_export.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from src.domain.model_families import NMS, ONE2ONE


EXPORT_DIR = Path(__file__).resolve().parents[2] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import export_worker  # noqa: E402  # follows the sys.path setup


def _job(head_kind: str | None, formats: list[str] | None = None) -> dict[str, Any]:
    job: dict[str, Any] = {
        'pt_file': '/app/pytorch_models/car.pt',
        'triton_name': 'car',
        'max_batch': 8,
        'formats': formats or ['trt'],
    }
    if head_kind is not None:
        job['head_kind'] = head_kind
    return job


def test_one_command_serves_both_head_kinds() -> None:
    nms = export_worker.build_command(_job(NMS))
    one2one = export_worker.build_command(_job(ONE2ONE))

    assert nms == one2one
    assert nms[1].endswith('ultralytics_export.py')
    assert nms[nms.index('--custom-model') + 1] == '/app/pytorch_models/car.pt:car:8'
    assert nms[nms.index('--formats') + 1 : nms.index('--formats') + 2] == ['trt']
    assert '--overwrite-config' in nms


def test_the_command_passes_the_requested_formats_through() -> None:
    cmd = export_worker.build_command(_job(NMS, ['onnx']))

    assert cmd[cmd.index('--formats') + 1 : cmd.index('--formats') + 2] == ['onnx']


def test_there_is_one_script_not_one_per_head_kind() -> None:
    assert not hasattr(export_worker, 'EXPORT_SCRIPT_Y26')


@pytest.mark.parametrize(
    ('kind', 'engine'), [(NMS, 'car_trt_end2end'), (ONE2ONE, 'car_trt')], ids=['nms', 'one2one']
)
def test_the_engine_a_job_loads_follows_its_head_kind(kind: str, engine: str) -> None:
    assert export_worker.engines_to_load(_job(kind)) == [engine]
    assert export_worker.engines_to_load(_job(kind, ['all'])) == [engine]


def test_an_onnx_only_job_builds_no_engine() -> None:
    assert export_worker.engines_to_load(_job(NMS, ['onnx'])) == []
    assert export_worker.engines_to_load(_job(ONE2ONE, ['onnx'])) == []


def test_a_job_without_a_head_kind_fails_closed() -> None:
    job = _job(None)

    with pytest.raises(ValueError, match='re-upload the model'):
        export_worker.build_command(job)
    with pytest.raises(ValueError, match='re-upload the model'):
        export_worker.engines_to_load(job)
