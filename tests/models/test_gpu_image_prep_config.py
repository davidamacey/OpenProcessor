"""The ``gpu_image_prep`` inputs the API relies on are the ones the model declares (#259)."""

from __future__ import annotations

import re
from pathlib import Path

from src.services.curation.reprocess_gpu_embed import BOXES_INPUT


_MODEL = Path(__file__).resolve().parents[2] / 'models' / 'gpu_image_prep'


def _inputs() -> dict[str, str]:
    text = (_MODEL / 'config.pbtxt').read_text()
    block = re.search(r'^input \[(.*?)^\]', text, re.S | re.M)
    assert block is not None
    return {
        m.group(1): m.group(0) for m in re.finditer(r'\{\s*name:\s*"(\w+)"[^}]*\}', block.group(1))
    }


def test_the_stored_boxes_input_is_optional_fp64_rows_of_four() -> None:
    entry = _inputs()[BOXES_INPUT]
    assert 'TYPE_FP64' in entry
    assert re.search(r'dims:\s*\[\s*-1,\s*4\s*\]', entry)
    assert 'optional: true' in entry  # ingest requests do not send it


def test_the_model_reads_the_input_the_config_declares() -> None:
    source = (_MODEL / '1' / 'model.py').read_text()
    assert f"get_input_tensor_by_name(request, '{BOXES_INPUT}')" in source
