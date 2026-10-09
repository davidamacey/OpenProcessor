"""An exporter's generated ``config.pbtxt`` never silently replaces a tracked,
differing one (``make export-models`` used to dirty the git tree)."""

from __future__ import annotations

import sys
from pathlib import Path


EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

from config_write import write_generated_config  # noqa: E402


def test_a_new_config_is_created(tmp_path: Path) -> None:
    target = tmp_path / 'config.pbtxt'
    assert write_generated_config(target, 'generated') == target
    assert target.read_text() == 'generated'


def test_an_identical_config_is_not_rewritten(tmp_path: Path) -> None:
    target = tmp_path / 'config.pbtxt'
    target.write_text('same')
    before = target.stat().st_mtime_ns
    assert write_generated_config(target, 'same') == target
    assert target.stat().st_mtime_ns == before


def test_a_differing_config_is_kept_and_the_generated_one_goes_beside_it(tmp_path: Path) -> None:
    target = tmp_path / 'config.pbtxt'
    target.write_text('hand tuned')
    written = write_generated_config(target, 'generated')
    assert written == tmp_path / 'config.pbtxt.generated'
    assert written.read_text() == 'generated'
    assert target.read_text() == 'hand tuned'


def test_overwrite_replaces_a_differing_config(tmp_path: Path) -> None:
    target = tmp_path / 'config.pbtxt'
    target.write_text('hand tuned')
    assert write_generated_config(target, 'generated', overwrite=True) == target
    assert target.read_text() == 'generated'
    assert not (tmp_path / 'config.pbtxt.generated').exists()


TRACKED_END2END = """output [
  {
    name: "num_dets"
    data_type: TYPE_INT32
    dims: [ 1 ]
  },
  {
    name: "det_boxes"
    data_type: TYPE_FP32
    dims: [ 300, 4 ]
  },
  {
    name: "det_scores"
    data_type: TYPE_FP32
    dims: [ 300 ]
  }
]
max_batch_size: 16
"""


def test_a_kept_config_still_takes_the_built_engines_output_dtypes(tmp_path: Path) -> None:
    # TRT 11.0 builds FP16 boxes/scores; a kept FP32 config makes Triton refuse the model.
    from config_write import sync_output_dtypes

    target = tmp_path / 'config.pbtxt'
    target.write_text(TRACKED_END2END)
    assert sync_output_dtypes(target, {'det_boxes': 'TYPE_FP16', 'det_scores': 'TYPE_FP16'})
    text = target.read_text()
    assert 'name: "det_boxes"\n    data_type: TYPE_FP16' in text
    assert 'name: "det_scores"\n    data_type: TYPE_FP16' in text
    assert 'name: "num_dets"\n    data_type: TYPE_INT32' in text
    assert 'max_batch_size: 16' in text
    assert not sync_output_dtypes(target, {'det_boxes': 'TYPE_FP16', 'det_scores': 'TYPE_FP16'})


def test_write_generated_config_syncs_dtypes_into_the_kept_config(tmp_path: Path) -> None:
    target = tmp_path / 'config.pbtxt'
    target.write_text(TRACKED_END2END)
    written = write_generated_config(
        target, 'generated', output_dtypes={'det_boxes': 'TYPE_FP16', 'det_scores': 'TYPE_FP16'}
    )
    assert written == tmp_path / 'config.pbtxt.generated'
    assert 'name: "det_scores"\n    data_type: TYPE_FP16' in target.read_text()
    assert 'max_batch_size: 16' in target.read_text()
