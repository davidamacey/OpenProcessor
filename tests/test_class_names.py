"""Tests for src/utils/class_names.py.

F-42 (fresh-start E2E findings 2026-09-25, round 2): a promoted model's
detections used to always come back labeled from the stock 80-class COCO
vocabulary regardless of which model actually produced the class ids
(src/utils/affine.py:344's bare ``COCO_CLASSES.get(...)``). These tests
cover the resolver in isolation, against a fake model repo -- no Triton,
no GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.utils import class_names as cn


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_cache_and_env(monkeypatch: pytest.MonkeyPatch):
    """Every test gets a clean class-name cache and no OP_TRITON_MODEL_REPO
    leaking in from the real environment."""
    cn.clear_class_name_cache()
    monkeypatch.delenv('OP_TRITON_MODEL_REPO', raising=False)
    yield
    cn.clear_class_name_cache()


def _point_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    return tmp_path


class TestLabelsFileResolution:
    def test_reads_the_models_own_labels_txt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models_dir = _point_at(tmp_path, monkeypatch)
        model_dir = models_dir / 'vehicle_detector_v1'
        model_dir.mkdir()
        (model_dir / 'labels.txt').write_text('car\ntruck\nbus\n')

        names = cn.get_class_names('vehicle_detector_v1')

        assert names == {0: 'car', 1: 'truck', 2: 'bus'}
        assert cn.resolve_class_name('vehicle_detector_v1', 0) == 'car'
        assert cn.resolve_class_name('vehicle_detector_v1', 2) == 'bus'

    def test_a_promoted_models_class_0_is_never_read_as_coco_person(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact F-42 symptom: a vehicle detector's class 0 ('car')
        must not come back as COCO's class 0 ('person'), or any other
        COCO word that happens to share the id."""
        models_dir = _point_at(tmp_path, monkeypatch)
        model_dir = models_dir / 'vehicle_detector_v1'
        model_dir.mkdir()
        (model_dir / 'labels.txt').write_text('car\ntruck\nbus\n')

        assert cn.resolve_class_name('vehicle_detector_v1', 0) == 'car'
        assert cn.resolve_class_name('vehicle_detector_v1', 0) != 'person'

    def test_unresolvable_id_renders_as_class_n_not_a_coco_guess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models_dir = _point_at(tmp_path, monkeypatch)
        model_dir = models_dir / 'vehicle_detector_v1'
        model_dir.mkdir()
        (model_dir / 'labels.txt').write_text('car\n')

        assert cn.resolve_class_name('vehicle_detector_v1', 7) == 'class_7'

    def test_blank_lines_in_labels_txt_are_gaps_not_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models_dir = _point_at(tmp_path, monkeypatch)
        model_dir = models_dir / 'gappy_v1'
        model_dir.mkdir()
        (model_dir / 'labels.txt').write_text('car\n\nbus\n')

        names = cn.get_class_names('gappy_v1')

        assert names == {0: 'car', 2: 'bus'}
        assert cn.resolve_class_name('gappy_v1', 1) == 'class_1'


class TestConfigLabelFilenameFallback:
    def test_falls_back_to_a_label_filename_referenced_in_config_pbtxt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models_dir = _point_at(tmp_path, monkeypatch)
        model_dir = models_dir / 'third_party_model'
        model_dir.mkdir()
        (model_dir / 'config.pbtxt').write_text(
            'name: "third_party_model"\n'
            'output [\n'
            '  { name: "output0" label_filename: "my_labels.txt" }\n'
            ']\n'
        )
        (model_dir / 'my_labels.txt').write_text('cat\ndog\n')

        names = cn.get_class_names('third_party_model')

        assert names == {0: 'cat', 1: 'dog'}

    def test_missing_referenced_label_filename_does_not_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models_dir = _point_at(tmp_path, monkeypatch)
        model_dir = models_dir / 'third_party_model'
        model_dir.mkdir()
        (model_dir / 'config.pbtxt').write_text(
            'output [ { label_filename: "does_not_exist.txt" } ]\n'
        )

        assert cn.get_class_names('third_party_model') == {}
        assert cn.resolve_class_name('third_party_model', 0) == 'class_0'


class TestStockCocoFallback:
    def test_stock_detector_falls_back_to_coco_when_labels_txt_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Safety net only -- the real stock model ships its own
        labels.txt (resolution step 1 handles it); this proves the
        fallback still works if that file is ever absent."""
        _point_at(tmp_path, monkeypatch)

        names = cn.get_class_names('yolov11_small_trt_end2end')

        assert names[0] == 'person'
        assert names[2] == 'car'
        assert len(names) == 80

    def test_an_unrecognized_model_with_no_labels_never_gets_coco_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The actual F-42 bug: a promoted model with no labels.txt (not
        yet written, or removed) must render class_{id} placeholders,
        never silently borrow COCO's vocabulary."""
        _point_at(tmp_path, monkeypatch)

        names = cn.get_class_names('some_promoted_model_v3')

        assert names == {}
        assert cn.resolve_class_name('some_promoted_model_v3', 0) == 'class_0'
        assert cn.resolve_class_name('some_promoted_model_v3', 0) != 'person'


class TestCaching:
    def test_result_is_cached_across_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models_dir = _point_at(tmp_path, monkeypatch)
        model_dir = models_dir / 'cached_v1'
        model_dir.mkdir()
        (model_dir / 'labels.txt').write_text('car\n')

        first = cn.get_class_names('cached_v1')
        (model_dir / 'labels.txt').write_text('truck\n')  # changed on disk
        second = cn.get_class_names('cached_v1')

        assert first == second == {0: 'car'}  # still the cached value

    def test_invalidate_forces_a_re_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The promote/unload hooks call this so a freshly promoted
        model's labels are visible on its very first request."""
        models_dir = _point_at(tmp_path, monkeypatch)
        model_dir = models_dir / 'cached_v1'
        model_dir.mkdir()
        (model_dir / 'labels.txt').write_text('car\n')
        cn.get_class_names('cached_v1')

        (model_dir / 'labels.txt').write_text('truck\n')
        cn.invalidate_class_names('cached_v1')

        assert cn.get_class_names('cached_v1') == {0: 'truck'}

    def test_invalidate_is_a_no_op_for_an_uncached_model(self) -> None:
        cn.invalidate_class_names('never_requested_model')  # must not raise
