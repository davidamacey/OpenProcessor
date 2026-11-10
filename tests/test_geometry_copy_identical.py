"""The GPU image-prep model cannot import ``src``, so it carries a copy of the geometry module.

The copy is the same bytes as ``src/domain/image_geometry.py``; one arithmetic serves the CPU
paths, the GPU model and the client un-map (``export/sync_model_seed.py`` ships the model dir).
"""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def test_the_model_directory_copy_is_byte_identical() -> None:
    domain = _ROOT / 'src' / 'domain' / 'image_geometry.py'
    copy = _ROOT / 'models' / 'gpu_image_prep' / '1' / 'image_geometry.py'
    assert copy.read_bytes() == domain.read_bytes()


def test_the_domain_module_imports_nothing_from_the_project() -> None:
    text = (_ROOT / 'src' / 'domain' / 'image_geometry.py').read_text()
    assert 'from src' not in text
    assert 'import src' not in text
