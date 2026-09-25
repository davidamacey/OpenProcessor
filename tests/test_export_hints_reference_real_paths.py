"""F-13a residual (fresh-start E2E findings 2026-09-25): export scripts'
missing-checkpoint hints must point at a script/target that actually
exists.

``export_mobileclip_image_encoder.py`` / ``export_mobileclip_text_encoder.py``
told an operator with a missing checkpoint to run
``bash scripts/track_e/setup_mobileclip_env.sh`` -- a directory that has
never existed in this repo (``scripts/track_e/`` is not a real path).
The actual downloader is ``make download-models``
(``scripts/lib/download.sh``). Both files' module docstrings also told
an operator to run the export scripts themselves from a stale
``/app/scripts/track_e/...`` path instead of their real location,
``/app/export/...``.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ENCODER_SRC = (REPO_ROOT / 'export' / 'export_mobileclip_image_encoder.py').read_text()
TEXT_ENCODER_SRC = (REPO_ROOT / 'export' / 'export_mobileclip_text_encoder.py').read_text()


def test_scripts_track_e_is_not_a_real_path() -> None:
    """Sanity check on the assumption the tests below rely on."""
    assert not (REPO_ROOT / 'scripts' / 'track_e').exists()


def test_download_models_make_target_exists() -> None:
    makefile = (REPO_ROOT / 'Makefile').read_text()
    assert '.PHONY: download-models' in makefile
    assert 'download-models:' in makefile


def test_image_encoder_missing_checkpoint_hint_points_at_a_real_target() -> None:
    assert 'scripts/track_e' not in IMAGE_ENCODER_SRC
    assert 'make download-models' in IMAGE_ENCODER_SRC


def test_text_encoder_missing_checkpoint_hint_points_at_a_real_target() -> None:
    assert 'scripts/track_e' not in TEXT_ENCODER_SRC
    assert 'make download-models' in TEXT_ENCODER_SRC


def test_image_encoder_docstring_references_its_own_real_path() -> None:
    assert '/app/export/export_mobileclip_image_encoder.py' in IMAGE_ENCODER_SRC


def test_text_encoder_docstring_references_its_own_real_path() -> None:
    assert '/app/export/export_mobileclip_text_encoder.py' in TEXT_ENCODER_SRC
