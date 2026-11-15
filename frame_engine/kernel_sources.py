"""The CUDA C sources of the engine's kernels.

The files live in ``native/frame_engine/kernels/``, the one copy both this engine and the native
engine compile (docs/design/v070_rebuild/17_native_engine_plan.md, section 3). Each module hands the
text of its file to CuPy unchanged; ``tests/frame_engine_tests/test_kernel_sources.py`` pins the
digests.
"""

from __future__ import annotations

from pathlib import Path


#: ``/app/frame_engine`` beside ``/app/native`` in the image, the checkout's two directories in a repo.
KERNELS_DIR = Path(__file__).resolve().parents[1] / 'native' / 'frame_engine' / 'kernels'


def kernel_source(name: str) -> str:
    """The exact text of ``kernels/<name>``: bytes decoded as UTF-8, no newline translation."""
    return (KERNELS_DIR / name).read_bytes().decode('utf-8')
