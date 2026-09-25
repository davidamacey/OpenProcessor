"""The API image's dependency set must stay resolvable.

perception_models' own requirements exact-pin timm==1.0.15 (and ~30 research
packages), which conflicts with open-clip-torch>=3.2 (timm>=1.0.17, the first
release with MobileCLIP2-S2). It is installed with --no-deps in the Dockerfile
instead; listing it in requirements.txt makes `docker compose build` fail with
ResolutionImpossible.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _requirement_lines() -> list[str]:
    lines = (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines()
    return [ln.split('#', 1)[0].strip() for ln in lines if ln.split('#', 1)[0].strip()]


def test_perception_models_not_resolved_with_its_own_pins() -> None:
    assert not [ln for ln in _requirement_lines() if ln.startswith('perception_models')]


def test_dockerfile_installs_pinned_perception_models_without_deps() -> None:
    dockerfile = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
    match = re.search(
        r'--no-deps\s*\\?\s*"perception_models @ git\+\S+@([0-9a-f]{40})"', dockerfile
    )
    assert match, 'perception_models must be installed --no-deps at a pinned commit'


def test_pe_encoder_runtime_deps_declared() -> None:
    names = {re.split(r'[<>=\[ ;@]', ln, maxsplit=1)[0].lower() for ln in _requirement_lines()}
    assert {'einops', 'regex', 'timm', 'ftfy', 'huggingface_hub'} <= names
