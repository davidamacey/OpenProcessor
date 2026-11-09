"""Pin a build input and review per-service container hardening: build inputs are immutable."""

from __future__ import annotations

import re

from compose._helpers import REPO_ROOT


SEGMENTER_DOCKERFILE = REPO_ROOT / 'docker' / 'segmenter' / 'Dockerfile'


def test_source_checkout_build_arg_defaults_to_a_full_commit() -> None:
    text = SEGMENTER_DOCKERFILE.read_text()
    match = re.search(r'^ARG SAM3_SHA=(\S+)$', text, re.MULTILINE)
    assert match, 'the source checkout stage must declare its revision build arg'
    assert re.fullmatch(r'[0-9a-f]{40}', match.group(1)), match.group(1)
