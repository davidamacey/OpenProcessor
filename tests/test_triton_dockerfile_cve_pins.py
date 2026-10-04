"""The triton image upgrades its python CVE-lagging packages explicitly."""

import re
from pathlib import Path


DOCKERFILE = Path(__file__).resolve().parents[1] / 'Dockerfile.triton'


def _min_version(package: str) -> tuple[int, ...]:
    match = re.search(rf"'{package}>=([0-9.]+)'", DOCKERFILE.read_text())
    assert match, f'Dockerfile.triton has no {package}>= upgrade pin'
    return tuple(int(p) for p in match.group(1).split('.'))


def test_anyio_pinned_past_cve_2026_63374():
    assert _min_version('anyio') >= (4, 14, 2)


def test_starlette_pin_kept():
    assert _min_version('starlette') >= (1, 3, 1)
