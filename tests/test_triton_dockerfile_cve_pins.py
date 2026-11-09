"""The triton image pins its python CVE-lagging packages exactly, at or past the fixed release."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECURITY = ROOT / 'docker' / 'triton' / 'requirements-security.txt'


def _pinned_version(package: str) -> tuple[int, ...]:
    match = re.search(rf'^{package}==([0-9.]+)\s*$', SECURITY.read_text(), re.M)
    assert match, f'{SECURITY.name} has no exact {package}== pin'
    return tuple(int(p) for p in match.group(1).split('.'))


def test_the_dockerfile_installs_the_security_pins() -> None:
    text = (ROOT / 'Dockerfile.triton').read_text().replace('\\\n', ' ')
    assert re.search(r'pip install[^\n]*--upgrade -r \S*requirements-security\.txt', text)


def test_anyio_pinned_past_cve_2026_63374() -> None:
    assert _pinned_version('anyio') >= (4, 14, 2)


def test_starlette_pin_kept() -> None:
    assert _pinned_version('starlette') >= (1, 3, 1)


def test_certifi_pinned_past_ghsa_6q48() -> None:
    assert _pinned_version('certifi') >= (2026, 10, 18)
