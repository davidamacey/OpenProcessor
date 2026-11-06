"""The control-plane packages (PostgreSQL, Valkey and Celery) are exact pins, equal everywhere.

Versions: docs/design/v070_rebuild/01_package_pins.md section 4 (the latest releases on the bump
day, D-70). The extras differ by environment on purpose: the images build `psycopg[c]`, the test
venv installs `psycopg[binary]`; the version is the same.
"""

from __future__ import annotations

import re
import tomllib
from importlib.metadata import requires, version
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]

RUNTIME = ('sqlalchemy', 'alembic', 'psycopg', 'celery', 'kombu', 'billiard', 'redis')
RUNTIME_ONLY = ('flower',)
LOGGING = ('asgi-correlation-id',)
TEST_ONLY = ('testcontainers',)


def _norm(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def _manifest(path: str) -> dict[str, Requirement]:
    out: dict[str, Requirement] = {}
    for line in (ROOT / path).read_text(encoding='utf-8').splitlines():
        spec = line.split('#', 1)[0].strip()
        if spec and not spec.startswith('-'):
            req = Requirement(spec)
            out[_norm(req.name)] = req
    return out


def _pyproject() -> dict[str, Requirement]:
    data = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    reqs = [Requirement(s) for s in data['project']['dependencies']]
    return {_norm(r.name): r for r in reqs}


def _exact(req: Requirement) -> str:
    (spec,) = req.specifier
    assert spec.operator == '==', f'{req} is not an exact pin'
    return spec.version


MANIFESTS = {
    'requirements.txt': _manifest('requirements.txt'),
    'requirements-test.txt': _manifest('requirements-test.txt'),
    'pyproject.toml': _pyproject(),
}


@pytest.mark.parametrize('name', RUNTIME + LOGGING)
def test_control_plane_package_is_one_exact_pin_in_every_manifest(name: str) -> None:
    versions = {path: _exact(reqs[name]) for path, reqs in MANIFESTS.items() if name in reqs}
    assert set(versions) == set(MANIFESTS), f'{name} missing from some manifest: {versions}'
    assert len(set(versions.values())) == 1, versions


@pytest.mark.parametrize('name', RUNTIME_ONLY)
def test_flower_is_pinned_where_its_service_runs(name: str) -> None:
    """Flower runs from the api image (02c section 4), so it is a runtime pin only."""
    assert _exact(MANIFESTS['requirements.txt'][name]) == _exact(MANIFESTS['pyproject.toml'][name])


@pytest.mark.parametrize('name', TEST_ONLY)
def test_testcontainers_is_a_test_pin_only(name: str) -> None:
    assert _exact(MANIFESTS['requirements-test.txt'][name])
    assert name not in MANIFESTS['requirements.txt']
    assert name not in MANIFESTS['pyproject.toml']


def test_psycopg_builds_from_source_in_images_and_binary_in_tests() -> None:
    assert 'c' in MANIFESTS['requirements.txt']['psycopg'].extras
    assert 'c' in MANIFESTS['pyproject.toml']['psycopg'].extras
    assert 'binary' in MANIFESTS['requirements-test.txt']['psycopg'].extras


def test_billiard_requeues_on_cold_shutdown() -> None:
    """billiard 4.3.0 acks a task interrupted by a cold shutdown instead of requeuing it."""
    assert Version(_exact(MANIFESTS['requirements.txt']['billiard'])) >= Version('4.3.1')


def test_redis_pin_satisfies_the_kombu_pin() -> None:
    kombu_reqs = [Requirement(r) for r in requires('kombu') or []]
    redis_reqs = [r for r in kombu_reqs if _norm(r.name) == 'redis']
    assert redis_reqs, 'installed kombu declares no redis requirement'
    redis_pin = _exact(MANIFESTS['requirements.txt']['redis'])
    assert all(redis_pin in r.specifier for r in redis_reqs), (redis_pin, redis_reqs)
    assert _exact(MANIFESTS['requirements.txt']['kombu']) == version('kombu')
