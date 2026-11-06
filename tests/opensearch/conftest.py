"""Fixtures of the OpenSearch lifecycle tests."""

from __future__ import annotations

import pytest

from src.config.curation_accessors import base_curation_config
from src.config.projects import ProjectRecord, new_project_record
from src.services.curation import index_bootstrap
from tests.opensearch.fake_cluster import FakeCluster, FakeGlobalClient


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts as a process that has bootstrapped nothing."""
    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', set())


@pytest.fixture
def cluster() -> FakeCluster:
    return FakeCluster()


@pytest.fixture
def global_client(cluster: FakeCluster) -> FakeGlobalClient:
    return FakeGlobalClient(cluster)


def make_record(slug: str, status: str = 'active') -> ProjectRecord:
    return new_project_record(slug, base_curation_config(), status=status)  # type: ignore[arg-type]


@pytest.fixture
def records() -> list[ProjectRecord]:
    return [make_record('alpha'), make_record('beta')]
