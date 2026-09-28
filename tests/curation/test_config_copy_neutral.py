"""No private-domain leakage in the generic config-editor surfaces
(any_domain_plan.md §3.4/§7.2, W3): ``/prompt_packs/schema`` must read as
neutral copy, excluding template bodies (templates intentionally carry
domain examples like "wheel"/"plate", not leaked *default* vocabulary).

W4 extends this file to cover ``/region_profiles/schema`` and
``/config/vocabulary`` once those routes exist.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch


# \bplate\b (not bare 'plate'): 'template'/'templates' are core, generic
# prompt-pack vocabulary here (str.format templates) and would otherwise
# false-positive on the substring 'plate' inside 'tem-PLATE'.
_LEAK_RE = re.compile(r'(?i)\bplate\b|licen[cs]e|vehicle|\blpr\b|bumper')


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = FakeConfigOpenSearch()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


# Structural identifiers, not human-facing copy: a code-shaped key like
# 'class_user_template' or 'combined_batch_rules' legitimately contains
# the substring 'plate' (tem-PLATE) with zero connection to the leak
# regex's real target (neutral help/label copy naming a private domain).
_STRUCTURAL_KEYS = frozenset({'field', 'fields', 'id', 'group', 'kind', 'name'})


def _assert_no_leak(value: object, *, key: str | None = None) -> None:
    if isinstance(value, str):
        if key in _STRUCTURAL_KEYS:
            return
        assert not _LEAK_RE.search(value), f'private-domain leak in {key!r}: {value!r}'
    elif isinstance(value, dict):
        for k, v in value.items():
            _assert_no_leak(v, key=k)
    elif isinstance(value, list):
        for v in value:
            _assert_no_leak(v, key=key)


def test_prompt_pack_schema_is_domain_neutral(app_client: TestClient) -> None:
    r = app_client.get('/curation/projects/default/prompt_packs/schema')
    assert r.status_code == 200, r.text
    _assert_no_leak(r.json())
