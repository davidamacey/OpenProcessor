"""Address spellings that reach a denied address by another route (W9 review
m9): IPv6 forms that embed 169.254.169.254, and full-width service names."""

from __future__ import annotations

import pytest

import src.services.labeling.vlm_url_policy as policy
from src.services.labeling.vlm_url_policy import parse_endpoint_url, url_denial


@pytest.fixture(autouse=True)
def _dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy, '_resolve', lambda _host: [])
    monkeypatch.setattr(policy, '_docker_gateway_addresses', lambda: frozenset())
    policy.reset_policy_caches()


@pytest.mark.parametrize(
    'host',
    [
        '[64:ff9b::a9fe:a9fe]',  # NAT64
        '[2002:a9fe:a9fe::]',  # 6to4
        '[::ffff:0:a9fe:a9fe]',  # SIIT
        '[::ffff:169.254.169.254]',  # IPv4-mapped
    ],
)
def test_every_ipv6_spelling_of_the_metadata_address_is_denied(host: str) -> None:
    assert url_denial(f'http://{host}/v1') is not None


def test_a_fullwidth_service_name_is_the_service_name() -> None:
    assert parse_endpoint_url('http://ｏｐｅｎｓｅａｒｃｈ:9200/v1').host == 'opensearch'
    assert url_denial('http://ｏｐｅｎｓｅａｒｃｈ:9200/v1') is not None
