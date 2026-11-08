"""The object reference stored as ``source_identifier`` and the fail-closed allow-list."""

from __future__ import annotations

import pytest

from src.services.curation.bulk_ingest.object_store import (
    ObjectRef,
    S3Config,
    S3NotAllowedError,
    S3NotConfiguredError,
)


HOSTILE_KEYS = [
    'plain/a.jpg',
    'with space/and+plus.jpg',
    'hash#and?query=1/x.jpg',
    'pct%20/already%2Fencoded.jpg',
    'ünï/códe/日本.png',
    'trailing//double//slash.jpg',
]


@pytest.mark.parametrize('key', HOSTILE_KEYS)
def test_a_reference_round_trips_whatever_the_key(key: str) -> None:
    ref = ObjectRef('bkt', key, 'abc123-4', 'v1.2/3', 10)
    parsed = ObjectRef.parse(ref.to_uri())
    assert (parsed.bucket, parsed.key, parsed.etag, parsed.version_id) == (
        'bkt',
        key,
        'abc123-4',
        'v1.2/3',
    )
    assert ref.to_uri().count('#') == 1  # a '#' in the key cannot split the reference


def test_a_different_etag_is_a_different_reference() -> None:
    assert ObjectRef('b', 'k.jpg', 'one').to_uri() != ObjectRef('b', 'k.jpg', 'two').to_uri()


def test_non_references_are_rejected() -> None:
    for bad in ('http://x/y.jpg', 's3://', 's3://bucket-only'):
        with pytest.raises(ValueError, match='s3'):
            ObjectRef.parse(bad)


def test_s3_is_off_unless_configured_and_the_allow_list_binds_bucket_and_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('OP_S3_ALLOWED', raising=False)
    with pytest.raises(S3NotConfiguredError):
        S3Config.from_env().check('any', 'k.jpg')

    monkeypatch.setenv('OP_S3_ALLOWED', 'whole, scoped/in/ ,s3://uri/form')
    config = S3Config.from_env()
    config.check('whole', 'anything/at/all.jpg')
    config.check('scoped', 'in/deep/x.jpg')
    config.check('uri', 'form/k')
    for bucket, key in (('scoped', 'out/x.jpg'), ('scoped', ''), ('unlisted', 'in/x.jpg')):
        with pytest.raises(S3NotAllowedError):
            config.check(bucket, key)
