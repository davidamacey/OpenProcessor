"""The webhook body: the notification shapes S3 and MinIO send are accepted, anything
else is refused with the path of the bad field."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.curation.bulk_ingest.s3_events import InvalidS3EventError, parse_s3_event


def _record(name: str = 'ObjectCreated:Put', **obj: Any) -> dict[str, Any]:
    return {
        'eventVersion': '2.1',
        'eventSource': 'aws:s3',
        'eventName': name,
        's3': {
            's3SchemaVersion': '1.0',
            'bucket': {'name': 'photos', 'arn': 'arn:aws:s3:::photos'},
            'object': {'key': 'in/a.jpg', 'size': 1024, 'eTag': 'abc123', **obj},
        },
    }


def test_an_aws_notification_is_parsed_with_version_and_size() -> None:
    body = {'Records': [_record(versionId='096fKKX.nfl', sequencer='0055AED6')]}
    [record] = parse_s3_event(body)
    assert (record.bucket, record.key, record.etag, record.size) == (
        'photos',
        'in/a.jpg',
        'abc123',
        1024,
    )
    assert record.version_id == '096fKKX.nfl'
    assert record.is_created


def test_a_minio_notification_with_prefixed_event_name_and_encoded_key() -> None:
    body = {
        'EventName': 's3:ObjectCreated:Put',
        'Key': 'photos/x',
        'Records': [
            _record(
                's3:ObjectCreated:CompleteMultipartUpload',
                key='in/my+photo%2B1%20%28a%29.jpg',
                eTag='"quoted-2"',
            )
        ],
    }
    [record] = parse_s3_event(body)
    assert record.key == 'in/my photo+1 (a).jpg'  # decoded exactly once
    assert record.etag == 'quoted-2'
    assert record.is_created


def test_deletes_parse_but_are_not_creates_and_need_no_etag() -> None:
    delete = _record('ObjectRemoved:Delete')
    del delete['s3']['object']['eTag']
    [record] = parse_s3_event({'Records': [delete]})
    assert not record.is_created


def test_the_test_event_s3_sends_on_configuration_has_no_records() -> None:
    assert (
        parse_s3_event({'Service': 'Amazon S3', 'Event': 's3:TestEvent', 'Bucket': 'photos'}) == []
    )


@pytest.mark.parametrize(
    ('body', 'path'),
    [
        ([], '$'),
        ('text', '$'),
        ({}, 'Records'),
        ({'Records': 'x'}, 'Records'),
        ({'Records': ['x']}, 'Records[0]'),
        ({'Records': [{'s3': {}}]}, 'Records[0].eventName'),
        ({'Records': [{'eventName': 'ObjectCreated:Put'}]}, 'Records[0].s3'),
        (
            {'Records': [{'eventName': 'ObjectCreated:Put', 's3': {'object': {}}}]},
            'Records[0].s3.bucket',
        ),
        ({'Records': [_record(key='')]}, 'Records[0].s3.object.key'),
        ({'Records': [_record(eTag='')]}, 'Records[0].s3.object.eTag'),
        ({'Records': [_record(size=-1)]}, 'Records[0].s3.object.size'),
        ({'Records': [_record(), _record(versionId=5)]}, 'Records[1].s3.object.versionId'),
    ],
)
def test_a_body_that_is_not_a_notification_names_the_first_bad_field(body: Any, path: str) -> None:
    with pytest.raises(InvalidS3EventError) as raised:
        parse_s3_event(body)
    assert raised.value.path == path
