"""Parse and validate S3 event notifications (the webhook body).

The shape is the one S3 publishes and S3-compatible stores (MinIO, Ceph) copy::

    {
        'Records': [
            {
                'eventName': 's3:ObjectCreated:Put',
                's3': {
                    'bucket': {'name': 'b'},
                    'object': {
                        'key': 'a+b%2Bc.jpg',
                        'size': 1,
                        'eTag': '...',
                        'versionId': '...',
                    },
                },
            }
        ]
    }

A record that does not have this shape is an error that names the offending field; a
record that is well formed but of no interest (a delete, another bucket, a non-image)
is not an error, it is simply ignored by the caller. Keys arrive URL-encoded (a space
is ``+``) and are decoded exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote_plus


MAX_RECORDS = 10_000


class InvalidS3EventError(ValueError):
    """The body is not an S3 event notification; ``path`` locates the first bad field."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f'{path}: {reason}')
        self.path = path
        self.reason = reason


@dataclass(frozen=True)
class EventRecord:
    event_name: str
    bucket: str
    key: str
    etag: str
    version_id: str | None
    size: int | None

    @property
    def is_created(self) -> bool:
        return self.event_name.removeprefix('s3:').startswith('ObjectCreated:')


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidS3EventError(path, 'expected a non-empty string')
    return value


def _record(raw: Any, index: int) -> EventRecord:
    base = f'Records[{index}]'
    if not isinstance(raw, dict):
        raise InvalidS3EventError(base, 'expected an object')
    name = _text(raw.get('eventName'), f'{base}.eventName')
    s3 = raw.get('s3')
    if not isinstance(s3, dict):
        raise InvalidS3EventError(f'{base}.s3', 'expected an object')
    bucket_obj, obj = s3.get('bucket'), s3.get('object')
    if not isinstance(bucket_obj, dict):
        raise InvalidS3EventError(f'{base}.s3.bucket', 'expected an object')
    if not isinstance(obj, dict):
        raise InvalidS3EventError(f'{base}.s3.object', 'expected an object')
    bucket = _text(bucket_obj.get('name'), f'{base}.s3.bucket.name')
    key = unquote_plus(_text(obj.get('key'), f'{base}.s3.object.key'))
    created = name.removeprefix('s3:').startswith('ObjectCreated:')
    etag = obj.get('eTag', obj.get('etag', ''))
    if created:
        etag = _text(etag, f'{base}.s3.object.eTag')
    size = obj.get('size')
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
        raise InvalidS3EventError(f'{base}.s3.object.size', 'expected a non-negative integer')
    version = obj.get('versionId')
    if version is not None and not isinstance(version, str):
        raise InvalidS3EventError(f'{base}.s3.object.versionId', 'expected a string')
    return EventRecord(
        event_name=name,
        bucket=bucket,
        key=key,
        etag=str(etag or '').strip('"'),
        version_id=version or None,
        size=size,
    )


def parse_s3_event(body: Any) -> list[EventRecord]:
    """Every record of a notification. The ``s3:TestEvent`` S3 sends when a
    notification is configured has no records and parses to an empty list."""
    if not isinstance(body, dict):
        raise InvalidS3EventError('$', 'expected a JSON object')
    if body.get('Event') == 's3:TestEvent':
        return []
    records = body.get('Records')
    if not isinstance(records, list):
        raise InvalidS3EventError('Records', 'expected a list')
    if len(records) > MAX_RECORDS:
        raise InvalidS3EventError('Records', f'more than {MAX_RECORDS} records')
    return [_record(raw, i) for i, raw in enumerate(records)]
