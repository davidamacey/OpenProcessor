"""The object-store seam of S3 ingest: references, configuration and the client.

Everything that talks to S3 or an S3-compatible store goes through
:class:`ObjectStore`, so tests run against an in-memory fake and the one real
implementation, :class:`Boto3ObjectStore`, is only built when a route or the
dispatcher first needs it.

Credentials are never handled here: boto3's default chain finds them (environment,
a mounted shared-credentials file, or an instance or task role). Which buckets and
prefixes may be read is deployment configuration (``OP_S3_ALLOWED``); an empty list
means S3 is off and every S3 operation is refused (fail closed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import parse_qs, quote, unquote

from src.config.env import env_bool, env_int, env_str


if TYPE_CHECKING:
    from datetime import datetime

DEFAULT_INFLIGHT_MB = 256
DEFAULT_MAX_OBJECT_MB = 256
DEFAULT_FETCH_WORKERS = 8
DEFAULT_LIST_WORKERS = 8
LIST_PAGE_SIZE = 1000

ERROR_KIND_GONE = 'gone'
ERROR_KIND_FORBIDDEN = 'forbidden'
ERROR_KIND_FETCH = 'fetch_error'


class ObjectStoreError(Exception):
    """Base of the classified object-store failures."""

    error_kind = ERROR_KIND_FETCH


class ObjectGoneError(ObjectStoreError):
    error_kind = ERROR_KIND_GONE


class ObjectForbiddenError(ObjectStoreError):
    error_kind = ERROR_KIND_FORBIDDEN


class ObjectFetchError(ObjectStoreError):
    error_kind = ERROR_KIND_FETCH


class S3NotConfiguredError(Exception):
    """``OP_S3_ALLOWED`` is empty: S3 is off."""


class S3NotAllowedError(Exception):
    """The bucket or prefix is outside ``OP_S3_ALLOWED``."""


@dataclass(frozen=True)
class ObjectRef:
    """One object version: the unit the queue, the fetch and the stored image
    reference are all made of."""

    bucket: str
    key: str
    etag: str = ''
    version_id: str | None = None
    size: int | None = None

    def to_uri(self) -> str:
        """``s3://bucket/key#etag=...[&version=...]`` with the key percent-encoded,
        the value stored as the image's ``source_identifier``."""
        frag = f'etag={quote(self.etag, safe="")}'
        if self.version_id:
            frag += f'&version={quote(self.version_id, safe="")}'
        return f's3://{self.bucket}/{quote(self.key, safe="/")}#{frag}'

    @classmethod
    def parse(cls, uri: str) -> ObjectRef:
        if not uri.startswith('s3://'):
            raise ValueError(f'not an s3 reference: {uri[:80]!r}')
        rest = uri[len('s3://') :]
        location, _, fragment = rest.partition('#')
        bucket, _, quoted_key = location.partition('/')
        if not bucket or not quoted_key:
            raise ValueError(f'no bucket or key in {uri[:80]!r}')
        query = parse_qs(fragment, keep_blank_values=True)
        return cls(
            bucket=bucket,
            key=unquote(quoted_key),
            etag=(query.get('etag') or [''])[0],
            version_id=query['version'][0] if query.get('version') else None,
        )


@dataclass(frozen=True)
class ListedObject:
    ref: ObjectRef
    last_modified: datetime


@dataclass(frozen=True)
class ListPage:
    objects: list[ListedObject]
    prefixes: list[str]
    next_token: str | None


class ObjectStore(Protocol):
    """Blocking calls; callers run them in a thread."""

    def list_page(
        self, bucket: str, prefix: str, *, token: str | None, delimiter: str | None
    ) -> ListPage: ...

    def get(self, ref: ObjectRef) -> tuple[ObjectRef, bytes]:
        """The object's bytes with the reference it actually had (the etag of the
        bytes returned, which differs from ``ref`` if the object changed after it was
        listed)."""
        ...

    def head(self, ref: ObjectRef) -> ObjectRef: ...

    def presign(self, ref: ObjectRef, expires_in: int) -> str: ...


@dataclass(frozen=True)
class S3Config:
    allowed: tuple[tuple[str, str], ...] = ()
    endpoint_url: str | None = None
    region: str | None = None
    path_style: bool = False
    inflight_mb: int = DEFAULT_INFLIGHT_MB
    max_object_mb: int = DEFAULT_MAX_OBJECT_MB
    fetch_workers: int = DEFAULT_FETCH_WORKERS
    list_workers: int = DEFAULT_LIST_WORKERS

    @property
    def configured(self) -> bool:
        return bool(self.allowed)

    @property
    def max_object_bytes(self) -> int:
        return self.max_object_mb * 1024 * 1024

    def check(self, bucket: str, key_or_prefix: str = '') -> None:
        """Raise unless ``bucket`` and the key or prefix lie inside the allow-list."""
        if not self.configured:
            raise S3NotConfiguredError(
                'S3 is not configured: set OP_S3_ALLOWED to bucket[/prefix] entries'
            )
        for allowed_bucket, allowed_prefix in self.allowed:
            if bucket == allowed_bucket and key_or_prefix.startswith(allowed_prefix):
                return
        raise S3NotAllowedError(f's3://{bucket}/{key_or_prefix} is outside OP_S3_ALLOWED')

    @classmethod
    def from_env(cls) -> S3Config:
        entries: list[tuple[str, str]] = []
        for item in env_str('OP_S3_ALLOWED', '').split(','):
            text = item.strip().removeprefix('s3://')
            if not text:
                continue
            bucket, _, prefix = text.partition('/')
            entries.append((bucket, prefix.lstrip('/')))
        return cls(
            allowed=tuple(entries),
            endpoint_url=env_str('OP_S3_ENDPOINT_URL') or None,
            region=env_str('OP_S3_REGION') or None,
            path_style=env_bool('OP_S3_PATH_STYLE'),
            inflight_mb=env_int('OP_S3_INFLIGHT_MB', DEFAULT_INFLIGHT_MB, minimum=1),
            max_object_mb=env_int('OP_S3_MAX_OBJECT_MB', DEFAULT_MAX_OBJECT_MB, minimum=1),
            fetch_workers=env_int('OP_S3_FETCH_WORKERS', DEFAULT_FETCH_WORKERS, minimum=1),
            list_workers=env_int('OP_S3_LIST_WORKERS', DEFAULT_LIST_WORKERS, minimum=1),
        )


def classify_client_error(exc: Exception) -> ObjectStoreError:
    """Map a botocore failure onto the three kinds the pipeline reports."""
    response: dict[str, Any] = getattr(exc, 'response', None) or {}
    code = str(response.get('Error', {}).get('Code', ''))
    status = int(response.get('ResponseMetadata', {}).get('HTTPStatusCode', 0) or 0)
    message = f'{type(exc).__name__}: {exc}'[:300]
    if code in {'NoSuchKey', 'NoSuchVersion', 'NoSuchBucket', '404', 'NotFound'} or status == 404:
        return ObjectGoneError(message)
    if code in {'AccessDenied', '403', 'Forbidden', 'InvalidAccessKeyId', 'SignatureDoesNotMatch'}:
        return ObjectForbiddenError(message)
    if 400 <= status < 500 and status not in {408, 429}:
        return ObjectForbiddenError(message)
    return ObjectFetchError(message)


def _strip_etag(value: Any) -> str:
    return str(value or '').strip('"')


class Boto3ObjectStore:
    """The production :class:`ObjectStore`. boto3 clients are thread-safe, so one
    client serves every thread of the process."""

    def __init__(self, config: S3Config) -> None:
        import boto3
        from botocore.config import Config

        pool = config.fetch_workers + config.list_workers + 4
        self._client = boto3.client(
            's3',
            endpoint_url=config.endpoint_url,
            region_name=config.region,
            config=Config(
                retries={'mode': 'standard', 'max_attempts': 5},
                s3={'addressing_style': 'path' if config.path_style else 'auto'},
                max_pool_connections=pool,
                connect_timeout=10,
                read_timeout=60,
            ),
        )

    def list_page(
        self, bucket: str, prefix: str, *, token: str | None, delimiter: str | None
    ) -> ListPage:
        params: dict[str, Any] = {'Bucket': bucket, 'Prefix': prefix, 'MaxKeys': LIST_PAGE_SIZE}
        if token:
            params['ContinuationToken'] = token
        if delimiter:
            params['Delimiter'] = delimiter
        try:
            resp = self._client.list_objects_v2(**params)
        except Exception as exc:
            raise classify_client_error(exc) from exc
        objects = [
            ListedObject(
                ObjectRef(
                    bucket, item['Key'], _strip_etag(item.get('ETag')), None, item.get('Size')
                ),
                item['LastModified'],
            )
            for item in resp.get('Contents', [])
        ]
        prefixes = [p['Prefix'] for p in resp.get('CommonPrefixes', [])]
        return ListPage(objects, prefixes, resp.get('NextContinuationToken'))

    def get(self, ref: ObjectRef) -> tuple[ObjectRef, bytes]:
        params: dict[str, Any] = {'Bucket': ref.bucket, 'Key': ref.key}
        if ref.version_id:
            params['VersionId'] = ref.version_id
        try:
            resp = self._client.get_object(**params)
            body = resp['Body'].read()
        except Exception as exc:
            raise classify_client_error(exc) from exc
        actual = ObjectRef(
            ref.bucket,
            ref.key,
            _strip_etag(resp.get('ETag')) or ref.etag,
            resp.get('VersionId') or ref.version_id,
            len(body),
        )
        return actual, body

    def head(self, ref: ObjectRef) -> ObjectRef:
        params: dict[str, Any] = {'Bucket': ref.bucket, 'Key': ref.key}
        if ref.version_id:
            params['VersionId'] = ref.version_id
        try:
            resp = self._client.head_object(**params)
        except Exception as exc:
            raise classify_client_error(exc) from exc
        return ObjectRef(
            ref.bucket,
            ref.key,
            _strip_etag(resp.get('ETag')) or ref.etag,
            resp.get('VersionId') or ref.version_id,
            resp.get('ContentLength'),
        )

    def presign(self, ref: ObjectRef, expires_in: int) -> str:
        params: dict[str, Any] = {'Bucket': ref.bucket, 'Key': ref.key}
        if ref.version_id:
            params['VersionId'] = ref.version_id
        url: str = self._client.generate_presigned_url(
            'get_object', Params=params, ExpiresIn=expires_in
        )
        return url


_stores: dict[int, Boto3ObjectStore] = {}


def get_object_store() -> ObjectStore:
    """The process's store (one per process id, so a forked worker builds its own).
    Raises :class:`S3NotConfiguredError` while S3 is off."""
    config = S3Config.from_env()
    if not config.configured:
        raise S3NotConfiguredError(
            'S3 is not configured: set OP_S3_ALLOWED to bucket[/prefix] entries'
        )
    pid = os.getpid()
    if pid not in _stores:
        _stores[pid] = Boto3ObjectStore(config)
    return _stores[pid]
