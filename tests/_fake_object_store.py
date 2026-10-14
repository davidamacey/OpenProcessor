"""An in-memory :class:`ObjectStore` for offline tests: buckets of versions with
modification times, ListObjectsV2-style paging and delimiters, and injectable failures."""

from __future__ import annotations

import hashlib
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.services.curation.bulk_ingest.object_store import (
    ListedObject,
    ListPage,
    ObjectFetchError,
    ObjectForbiddenError,
    ObjectGoneError,
    ObjectRef,
)


if TYPE_CHECKING:
    from src.services.curation.bulk_ingest.object_store import ObjectStoreError

PAGE = 3


class FakeObjectStore:
    def __init__(self, page_size: int = PAGE) -> None:
        self.page_size = page_size
        self.objects: dict[tuple[str, str], tuple[bytes, datetime]] = {}
        self.errors: dict[tuple[str, str], ObjectStoreError] = {}
        self.list_calls: list[tuple[str, str, str | None, str | None]] = []
        self.get_calls: list[str] = []
        self.lock = threading.Lock()
        self.fail_listing_after: int | None = None

    def put(
        self, bucket: str, key: str, data: bytes, *, modified: float = 1_700_000_000.0
    ) -> ObjectRef:
        self.objects[(bucket, key)] = (data, datetime.fromtimestamp(modified, UTC))
        return self.ref(bucket, key)

    def ref(self, bucket: str, key: str) -> ObjectRef:
        data, _ = self.objects[(bucket, key)]
        return ObjectRef(
            bucket, key, hashlib.md5(data, usedforsecurity=False).hexdigest(), None, len(data)
        )

    def fail(self, bucket: str, key: str, kind: str) -> None:
        self.errors[(bucket, key)] = {
            'gone': ObjectGoneError('NoSuchKey'),
            'forbidden': ObjectForbiddenError('AccessDenied'),
            'transient': ObjectFetchError('503 SlowDown'),
        }[kind]

    def list_page(
        self, bucket: str, prefix: str, *, token: str | None, delimiter: str | None
    ) -> ListPage:
        with self.lock:
            self.list_calls.append((bucket, prefix, token, delimiter))
            if (
                self.fail_listing_after is not None
                and len(self.list_calls) > self.fail_listing_after
            ):
                raise ObjectFetchError('listing unavailable')
        keys = sorted(k for (b, k) in self.objects if b == bucket and k.startswith(prefix))
        prefixes: list[str] = []
        entries: list[tuple[str, str]] = []
        for key in keys:
            rest = key[len(prefix) :]
            if delimiter and delimiter in rest:
                common = prefix + rest.split(delimiter)[0] + delimiter
                if common not in prefixes:
                    prefixes.append(common)
            else:
                entries.append(('o', key))
        merged = [('p', p) for p in prefixes] + entries
        start = int(token) if token else 0
        window = merged[start : start + self.page_size]
        next_token = str(start + self.page_size) if start + self.page_size < len(merged) else None
        return ListPage(
            [
                ListedObject(self.ref(bucket, k), self.objects[(bucket, k)][1])
                for t, k in window
                if t == 'o'
            ],
            [k for t, k in window if t == 'p'],
            next_token,
        )

    def get(self, ref: ObjectRef) -> tuple[ObjectRef, bytes]:
        with self.lock:
            self.get_calls.append(ref.key)
        if (ref.bucket, ref.key) in self.errors:
            raise self.errors[(ref.bucket, ref.key)]
        if (ref.bucket, ref.key) not in self.objects:
            raise ObjectGoneError('NoSuchKey')
        data, _ = self.objects[(ref.bucket, ref.key)]
        return self.ref(ref.bucket, ref.key), data

    def head(self, ref: ObjectRef) -> ObjectRef:
        if (ref.bucket, ref.key) not in self.objects:
            raise ObjectGoneError('NoSuchKey')
        return self.ref(ref.bucket, ref.key)

    def presign(self, ref: ObjectRef, expires_in: int) -> str:
        return f'https://fake.example/{ref.bucket}/{ref.key}?expires={expires_in}'
