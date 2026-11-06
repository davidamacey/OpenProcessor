"""A fake Valkey and project seeding shared by the event tests."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import psycopg
import redis


class FakeBroker:
    """In-memory pub/sub: what a Valkey server does for ``PUBLISH`` and ``SUBSCRIBE``."""

    def __init__(self) -> None:
        self.pubsubs: list[FakePubSub] = []
        self.published: list[str] = []
        self.drop_wakeups = False
        self.down = False
        self.on_publish: Any = None

    def publish(self, channel: str) -> None:
        if self.down:
            raise redis.ConnectionError('valkey is down')
        if self.on_publish is not None:
            self.on_publish(channel)
        self.published.append(channel)
        if self.drop_wakeups:
            return
        for ps in self.pubsubs:
            if channel in ps.channels:
                ps.queue.put_nowait({'type': 'message', 'channel': channel, 'data': '1'})

    def disconnect_subscribers(self) -> None:
        for ps in self.pubsubs:
            ps.broken = True


class FakeSyncRedis:
    def __init__(self, broker: FakeBroker) -> None:
        self.broker = broker

    def publish(self, channel: str, _message: Any) -> int:
        self.broker.publish(channel)
        return 1


class FakePubSub:
    def __init__(self, broker: FakeBroker) -> None:
        self.broker = broker
        self.channels: set[str] = set()
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.broken = False
        broker.pubsubs.append(self)

    def _check(self) -> None:
        if self.broker.down or self.broken:
            raise redis.ConnectionError('connection lost')

    async def subscribe(self, *channels: str) -> None:
        self._check()
        self.channels.update(channels)

    async def unsubscribe(self, *channels: str) -> None:
        self._check()
        self.channels.difference_update(channels)

    async def get_message(
        self,
        ignore_subscribe_messages: bool = False,  # noqa: ARG002  # the real signature
        timeout: float = 0.0,
    ):
        self._check()
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except TimeoutError:
            return None

    async def aclose(self) -> None:
        if self in self.broker.pubsubs:
            self.broker.pubsubs.remove(self)


class FakeAsyncRedis:
    def __init__(self, broker: FakeBroker) -> None:
        self.broker = broker

    def pubsub(self) -> FakePubSub:
        if self.broker.down:
            raise redis.ConnectionError('valkey is down')
        return FakePubSub(self.broker)

    async def aclose(self) -> None:
        return None


def seed_project(pg: Any, slug: str) -> uuid.UUID:
    project = uuid.uuid4()
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        conn.execute('INSERT INTO project_slugs (slug) VALUES (%s)', (slug,))
        conn.execute(
            'INSERT INTO projects (id, slug, display_name, status, index_prefix, storage_prefix, '
            "model_prefix, mlflow_experiment, created_by) VALUES (%s, %s, %s, 'active', %s, %s, %s, %s, 'test')",
            (project, slug, slug, f'{slug}__', f'projects/{slug}', f'{slug}__', slug),
        )
    return project


def count_events(pg: Any, project: uuid.UUID | None = None) -> int:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        if project is None:
            return conn.execute('SELECT count(*) FROM events').fetchone()[0]
        return conn.execute(
            'SELECT count(*) FROM events WHERE project_id = %s', (project,)
        ).fetchone()[0]


def count_global_events(pg: Any) -> int:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        return conn.execute('SELECT count(*) FROM events WHERE project_id IS NULL').fetchone()[0]
