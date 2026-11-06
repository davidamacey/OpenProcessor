"""The Redis transport used by every worker survives a collected connection with unacked tasks.

Reproduces the sequence of ``celery.worker.consumer.Consumer.on_connection_error_after_connected``
(``connection.collect()`` then ``connection.close()``) at the layer where kombu 5.7.0 fails.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kombu.transport import redis as kombu_redis
from redis.exceptions import ConnectionError as RedisConnectionError

from src.tasks.runtime import redis_transport


def _channel_with_unacked(channel_class: type[kombu_redis.Channel]) -> Any:
    """A channel whose transport was collected (``client`` is None) and which holds one unacked tag."""
    channel = channel_class.__new__(channel_class)
    channel.connection = SimpleNamespace(client=None)
    channel._pool = None
    qos = channel_class.QoS(channel)
    qos._delivered['tag-1'] = object()
    return qos


def test_stock_kombu_crashes_restoring_unacked_after_collect() -> None:
    """The control: this is the failure the worker hits (kombu issue 970)."""
    qos = _channel_with_unacked(kombu_redis.Channel)
    try:
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'hostname'"):
            qos.restore_unacked()
    finally:
        qos._delivered.clear()


def test_our_transport_leaves_the_unacked_messages_to_the_broker() -> None:
    qos = _channel_with_unacked(redis_transport.Channel)
    qos.restore_unacked()
    assert not qos._delivered


def test_a_late_ack_on_a_collected_channel_is_a_connection_error() -> None:
    """Without it the ack of a task that outlived the connection kills the reconnected consumer."""
    channel = redis_transport.Channel.__new__(redis_transport.Channel)
    channel.connection = SimpleNamespace(client=None)
    with pytest.raises(RedisConnectionError), channel.conn_or_acquire():
        pytest.fail('no client may be built from a collected transport')
    channel.connection = None  # a closed channel drops its transport altogether
    with pytest.raises(RedisConnectionError), channel.conn_or_acquire():
        pytest.fail('no client may be built from a closed channel')
    stock = kombu_redis.Channel.__new__(kombu_redis.Channel)
    stock.connection = SimpleNamespace(client=None)
    stock._pool = None
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'hostname'"):
        stock.conn_or_acquire().__enter__()


def test_a_live_connection_still_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    qos = _channel_with_unacked(redis_transport.Channel)
    qos.channel.connection = SimpleNamespace(client=object())
    restored: list[Any] = []
    monkeypatch.setattr(
        kombu_redis.QoS, 'restore_unacked', lambda _self, client=None: restored.append(client)
    )
    qos.restore_unacked('given-client')
    qos._delivered.clear()  # kombu restores a QoS' unacked messages again at interpreter exit
    assert restored == ['given-client']


def test_the_transport_is_wired_into_the_channel() -> None:
    assert redis_transport.Transport.Channel is redis_transport.Channel
    assert redis_transport.Channel.QoS is redis_transport.QoS
