"""The kombu Redis transport that survives a collected connection with tasks still running.

kombu 5.7.0 (the latest release) makes a worker die when the broker connection drops while an
``acks_late`` task is running. Celery's ``Consumer.on_connection_error_after_connected`` calls
``connection.collect()`` and then ``connection.close()``. ``collect()`` sets the transport's
``client`` to ``None``; ``close()`` then closes the default channel, whose ``close`` restores the
unacked messages and so builds a fresh Redis client from ``self.connection.client.hostname``:
``AttributeError: 'NoneType' object has no attribute 'hostname'``. Celery does not treat that as a
connection error, the consumer start raises "Unrecoverable error", and the worker begins a warm
shutdown that waits for every running task before the container can restart. For a worker of long
jobs that is minutes to hours of a queue nobody consumes. Upstream reports the same traceback
since 2018 (https://github.com/celery/kombu/issues/970, /issues/582).

The same happens later when the task that was running finishes: its ack goes through the stale
channel and builds a client from the same ``None`` (``Unrecoverable error: ... no attribute
'client'``), killing the consumer that had reconnected by then.

Any Redis call needs a client, and the transport that owned the credentials is already gone, so
it cannot succeed. This transport turns that into the ``ConnectionError`` kombu and Celery
already handle (``maybe_close_channel`` ignores it, ``ack_log_error`` logs "Couldn't ack"), and
the restore forgets its tags. The messages stay in the broker's ``unacked`` hash and the visibility
timeout (``VISIBILITY_TIMEOUT_S``) returns them, which is Celery's documented behaviour for an
``acks_late`` task whose connection was lost (``worker_cancel_long_running_tasks_on_connection_loss``). Registered through ``broker_transport`` in
``src/tasks/runtime/gpu_app.py`` (https://docs.celeryq.dev/en/stable/userguide/configuration.html
#broker-transport).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from kombu.transport import redis
from redis.exceptions import ConnectionError as RedisConnectionError


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)


class QoS(redis.QoS):
    """``restore_unacked`` that does nothing once the transport has been collected."""

    def restore_unacked(self, client: Any = None) -> None:
        if client is None and self.channel.connection.client is None:
            logger.warning(
                'broker connection already collected: leaving %d unacked message(s) to their '
                'tasks and the visibility timeout',
                len(self._delivered),
            )
            self._delivered.clear()
            return
        super().restore_unacked(client)


class Channel(redis.Channel):
    QoS = QoS

    @contextmanager
    def conn_or_acquire(self, client: Any = None) -> Iterator[Any]:
        if client is None and self.connection.client is None:
            raise RedisConnectionError('the broker connection was collected')
        with super().conn_or_acquire(client) as acquired:
            yield acquired


class Transport(redis.Transport):
    Channel = Channel
