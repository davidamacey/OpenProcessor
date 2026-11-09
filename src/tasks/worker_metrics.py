"""One ``/metrics`` HTTP server per Celery worker container.

The pool children write their metrics into ``PROMETHEUS_MULTIPROC_DIR`` and the worker's main
process serves the merged view, as the client's multiprocess guide describes
(https://prometheus.github.io/client_python/multiprocess/). ``worker_ready`` fires in the main
process once the pool is up. A bound port that is taken or an unset directory is logged and the
worker keeps running unscraped: metrics never stop a job.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from celery.signals import worker_ready
from prometheus_client import CollectorRegistry, multiprocess, start_http_server

from src.config.env import env_optional_str, env_str


logger = logging.getLogger(__name__)

DEFAULT_PORT = '9808'
_started = threading.Event()


def start_server() -> int | None:
    """Serve the merged multiprocess registry; returns the bound port or ``None``."""
    raw = env_str('OP_WORKER_METRICS_PORT', DEFAULT_PORT).strip()
    if _started.is_set() or raw in ('', '0', 'off'):
        return None
    if not env_optional_str('PROMETHEUS_MULTIPROC_DIR'):
        logger.warning('worker_metrics_disabled reason=no PROMETHEUS_MULTIPROC_DIR')
        return None
    try:
        port = int(raw)
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        start_http_server(port, registry=registry)
    except (ValueError, OSError) as exc:
        logger.warning('worker_metrics_not_started port=%s error=%s', raw, exc)
        return None
    _started.set()
    return port


def _on_worker_ready(**_: Any) -> None:
    start_server()


def install() -> None:
    worker_ready.connect(_on_worker_ready, weak=False)
