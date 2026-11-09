"""Forget a process's live-gauge files when it leaves (``PROMETHEUS_MULTIPROC_DIR``).

In multiprocess mode ``/metrics`` reads every file in the directory, and a process that exits
without telling the library leaves its live gauges behind: 1,000 such file sets made a scrape
take 2.8 s instead of 5 ms. The documented remedy is ``multiprocess.mark_process_dead(pid)``
(https://prometheus.github.io/client_python/multiprocess/). Counters and histograms stay: they are
history the next scrape still needs.
"""

from __future__ import annotations

from src.config.env import env_optional_str


def forget_process(pid: int) -> None:
    """Remove ``pid``'s live-gauge files; a no-op when no multiprocess directory is configured."""
    if env_optional_str('PROMETHEUS_MULTIPROC_DIR') is None:
        return
    from prometheus_client import multiprocess

    multiprocess.mark_process_dead(pid)
