"""Step 3a of the 0.6.0 upgrade: add the compact text normalizer to every upgraded items index.

Analysis settings are static, so an existing index takes them only while closed:
``_close``, ``PUT _settings``, ``_open``
(https://docs.opensearch.org/latest/api-reference/index-apis/update-settings/). The step is
bounded and safe (review fix RR-8, 02c section 2.2):

* one index at a time, in the order given (slug order); each is closed only while its own
  settings call runs;
* the reopen is in a ``finally`` and waits for at least yellow health, bounded at 120 s;
* ``_analyze`` with the normalizer verifies the reopened index; a mismatch stops the upgrade;
* a refused settings call leaves the index open and raises :class:`AnalysisSettingsError` naming it;
* an index that already carries the normalizer is skipped, so a re-run issues no write.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.clients.curation_opensearch.text_analysis import ITEMS_ANALYSIS, NORMALIZER_NAME


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

HEALTH_WAIT = '120s'
# What the normalizer must make of this text: separators dropped, accent folded, upper-cased.
PROBE_TEXT = 'ab-1 cé 9'
PROBE_EXPECTED = 'AB1CE9'
# A planning estimate for the dry run only (the real run prints the measured seconds).
ESTIMATE_BASE_S = 5.0
ESTIMATE_PER_GIB_S = 2.0
GIB = 1024**3


class AnalysisSettingsError(RuntimeError):
    """An index could not take the analysis settings; the upgrade stops before ``delete_sources``."""

    def __init__(self, index: str, detail: str):
        super().__init__(f'analysis settings on {index}: {detail}')
        self.index = index


@dataclass
class AnalysisReport:
    lines: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _has_normalizer(settings: dict[str, Any]) -> bool:
    for cfg in settings.values():
        analysis = ((cfg.get('settings') or {}).get('index') or {}).get('analysis') or {}
        if NORMALIZER_NAME in (analysis.get('normalizer') or {}):
            return True
    return False


async def _doc_count_and_bytes(client: Any, index: str) -> tuple[int, int]:
    stats = await client.indices.stats(index=index, metric='docs,store')
    total = (stats.get('_all') or {}).get('primaries') or {}
    return int((total.get('docs') or {}).get('count', 0)), int(
        (total.get('store') or {}).get('size_in_bytes', 0)
    )


async def dry_run(
    client: Any, indexes: Sequence[str], *, say: Callable[[str], None] = lambda _: None
) -> AnalysisReport:
    """List every index with its document count, size and estimated downtime; change nothing."""
    report = AnalysisReport()
    total = len(indexes)
    for n, index in enumerate(indexes, start=1):
        settings = await client.indices.get_settings(index=index)
        if _has_normalizer(settings):
            line = f'dry run {n}/{total} {index} already has {NORMALIZER_NAME}, skipped'
            report.skipped.append(index)
        else:
            docs, size = await _doc_count_and_bytes(client, index)
            estimate = ESTIMATE_BASE_S + ESTIMATE_PER_GIB_S * size / GIB
            line = (
                f'dry run {n}/{total} {index} {docs} documents, {size} bytes: closed for about '
                f'{estimate:.0f} s (planning estimate)'
            )
            report.changed.append(index)
        report.lines.append(line)
        say(line)
    return report


async def _reopen(client: Any, index: str) -> str:
    """Open the index and wait (bounded) for at least yellow; returns the health colour."""
    await client.indices.open(index=index)
    health = await client.cluster.health(index=index, wait_for_status='yellow', timeout=HEALTH_WAIT)
    if health.get('timed_out'):
        raise AnalysisSettingsError(
            index, f'did not reach yellow health within {HEALTH_WAIT} after reopening'
        )
    return str(health.get('status', 'unknown'))


async def _verify(client: Any, index: str) -> None:
    answer = await client.indices.analyze(
        index=index, body={'normalizer': NORMALIZER_NAME, 'text': PROBE_TEXT}
    )
    tokens = [t.get('token') for t in answer.get('tokens', [])]
    if tokens != [PROBE_EXPECTED]:
        raise AnalysisSettingsError(
            index,
            f'{NORMALIZER_NAME} gave {tokens!r} for {PROBE_TEXT!r}, expected {PROBE_EXPECTED!r}',
        )


async def apply(
    client: Any, indexes: Sequence[str], *, say: Callable[[str], None] = lambda _: None
) -> AnalysisReport:
    """Add the normalizer to each index that lacks it. Raises :class:`AnalysisSettingsError`."""
    report = AnalysisReport()
    total = len(indexes)
    for n, index in enumerate(indexes, start=1):
        if _has_normalizer(await client.indices.get_settings(index=index)):
            line = f'{n}/{total} {index} already has {NORMALIZER_NAME}, skipped'
            report.skipped.append(index)
        else:
            line = await _apply_one(client, index, n, total)
            report.changed.append(index)
        report.lines.append(line)
        say(line)
    return report


async def _apply_one(client: Any, index: str, n: int, total: int) -> str:
    closed_at = time.monotonic()
    settings_error: OpenSearchException | None = None
    try:
        await client.indices.close(index=index)
        await client.indices.put_settings(index=index, body={'index': {'analysis': ITEMS_ANALYSIS}})
    except OpenSearchException as exc:
        settings_error = exc
    finally:
        # Reopen whatever happened to the settings call: an index is never left closed.
        colour = await _reopen(client, index)
    if settings_error is not None:
        raise AnalysisSettingsError(
            index,
            f'the close or settings call was refused ({settings_error}); the index is open and unchanged',
        ) from settings_error
    seconds = time.monotonic() - closed_at
    await _verify(client, index)
    return f'{n}/{total} {index} closed {seconds:.1f} s, reopened, health {colour}'
