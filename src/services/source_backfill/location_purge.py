"""The location-purge step: bring every stored coordinate of the project within its policy.

``drop`` removes every position field from the images index (`capture.location`, altitude,
accuracy, heading, precision); ``store_coarse`` rounds a stored exact position to 0.01 degree and
removes the finer detail; ``store`` changes nothing. The step runs as bounded `update_by_query`
calls (OpenSearch `max_docs`, `conflicts=proceed`) over the documents that still violate the
policy, so a re-run finds nothing and writes nothing. Its record is counts only: no coordinate,
image id or path is returned, logged or audited.

Another process may serve the previous policy for up to ``POLICY_CACHE_TTL_S`` after a switch, so
the first pass waits that long (once per run) before scanning; an image written by such a process
in the window is then still caught.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Final

from src.clients.curation_opensearch.capture_fields import LOCATION_FIELDS
from src.config.curation_accessors import images_index
from src.core.logging import get_logger
from src.domain.location_policy import COARSE_DECIMALS
from src.services.curation.location_policy_store import POLICY_CACHE_TTL_S


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

PURGE_BATCH: Final = 5000
SETTLE_S: Final = POLICY_CACHE_TTL_S + 1

_DROP_SCRIPT = (
    "if (ctx._source.capture == null) { ctx.op = 'noop' } "
    'else { for (f in params.fields) { ctx._source.capture.remove(f) } }'
)
_COARSE_SCRIPT = """
def c = ctx._source.capture;
if (c == null || c.location == null) { ctx.op = 'noop' }
else {
  def lat = Math.round(c.location.lat * params.scale) / params.scale;
  def lon = Math.round(c.location.lon * params.scale) / params.scale;
  c.remove('altitude_m'); c.remove('location_accuracy_m'); c.remove('heading_deg');
  if (lat == 0 && lon == 0) { c.remove('location'); c.remove('location_precision') }
  else { c.location = ['lat': lat, 'lon': lon]; c.location_precision = 'coarse' }
}
"""


def _query(policy: str) -> dict[str, Any]:
    if policy == 'drop':
        return {
            'bool': {'should': [{'exists': {'field': f'capture.{f}'}} for f in LOCATION_FIELDS]}
        }
    return {
        'bool': {
            'filter': [{'exists': {'field': 'capture.location'}}],
            'must_not': [{'term': {'capture.location_precision': 'coarse'}}],
        }
    }


def _script(policy: str) -> dict[str, Any]:
    if policy == 'drop':
        return {
            'lang': 'painless',
            'source': _DROP_SCRIPT,
            'params': {'fields': list(LOCATION_FIELDS)},
        }
    return {
        'lang': 'painless',
        'source': _COARSE_SCRIPT,
        'params': {'scale': float(10**COARSE_DECIMALS)},
    }


async def purge_pass(client: AsyncOpenSearch, policy: str) -> int:
    """One bounded `update_by_query` pass; the number of documents it changed or skipped on a
    version conflict. `conflicts=proceed` counts a document a concurrent writer touched as a
    conflict instead of failing, so a pass that only skipped must not read as "nothing left": the
    caller runs another pass until neither a change nor a conflict is reported (OpenSearch docs,
    update-by-query API: `version_conflicts`)."""
    if policy == 'store':
        return 0
    resp = await client.update_by_query(
        index=images_index(),
        body={'query': _query(policy), 'script': _script(policy)},
        max_docs=PURGE_BATCH,
        conflicts='proceed',
        refresh=True,
    )
    return int(resp.get('updated', 0)) + int(resp.get('version_conflicts', 0))


async def settle() -> None:
    """Wait for every process's cached policy to be the new one."""
    await asyncio.sleep(SETTLE_S)


__all__ = ['PURGE_BATCH', 'SETTLE_S', 'purge_pass', 'settle']
