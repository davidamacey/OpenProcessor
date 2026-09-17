"""Phase 3 — SAM3 fan-out telemetry (wait/inflight/response histograms).

Pins:

* Every ``segment_plate`` call observes a sample in each of the three
  histograms (wait, inflight, response), regardless of outcome.
* The ``outcome`` label tracks the call result accurately —
  ``hit`` when a candidate is returned, ``miss`` when SAM3 returned no
  candidates, ``error`` when the host returned a 5xx or bad JSON.

Tests use ``httpx.MockTransport`` so they are hermetic (~ms each) and
do not require a live SAM3 service.
"""

from __future__ import annotations

import httpx
import pytest

from scripts.curation.worker.client import Sam3Client
from src.services.curation.metrics import (
    KB_SAM3_REQUEST_INFLIGHT_SECONDS,
    KB_SAM3_REQUEST_RESPONSE_SECONDS,
    KB_SAM3_REQUEST_WAIT_SECONDS,
)


pytestmark = pytest.mark.asyncio


_CROP_BYTES = b'\xff\xd8\xff\xe0fake-jpeg'
_HOST = 'http://sam3-telemetry-fake:7000'


def _hist_obs_count(hist, *, host: str, outcome: str) -> float:
    """Return the total observation count for the labelled histogram.

    ``prometheus_client.Histogram`` stores per-bucket (non-cumulative)
    counts internally, so total observations = sum across all buckets.
    """
    child = hist.labels(host=host, outcome=outcome)
    return sum(b.get() for b in child._buckets)


def _build_client(handler):
    transport = httpx.MockTransport(handler)
    httpx_client = httpx.AsyncClient(transport=transport, timeout=5.0)
    return Sam3Client(base_url=_HOST, client=httpx_client)


async def test_wait_inflight_response_histograms_observed():
    """A successful hit increments all three histograms exactly once."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={'candidates': [{'bbox_norm': [0.1, 0.1, 0.5, 0.5], 'score': 0.91}]},
        )

    sam = _build_client(handler)
    before = {
        'wait': _hist_obs_count(KB_SAM3_REQUEST_WAIT_SECONDS, host=_HOST, outcome='hit'),
        'inflight': _hist_obs_count(KB_SAM3_REQUEST_INFLIGHT_SECONDS, host=_HOST, outcome='hit'),
        'response': _hist_obs_count(KB_SAM3_REQUEST_RESPONSE_SECONDS, host=_HOST, outcome='hit'),
    }

    candidate = await sam.segment_plate(_CROP_BYTES)
    assert candidate is not None
    assert candidate.source == 'sam3'

    after = {
        'wait': _hist_obs_count(KB_SAM3_REQUEST_WAIT_SECONDS, host=_HOST, outcome='hit'),
        'inflight': _hist_obs_count(KB_SAM3_REQUEST_INFLIGHT_SECONDS, host=_HOST, outcome='hit'),
        'response': _hist_obs_count(KB_SAM3_REQUEST_RESPONSE_SECONDS, host=_HOST, outcome='hit'),
    }
    assert after['wait'] == before['wait'] + 1
    assert after['inflight'] == before['inflight'] + 1
    assert after['response'] == before['response'] + 1

    await sam.aclose()


async def test_outcome_label_correct_on_hit_miss_error():
    """Vary the mock response: hit candidate, empty list, and 500 error.

    Each path must populate the histogram under the correct outcome
    label (``hit`` / ``miss`` / ``error``) and not cross-pollute.
    """
    host = 'http://sam3-outcome-fake:7000'

    # Three responses, one per call: hit, miss (empty list), error (500).
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    'candidates': [
                        {'bbox_norm': [0.0, 0.0, 1.0, 1.0], 'score': 0.7},
                    ]
                },
            ),
            httpx.Response(200, json={'candidates': []}),
            httpx.Response(500, json={'error': 'boom'}),
        ]
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return next(responses)

    transport = httpx.MockTransport(handler)
    httpx_client = httpx.AsyncClient(transport=transport, timeout=5.0)
    sam = Sam3Client(base_url=host, client=httpx_client)

    before = {
        outcome: _hist_obs_count(KB_SAM3_REQUEST_INFLIGHT_SECONDS, host=host, outcome=outcome)
        for outcome in ('hit', 'miss', 'error')
    }

    hit = await sam.segment_plate(_CROP_BYTES)
    miss = await sam.segment_plate(_CROP_BYTES)
    err = await sam.segment_plate(_CROP_BYTES)

    assert hit is not None
    assert miss is None
    assert err is None

    after = {
        outcome: _hist_obs_count(KB_SAM3_REQUEST_INFLIGHT_SECONDS, host=host, outcome=outcome)
        for outcome in ('hit', 'miss', 'error')
    }
    assert after['hit'] == before['hit'] + 1
    assert after['miss'] == before['miss'] + 1
    assert after['error'] == before['error'] + 1

    await sam.aclose()
