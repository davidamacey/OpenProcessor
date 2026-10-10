"""The JSON serializer every OpenSearch client of this repo is built with.

It is the opensearch-py serializer with the encode time recorded under the
``bulk_serialize`` ingest stage, so the cost of turning documents (an items
document carries a 1,024-float vector) into request bodies is a named line
rather than part of the unattributed remainder.
"""

from __future__ import annotations

from typing import Any

from opensearchpy.serializer import JSONSerializer

from src.utils.stage_timing import stage_timer


class TimedJSONSerializer(JSONSerializer):
    """``JSONSerializer`` whose ``dumps`` is timed."""

    def dumps(self, data: Any) -> Any:
        with stage_timer('bulk_serialize'):
            return super().dumps(data)
