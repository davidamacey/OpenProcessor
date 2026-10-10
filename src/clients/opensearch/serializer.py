"""The JSON serializer every OpenSearch client of this repo is built with.

An items document carries a 1,024-float vector and a bulk request carries a
dozen documents, so encoding bodies is a real share of ingest CPU. This is the
opensearch-py serializer with ``orjson`` doing the encoding (it writes floats in
C where ``json.dumps`` is several times slower) and the time recorded under the
``bulk_serialize`` ingest stage.

Anything ``orjson`` does not take natively (numpy scalars and arrays, datetimes,
integers beyond 64 bits) goes to the same ``default`` hook, or to the stock
encoder, as before. The one difference is a NaN or infinity, which ``orjson``
writes as ``null`` where ``json.dumps`` writes a token OpenSearch rejects.
"""

from __future__ import annotations

from typing import Any

import orjson
from opensearchpy.serializer import JSONSerializer

from src.utils.stage_timing import stage_timer


_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_PASSTHROUGH_DATETIME


class OrjsonSerializer(JSONSerializer):
    """``JSONSerializer`` that encodes with ``orjson`` and times ``dumps``."""

    def dumps(self, data: Any) -> Any:
        if isinstance(data, str):
            return data
        with stage_timer('bulk_serialize'):
            try:
                return orjson.dumps(data, default=self.default, option=_OPTIONS).decode()
            except TypeError:
                # Not encodable by orjson (a 65-bit integer, a type its default hook cannot
                # take): the stock encoder gives the stock answer or the stock error.
                return super().dumps(data)
