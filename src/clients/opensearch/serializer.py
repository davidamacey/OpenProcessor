"""The JSON serializer every OpenSearch client of this repo is built with.

An items document carries a 1,024-float vector and a bulk request carries a
dozen documents, so encoding bodies is a real share of ingest CPU. This is the
opensearch-py serializer with ``orjson`` doing the encoding (it writes floats in
C where ``json.dumps`` is several times slower) and the time recorded under the
``bulk_serialize`` ingest stage.

Anything ``orjson`` does not take natively (numpy scalars and arrays, datetimes,
integers beyond 64 bits) goes to the same ``default`` hook, or to the stock
encoder, as before. The one difference is a NaN or infinity, which ``orjson``
writes as ``null`` where ``json.dumps`` writes a token OpenSearch rejects. Responses are decoded
with ``orjson`` too (a page of vectors is most of the fetch's client CPU), falling back to the
stock decoder for anything it will not read.
"""

from __future__ import annotations

from typing import Any

import orjson
from opensearchpy.serializer import JSONSerializer

from src.utils.stage_timing import stage_timer


_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_PASSTHROUGH_DATETIME


class OrjsonSerializer(JSONSerializer):
    """``JSONSerializer`` that encodes and decodes with ``orjson`` and times ``dumps``."""

    def loads(self, s: Any) -> Any:
        # A page of 2,000 documents with their vectors is about 25 MB of floats; orjson reads
        # it in half the time ``json.loads`` takes and returns the same objects. A body it will
        # not read (NaN or Infinity tokens, a lone surrogate) goes to the stock decoder, which
        # gives the stock answer or the stock error. The one difference: an integer beyond 64
        # bits comes back as a float; no field of an index this repo creates holds one.
        try:
            return orjson.loads(s)
        except orjson.JSONDecodeError:
            return super().loads(s)

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
