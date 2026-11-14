"""`serialize_item` output over 1,000 varied documents is pinned byte for byte."""

from __future__ import annotations

import hashlib
import json

from curation.wire_golden_docs import golden_docs
from src.services.curation.wire import serialize_item


GOLDEN_SHA256 = 'f50d0138c89552b38e8a7b09c12b0d5803583b035e5ba5b3671a9e5889225898'


def test_serialize_item_output_is_unchanged_over_1000_documents() -> None:
    state_fields = {'a': 'embedding_state_a', 'b': 'embedding_state_b'}
    blob = json.dumps(
        [
            serialize_item(doc, fid, state_fields=state_fields, api_prefix='/p')
            for fid, doc in golden_docs()
        ],
        sort_keys=False,
        default=str,
    ).encode()
    assert hashlib.sha256(blob).hexdigest() == GOLDEN_SHA256
