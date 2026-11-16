#!/usr/bin/env python3
"""Goldens for the BYTES-tensor codec of ``op_engine_client`` (plan 18a slice R1e).

KServe v2 carries a BYTES tensor in ``raw_input_contents`` as each element's 4-byte little-endian
length followed by its bytes. This script packs element lists with Triton's own
``tritonclient.utils.serialize_byte_tensor`` (what ``src/clients/frame_engine.py`` sends through)
and writes the bytes it produced to ``native/op_engine_client/tests/goldens/bytes_codec.json``;
the Rust tests pack the same elements and read the same bytes back. An element too long to
write out is described by a byte and a repeat count, and its packed form by the SHA-256 of the bytes.

The data is committed; the Rust tests read it. ``--check`` regenerates into memory and compares
with the file. The script is deleted with the Python client when the Rust server replaces it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'native' / 'op_engine_client' / 'tests' / 'goldens'
SCRIPT = 'scripts/codegen/frame_engine_bytes_goldens.py'

#: Elements as hex strings, so a case reads the same in every language.
SMALL_CASES: list[tuple[str, list[str]]] = [
    ('one_element', ['616263']),
    ('empty_element', ['']),
    ('no_element', []),
    ('three_elements', ['61', '', '626364']),
    ('zero_bytes', ['000000', '00']),
    ('utf8_text', [bytes('café STOP 中', 'utf-8').hex()]),
    ('length_looks_like_a_prefix', ['01000000ff']),
    ('every_byte_value', [bytes(range(256)).hex()]),
    ('jpeg_start_and_end', ['ffd8ffe000104a464946', 'ffd9']),
]
#: ``(id, byte, times)``: one element of ``times`` copies of ``byte``, past 16 and 24 bit lengths.
LONG_CASES: list[tuple[str, int, int]] = [
    ('long_65536', 0x5A, 65_536),
    ('long_70000', 0xA5, 70_000),
    ('long_16777216', 0x11, 16_777_216),
]


def _pack(elements: list[bytes]) -> bytes:
    import numpy as np
    from tritonclient.utils import serialize_byte_tensor

    packed = serialize_byte_tensor(np.array(elements, dtype=np.object_))
    return b'' if packed.size == 0 else bytes(packed.item())


def render() -> dict[str, str]:
    small = []
    for case_id, hexes in SMALL_CASES:
        packed = _pack([bytes.fromhex(h) for h in hexes])
        small.append({'id': case_id, 'elements_hex': hexes, 'packed_hex': packed.hex()})
    long = []
    for case_id, byte, times in LONG_CASES:
        packed = _pack([bytes([byte]) * times])
        long.append(
            {
                'id': case_id,
                'byte': byte,
                'times': times,
                'packed_len': len(packed),
                'packed_sha256': hashlib.sha256(packed).hexdigest(),
            }
        )
    body = {
        'generated_by': SCRIPT,
        'sources': ['tritonclient.utils.serialize_byte_tensor'],
        'small': small,
        'long': long,
    }
    return {'bytes_codec.json': json.dumps(body, sort_keys=True, separators=(',', ':')) + '\n'}


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import frame_engine_goldens

    return int(frame_engine_goldens.main(argv, render, SCRIPT, out=OUT))


if __name__ == '__main__':
    raise SystemExit(main())
