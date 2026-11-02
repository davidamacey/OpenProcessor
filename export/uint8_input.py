"""Command-line wrapper for ``src/utils/uint8_input.py`` (the in-graph uint8 input prefix).

The implementation lives in ``src`` so the API image (promote step) and the export scripts
share one copy.

Usage (for build scripts that shell out to ``trtexec``)::

    python export/uint8_input.py IN.onnx OUT.onnx [--mean M M M --std S S S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.uint8_input import add_uint8_input


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', maxsplit=1)[0])
    parser.add_argument('onnx_in')
    parser.add_argument('onnx_out')
    parser.add_argument('--mean', type=float, nargs=3)
    parser.add_argument('--std', type=float, nargs=3)
    args = parser.parse_args(argv)
    mean = tuple(args.mean) if args.mean else None
    std = tuple(args.std) if args.std else None
    print(add_uint8_input(args.onnx_in, args.onnx_out, mean=mean, std=std))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
