#!/usr/bin/env python3
"""Regenerate (or ``--check``) every committed API contract under ``contracts/``.

Runs, in order:

* ``export_region_status_to_ts.py`` -> ``contracts/ts/regionStatus.ts``
* ``export_api_contracts.py``       -> ``contracts/json/item_wire.json``,
  ``contracts/ts/itemWire.ts``, ``contracts/ts/classSources.ts``,
  ``contracts/openapi/curation.json``

Usage:

    python3 scripts/codegen/generate_contracts.py          # or: make contracts
    python3 scripts/codegen/generate_contracts.py --check  # or: make contracts-check
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f'_contracts_{name}', _HERE / f'{name}.py')
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load {name}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--check', action='store_true', help='Do not write; exit 1 on drift.')
    args = parser.parse_args(argv)
    flag = ['--check'] if args.check else []

    # Run both even if the first fails so one invocation reports all drift.
    region_rc = _load('export_region_status_to_ts').main(flag)
    api_rc = _load('export_api_contracts').main(flag)
    return 1 if region_rc or api_rc else 0


if __name__ == '__main__':
    sys.exit(main())
