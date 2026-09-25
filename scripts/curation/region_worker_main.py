#!/usr/bin/env python3
"""Compatibility shim — the detection worker lives in ``scripts/curation/worker/``.

Existing call sites that do ``python -m scripts.curation.region_worker_main``
or import ``scripts.curation.region_worker_main`` continue to work via the
re-exports below.
"""

from __future__ import annotations

import sys

from scripts.curation.worker import *  # noqa: F403
from scripts.curation.worker import run  # noqa: F401
from scripts.curation.worker.__main__ import main, parse_args  # noqa: F401
from scripts.curation.worker.bulk_writer import _bulk_update  # noqa: F401
from scripts.curation.worker.cascade import _process_crop  # noqa: F401
from scripts.curation.worker.state import (  # noqa: F401
    _class_group,
    _is_secondary_shape,
    _ItemTask,
    _wait_for_sentinel_clear,
    region_profile,
)


if __name__ == '__main__':
    sys.exit(main())
