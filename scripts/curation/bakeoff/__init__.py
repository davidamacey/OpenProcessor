"""License-plate detector bake-off harness.

Scores any plate detector against a frozen YOLO test split using a single
shared pycocotools metric, so in-house and public models are compared on
identical ground. See :mod:`run` for the CLI entrypoint.
"""

from __future__ import annotations
