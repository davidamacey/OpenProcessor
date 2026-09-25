"""Detector bake-off harness.

Scores any object detector against a frozen YOLO test split using a single
shared pycocotools metric, so in-house and public models are compared on
identical ground, per class. What is measured (which eval classes, cascade
context classes, Triton model, metric thresholds, plugin backends and dataset
converters) comes from a
:class:`~scripts.curation.bakeoff.profile.BakeoffProfile`. Domain-specific
reference configurations are opt-in and live outside this package, under the
repo's ``examples/bakeoff/`` tree (loaded only by an explicit profile path).
See :mod:`run` for the single-model CLI and :mod:`bakeoff_runner` for the job
runner.

The harness lives under ``scripts/`` rather than ``src/`` because it runs in
its own evaluator container (a newer detection stack than the API image
pins), not inside the API process; the API only drops job specs for it.
"""

from __future__ import annotations
