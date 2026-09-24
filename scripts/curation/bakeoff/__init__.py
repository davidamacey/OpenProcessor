"""Detector bake-off harness.

Scores any object detector against a frozen YOLO test split using a single
shared pycocotools metric, so in-house and public models are compared on
identical ground. What is measured (target class, cascade context classes,
Triton model, metric thresholds) comes from a
:class:`~scripts.curation.bakeoff.profile.BakeoffProfile`; domain-specific
reference configurations live under ``examples/``. See :mod:`run` for the
single-model CLI and :mod:`bakeoff_runner` for the job runner.

The harness lives under ``scripts/`` rather than ``src/`` because it runs in
its own evaluator container (a newer detection stack than the API image
pins), not inside the API process; the API only drops job specs for it.
"""

from __future__ import annotations
