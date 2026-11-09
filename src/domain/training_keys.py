"""Storage keys and fixed names of one training run: the log, the served metric artifacts.

A run's ``artifact_prefix`` is ``projects/<slug>/training/<job id>`` (the ``projects`` storage
root, 02c section 5). Both the API (reads) and the trainer (writes) build keys here, so the two
sides cannot disagree.
"""

from __future__ import annotations

from src.domain.storage_keys import make_key, parse_key


RUN_LOG_NAME = 'run.log'

# The cancel reason a campaign gives the queued runs it no longer needs; the wire calls them
# ``skipped``.
SKIPPED_ERROR_CODE = 'auto_skip_threshold_met'

# GET {api_prefix}/train/artifacts/{job_id}/{name} -- exact basenames only (no path traversal is
# even expressible: no ``/`` is a valid character in any of these). Metrics/plots only --
# deliberately excludes Ultralytics' ``train_batch*.jpg``/``val_batch*.jpg`` (actual
# training/validation images, not aggregate metrics) and ``labels.jpg`` (a dataset-content
# visualization), any of which could leak imagery a deployment doesn't want served over this route.
RUN_ARTIFACT_WHITELIST = frozenset(
    {
        'confusion_matrix.png',
        'confusion_matrix_normalized.png',
        'results.png',
        'results.csv',
        'BoxF1_curve.png',
        'BoxP_curve.png',
        'BoxPR_curve.png',
        'BoxR_curve.png',
    }
)
REGISTRY_SNAPSHOT_NAME = 'registry_snapshot.json'


def run_prefix(slug: str, job_id: str) -> str:
    return make_key('projects', slug, 'training', job_id)


def run_object_key(prefix: str, name: str) -> str:
    """``<prefix>/<name>``; ``prefix`` must itself be a valid key and ``name`` one segment."""
    parsed = parse_key(prefix)
    return make_key(parsed.root, *parsed.parts, name)
