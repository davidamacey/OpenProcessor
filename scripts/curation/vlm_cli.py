"""Command line of the VLM labeling worker."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.services.projects.script_binding import add_project_argument


DEFAULT_API = os.environ.get('OP_API', 'http://localhost:4603')
DEFAULT_OS = os.environ.get('OPENSEARCH_URL', 'http://localhost:4607')

# Default thresholds match ``pipeline_auto_label``'s skip logic so the
# worker and the on-demand pipeline make the same decisions.
# Skip VLM classify when the classifier already labeled the crop with at
# least this confidence. Raised 0.70 -> 0.80 to align with
# pipeline.classifier_confidence_skip_vlm and the detection worker's combined
# path's own low-confidence threshold. The 0.70-0.80 band was sending high-confidence classifier crops
# to the VLM and surfacing them in the vlm_low_conf review tab as
# "classifier 95.9 %, VLM medium" — noise the human review queue doesn't need.
DEFAULT_CLASSIFIER_CONF_SKIP = 0.80


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Async polling worker that drains uncertain crops to the VLM.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--api', default=DEFAULT_API, help='triton-api base URL')
    p.add_argument('--opensearch', default=DEFAULT_OS, help='OpenSearch base URL')
    p.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Crops fetched per poll; must be >= vlm_batch_size * concurrency.',
    )
    p.add_argument(
        '--vlm-batch-size',
        type=int,
        default=32,
        help='Crops per label_batch call.',
    )
    p.add_argument(
        '--concurrency',
        type=int,
        default=8,
        help='Concurrent label_batch calls in flight (single-project mode only).',
    )
    p.add_argument(
        '--poll-interval',
        type=float,
        default=5.0,
        help='Seconds to wait when the queue is empty before polling again.',
    )
    p.add_argument(
        '--idle-stop-after',
        type=int,
        default=6,
        help='Consecutive empty polls before the worker exits.',
    )
    p.add_argument(
        '--until-empty',
        action='store_true',
        help='One drain pass: exit on the first empty poll regardless of idle-stop-after.',
    )
    p.add_argument(
        '--continuous',
        action='store_true',
        help='Run forever -- never exit on idle (single-project mode only).',
    )
    p.add_argument(
        '--classifier-conf-skip',
        '--classifier-conf-skip',
        dest='classifier_conf_skip',
        type=float,
        default=DEFAULT_CLASSIFIER_CONF_SKIP,
        help='Skip the VLM for classifier-labeled crops at or above this confidence.',
    )
    # GPU arbiter sentinel (single-project mode): pauses while the
    # trainer holds a single-GPU run. See src/services/training/gpu_arbiter.py.
    p.add_argument(
        '--pause-sentinel',
        default=os.environ.get(
            'OP_PAUSE_SENTINEL',
            str(
                Path(os.environ.get('OP_STATE_DIR', '/var/lib/openprocessor'))
                / 'vlm_worker'
                / 'pause.sentinel'
            ),
        ),
        help='File whose presence pauses the worker (GPU arbiter writes it).',
    )
    p.add_argument(
        '--pause-poll-interval',
        type=float,
        default=10.0,
        help='Seconds to sleep between sentinel checks while paused.',
    )
    add_project_argument(p)
    # Default: every active project; --project restricts to one.
    p.set_defaults(project=None)
    return p.parse_args(argv)
