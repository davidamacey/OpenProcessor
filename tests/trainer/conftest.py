"""The trainer modules are a flat directory on ``PYTHONPATH`` in their image; so they are here."""

from __future__ import annotations

import sys
from pathlib import Path


TRAINER_DIR = Path(__file__).resolve().parents[2] / 'docker' / 'trainer'
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))
