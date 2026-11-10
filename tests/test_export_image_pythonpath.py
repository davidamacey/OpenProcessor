"""The export image runs ``python /app/export/<script>.py``; the script directory, not /app, is
then ``sys.path[0]``, so ``from src...`` imports (ultralytics_export.py) need PYTHONPATH=/app."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_export_stage_puts_app_on_pythonpath() -> None:
    text = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
    stage = re.search(r'^FROM runtime-base AS export$(.*?)^FROM ', text, re.M | re.S)
    assert stage is not None
    assert re.search(r'^ENV PYTHONPATH=/app$', stage.group(1), re.M)
