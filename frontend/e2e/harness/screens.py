"""Full-page screenshots at desktop and phone width, saved under `artifacts_local/screens/<folder>/`."""

from __future__ import annotations

from pathlib import Path
from typing import Any


SCREENS = Path(__file__).resolve().parents[3] / 'artifacts_local' / 'screens'
WIDTHS = (('desktop', 1280), ('430', 430))


def save_screens(page: Any, folder: str, name: str) -> None:
    out = SCREENS / folder
    out.mkdir(parents=True, exist_ok=True)
    for label, width in WIDTHS:
        page.set_viewport_size({'width': width, 'height': 1000})
        page.screenshot(path=str(out / f'{name}-{label}.png'), full_page=True)
    page.set_viewport_size({'width': 1280, 'height': 720})
