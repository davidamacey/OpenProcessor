"""Full-page screenshots for stubbed tests.

`save_screens` writes desktop and phone widths under `artifacts_local/screens/<folder>/`; `shot` writes into
the directory an environment variable names and does nothing when it is unset.
"""

from __future__ import annotations

import os
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


def shot(
    page: Any,
    name: str,
    *,
    env_var: str = 'OP_E2E_SHOT_DIR',
    widths: tuple[int, ...] = (),
) -> None:
    """Full-page screenshot into ``$env_var``; one file per width (``name-<width>``) when ``widths`` is given."""
    out = os.environ.get(env_var)
    if not out:
        return
    Path(out).mkdir(parents=True, exist_ok=True)
    if not widths:
        page.screenshot(path=str(Path(out) / f'{name}.png'), full_page=True)
        return
    for width in widths:
        page.set_viewport_size({'width': width, 'height': 1000})
        page.screenshot(path=str(Path(out) / f'{name}-{width}.png'), full_page=True)
    page.set_viewport_size({'width': 1280, 'height': 720})
