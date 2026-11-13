"""Projects page row actions live in each row's More menu."""

from __future__ import annotations

import time
from typing import Any


def _open_menu(page: Any, slug: str) -> None:
    trigger = page.locator(f'[data-testid="project-more-{slug}"] button[aria-haspopup]')
    if trigger.count() and trigger.get_attribute('aria-expanded') != 'true':
        page.keyboard.press('Escape')  # close any other row's open menu first
        trigger.click()


def row_action(page: Any, action: str, slug: str) -> Any:
    """The menu item `project-<action>-<slug>`, opening the row's More menu first."""
    _open_menu(page, slug)
    return page.get_by_test_id(f'project-{action}-{slug}')


def wait_row_action(page: Any, action: str, slug: str, timeout_ms: int = 45000) -> Any:
    """Wait for a menu item that appears after a state change, reopening the menu as needed."""
    deadline = time.monotonic() + timeout_ms / 1000
    item = page.get_by_test_id(f'project-{action}-{slug}')
    while time.monotonic() < deadline:
        _open_menu(page, slug)
        if item.count() and item.is_visible():
            return item
        page.wait_for_timeout(200)
    raise AssertionError(f'project-{action}-{slug} never appeared in its More menu')
