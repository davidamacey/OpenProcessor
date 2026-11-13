"""Waits on the stub's answers and the browser's paint pipeline."""

from __future__ import annotations

from typing import Any


class _Handled:
    """Context manager whose `.value` is the matched *request*, resolved only
    once the stub has answered it."""

    def __init__(self, manager):
        self._manager = manager
        self._info = None

    def __enter__(self):
        self._info = self._manager.__enter__()
        return self

    def __exit__(self, *exc):
        return self._manager.__exit__(*exc)

    @property
    def value(self):
        return self._info.value.request


def expect_handled(page, predicate, timeout=None):
    """Wait for the stub to have answered a matching request, not just for the
    browser to send it: the handlers record bodies, and on a slow runner the
    recording can trail the send."""
    return _Handled(page.expect_response(lambda resp: predicate(resp.request), timeout=timeout))


def wait_for_url_state(page, has=(), lacks=(), timeout=None) -> None:
    """Wait until the URL (colons decoded: the app writes `%3A`, a test reads `:`) holds every
    part of `has` and none of `lacks`. The URL is written by an effect after the click, so a bare
    `assert ... in page.url` can read the old one."""
    from harness.constants import ACTION_TIMEOUT_MS

    def settled(url: str) -> bool:
        url = url.replace('%3A', ':')
        return all(p in url for p in has) and not any(p in url for p in lacks)

    page.wait_for_url(settled, timeout=timeout or ACTION_TIMEOUT_MS)


def settled_box(page, locator, frames=3):
    """The locator's bounding box once it has not moved for `frames` painted frames (a zoom or a
    pan animates); the paint pipeline is the clock, not a sleep."""
    last = None
    still = 0
    while still < frames:
        wait_for_paint(page)
        box = locator.bounding_box()
        still = still + 1 if box == last else 0
        last = box
    return last


def wait_for_paint(page: Any) -> None:
    """Wait for two real animation frames instead of an arbitrary sleep.

    A handful of interactions (mid-drag Escape, rapid Escape presses) have
    no app-exposed DOM/network signal to wait on — the assertion that
    follows is about the ABSENCE of a console error, not a state change a
    selector can observe. Sleeping a fixed duration there is exactly the
    flake-under-load pattern this whole rewrite removes: this instead
    waits on the browser's own paint pipeline (two rAF callbacks
    guarantees at least one full frame was rendered), which naturally
    slows down under host contention the same way real waiting would,
    without picking an arbitrary millisecond budget.
    """
    page.evaluate(
        '() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))'
    )
