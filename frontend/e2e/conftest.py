"""Shared fixtures for the stubbed-backend Playwright/pytest e2e suite.

See docs/design/test-audit-2026-09-24.md recommendation 5 (P1-1) and
CLAUDE.md's "Development" section.

The harness lives in ``harness/`` and is loaded as plugins below:
``server`` (browser launch args, the built app under ``vite preview``),
``stub`` (the fail-closed API stub and its defaults), ``assets`` (the app's
own files served through Playwright), ``page_log`` (timeouts and failure
dumps), ``waits`` and ``constants`` (imported by tests).

Fail-closed contract (the whole point of this harness — see the R2 finding
in the audit doc about the old scripts/playwright_*.py stubs failing
open):

  * Every request under the configured API prefix (default ``/curation``)
    is routed through ``Stub``. A path/method a test did not register via
    ``stub.on(...)`` gets a ``501`` and is recorded in ``stub.unhandled`` —
    never a silent ``200 {}``.
  * At teardown, the ``stub`` fixture asserts ``unhandled == []`` and
    that at least one request was actually handled — a prefix/route
    mismatch fails the test instead of passing on an empty page.

The app under test is a real production build: ``npm run build`` once
per session, then ``vite preview`` serves the static output. The stubs
intercept in the browser, so a static build is enough — no dev server,
no SSR.
"""

pytest_plugins = [
    "harness.server",
    "harness.stub",
    "harness.page_log",
]
