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

Stability rules (an intermittent `expect_response` timeout is a lost
action or a race, never "CI is slow"; raising a timeout hides it):

  * Run the suite as a slow runner would: `E2E_CPU_THROTTLE=4` slows every
    page 4x (CDP `Emulation.setCPUThrottlingRate`), and `taskset -c 0` pins
    the browser and pytest to one core. Run it with `E2E_APP_URL` set to a
    preview you started, `-n 2`, and repeat; one pass proves little.
  * Wait for the stub to have ANSWERED (`expect_handled`), not only for the
    browser to have sent: handlers record bodies when they answer.
  * Never mark "before/after" points by an index into `stub.handled`: a
    request sent just before the mark is recorded after it. Listen to
    `page.on("request")` from the action on.
  * After a response, the app keeps working (re-seeding state from it). A
    key pressed right after `expect_handled` can meet that, so a failure of
    the form "second action hit the first action's target" is an app race
    to fix in `src/`, as the per-box selection was.

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
