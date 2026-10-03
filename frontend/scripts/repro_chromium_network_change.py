"""Reproduce the e2e boot flake: Chromium aborts queued localhost requests
with net::ERR_NETWORK_CHANGED when a docker container starts on the host.

Runs 40 slow fetches against a local server (Chromium opens 6 connections
per origin, so 34 wait in its socket pool), starts a short-lived container
meanwhile, and counts the fetches that failed. `plain` lets Chromium load
the requests (expect 34/40 to fail every iteration); `routed` answers them
through `page.route` + `route.fetch()`, the way e2e/conftest.py's
`serve_app_through_harness` serves the app (expect 0).

The container event is host-wide: it can fail requests in any other
browser running on this machine, so only run it when no other e2e or
Playwright run is active.

usage: e2e/.venv/bin/python scripts/repro_chromium_network_change.py {plain|routed} [N]
needs: docker and the alpine:latest image
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from playwright.sync_api import sync_playwright

QUEUED_FETCHES = 40
SLOW_RESPONSE_S = 8
# The host-side veth only gains (and then loses) its IPv6 link-local
# address after a second or two; a container that exits at once may not
# trigger the change.
CONTAINER_LIFETIME_S = "4"


class _Slow(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server's naming
        if self.path.startswith("/slow"):
            time.sleep(SLOW_RESPONSE_S)
        body = b"<html><body>ok</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("plain", "routed"):
        print(__doc__)
        return 2
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Slow)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://localhost:{server.server_address[1]}"

    failed_iterations = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--disable-gpu"])
        for i in range(iterations):
            page = browser.new_page()
            if mode == "routed":
                page.route(f"{base}/**", lambda route: route.fulfill(response=route.fetch()))
            page.goto(base)
            page.evaluate(
                f"""() => {{ window.__r = Promise.all([...Array({QUEUED_FETCHES}).keys()].map(i =>
                    fetch('/slow' + i + '?' + Math.random())
                      .then(r => r.ok ? 'ok' : 'bad', e => String(e)))); }}"""
            )
            time.sleep(0.5)
            subprocess.run(
                ["docker", "run", "--rm", "alpine:latest", "sleep", CONTAINER_LIFETIME_S],
                check=True,
            )
            bad = [r for r in page.evaluate("() => window.__r") if r != "ok"]
            failed_iterations += bool(bad)
            print(f"{mode} iteration {i}: {len(bad)}/{QUEUED_FETCHES} fetches failed", flush=True)
            page.close()
        browser.close()
    server.shutdown()
    print(f"{mode}: {failed_iterations}/{iterations} iterations had aborted requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
