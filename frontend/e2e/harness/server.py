"""Session fixtures: the headless browser launch args and the app under test
(a production build served by `vite preview`)."""

from __future__ import annotations

import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

import pytest

from harness.constants import ROOT


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args: dict[str, Any]) -> dict[str, Any]:
    """Force software rendering off entirely.

    On this machine, headless chromium's default GPU-process init
    (`--use-gl=disabled`'s ANGLE/EGL path) crash-loops the GPU process
    repeatedly, pinning the renderer at ~100% CPU for a couple of minutes
    before `Browser.new_page()` itself raises `TargetClosedError` —
    `--disable-gpu` on top of pytest-playwright's own `--headless=new`
    sidesteps the crash loop entirely (verified: `new_page()` returns in
    under a second with it, hangs for minutes without it).
    """
    return {**browser_type_launch_args, "args": ["--disable-gpu"]}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stop_group(proc: subprocess.Popen[str]) -> None:
    """SIGTERM the server's whole process group, SIGKILL it after a grace."""
    import os
    import signal

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=15)


def _wait_for_server(base_url: str, proc: subprocess.Popen[str], timeout: float = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"vite preview exited early (code {proc.returncode}):\n{out}")
        try:
            urllib.request.urlopen(base_url, timeout=1)
            break
        except urllib.error.URLError:
            time.sleep(0.5)
    else:
        _stop_group(proc)
        raise RuntimeError(f"vite preview did not come up within {timeout}s at {base_url}")

    # Warm-up request: the port accepting connections doesn't guarantee the
    # static handler's own first-request bookkeeping (route table build,
    # first disk reads into the OS page cache) is done — under host
    # contention that first real page load can be meaningfully slower than
    # every one after it. A single throwaway fetch here, outside any test's
    # own timeout budget, absorbs that cost once instead of it landing on
    # whichever test happens to run first.
    try:
        urllib.request.urlopen(base_url, timeout=max(timeout, 10))
    except urllib.error.URLError:
        pass  # the real per-test navigation will surface a genuine failure


@pytest.fixture(scope="session")
def app_url() -> Any:
    """Build the app once, serve it via `vite preview`, tear down at session end.

    Machine is heavily loaded (see CLAUDE.md) — build gets a generous timeout
    and preview startup is polled, never a fixed sleep.

    Set `E2E_APP_URL` to point at an already-running `vite preview` (or any
    static server of a build you built yourself) and skip both the build
    and the server lifecycle — a fast inner loop for local iteration; not
    used in CI.
    """
    import os

    preset = os.environ.get("E2E_APP_URL")
    if preset:
        yield preset.rstrip("/")
        return

    build = subprocess.run(
        ["npm", "run", "build"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if build.returncode != 0:
        raise RuntimeError(f"npm run build failed:\n{build.stdout}\n{build.stderr}")

    port = _free_port()
    proc = subprocess.Popen(
        ["npx", "vite", "preview", "--port", str(port), "--strictPort"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # Own process group so the whole tree (npx wrapper + real server)
        # can be stopped; terminating the wrapper alone orphans the server.
        start_new_session=True,
    )
    base_url = f"http://localhost:{port}"
    try:
        _wait_for_server(base_url, proc)
        yield base_url
    finally:
        _stop_group(proc)
