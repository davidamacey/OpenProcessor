"""Constants shared by the stubbed e2e harness and its tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

API_PREFIX = "/curation"

# Single, documented action-timeout budget for the whole stubbed suite.
#
# Every explicit `.wait_for(timeout=...)`/`.click(timeout=...)` call used to
# hardcode a bare `15000` literal, independently, in ~40 files. On a heavily
# loaded host (concurrent docker builds, GPU engine compiles pegging most
# cores — see CLAUDE.md) that fixed 15s budget was occasionally too tight
# for hydration + the stub's request round trip to finish, not a real app
# defect: a controlled repro (pinning `vite preview` and pytest to a single
# CPU core contended by an equal-priority stress load) reproduced both a
# ~10x slowdown (13 tests: ~26s -> ~261s) and, under heavier contention, the
# preview server failing to even finish starting inside its own startup
# window. 45s gives real headroom for that kind of contention while still
# failing fast (well under pytest's own per-test wall clock) on an actual
# regression. Import this into a test file instead of writing a new literal.
ACTION_TIMEOUT_MS = 45000

# 1x1 transparent GIF — grids only need the <img> to resolve, not real pixels.
TRANSPARENT_GIF = bytes.fromhex(
    "47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b"
)
