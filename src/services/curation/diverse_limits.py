"""Upper bounds on ``k`` for diversity selection — the one place both
routes and the served ``/methods`` entry read them from."""

from __future__ import annotations


# ``GET /crops?order=diverse&k=`` (rank just the first k picks of a browse).
DIVERSE_BROWSE_MAX_K = 10_000
# ``POST /select/diverse`` body ``k`` (a materialized selection).
DIVERSE_SELECT_MAX_K = 50_000
