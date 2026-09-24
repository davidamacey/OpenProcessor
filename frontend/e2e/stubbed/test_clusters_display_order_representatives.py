"""DQ-M4 (docs/design/data-quality-pass-2026-09-24.md): at the default
sort ("purity asc"), `/clusters` representatives used to be requested via
a server-order (`_count desc`) `offset`/`limit` window that never lined up
with the client-side sorted display order — 4 of the first 8 cards
rendered blank until a scroll forced the window far enough to cover them.

Fixed by fetching each display-window card's representatives individually
via `GET /clusters?cluster_id=<id>` (src/routes/clusters/+page.svelte,
src/lib/clusters/displayOrderRepresentatives.ts) instead of a single
windowed call in server order.

This test sets up 30 candidate clusters returned in size-desc (id
ascending) server order, each with NO representatives — the server-order
window a pre-fix client would have requested therefore never fills
anything. Purity is assigned in reverse (`purity = (31 - id) / 100`), so
the default purity-asc sort's first screenful (page size 24) is exactly
the *last* 24 clusters by id (30 down to 7) — 6 of which (ids 25-30) never
appear in ANY server-order window a naive client would ask for early on.
Asserting per-cluster_id requests were made for those ids proves the fetch
follows display order, not server order.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

CLASSES: list[dict] = []
METHODS = {"strategies": [], "flags": {}}

N_CLUSTERS = 30


def cluster_item(cluster_id: int) -> dict:
    return {
        "cluster_id": cluster_id,
        "cluster_kind": "candidate",
        "size": 100 - cluster_id,  # size-desc server order == id ascending
        "validated_count": 0,
        "labelled_count": 0,
        "dominant_class_id": None,
        "dominant_class_name": None,
        "dominant_count": 0,
        "purity": (N_CLUSTERS + 1 - cluster_id) / 100,  # reversed vs. id
        "purity_tier": "noisy",
        "promotable": False,
        "is_unlabeled": True,
        "n_subclusters": 0,
        "updated_at": None,
        "representatives": [],  # nothing filled by the main list call
    }


CLUSTERS_LIST = {
    "items": [cluster_item(i) for i in range(1, N_CLUSTERS + 1)],
    "total": N_CLUSTERS,
    "total_class_clusters": 0,
    "total_candidate_clusters": N_CLUSTERS,
    "cluster_id_offset": 10000,
    "purity_thresholds": {
        "pure_min": 0.8,
        "mixed_min": 0.5,
        "promote_min_members": 5,
        "promote_min_labelled_share": 0.3,
    },
}


def test_clusters_fetch_representatives_in_display_order(stub, page, app_url):
    per_id_calls: list[int] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on("GET", r"/regions(\?|$)", {"items": [], "total": 0})

    def clusters_handler(request, _match):
        qparams = parse_qs(urlparse(request.url).query)
        cid = qparams.get("cluster_id", [None])[0]
        if cid is not None:
            per_id_calls.append(int(cid))
            item = cluster_item(int(cid))
            item["representatives"] = [{"crop_id": f"crop-{cid}"}]
            return (200, {**CLUSTERS_LIST, "items": [item], "total": 1})
        return (200, CLUSTERS_LIST)

    stub.on("GET", r"/clusters(\?|$)", clusters_handler)

    page.goto(f"{app_url}/clusters")
    page.wait_for_timeout(2000)

    # The 6 clusters (ids 25-30) that are in the purity-asc first window
    # but NOT in the size-desc server-order first window must have been
    # fetched individually.
    display_only_ids = {25, 26, 27, 28, 29, 30}
    fetched = set(per_id_calls)
    missing = display_only_ids - fetched
    assert not missing, (
        f"display-order-only clusters {missing} were never fetched individually "
        f"— representatives are still following server order, not display order. "
        f"Fetched: {sorted(fetched)}"
    )

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"
