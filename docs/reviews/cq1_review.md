# CQ1: typed cluster list with server sort, filter and quality fields

Branch `feat/v070-cluster-outputs`, from `main` at `0e2d3fe7` (merge base; `origin/main` had not moved
when this was written). Head: the commit that adds this file. Plan: `docs/design/ship_track_plan.md`
section 6 (on branch `docs/ship-track-plan`). Issues: none closed.

## What changed

- `3559ff88` feat(curation): typed cluster list with server-side sort, filter and quality fields.
- `1071a8d8` perf(curation): count review tabs in one filters aggregation and add the cluster list probe.
- This commit: the review package.

Summary of the first commit:

- `GET {P}/clusters` answers a documented `ClusterListResponse` (declared through `responses=`, like the
  other wire-dict routes; the shapes are TypedDicts in `src/schemas/curation/clusters.py`). Validating
  about 600 pydantic cards per request cost more than the aggregation, so cards stay plain dicts.
- New query parameters: `sort`, `dominant_class`, `detector_class`, `min_size`, `max_size`,
  `validation_state`, `flag`, `needs_review`. Representatives are paged over the served order.
- New card fields: `detector_top_class`, `detector_top_share`, `detector_conf_mean`, `validation_state`,
  `review_counts`, `needs_review_count`, `flags`. New response fields: `sort`, `sort_options`,
  `flag_vocabulary`, `truncated`.
- Removed: `label_purity` (same value as `label_agreement`).
- Behaviour change: frozen test-holdout items are no longer counted in any card number (before, only
  excluded items were left out). This follows the plan's "holdout and excluded members never counted".
- Card building moved out of the router into `src/services/curation/cluster_cards.py`; the flag rules
  are in the pure module `src/domain/cluster_flags.py`.
- `make contracts` regenerated `contracts/openapi/curation.json` in the same commit; `GET /clusters`
  left the `LEGACY_DICT` allowlist of `tests/test_contract_typed_responses.py`.
- Docs: `docs/design/curation_api_contract.md` (Clusters section) and `CHANGELOG.md`.

## Acceptance

| Bar | Measured | Met | Raw data |
|---|---|---|---|
| Contracts regenerated | `make contracts` equivalent run; pre-commit contract hooks pass | yes | `contracts/openapi/curation.json` |
| `GET /clusters` p95 at 100k items (592 clusters: 80 class and 512 candidate, `limit=50`, `per_cluster=4`) at most 400 ms | median over 4 repetitions of p95: 133.9 ms (125.0 to 147.9) | yes | `artifacts_local/bench/v070/CQ1/final100k/` |
| Within 1.5x of the same call on `main` in the same session | `main` 97.6 ms (91.5 to 107.6), so 1.37x; p50 111.1 against 87.1 ms, 1.28x | yes, but see the caveat | same |
| 1M synthetic recorded | new: p50 148.1 ms, p95 164.4, 156.9, 165.4 (10 calls per repetition); `main`: p50 114.4, p95 458.2 (one slow call), 121.4, 123.2 | recorded | `.../final1m/` |

Caveats, stated plainly:

- Every repetition ran at a 1-minute load average between 17 and 22, above the protocol's 16. They are
  all marked `noisy` in the raw JSON and the host never calmed below 17 during the session (another
  session's work). The ratio is from interleaved runs (main, new, main, new) so both saw the same load,
  but the absolute figures should be read as upper bounds.
- An earlier set of runs before the last optimisation (`.../100k/`, load 35 to 40, a differently
  seeded copy of the same distribution) gave 96.9 ms on `main` against 154.5 ms (1.59x). The
  optimisation (one `filters` aggregation for the five review tabs, dropping a separate `exists`
  filter, plain dicts instead of pydantic cards) is what brought the ratio under 1.5x. The repeat at
  load 24 to 25 showed `main` 68.1 ms against 105.2 ms (1.54x) before the final step, so the margin is
  not large; re-run it on a quiet host (below load 16) before relying on 1.37x.
- The plan names an "SP6 probe" on the "SP stack"; neither exists yet. The probe is
  `scripts/bench/cluster_list_probe.py` (new). The "stack" is two throw-away API containers (the
  `0.6.0` image with `main`'s `src` and with this branch's `src` mounted read-only, one worker each,
  no GPU) on the `optest1` network against its OpenSearch. The items are synthetic documents with
  COCO class names and no images or vectors (the list reads only item fields); the 1M project was
  seeded by an earlier revision of the probe that differed only in the `class_source` values.
- Only the aggregation was measured, not the browser. The response is about 520 KB at 592 clusters
  (345 KB on `main`); the UI can request fewer with `max_clusters` and `kind`.

## How it was tested

Red first (all seen failing before the code existed, as import errors on the missing modules at
`0e2d3fe7` plus the work tree):

- `tests/curation/test_cluster_flags.py` (every threshold boundary, vocabulary order and uniqueness).
- `tests/curation/test_cluster_cards.py`: each sort on a fixed fixture and stable on ties; each filter
  removes exactly the expected cards; `truncated` at `max_clusters` and independent of the filters;
  response validates strictly against `ClusterListResponse`; review counts equal what the review
  clauses match for the same cluster (the `/review/{tab}` query built by `build_review_request` plus the
  cluster id, counted on the query-capable fake, and the route's own totals across clusters);
  holdout and excluded members never counted; route-level sort, filter and 422 on bad values.
- `tests/curation/cross_project_leak/test_cluster_list_isolation.py`: the cross-project row (cards and
  representatives from the bound project only, only its own indexes reached).
- Existing cluster-card tests moved to `tests/curation/_cluster_cards_helpers.py` and `label_purity` to
  `label_agreement`; the duplicate-name test was deleted with the field.

Revert checks (each change made on its own, the suite watched failing, then restored):

| Mutation | Failing test |
|---|---|
| holdout guard removed from the aggregation query | `test_the_aggregation_counts_each_review_tab_and_hides_the_holdout` |
| `truncated` off by one | `test_truncated_when_the_aggregation_hits_max_clusters` |
| sort direction flipped | `test_each_sort_orders_the_fixture[needs_review_desc]` |
| cluster_id tie-break removed | `test_each_sort_orders_the_fixture[needs_review_desc]` |
| `mixed_detector_classes` threshold 0.8 to 0.9 | `test_mixed_detector_classes_boundaries[0.8-10-False]` |
| `min_size` made exclusive | `test_each_filter_removes_exactly_the_expected_cards[over3]` |
| review counts built with `include_test=True` | survived: the outer query already hides the holdout, so the review clause's own guard is redundant (equivalent mutant) |

Suites: `tests/curation` plus `tests/test_contract_typed_responses.py` offline, 6,167 passed and 2
failed on the first full run. Both failures came from the new probe script (`test_class_sources.py`
and `test_script_api_prefix.py` scan `scripts/`); fixed and both files re-run green. The frontend contract
tests (`npx vitest run src/lib/contract src/lib/api.clusters.test.ts`, 29 files, 999 tests) pass against
the regenerated contract. Pre-commit (all hooks) passes on both code commits.

## Risks and what was not done

- The aggregation now carries three more sub-aggregations per cluster bucket (detector top class, mean detector confidence and one `filters` aggregation with five filters). On a 1M-item index it
  measured about 35 ms slower than `main`, so it scales with clusters, not items; a deployment with
  many thousands of clusters should lower `max_clusters` (the response says when it is cut).
- Sort and the card-level filters run on at most `max_clusters` aggregated buckets; `truncated` says
  when that was not everything. A filter that matches only small clusters can be hidden by the cut.
- `GET /clusters/representatives` is kept (the plan deletes only the frontend's use of it, in UI1).
- Cropwright still reads `label_purity` with a fallback (`frontend/src/lib/api_clusters.ts`); UI1
  must switch its readers to `label_agreement` in the same merge, as the plan says. Nothing in
  `frontend/` was touched.
- Not done: the CQ2 and CQ3 waves (see their own review files, if present).

## Decisions needed from the owner (gate ids)

None for CQ1. Open for the UI lane: whether to ask for `max_clusters` below the default 1000 once the
list is virtualised.

## Five-minute check for the owner

On a stack running this branch's `src`:

```bash
P=http://127.0.0.1:5603/curation/projects/<project>
curl -s "$P/clusters?per_cluster=0&sort=needs_review_desc&limit=1" | python3 -c \
 "import sys,json; d=json.load(sys.stdin); c=d['items'][0]; print(d['sort'], d['truncated'], d['total']); \
  print({k:c[k] for k in ('cluster_id','needs_review_count','review_counts','flags','detector_top_class','detector_top_share','validation_state')})"
curl -s "$P/clusters?per_cluster=0&flag=mixed_detector_classes&validation_state=none" | python3 -c \
 "import sys,json; print([c['cluster_id'] for c in json.load(sys.stdin)['items']])"
```

Expected: the first card has the largest `needs_review_count`, and `review_counts` are the per-tab counts of the
review queue for that cluster (the tests pin them to the queries `/review/{tab}` runs). The second call lists
only clusters carrying that flag with no validated member. Latency: `.venv/bin/python scripts/bench/cluster_list_probe.py probe --api <url> --project <slug> --reps 3 --calls 30`.
