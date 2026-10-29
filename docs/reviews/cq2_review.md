# CQ2: in-cluster filters and facets

Branch `feat/v070-cluster-outputs`, on top of CQ1 (`docs/reviews/cq1_review.md`), merge base `main` at
`0e2d3fe7` (unchanged). Head: the commit that adds this file. Plan: section 6, CQ2.

## What changed

- `8f0a35b4` feat(curation): detector filters in the item filter and a cluster facets route.
- This commit: the review package and the probe's `--endpoint` option.

In the first commit:

- `ItemFilter` (and so every route that takes the shared filter: `GET /crops`, `GET /review/{tab}`,
  `GET /clusters`, stats, selections and the reprocess filter) gains `detector_class_names`,
  `detector_conf_min`, `detector_conf_max` and `review_tab`. Query names: `detector_class_name`
  (repeatable), `detector_conf_min`, `detector_conf_max`, `review_tab`.
  - `detector_class_name` matches the stored `detector_class_name` by name with the same
    normalization as `class_name`.
  - `detector_conf_max` is an exclusive bound (see the decision below).
  - `review_tab` applies a `SIMPLE_TABS` tab's own clauses (the ones `GET /review/{tab}` runs, holdout
    hidden). `regions` and unknown ids are a 400.
- `GET /crops` `sort` accepts `detector_confidence` (missing values last in either direction).
- New route `GET {P}/clusters/{cluster_id}/facets`, typed `ClusterFacets`: `detector_classes` (top
  20), `class_sources`, `validation` (validated, unvalidated), `review_tabs` (the five tabs of
  `CLUSTER_REVIEW_TABS`), `detector_confidence` (ten bins of 0.1, each with the `min` and `max` to send;
  the last has `max: null`) and `detector_confidence_missing`. Holdout and excluded items are not
  members. An unknown or empty cluster answers zeros.
- `make contracts` run; docs (`docs/design/curation_api_contract.md`) and `CHANGELOG.md` updated.
- Frontend: only the key maps that the contract tests pin to the `ItemFilter` and `ReprocessFilter`
  schemas (`frontend/src/lib/types_itemFilter.ts`, `types_import.ts` and the two contract tests).
  Without them `npx vitest run src/lib/contract` fails on the added schema properties. No route or
  component was touched.

## Acceptance

| Bar | Measured | Met | Raw data |
|---|---|---|---|
| Facets p95 at most 150 ms for a 2,000-member cluster at 100k items | cluster 10001 (1,871 members): p95 16.0, 19.5, 15.5 ms (p50 about 12); cluster 10000 (3,044 members): p95 18.7, 17.3, 22.0 ms | yes | `artifacts_local/bench/v070/CQ2/facets_*.json` |
| Counts equal what the filtered `GET /crops` returns, per facet value | asserted for every detector class, class source, validation state, review tab and confidence bin | yes | tests below |

Measured on the same throw-away stack as CQ1 (the project `cq1b100k`: 100,000 synthetic items, 592
clusters; one API container with this branch's `src`, no GPU), 30 calls per repetition, 3 repetitions.
The 1-minute load average was 46 to 48 for these runs (above the protocol's 16, so all are marked
`noisy`); the bar is passed by a factor of seven, so noise does not change the verdict.

## How it was tested

Red first (seen failing before the code: 9 of 13 in `test_item_filter_detector.py` and a collection
error for the facets module):

- `tests/curation/test_item_filter_detector.py`: clause shapes (name spellings, inclusive minimum and
  exclusive maximum, an empty band refused, `review_tab` equal to the review queue's own clause,
  `regions` and unknown ids refused); end to end through `GET /crops` (class filter, band, `review_tab`
  returning the same ids as `GET /review/outliers`, combinations, the 400s); the `detector_confidence`
  sort clause the route sends.
- `tests/curation/test_cluster_facets.py`: shape validated strictly against `ClusterFacets`; holdout,
  excluded and other clusters' values never appear; per facet value the count equals the `total` of the
  filtered `GET /crops` (detector class, class source, validated and unvalidated, each review tab, each
  of the ten confidence bins, including a value exactly on a bin edge); the bins plus the missing count
  equal the total; the class list is capped at 20; an unknown cluster answers zeros.
- `tests/curation/cross_project_leak/test_cluster_list_isolation.py`: facets leak row (own project's
  markers and indexes only). The every-route sweep also covers the new route.
- The query-capable fake (`tests/curation/query_fakes.py`) gained `range` aggregations; its docstring
  lists them.

Revert checks (each change made alone, the tests watched failing, then restored):

| Mutation | Failing tests |
|---|---|
| upper confidence bound made inclusive | band clause test, `test_crops_filter_by_detector_confidence_band`, histogram bins test |
| facets include the holdout | shape, "never appear" and detector class count tests |
| facets report only two review tabs | shape and review tab count tests |
| `unvalidated` filter counts validated | validation counts test |
| class list cap 20 raised to 30 | cap test |

Suites: the new and touched cluster tests above, the leak package, `npx vitest run src/lib` (4,538 tests,
all green after the key-map update), pre-commit (all hooks) on the code commit. The whole offline suite
was run once for CQ1 (6,167 passed) and not repeated.

## Risks and what was not done

- Every route that takes the shared filter now accepts the four new parameters, so an unrelated route
  (stats, selection writes, reprocess) can be narrowed by `review_tab`. That is the intent of the shared
  filter, but it widens what a bulk write can target; a selection by `review_tab` writes the same items
  the tab would list, and the lock rule is unchanged (automated writers still skip locked items).
- `review_tab` always hides the test holdout through the tab's own guard, even on a route that is
  asked to include it.
- `GET /clusters` now has both `detector_class` (keeps cards whose detector top class matches) and
  `detector_class_name` (narrows the members every card counts). Both are documented; a UI should use
  the first for the card list and the second for the in-cluster grid.
- Only `svelte-check` was not run on the frontend (the change adds optional fields to two interfaces and
  keys to two test maps); `npx vitest run src/lib` covers the contract.

## Decisions needed from the owner (gate ids)

- D-CQ2-1: `detector_conf_max` is exclusive (and `detector_conf_min` inclusive) so the ten histogram
  bins never overlap and each bin count equals its filtered list. The plan did not say; the shared
  `conf_max` stays inclusive. Say if the two should match (then the last bin cannot be exact).

## Five-minute check for the owner

On a stack running this branch's `src`:

```bash
P=http://127.0.0.1:5603/curation/projects/<project>
curl -s "$P/clusters/<cluster_id>/facets" | python3 -m json.tool | head -60
curl -s "$P/crops?cluster_id=<cluster_id>&detector_class_name=<a listed value>&page_size=1" | python3 -c \
 "import sys,json; print(json.load(sys.stdin)['total'])"
```

Expected: the second number equals that class's `count` in the facets; the same holds for
`class_source`, `label_validated`, `review_tab` and a bin's `detector_conf_min` / `detector_conf_max`.
Latency: `.venv/bin/python scripts/bench/cluster_list_probe.py probe --api <url> --project <slug> --endpoint clusters/<id>/facets --reps 3 --calls 30`.
