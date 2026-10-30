# UI1 and UI2: cluster list sort and filter, in-cluster facets

Branch `feat/v070-cluster-ui`, merge base `0e2d3fe7`; `feat/v070-cluster-outputs` (CQ1, CQ2) merged first
(`9bcecdbd`, no conflicts). Head: the commit that adds this file. Plan: `docs/design/ship_track_plan.md`
sections 8.2 and 8.3 (on branch `docs/ship-track-plan`). Issues: none closed.

## What changed

- `9bcecdbd` Merge branch 'feat/v070-cluster-outputs' into feat/v070-cluster-ui
- `94d87030` feat(frontend): render the cluster list as the server sorts and filters it (UI1; also carries the API wrappers, types and contract pin for the facets route, and the cited-paths baseline tightening)
- `f90e6fe4` feat(frontend): add facet chips and a detector-confidence sort to the cluster page (UI2)
- This commit: the review package.

Summary:

- `GET {P}/clusters` is rendered as served. The page sends `sort` and the card filters (dominant class, detector
  class, size range, validation state, flags, needs review) as query parameters and shows the served order. The
  toolbar renders the served `sort_options` and `flag_vocabulary`. The choice is in the URL and in session
  storage (URL wins), and "Clear filters" resets it.
- Cards show the detector top class with share and mean confidence, `label_agreement`, the cohesion tier, the
  validation state, the review count (per-tab counts as tooltip) and the flags (served label, served description
  as tooltip). A `truncated` response shows a banner with the served count. Loading, error (with Retry) and
  filtered-empty states.
- Deleted: the client sort (`sortClusters`), the client "Unlabeled only" filter, `displayOrderRepresentatives.ts`
  and the per-card representative requests. Thumbnails come from the list call's representatives window
  (`offset`/`limit`): the first call carries the first 24 cards' thumbnails, each further window is one more call.
  Every reader of `label_purity` now reads `label_agreement`.
- UI2: the cluster page has facet chips from `GET {P}/clusters/{id}/facets` (detector class, validation, review
  tab, detector-confidence band, each with the served count; label-source options carry their count). A chip
  sends the matching `GET /crops` parameter (`detector_class_name`, `label_validated`, `review_tab`,
  `detector_conf_min` and `detector_conf_max` from the served bin); a sort by detector confidence replaces the
  core-first order. The panel is folded by default under 768 px.
- Found while doing it: the search badge lookup sent `limit=0` (the server answers 422); it sends `per_cluster=0`.

## Acceptance

| Bar | Measured | Met | Raw data |
|---|---|---|---|
| First screen of cards with thumbnails in at most 2 requests (512 clusters) | 1 list request, 24 cards, 96 of 96 thumbnails loaded (527 clusters served) | yes | `frontend/artifacts_local/UI1_UI2/report.json` |
| Each served sort renders the served order | DOM order equals the API order for the same query, for all 10 sorts | yes | same |
| Next window | 1 more request per window on scroll | yes | same |
| Facet chip count equals the filtered list total | person chip 108, grid total 108 (footer `108 / 108`); review tab Outliers 35, footer `35 / 35` | yes | same |
| Filter and sort survive reload and a trip into a cluster and back | flag button pressed after reload and after back | yes | same |
| axe clean | 0 violations on list, filtered list, filtered-empty and cluster page with facets (live); stubbed gate extended with two rows | yes | same |
| No console error, no 4xx/5xx on touched routes | 0 and 0 | yes | same |
| check, lint, knip, build, vitest | all green; vitest 476 files, 4954 tests | yes | |
| stubbed e2e | 314 passed (full suite, after the host load fell); axe gate 10 passed | yes | |

## How it was tested

Red first (seen failing, then green): `contract/clusterListContract.test.ts` (written with the types, mutation
checked), `clusterListFilter.test.ts` and `persistedFilter.test.ts` (module missing), `api.clusters.list.test.ts` (5 of 7
failed), `clusterListController.svelte.test.ts` (12 of 12 failed before the controller), `ClusterGridCard.quality.test.ts`
(3 of 7 failed), `clusterList.scan.test.ts` (5 of 11 failed before the deletions). The facet bar and filter bar component tests
and `clusterDetailController.facets.svelte.test.ts` were written beside the code; to show they bite, the sort and the
detector-class parameter were removed from the code and 4 of 18 tests failed, then the code was restored.

Live: a throwaway API container (`uiwave-api`, the 0.6.0 image with this branch's `src` and `scripts` mounted
read-only, no GPU) on the `optest1_triton_net` network against the shared OpenSearch, a new project `uiwave1`
seeded with 1,980 public COCO annotations (real images from the public COCO sample, 527 clusters), a build with
`PUBLIC_TRITON_API_URL` pointing at it served by `vite preview` on port 5821. Chromium headless, `--no-sandbox
--disable-dev-shm-usage --disable-gpu`. The project was deleted through the API (index list empty), the container
removed and the preview stopped. `default`, `cocotest`, `hiresp`, `nas20k` and `cq3p1` were not touched.

Screenshots (all opened and viewed; full height with the plan's recipe extended to the inner scroll area, capped at
2600 px at 1600 and 5200 px at 430), under `frontend/artifacts_local/UI1_UI2/`:
`d_clusters_default`, `m_clusters_default` (viewed downscaled), `d_sort_needs_review_desc`, `d_clusters_filtered`,
`m_clusters_filtered`, `d_clusters_filtered_empty`, `m_clusters_filtered_empty`, `d_clusters_error`,
`d_cluster_detail_facet_selected`, `m_cluster_detail_facets_folded`, `m_cluster_detail_facet_selected` (viewed
downscaled), `d_cluster_detail_review_tab`. Also captured, with the order checked against the API instead of by eye:
the other nine `d_sort_*` shots, `m_sort_*`, `d_cluster_detail_facets`, `m_cluster_detail_facets`.

## Risks and what was not done

- The truncated banner is covered by a stubbed e2e test only: the live list asks for up to 2000 clusters and the
  seeded project has 527.
- The cluster page's default order (core first) needs vectors; the CPU-only stack has none, so the live cluster page
  showed "No crops" until the sort by detector confidence was chosen. Facet chips and counts were checked in that
  sort. With vectors the default order works as before.
- Stubbed e2e: under host load (load average 20 to 120 from other sessions) real clicks intermittently hung at
  Playwright's "visible, enabled and stable" check, including on unchanged code (`test_cluster_bulk_undo.py` failed on
  the pre-change tree too). A first full run under that load gave 278 passed and 36 failed; the failures were such
  hangs except two real ones this wave caused (`test_clusters_error_state_shows_no_counts` looked for the old error
  copy, `test_ignored_mode_hides_cluster_grid_controls` for the old sort control), both fixed. After the load fell
  (about 10) the full suite gave 314 passed.
- The facet panel is a `<details>`; its open state is not persisted.
- The detector class picker lists the classes seen on served cards (kept across filters); the contract serves no
  detector class registry.
- `GET /clusters/representatives` stays on the backend; nothing in the frontend reads it.
- UI3 (bulk label with undo), UI4 and UI5 are not started.

## Decisions needed from the owner (gate ids)

None.

## Five-minute check for the owner

`cd frontend && npx vitest run src/lib/clusters src/lib/components/clusters src/lib/contract/clusterListContract.test.ts`
(all green), then `node scripts/run-e2e.mjs e2e/stubbed/test_cluster_list_filters.py e2e/stubbed/test_cluster_facets.py
e2e/stubbed/test_a11y_axe.py`. On a stack running this branch: open `/p/<project>/clusters`, pick a sort and a flag
(the URL gains them, a reload keeps them), open a cluster and click a detector class chip: the footer total equals the
chip's count.
