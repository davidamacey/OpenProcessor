# 12b Wave 2 lane specs: the Workspace shell and the backend for every view

| Field | Value |
|---|---|
| Status | spec; wave 2 starts after wave 1 (`wf-1a` to `wf-1d`) merges into `int/b4-stateless` |
| Interfaces | [12a_workflow_interfaces.md](12a_workflow_interfaces.md): every model, route, module and name used below is defined there; section numbers below (`12a §3`) point into it |
| Order | [12e_workflow_execution_order.md](12e_workflow_execution_order.md) |
| Lanes | 2A Workspace shell (frontend + R-22), 2B Shared filter and map data (backend + filter state), 2C View models and served boxes (backend + box layer), 2D Annotation writes (backend, **high risk**), 2E Cursor paging and crop sorts (backend + pager) |
| Base | `int/b4-stateless` at `1685723a9` (wf-1b, wf-1d, cov-live merged); wf-1a and wf-1c merge before wave 2 starts |

## Rules every lane follows

- Brief: `artifacts_local/briefs/workflow_lanes_1111.md` in the main checkout (binding rules and code standards),
  root `CLAUDE.md`, `frontend/CLAUDE.md`, `docs/design/engineering_standards.md`.
- Worktree: `git worktree add -b lane/wf-<id> ../OpenProcessor-wt-wf-<id> int/b4-stateless`. Link
  `.venv` and `frontend/e2e/.venv` to the main checkout's copies; `cd frontend && npm ci`.
- One concern per commit, conventional message, hooks on. Each commit body cites the official docs pages read
  and pastes the red line of its first failing test.
- Targeted tests only: the lane's pytest and vitest files, the stubbed e2e files it touches,
  `npm run check`, ruff and mypy on touched Python. No full suite, no `--all-files`.
- Live harness: `flock ~/host-coord/heavy.lock nice -n 19 docker compose -p op-wf-<id> -f docker/test/compose.yml up -d --wait`,
  then `.venv/bin/python -m pytest tests/live/<file> -q --no-cov -m live`, then tear down with `down -v --remove-orphans`.
  Write the owner line before and clear it after. Never use the default project name.
- Live Cropwright tier: write or adapt the tests; the coordinator runs them after the merge.
- Public data only. Never touch op-gpu, set P or the baseline projects.
- No waiting loops on `pgrep -f`.

---

## Lane 2A: Workspace shell, Clusters and Detections views, class picker

**Size:** L, 5 days. **Model:** Sonnet. **Merge:** last in wave 2, after 2B, 2E, 2C, 2D.

### Goal and outcome

`/p/<project>/workspace` replaces `/clusters` and `/clusters/[id]`. The user sees the filter bar, the view
bar (Clusters, Detections; later lanes add views), the Queues and Classes rail, the details panel and the bulk
bar. Picking a cluster card opens the cluster scope (today's cluster-detail grid) in place. A pure cluster takes
a class letter plus `Enter` to label. The selection and filter survive a view switch, and Back restores the
previous view and filter.

### Files

Create:
- `frontend/src/routes/p/[project]/workspace/+page.svelte`
- `frontend/src/lib/workspace/workspaceUrl.ts`, `workspaceUrl.test.ts`
- `frontend/src/lib/workspace/workspaceController.svelte.ts`, `workspaceController.test.ts`
- `frontend/src/lib/workspace/selection.ts`, `selection.test.ts`
- `frontend/src/lib/workspace/viewRegistry.ts`, `detailsRegistry.ts`
- `frontend/src/lib/workspace/views/ClustersView.svelte`, `DetectionsView.svelte`
- `frontend/src/lib/workspace/details/ClusterDetails.svelte`, `ItemDetails.svelte`, `SelectionDetails.svelte`
- `frontend/src/lib/components/workspace/{FilterBar,ViewBar,SortSelect,WorkspaceRail,DetailsPanel,BulkBar}.svelte` with tests
- `frontend/src/lib/components/state/ViewState.svelte`, `ViewState.test.ts`
- `frontend/src/lib/components/ClassPicker.svelte`, `frontend/src/lib/classes/classPicker.svelte.ts`, tests
- `frontend/src/lib/stores/nextSteps.svelte.ts`, `frontend/src/lib/api_workflow.ts`, `frontend/src/lib/types_workflow.ts`
- `frontend/e2e/stubbed/test_workspace_shell.py`
- `docs-site/docs/cropwright/user-guide/workspace.md`
- backend: `src/utils/class_names.py` (`suggested_class_name`), tests `tests/curation/test_class_create_suggestion.py`

Change:
- `frontend/src/lib/projectPaths.ts` (`PROJECT_SECTIONS :15-31`: add `workspace`, remove `clusters`; `SECTION_LABELS :35`; `legacyRedirectTarget :98`)
- `frontend/src/routes/p/[project]/+layout.svelte` (nav `:348-425`: Workspace replaces Clusters; `showSidebar :199` removed (the rail is part of the page); `selectClass :210-236` deleted; breadcrumbs `:240-255`)
- `frontend/src/lib/components/ClassSidebar.svelte` (its body becomes the Classes half of `WorkspaceRail`; delete the file once `WorkspaceRail` renders it)
- `frontend/src/lib/stores/keyboard.svelte.ts` (`:215-247` dispatch with includes, D-12)
- `frontend/src/lib/keymapFallback.ts` and `src/config/keymap_actions.json` (12a §7: `workspace` context and actions; `cluster` includes `workspace`; `cluster.select_all`, `cluster.confirm`, `cluster.undo` changes)
- `frontend/src/lib/api_select.ts` (`getSelectionSummary`), `frontend/src/lib/api_classes.ts` (`createClass` refusal), `frontend/src/lib/api_cropWrites.ts` (`getExcludeReasons`; delete `ExcludeReason`)
- `frontend/src/lib/clusters/clusterDetailController.svelte.ts` (`:25` reads the uid from `ws.url.scope` instead of `page.params.id`), `clusterDetailActions.svelte.ts` (delete `EXCLUDE_REASONS :111-117`; read served reasons)
- `frontend/src/lib/components/ClusterBadge.svelte:31`, `ClusterGridCard.svelte:45`, `clusters/similarAction.ts:15`, `clusterSearchModes.svelte.ts:304-308` (links to `/workspace?scope=cluster:<uid>` and `/workspace?...`)
- `src/routers/curation/classes_write.py:57-76`, `src/routers/curation/_class_models.py:62-70` (R-22)
- `frontend/src/lib/contract/endpointCatalog.test.ts` (`SCANNED_FILES`: add `api_workflow.ts`)
- `frontend/e2e/stubbed/test_a11y_axe.py` (`PAGES :123`), `test_narrow_viewport_430.py` (`ROUTES :52`), the repointed `test_cluster_*.py`, `test_item_filter.py`, `test_selection_actions.py`, `test_keymap.py`, `test_project_switch.py`
- `frontend/e2e/live/test_route_sweep.py` (`ROUTES :94-112`), `test_data_agreement.py`
- `frontend/CLAUDE.md` route table, `frontend/CHANGELOG.md`, `CHANGELOG.md`, `docs-site/sidebars.ts`

Delete: `frontend/src/routes/p/[project]/clusters/` (pages; the 10 tests move into `frontend/src/lib/workspace/`
or the matching controller test files, adapted, never dropped), `ClusterDetailToolbar.svelte`, `ClassSidebar.svelte`,
`docs-site/docs/cropwright/user-guide/clusters.md` (content folded into `workspace.md`).

### Steps and commits

1. `feat(curation): suggest a valid class name when a create is refused` (R-22). Red: `test_class_create_suggestion.py::test_traffic_light_is_refused_with_traffic_light_suggested`. `make contracts`.
2. `feat(frontend): dispatch keymap actions through context includes` (D-12). Red: `keyboard.svelte.test.ts::dispatches an action registered in an included context` and `::prefers the current context over an included one`.
3. `feat(keymap): workspace context and actions` (12a §7.2, backend JSON + fallback + `make contracts`). Red: `tests/curation/test_keymap_vocabulary.py::test_workspace_actions_are_served` and the vitest fallback-parity test.
4. `feat(frontend): workspace URL state and controller` (`workspaceUrl.ts`, controller, `selection.ts`, registries; no page yet). Red: the vitest cases below.
5. `feat(frontend): shared view state, sort select and class picker components`.
6. `feat(frontend): workspace page with the Clusters and Detections views` (moves the clusters pages in, deletes the route, repoints links). Tests move in the same commit.
7. `feat(frontend): workspace rail, details panel and bulk bar` (deletes `ClusterDetailToolbar`, `ClassSidebar`, `EXCLUDE_REASONS`).
8. `feat(frontend): workspace nav, 430 px drawers and the narrow filter button` (layout, `projectPaths`).
9. `docs(cropwright): workspace user guide; changelogs`.

### Interfaces

Consumes: 12a §3 (`ItemFilterState` with 2B's keys), §4 (`createPager` from 2E; **do not edit any `fetchPage`
closure or `getCrops` call**), §5, §6, §7, R-12 (2C), R-19 (2C), R-22, R-26 (2D), R-5, R-8. Provides: §6 controller
and registries, §10.2 shell components, `ClassPicker`, `ViewState`, `nextStepsStore`.

### Red tests first

| File | Test | Asserts |
|---|---|---|
| `frontend/src/lib/workspace/workspaceUrl.test.ts` | `round-trips every parameter` | `writeWorkspaceUrl(parseWorkspaceUrl(x))` equals x for a URL with every 12a §6.1 key |
| same | `ignores an unknown view and keeps the rest` | `view=nope` parses to `null` |
| `workspaceController.test.ts` | `switching views keeps filter, queue, scope and selection` | after `setView('detections')`, `itemQuery()` is unchanged and `selection` is the same object |
| same | `back restores the previous view and filter` | simulate two `page.url` values; `view` and `itemQuery()` follow the URL |
| same | `uses the served default view when the URL has none` | `nextStepsStore.defaultView='clusters'`; `view === 'clusters'`; one `goto` with `replace: true` |
| same | `the queue reaches every view as review_tab` | `setQueue('to_label')`; `itemQuery().review_tab === 'to_label'` |
| same | `cluster scope adds the cluster to the filter` | `setScope({kind:'cluster',id})`; `itemBody().cluster_uids` equals `[id]` |
| `selection.test.ts` | one case per row of 12a §5.1 | exact `ItemSelection` bodies |
| same | `a second select-all on a full page asks for all matching` | the controller exposes `pendingSelectAll` with the served `total` |
| `ClassPicker.test.ts` | `offers create when nothing matches`, `turns a refusal into the served suggestion`, `assigns the created class in the same gesture`, `hides deprecated classes`, `adds no window keydown listener` | as named |
| `BulkBar.test.ts` | `shows the served summary text`, `dry-runs before confirm`, `renders served exclude reasons` | as named |
| `WorkspaceRail.test.ts` | `class counts are the served count`, `queue counts are the served count`, `no client sort` (scan of the file for `.sort(`) | as named |
| `keyboard.svelte.test.ts` | the two in step 2 | as named |
| `tests/curation/test_class_create_suggestion.py` | three cases of 12a R-22 | status, body `error`, `suggested_name` |

### Other tiers

- Vitest: the moved `routes/p/[project]/clusters/**/*.test.ts` pass at their new home;
  `clusterList.scan.test.ts` gains `ClustersView.svelte` in its file list; `keymap.literals.scan.test.ts` passes.
- Stubbed e2e: new `test_workspace_shell.py`: (1) open Workspace, Clusters view, click a card, the scope chip
  shows, press the class letter, the confirm shows the dry-run count, `Enter` applies (stub records
  `PUT /crops/batch_label` with `selection.filter.cluster_uids`); (2) set a class filter, switch to Detections,
  chips stay, select 3 cards, switch back, the bulk bar still says the served summary; (3) Back restores the
  Clusters view; (4) at 430 px the rail and details are drawers, the filter bar is one button. Repoint
  `test_cluster_*.py`, `test_item_filter.py`, `test_selection_actions.py`, `test_keymap.py`, `test_project_switch.py`.
  `test_a11y_axe.py` rows: `workspace-clusters`, `workspace-cluster-scope`, `workspace-detections`,
  `workspace-class-picker`. `test_narrow_viewport_430.py` rows for the same three URLs.
- Live Cropwright tier (coordinator): `test_route_sweep.py` swaps `/clusters` rows for `/workspace?view=clusters`,
  `/workspace?view=detections`, `/workspace?scope=cluster:<served uid>`; `test_data_agreement.py`: the rail's class
  counts equal `GET /classes` `count`; the Detections total equals `GET /crops` `total` for the same filter.
- `projectRouting.test.ts`: `/p/default/clusters` and `/p/default/clusters/<uid>` are the app's 404.

### Visual checks

Screens at 1600 x 2000 and 430 x 900, saved under `artifacts_local/screens/workflow_2a/`, on the stubbed
harness (seed from `frontend/e2e/fixtures/wire.py`) and, after merge, on `coco-e2e`. Pages and states:
Clusters view with cards; the empty Clusters view with the pending-partition banner (wf-1a); a cluster scope
with facet chips and a selection; Detections with a filter and the hidden note; the class picker in assign and
create modes with a refusal; the bulk bar confirm. "Right" means: a reviewer can tell where to start (the
next-step chip), sees which cluster is open (scope chip, header), sees what is selected and what an action will
change (served summary and dry run) before pressing `Enter`. Alignment on the 4 px grid; no clipped chip text at
430 px; focus ring visible on every control reached by Tab.

### Docs and changelog

`docs-site/docs/cropwright/user-guide/workspace.md` (concise: what the page is for, the views, the rail, the
batch flow in five steps, keys by glyph), `keyboard-shortcuts.md` (workspace context), remove `clusters.md` and
its sidebar line, fix links in `assist.md`, `item-filter.md`, `dashboard.md`. `frontend/CLAUDE.md` route table:
`/workspace` replaces `/clusters`, `/clusters/[id]`; the "Cluster list and facets" and "StrategyBar" rules name
the Workspace. Frontend CHANGELOG `[Unreleased]`: "Added: Workspace page (`/workspace`) with Clusters and
Detections views, served queues and classes rail, details panel, bulk bar with a served selection summary, class
picker with inline create. Removed: `/clusters` and `/clusters/[id]`." Root CHANGELOG: R-22 under Changed.

### Risks and do not

- Do not copy the clusters page code into the views and keep the old route "for now". Move, then delete.
- Do not read `page.url.searchParams` outside `workspaceUrl.ts`.
- Do not sort or filter the served class list for the rail (only the picker filters the bounded registry by
  typed text, allowed by plan §4.7).
- Do not edit `pager.svelte.ts` or any `fetchPage`; 2E owns them.
- Do not add a window keydown listener; do not print a literal key.
- Risk: the clusters page holds several search modes (`clusterSearchModes.svelte.ts`, 546 lines). Keep it working
  inside the Clusters view as is; its URL keys `q`, `similar_to` go through `workspaceUrl.ts`.

### Dependencies and merge

Needs 2B (`ItemFilterState` keys), 2E (pager), 2C (R-12, R-19), 2D (R-26) merged before its contract and e2e
tests can pass. Code in parallel against 12a; rebase onto the integration branch after those four merge, then run
the targeted tests. Merges last in wave 2.

### Done when

- [ ] `/workspace` serves Clusters and Detections; `/clusters` routes deleted; every link repointed (`rg "/clusters" frontend/src` finds only API paths).
- [ ] The red tests above went red then green; the moved tests pass.
- [ ] Stubbed e2e files touched pass with `-n 2`; axe rows pass; 430 px rows pass.
- [ ] Screens viewed and judged against "label a cluster in one batch".
- [ ] Docs and changelogs updated; `scripts/docs/check_docs_vs_code.py` clean.
- [ ] Standards gate (12a §13).

---

## Lane 2B: Shared filter, area filters and map data

**Size:** M-L, 4 days. **Model:** Sonnet. **Merge:** first in wave 2.

### Goal and outcome

Every list, count and bulk route takes the same filter, now with `cluster_uids`, `geo_bbox`, `geo_polygon`,
`has_location` and `near_dup_group`, snapped to the location policy, with `join_truncated` served. Map cells serve
their `bounds` and single-photo pins; the map header can say how many photos have no location. `hidden` counts
follow the filter.

### Files

Create: `src/services/curation/image_join.py`, `src/schemas/curation/views.py`, `tests/curation/test_item_filter_geo.py`,
`tests/curation/test_image_join.py`, `frontend/src/lib/itemFilter/itemFilterState.geo.test.ts`,
`frontend/src/lib/itemFilter/filterKeys.contract.test.ts`.

Change: `src/services/curation/item_filter.py` (`:41-111`, `:349-385`), `src/routers/curation/_item_filter_params.py`
(`:34-147`), `src/domain/location_policy.py` (snap functions), `src/services/curation/map_cells.py` (`:74-150`,
`:195-251`; use `image_join`), `src/schemas/curation/map.py` (`:17`, `:42`), `src/services/curation/image_view.py`
(wf-1b; geo moves out), `src/routers/curation/images_view.py` (wf-1b; params move to the filter),
`src/services/curation/hidden_counts.py` (wf-1a, D-15), `src/routers/curation/_item_models.py` and
`src/schemas/curation/images.py` (one `HiddenCounts`, C-1), every caller in 12a §3.2 and R-20,
`src/services/curation/item_selection.py:74` (join), the `cluster_uid` callers listed in 12a R-20,
`frontend/src/lib/itemFilter/itemFilterState.svelte.ts` (new properties, `URL_KEYS :365-380`, `chips()` `:274`),
`frontend/src/lib/types_itemFilter.ts` (`:40-111`), `frontend/src/lib/api_clusters.ts:118-141`
(`cluster_uid` mapping), `frontend/src/lib/clusters/clusterDetailController.svelte.ts` (the `/crops` query's
`cluster_uid` stays the same spelling; the body form becomes `cluster_uids`), `tests/curation/query_fakes.py`
(`geo_polygon`, `geo_bounding_box` evaluation if missing), leak tables.

### Steps and commits

1. `refactor(curation): one HiddenCounts and SortOption model` (move only; C-1).
2. `feat(curation): hidden counts follow the request filter` (D-15). Red: `test_hidden_counts.py::test_hidden_counts_only_items_matching_the_filter`.
3. `refactor(curation): image join module from the map and gallery joins` (move `_box_query`, `_joined_image_ids`, the geo and near-dup parts of `_image_level_clauses` verbatim; repoint).
4. `feat(curation): snap area filters to the location policy` (domain functions). Red: `test_item_filter_geo.py::test_coarse_policy_rounds_a_box_outward`.
5. `feat(curation): area, location and near-dup group in the shared item filter` (model, query params, `ImageJoinMissing`, every caller passes the join; `join_truncated` on `/crops`, `/review/{tab}`, `/clusters`, `/images`, `/map/cells`, `/stats/dataset`). Red: the parametrized route sweep. `make contracts`.
6. `feat(curation): filter by several clusters` (`cluster_uids` replaces `cluster_uid`; callers swapped). Red: `test_item_filter_geo.py::test_cluster_uids_narrows_every_filter_route` (parametrized). `make contracts`.
7. `feat(curation): map cells serve bounds, single-photo pins and the unlocated count` (R-9, R-21). Red: `tests/curation/test_map.py::test_cell_serves_its_bounds`, `::test_single_photo_cell_at_max_precision_serves_its_image`, `::test_meta_counts_images_without_location`.
8. `feat(frontend): area, location, near-dup group and cluster keys in the item filter state`. Red: the two vitest files.
9. `feat(frontend): region text match mode on the filter bar` (`regionTextMatch` in `ItemFilterState`, URL key `region_text_match`, a served `filter_specs` entry with labels, rendered by `ServedFilterField.svelte` beside region text; cov-live finding). Red: `itemFilterState.geo.test.ts::region text match round-trips and reaches the query`, `tests/curation/test_review_filter_specs.py::test_region_text_match_spec_is_served`.

### Interfaces

Provides 12a §3 entirely, R-9, R-20 (except `origin=drawn`), R-21, D-3, D-4, D-5, D-15, `HiddenCounts` and
`SortOption` homes. Consumes nothing new.

### Red tests first

| File | Test | Asserts |
|---|---|---|
| `tests/curation/test_item_filter_geo.py` | `test_geo_bbox_narrows_every_filter_route[<route>]` | for each route found by walking `app.routes` for the `item_filter_query` dependency, a seeded project with 2 located and 1 unlocated image returns only the items or images inside the box, and `join_truncated` is false |
| same | `test_geo_polygon_matches_a_triangle` | only the image inside the triangle |
| same | `test_bad_polygon_is_refused_with_the_served_text` | 422, message names 3 to 64 vertices |
| same | `test_coarse_policy_rounds_a_box_outward` / `test_drop_policy_matches_nothing` | per D-5 |
| same | `test_filter_without_join_fails_closed` | calling `item_filter_clauses` with `geo_bbox` and no join raises `ImageJoinMissing` |
| same | `test_join_truncated_is_served_past_the_cap` | `JOIN_MAX_IMAGES` monkeypatched to 1 |
| same | `test_near_dup_group_matches_item_and_image_level_members` | both levels |
| same | `test_cluster_uids_narrows_every_filter_route[<route>]` | two clusters, the union is returned |
| `tests/curation/test_hidden_counts.py` (wf-1a file) | `test_hidden_counts_only_items_matching_the_filter` | `class_name=car` counts only held-out cars |
| `tests/curation/test_map.py` | three tests of step 7 | as named |
| `itemFilterState.geo.test.ts` | `round-trips geo_bbox, geo_polygon, has_location, cluster_uid` and `query and body forms agree` | as named |
| `filterKeys.contract.test.ts` | `every /crops filter param is a URL key or listed as not on the bar` | reads the contract |

### Other tiers

- Live harness: extend `tests/live/test_live_capture_map.py` with `test_area_filters_on_crops_and_images` (the
  harness's GPS fixtures: a box around one photo returns its items on `/crops` and the image on `/images`; a
  polygon likewise; `n_without_location` matches).
- Contract: `tests/test_contract_typed_responses.py`; leak sweep rows for unchanged paths still pass (no new
  paths in this lane).
- Stubbed e2e: `test_item_filter.py` and `test_map_page.py` still pass on the regenerated stubs (`contract.fill`).

### Visual checks

None of its own (backend lane). The map header screenshot lands with 3D.

### Docs and changelog

`docs-site/docs/cropwright/user-guide/item-filter.md` (area, location, near-dup group, several clusters).
`docs/design/curation_api_contract.md` (filter table). Root CHANGELOG under Changed: "The shared item filter
takes `cluster_uid` (repeatable), `geo_bbox`, `geo_polygon`, `has_location` and `near_dup_group` on every list,
count and bulk route; area filters are snapped to the location policy; list responses serve `join_truncated`;
`hidden` counts follow the filter. Map cells serve `bounds` and single-photo `image_id`; `GET /map` serves
`n_without_location`."

### Risks and do not

- Do not copy a coordinate onto items. The join is the design (D-3).
- Do not catch `ImageJoinMissing`. It is a programming error and must surface in tests.
- Do not let an area filter bypass snapping on any route; the sweep test is the guard.
- `cluster_uid` swaps touch job-starting code (`autolabel/selection.py`, `assist/scope.py`): run their unit tests.
- Do not delete `GET /map/images` (3D deletes it with its caller).

### Dependencies and merge

None upstream. Merges first in wave 2; 2E, 2C, 2D and 2A rebase on it.

### Done when

- [ ] All red tests green; the parametrized sweep covers every filter route (print the list in the commit body).
- [ ] `make contracts` committed; the endpoint catalog and contract tests pass.
- [ ] Live harness case passes under `-p op-wf-2b`, torn down.
- [ ] Docs, changelog, docs check clean.
- [ ] Standards gate (12a §13).

---

## Lane 2C: View models and served boxes: table, selection summary, image locate, next steps, box labels

**Size:** L, 5 days. **Model:** Sonnet. **Merge:** third in wave 2 (after 2B and 2E).

### Goal and outcome

The backend serves every view model the wave-3 views need: the Table (R-18), the selection summary (R-19),
locate in the image list (R-17), the next steps with the default view (R-12), and one rule for every box label
(D-16, R-24). The frontend stops computing box labels: `SourceImageOverlay` (Review, crop dialog, config
previews) and the region editor draw only served `boxes` and served display names, through one `BoxLayer`. The
region overlay no longer says "Object 1/2/3" (wf-1d finding).

### Files

Create: `src/routers/curation/{table_view,selection_summary,workflow_steps}.py`,
`src/routers/curation/_image_query_params.py`, `src/services/curation/{table_view,table_cells,selection_summary,workflow_steps}.py`,
`src/schemas/curation/{table,selection,workflow_steps}.py`, tests `tests/curation/test_table_view.py`,
`test_selection_summary.py`, `test_images_locate.py`, `test_workflow_steps.py`, `test_box_display_names.py`.

Frontend create: `frontend/src/lib/components/boxes/BoxLayer.svelte` + `BoxLayer.test.ts`,
`frontend/src/lib/boxes/boxTone.ts` + test. Frontend change: `frontend/src/lib/components/SourceImageOverlay.svelte`
(`:175-320`: delete `itemLabelInfo` and the `DrawBox` derivation from `items`; mount `BoxLayer` over the served
`boxes`; `extraShapes` stay), `SourceImageOverlay.test.ts`, `SourceImageOverlay.projectCache.test.ts`, the region
label of `MultiBoxCanvas` boxes in `frontend/src/lib/review/reviewSlotEditController.svelte.ts` and
`multiBoxRegionController.svelte.ts` (the `CanvasBox.label` comes from the region box's served `display_name`),
`frontend/src/lib/types.ts` (`ImageBox.label_kind`; region box `display_name`), `frontend/src/lib/api_cropMap.ts`
(`RAW_CROP_KEYS` if the region wire key list lives there), `frontend/e2e/stubbed/test_source_overlay.py`,
`frontend/CHANGELOG.md`.

Change: `src/routers/curation/images_view.py` (params to `ImageViewQuery`; add locate before
`/images/{image_id}`), `src/services/curation/image_view.py` (locate; `list_cursor`), `src/services/curation/image_boxes.py`
(wf-1b; label from the display name, `label_kind`), `src/schemas/curation/images.py` (`BoxView.label_kind`),
`src/services/curation/wire.py` (`display_name_of` public; region `display_name`), `src/domain/region_boxes.py`
(`region_box_display_name`), `src/routers/curation/_item_models.py` (region box wire field),
`src/routers/curation/__init__.py`, leak tables and bodies, `contracts/`.

### Steps and commits

1. `refactor(curation): image list parameters as one dependency` (`ImageViewQuery`; move only).
2. `feat(curation): one display name for every drawn box` (D-16, R-24). Red: `test_box_display_names.py::test_item_box_label_is_the_item_display_name`, `::test_proposed_and_detector_names_are_served_with_their_kind`, `::test_region_box_display_name_is_its_text_else_slot_label_and_order`. `make contracts`.
3. `feat(curation): locate an image in the gallery order` (R-17). Red: `test_images_locate.py::test_index_and_cursors_walk_to_the_image`, `::test_filtered_out_image_says_so`.
4. `feat(curation): selection summary` (R-19). Red: `test_selection_summary.py::test_counts_equal_crops_with_the_same_filter`, `::test_image_selection_expands_to_matching_items`, `::test_locked_counts_class_box_and_geometry_locks`, `::test_over_the_bulk_cap_is_refused`.
5. `feat(curation): table view model` (R-18). Red: `test_table_view.py::test_every_sortable_column_sorts_server_side` (parametrized over the served columns), `::test_cells_equal_the_item_wire`, `::test_cursor_walks_without_repeats`, `::test_unknown_column_is_refused`.
6. `feat(curation): next steps and the workspace default view` (R-12). Red: `test_workflow_steps.py` one case per step; `::test_default_view_is_gallery_without_a_serving_partition`, `::test_default_view_is_clusters_with_one` (db).
7. `feat(frontend): draw served boxes and display names only` (`BoxLayer`, `boxTone`, the overlay swap, region labels). Red: `BoxLayer.test.ts::draws one rect per served box at served coordinates`, `::shows the served label verbatim`, `::one class per tone`; `SourceImageOverlay.test.ts::draws the served boxes, not item-derived labels` (fails today: the overlay reads `items`); `reviewSlotEditController.test.ts::region box labels are the served display names`.

### Interfaces

Consumes: 12a §3 (2B), §4 `list_cursor` (2E), `selection_query`, `serialize_item`, `item_facts` (wf-1b),
`is_locked_item`, `is_locked_box`. For `is_locked_geometry` (2D), R-19's `locked` counts the class and box locks
now, and 2D adds the geometry clause to the same `filters` aggregation (2D's commit, named in its spec).
Provides R-12, R-17, R-18, R-19, R-24, `ImageViewQuery`.

### Other tiers

- Leak sweep rows: `('GET','/images/locate')` with `image_id`, `('GET','/table')`, `('GET','/table/columns')`,
  `('POST','/selection/summary')` body `{'crop_ids': [...]}`, `('GET','/next_steps')`.
- `tests/test_contract_typed_responses.py`; `tests/curation/test_list_route_item_uniqueness.py` gains `/table`.
- db tier: `make test-db` subset `-k workflow_steps`.

### Visual checks

The crop dialog, the Review source pane and the region edit mode at 1600 and 430 on the stub: every box carries
the served name (class, "car (proposed)", "person (detector)", region text or "<slot label> 2"), tones match the
legend, a selected box is distinct. The Table and next-step screens land in 3F and 3E.

### Docs and changelog

`docs/design/curation_api_contract.md` (new routes). Root CHANGELOG Added: "`GET /table/columns`,
`GET /table`, `POST /selection/summary`, `GET /images/locate`, `GET /next_steps`; boxes and region boxes serve
one display name." Frontend CHANGELOG Changed: "Image overlays draw the served boxes and names; region boxes
show their read text or the region name and number instead of 'Object N'."

### Risks and do not

- Do not format cells in the browser later: every `TableCell.text` is final here.
- Do not compute `workspace_default_view` anywhere but `workflow_steps.py`.
- Do not leave any label or colour rule in `SourceImageOverlay`; `boxTone.ts` is the only tone-to-class map.
- `serving_partition_id` defaults `level='class'`: pass the item level explicitly (fact from
  `src/services/clustering/assign.py:82-93`).
- R-19 must be one aggregation (no fan-out in handlers, engineering standards §1.9).

### Dependencies and merge

Needs 2B and 2E merged (rebase before the final run). Merges before 2D and 2A.

### Done when

- [ ] Red tests seen failing then green; leak rows added; contracts regenerated.
- [ ] Docs check clean; changelog line.
- [ ] Standards gate (12a §13).

---

## Lane 2D: Annotation writes and the geometry lock (high risk)

**Size:** L, 5 days. **Model:** Sonnet at high effort. **Independent review before merge** (checklist below).
**Merge:** fourth in wave 2.

### Goal and outcome

A person can create an item by drawing a box, move or resize an item box, and soft-delete one, each saved at
once, undoable, never overwriting a newer edit, and never undone by a machine re-run. A re-detect leaves drawn,
moved and deleted boxes exactly as the person left them.

### Files

Create: `src/routers/curation/{image_items,crops_geometry}.py`, `src/services/curation/{image_item_write,item_geometry}.py`,
`src/domain/item_geometry.py`, `src/schemas/curation/exclusion.py`, tests `tests/curation/test_image_items.py`,
`test_item_geometry.py`, `test_geometry_lock.py`, `test_exclude_reasons.py`, live `tests/live/test_live_annotation.py`.

Change: `src/clients/occ_locks.py` (`is_locked_geometry`, `lock_includes :126`),
`src/clients/curation_opensearch/{ensure_fields,bodies_core}.py`, `src/services/curation/index_bootstrap.py:183-208`,
`src/services/curation/edit_history.py:46-155` (`GEOMETRY`), `src/services/curation/reprocess_detect.py:104-221`,
`src/services/curation/proposal_merge.py:30-160`, `src/services/curation/item_delete.py:43-75` callers,
`src/services/curation/open_vocab_run.py:441,456`, `src/services/curation/dataset_import/reconcile.py:98`,
`src/services/curation/source_image_cache.py:68-75` (`evict_crop_cache`), `src/services/curation/wire.py:251-402`
(geometry fields, versioned `thumbnail_url :399`), `src/routers/curation/_item_models.py:97-294`,
`src/routers/curation/images.py` (crop thumbnail `v`), `src/services/curation/image_boxes.py` (locks on `BoxView`),
`src/services/curation/selection_summary.py` (geometry lock in `locked`), `src/services/curation/item_filter.py`
(`origin=drawn`, `Origin :41`), `src/routers/curation/_crop_write_models.py:31-41`, `src/routers/curation/crops_exclude.py`,
`src/schemas/curation/config_common.py:24` (`ErrorCode`), leak tables and bodies, `contracts/`.

### Steps and commits

1. `feat(curation): geometry fields in the items mapping` (migration + body; `test_index_schema_marker.py` passes).
2. `feat(curation): geometry lock predicate` (`is_locked_geometry`, `lock_includes`). Red: `test_geometry_lock.py::test_predicate_cases` (human, import, human not-an-object, holdout, detector: each true or false).
3. `feat(curation): re-detect never moves, deletes or duplicates a locked item` (D-10). Red: `test_geometry_lock.py::test_redetect_keeps_a_drawn_item`, `::test_redetect_keeps_a_moved_item_and_does_not_recreate_its_old_box`, `::test_redetect_does_not_recreate_a_human_deleted_item`, `::test_open_vocab_and_import_reconcile_respect_the_lock`.
4. `feat(curation): served exclude reasons with not_an_object` (R-16 enum, R-26 route). Red: `test_exclude_reasons.py::test_reasons_are_served`, `::test_unknown_reason_is_refused`.
5. `feat(curation): draw a box to create an item` (R-14, D-8, D-9, D-11). Red: `test_image_items.py` (table below).
6. `feat(curation): move or resize an item box with optimistic concurrency, and undo it` (R-15). Red: `test_item_geometry.py`.
7. `feat(curation): drawn origin filter and geometry fields on the wire` (`origin=drawn`, `geometry_*`, `geometry_locked`, `lock_label`, versioned thumbnail, `BoxView` locks, R-19 `locked`). `make contracts`.
8. `test(curation): live annotation round trip` (`tests/live/test_live_annotation.py`).

### Interfaces

Provides R-14, R-15, R-16, R-26, `origin=drawn`, `BoxView.locked/lock_label/geometry_revision`, `evict_crop_cache`.
Consumes `occ_update_one` (`occ.py:105`), `occ_skip_on_conflict_bulk` (`:183`), `build_item_doc`
(`item_doc.py:160`), `region_seed_status`, `class_id_for_name`, `enqueue_pending_bound`, `reembed_items`,
`record_edit`, `find_edit_undo`, `restore_edit_state`.

### Red tests first

| File | Test | Asserts |
|---|---|---|
| `test_image_items.py` | `test_drawn_item_has_human_class_and_geometry` | stored doc: `class_source`, `label_source` human, `class_validated`, `geometry_source` human, `geometry_revision` 1, `crop_area_norm`, `crop_rank_in_image` |
| same | `test_class_is_resolved_by_name_and_deprecated_is_refused` | 201 for an active name; 422 `unknown_class` for a deprecated or unknown one |
| same | `test_repeat_with_the_same_request_id_returns_the_first_item` | second call 200, `created: false`, one doc |
| same | `test_box_refusals` | out of image, too small (`min_side` served), degenerate: each 422 with its code |
| same | `test_foreign_image_is_404` | another project's image id |
| same | `test_region_seed_set_when_the_class_is_a_region_parent` | `region_status` pending detection |
| same | `test_new_item_on_a_held_out_image_is_held_out` | D-11 |
| same | `test_embed_failure_keeps_the_item_deferred` | encoder raises; 201, `embedding_state` deferred, message served |
| same | `test_near_dup_pending_marked_added` | `enqueue_pending_bound` called with `added` |
| `test_item_geometry.py` | `test_move_bumps_revision_and_keeps_detector_box` | `geometry_revision` +1, `detector_bbox_norm` = old box on first move only |
| same | `test_stale_revision_is_409_with_the_current_item` | body shape of 12a R-15 |
| same | `test_move_drops_the_vector_and_re_embeds` | vector field gone before embed; `embedding_state` after |
| same | `test_crop_cache_file_is_evicted` | the cache file is gone |
| same | `test_regions_outside_are_refused_unless_dropped` | 422 with `box_ids`; resend with `drop_region_box_ids` succeeds and the boxes are gone |
| same | `test_undo_restores_the_previous_geometry` / `test_undo_with_nothing_left_is_409` | pairing rule |
| same | `test_thumbnail_url_carries_the_revision` | wire `thumbnail_url` ends `?v=2` |

### Other tiers

- Live harness (`tests/live/test_live_annotation.py`, seeded items per D-19): draw a box on a seeded image, the item
  exists with a human class; move it, `geometry_revision` 2 and the crop thumbnail bytes differ; delete it
  (`not_an_object`), un-exclude restores; two concurrent PATCHes with the same expected revision: one 200, one 409;
  undo restores the geometry.
- Live Cropwright write tier (coordinator, D-19): `frontend/e2e/live/test_writes.py` gains
  `test_draw_move_delete_survive_a_redetect` on a throwaway public COCO project: draw, move, delete, then
  `POST /images/{id}/reprocess` with scope `detect`, then assert all three as the person left them and the new item
  reaches `embedding_state` embedded.
- Leak sweep rows: `('POST','/images/{image_id}/items')`, `('PATCH','/crops/{crop_id}/geometry')`,
  `('POST','/crops/{crop_id}/geometry/undo')`, `('GET','/exclude_reasons')`.

### Visual checks

None (backend). The editor screens land in 3C.

### Docs and changelog

`docs/design/curation_api_contract.md`, `docs/opensearch_schema_design.md` (three fields); the user page
is 3C's. Root CHANGELOG Added:
"`POST /images/{image_id}/items`, `PATCH /crops/{crop_id}/geometry`, `POST /crops/{crop_id}/geometry/undo`,
`GET /exclude_reasons`; the lock rule covers item geometry: re-detect, open-vocabulary runs and import reconcile
never move, delete or re-create a drawn, moved or human-deleted item."

### Risks and do not

- Do not write `bbox_norm` outside `item_geometry.py` and `image_item_write.py`.
- Do not hold a lock in process; OCC only (owner: multi-user, about 30 workers).
- Do not enqueue `reprocess` from these routes (D-8).
- Do not change `crop_id` on a move. The doc id is the identity; the content hash is only the creation key.
- Re-ingest of the same image: confirm `occ_upsert._merge_preserving_human` (`occ_upsert.py:393-423`) keeps
  `class_excluded` and the new geometry fields of an existing doc; add
  `test_geometry_lock.py::test_reingest_keeps_a_human_deleted_and_a_moved_item`.
- Combine copies recompute `crop_id` from `bbox_norm` (`projects/combine/copy_docs.py:178`): a copy of a moved
  item gets a new id in the target project; that is correct (a new project), and `geometry_source` is copied.
  Add a combine test that the field is copied.

### Independent review checklist (a separate Sonnet reviewer with no lane context; Opus only if the owner asks)

1. Every automated writer that creates, deletes or re-indexes items calls `is_locked_geometry`: `rg -n "index_items|delete_items|op_type" src/services` and check each caller (reprocess detect, open vocabulary run, dataset import chunk, reconcile and undo, combine).
2. IoU matching against `detector_bbox_norm` is used, not only `bbox_norm` (D-10), and the threshold is `LABEL_IOU_MATCH`.
3. All three geometry writes go through `occ_update_one` with the revision check inside the merger; no `client.update` without `if_seq_no`.
4. The idempotent create cannot produce two docs under concurrency (two identical requests in parallel: one `created: true`, one `false`).
5. Undo pairing for `geometry` mirrors `find_edit_undo`; an undo of an undo is not restorable.
6. The vector is dropped before the re-embed, so a failed embed never leaves an old vector on a moved box.
7. Holdout: no route lets an automated writer touch a held-out item; a human edit of one is allowed and keeps `test_holdout`.
8. The wire's `geometry_locked` and `lock_label` come from the same predicate the writers use.
9. Mapping migration is registered, is in the body, and the schema fingerprint changed.
10. The live tests ran under their own compose project and were torn down; the write tier ran only on a throwaway public project.

### Dependencies and merge

Needs 2B (filter) and 2C (`image_boxes`, `selection_summary`) merged before its final rebase. Merges before 2A.

### Done when

- [ ] Every red test above seen failing then green; the review checklist answered in writing by the reviewer, and every finding fixed or filed.
- [ ] Live harness case green under `-p op-wf-2d`; the write-tier test written for the coordinator.
- [ ] Contracts, docs and changelog; docs check clean.
- [ ] Standards gate (12a §13).

---

## Lane 2E: Cursor paging and crop sorts

**Size:** L, 4 days. **Model:** Sonnet. **Merge:** second in wave 2.

### Goal and outcome

`/crops`, `/review/{tab}` and `/regions` page by cursor past 10,000 items; `/crops` serves its sort options with
labels and sorts by class, detector class and validated; a crop can be located in any `/crops` order. Every grid
pages through the one pager, so no view ever sees a page number.

### Files

Create: `src/services/curation/list_cursor.py`, `tests/curation/test_list_cursor.py`,
`tests/curation/test_crops_cursor.py`, `tests/curation/test_crops_locate.py`.

Change: `src/services/curation/crop_listing.py:24-110`, `src/services/curation/crop_browse.py:13-120`,
`src/routers/curation/crops_list.py:36-270` (cursor, sort options, locate route before `/crops/{crop_id}`),
`src/routers/curation/review.py:118-334`, `src/services/curation/review_request.py:26-151`,
`src/routers/curation/_page_models.py:18-52`, `src/routers/curation/_item_models.py:297-314`,
`src/services/curation/region_rows.py:206`, the `/regions` router, `src/routers/curation/_common.py:54-71` (delete
the guard), `src/services/curation/image_view.py` (wf-1b; uses `list_cursor`), `src/schemas/curation/views.py`
(`SortOption` from clusters), `frontend/src/lib/pager.svelte.ts` and its test, `frontend/src/lib/api_cropRead.ts:17-29`,
`frontend/src/lib/api_review.ts:150-200`, the regions API wrapper, every `fetchPage` closure
(`itemFilter/matchingItems.svelte.ts:19`, `clusters/clusterDetailController.svelte.ts`, `clusters/clusterListController.svelte.ts`
if it pages crops, `review/reviewQueueLoader.svelte.ts` including `jumpToPendingCrop :218`, `slots/slotGalleryController.svelte.ts`,
`clusters/firstLoad.ts`, `clusters/clusterControllerTestKit.ts`), `frontend/src/lib/types.ts` (`PaginatedResponse`
becomes `CursorPage`), stubs in `frontend/e2e/harness/stub.py` and `fixtures/wire.py`.

### Steps and commits

0. `test(frontend): one screenshot helper for stubbed tests` (`frontend/e2e/harness/screens.py` `shot(page, name, *, env_var)`; the nine local `_shot` copies listed in 12a §2.1 import it; behaviour identical).
1. `refactor(curation): one list cursor module` (move from `image_view.py`; tests move).
2. `feat(curation): crop sort options with labels and class sorts` (R-4). Red: `test_crops_cursor.py::test_sort_options_are_served_with_labels`, `::test_sort_by_class_name_and_validated`.
3. `feat(curation): cursor paging on crops, review queues and regions` (R-11; `page` removed). Red: `test_crops_cursor.py::test_reads_past_ten_thousand` (fake with 10,050 docs), `::test_cursor_of_another_sort_is_refused`, `tests/curation/test_review_cursor.py::test_queue_reads_past_ten_thousand`. `make contracts`.
4. `feat(curation): locate a crop in any crops order` (R-25) and locate answers with cursors. Red: `test_crops_locate.py::test_index_and_cursors`, `::test_filtered_out`.
5. `feat(frontend): cursor pager for every grid` (pager + closures + types + stubs in one commit, because removing `page` breaks every caller). Red: `pager.svelte.test.ts::appends by next_cursor`, `::loadAt starts at a located cursor`, `::ignores a stale response after a reset`.

### Interfaces

Provides 12a §4, R-4, R-11, R-25, `SortOption` home. Consumes 2B's filter (rebase).

### Other tiers

- Vitest for each controller whose closure changed (existing tests adapt; none deleted).
- Stubbed e2e: every test that stubs `/crops`, `/review/` or `/regions` with `page` is repointed to the cursor shape;
  `test_review_deep_link_locate.py` asserts one request lands on the located item.
- Live harness: `tests/live/test_live_paging.py::test_crops_cursor_reads_past_the_window` with a bulk-seeded
  10,100-item cohort (the seed script's bulk path), under `-p op-wf-2e`.
- Leak sweep row: `('GET','/crops/locate')` with `crop_id`.

### Visual checks

The existing review and clusters pages at 1600 px: scrolling loads more, "N of total" shows the served index.

### Docs and changelog

Root CHANGELOG Changed: "`/crops`, `/review/{tab}` and `/regions` page with `cursor` and `next_cursor`; `page` and
`page_size` are removed; lists read past 10,000 items. `/crops` serves `sort_options` and sorts by class name,
detector class and validated. Added `GET /crops/locate`." Frontend CHANGELOG Changed: "Every grid pages by cursor."

### Risks and do not

- Do not keep `page` beside `cursor` (no two mechanisms).
- The `order=` modes on `/crops` (`outliers`, `core_first`, `diverse`; `crops_list.py:141-154`) rank in memory over
  a bounded set (`ordered_crops_page`): serve them as one bounded page with `next_cursor: null` and say so in the
  route description; do not fake a cursor.
- Sort values containing nulls: keep `missing: _last` (`crop_browse.parse_crop_sort`) and the `crop_id` tiebreak so
  `search_after` is total.

### Dependencies and merge

Needs 2B merged (rebase). Merges before 2C and 2A.

### Done when

- [ ] Red tests seen failing then green; no `page=` left in `frontend/src` API calls (`rg "page:" frontend/src/lib/api_*.ts`).
- [ ] Contracts regenerated; endpoint catalog test passes.
- [ ] Live paging case green and torn down.
- [ ] Standards gate (12a §13).
