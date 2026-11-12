# 12a Workflow interfaces: the shared contracts for waves 2 to 4

| Field | Value |
|---|---|
| Status | **spec, binding for every lane of waves 2 to 4.** Written 2026-11-11 on `int/b4-stateless` at `f544183e1`, after reading the wave-1 worktrees (`wf-1a` to `wf-1d`) as they stood that afternoon |
| Parent plan | [12_review_workflow_audit.md](12_review_workflow_audit.md) (the why); this file is the what, exactly |
| Lane specs | [12b wave 2](12b_wave2_lane_specs.md), [12c wave 3](12c_wave3_lane_specs.md), [12d wave 4](12d_wave4_lane_specs.md), [12e order](12e_workflow_execution_order.md) |
| Reader | a Sonnet lane with no memory of this session. Every name below is final: copy it, do not rename it. If something here is wrong, stop and tell the coordinator; do not invent a second version |
| Line numbers | `file:line` refer to `int/b4-stateless` at `1685723a9` (wave-1 lanes `wf-1b` and `wf-1d` merged) for files those lanes touched, else `f544183e1`. Marks `(wf-1a)` and `(wf-1c)` are the worktrees still to merge. When a line moved, re-find by symbol name (`rg -n 'def image_page'`) |

---

## 0. How to use this file

1. Section 1 lists the decisions that close gaps or risks in the plan. They override the plan where they differ.
2. Section 2 names the **one home** of each cross-cutting concern. Before writing a helper, model, query
   builder, component or API function, look it up there. If it is listed, import it. If it is not listed and
   two places need it, extract it into the named home in a separate commit.
3. Sections 3 to 8 are the shared contracts: the item filter, cursors, the selection, the Workspace URL state,
   the keymap and the shared view models.
4. Section 9 specifies every route, R-1 to R-27. Section 10 specifies the shared frontend modules. Section 11
   lists the existing code to reuse and the code to delete. Section 12 sets the visual rules.
5. Section 13 is the standards gate. Every lane copies it into its done-when list.

---

## 1. Decisions

| # | Decision | Why (docs, code facts) |
|---|---|---|
| D-1 | **Cursor paging uses `search_after` with a unique tiebreak, never a point-in-time (PIT).** The cursor is opaque URL-safe base64 of `{"sort": <sort id>, "after": <sort values>}`. It is refused (400 `cursor_invalid`) when the sort differs. One helper module, `src/services/curation/list_cursor.py`, holds it (extracted from `image_view.py:94-110` (wf-1b)). | The project guard keeps PIT ids per process (`src/services/projects/guard.py:41,69,85-95`) and refuses a PIT id it has not seen (`guard_policy.py:153-166`). With about 30 API workers a PIT handed to the browser would be refused on the next worker. `search_after` with a tiebreak is OpenSearch's documented deep-paging method (https://docs.opensearch.org/latest/search-plugins/searching-data/paginate/). Concurrent writes can shift rows between pages; the UI accepts that, and every list dedups by id (`pager.svelte.ts:16-64`). |
| D-2 | **Cursor paging moves from wave 4 to wave 2 (lane 2E)**, for `/crops`, `/review/{tab}` and `/regions`. `page` is removed in the same change. | The owner rule is no write-then-rewrite. Wave 3 views would otherwise be written against `page` and rewritten in wave 4. All grids already page through one abstraction, `createPager` (`frontend/src/lib/pager.svelte.ts:64`), so 2E changes the pager and the `fetchPage` closures once, and the views never see a page number. |
| D-3 | **Geo filters on item routes join through the images index.** `geo_bbox`, `geo_polygon`, `has_location` and the image level of `near_dup_group` are image predicates. One async resolver resolves them to at most `JOIN_MAX_IMAGES` (10,000) image ids (`map_cells.py:49`). Every list response that accepts the shared filter serves `join_truncated`. | Items carry no coordinate by design (`src/clients/curation_opensearch/capture_fields.py:1-6`: "No coordinate is ever copied to an item"). The map already joins this way (`map_cells.py:112-150`). |
| D-4 | **`item_filter_clauses` fails closed.** It raises `ImageJoinMissing` when the filter holds an image predicate and the caller passed no `ImageJoin`. | A route that forgets the join must 500 in tests. It must never silently ignore an area filter. |
| D-5 | **Area filters are snapped to the location policy before querying.** Under `store_coarse`, `geo_bbox` edges are rounded **outward** and polygon vertices are rounded to `COARSE_DECIMALS` (2, `src/domain/location_policy.py:20`). Under `store` they are unchanged. Under `drop` they match nothing. | Plan §4.8 privacy rule: a drawn area can never reveal positions finer than the policy serves. |
| D-6 | **The selection is not in the URL.** It lives in `WorkspaceController.selection` and survives view switches because the shell stays mounted. Filter, queue, scope, view, sort and focus are in the URL. | An explicit selection can hold 5,000 ids (`MAX_SELECTION_IDS`, `item_selection.py:27`). That is far past safe URL length, and the e2e and nginx limits sit near 8 KB. A deep link reproduces the screen; it does not reproduce a selection, the way Label Studio and CVAT links work too. This narrows plan §4.6 "all of it lives in the URL". |
| D-7 | **A selection is an `ItemSelection` with no exclusion list.** Gallery selection = the filter plus `image_ids`; Clusters selection = the filter plus `cluster_uids`; "all matching" = the filter. `ItemFilter.image_ids` and the new `cluster_uids` accept up to `MAX_SELECTION_IDS` in a **body**. Their query parameters stay capped at 200 (`_item_filter_params.py:117`). | `ItemSelection` is `crop_ids` xor `filter` (`item_selection.py:43-71`). No exclusion field exists, and adding one would be a second selection mechanism. Selections above 200 images only ever travel in POST bodies (bulk writes, R-19). |
| D-8 | **The drawn-box follow-up embeds inline, not through the `reprocess` job.** R-14 and R-15 embed the one item after the write with `reembed_items(..., only_missing=True)`, the same in-request pattern as `refresh_box_embeddings` (`src/services/curation/region_box_refresh.py:50-102`). On failure the write stands, the item stays `deferred`, and the response says so. | `reprocess` has `exclusive_group='reprocess'` (`src/config/job_kinds/families_a.py:29-38`), so a second box drawn while one job runs would get 409 `job_busy` (`src/services/jobs/service.py:151-158`). No coalescing helper exists. The plan's `followup_job_id` is dropped. |
| D-9 | **Idempotent create without a new table.** A drawn item's `crop_id` = `sha256(f'{image_id}|drawn|{request_id}')[:32]`, written with `op_type=create`. A retry with the same `request_id` hits the create conflict, reads the stored doc and returns it with 200. | `crop_id` is a content hash today (`src/services/detection/geometry.py:61-70`). Making the hash input the request id makes a retry land on the same doc. |
| D-10 | **The geometry lock also protects the machine's original box.** On the first human move the item keeps `detector_bbox_norm` (the stored box before the move). A re-detect drops a detection whose IoU against a geometry-locked item's `bbox_norm` **or** `detector_bbox_norm` is at or above `LABEL_IOU_MATCH` (0.5). Soft-deleted items (`not_an_object`) stay as documents, so their boxes block re-creation too. | `reprocess_detect.py:179-221` merges by IoU against **current** boxes (`proposal_merge.py:30,138`). Without D-10, an item moved away from the detector's spot would be re-created there as a duplicate. |
| D-11 | **A new item inherits the hold-out.** A box drawn on an image whose items are in the frozen test holdout is created with `test_holdout: true` and the image's `dataset_split`. | Otherwise a new object on a test image leaks into training. |
| D-12 | **Existing keymap ids and contexts are kept; new contexts `workspace` and `editor` are added.** `cluster.*` becomes the Workspace grid context (relabelled), `review.*` becomes the Review view's context, and `box_edit.*` stays the region editor's. `keyboardStore` dispatch gains context `includes`: the current context first, then its includes in order, then `global`. | Renaming ids would orphan saved per-project overrides (`GET/PUT /keymap`). Dispatch today fires only `global` or the exact scope (`frontend/src/lib/stores/keyboard.svelte.ts:242`), so a `workspace.*` action could not coexist with grid actions. The owner named the editor scope `editor.*` (answer 4). |
| D-13 | **Grid `Enter` opens Review at the focused card.** `cluster.confirm` (today `enter`, `keymapFallback.ts:209`) moves to `shift+enter`. | Plan §4.6 keyboard table; one meaning per key (§4.9, the CVAT `N` lesson). |
| D-14 | **Redo exists only in the image editor.** Grids and Review keep undo only (`cluster.undo` `z`, now also `ctrl+z` and `meta+z`). The shortcut overlay stays on `~` (`keymapFallback.ts:107`); the plan's `?` is dropped because the action already has its maximum of 3 combos. | Grid and Review undo is a served label op or a single-write inverse. A redo of a bulk op has no served route. |
| D-15 | **`hidden` counts follow the request's filter.** `hidden.test_holdout` and `hidden.excluded` count the items that match the request's item filter (and joins) but are removed by the visibility clauses. One function: `hidden_counts(opensearch, filter_clauses, *, include_test, include_excluded)` in `src/services/curation/hidden_counts.py` (wf-1a). | Wave 1 built two meanings (section 14, conflict C-1). A project-wide "25 hidden" under a `class=car` filter is wrong. |
| D-16 | **Box labels have one rule.** `BoxView.label` for an item is the item wire's `display_name` (wf-1d `_display_name`, `src/services/curation/wire.py`), or `"unlabeled"` when `display_name_kind == 'none'`. `BoxView` gains `label_kind`. Region boxes get a served `display_name` (R-24). `SourceImageOverlay` stops computing labels (`itemLabelInfo`) and draws served `boxes`. | wf-1b `image_boxes.item_label` ignores the proposed and detector names that wf-1d serves. Two rules is a bug waiting (section 14, C-2). |
| D-17 | **Queues are served filter presets.** The queue id is sent as `review_tab` on every view's list route and as the path of `/review/{tab}` in the Review view. The URL spells it `queue=` only. | Plan §4.6 "a queue is a served filter preset". `ItemFilter.review_tab` already validates against `SIMPLE_TABS` (`item_filter.py:96,213-223`). |
| D-18 | **The Review view over a non-queue query walks `/crops`**, in the Detections view's served sort, located with the new `GET /crops/locate` (R-25). | Plan §4.6: "`Enter` opens Review at the focused item over the same query and order" needs a locate on `/crops`. |
| D-19 | **The live harness cannot ingest COCO** (fake Triton, `tests/live/test_live_capture_map.py:3-9`). Write-path live tests run on seeded items in `tests/live/`. The real COCO ingest checks (draw, embed, re-detect) run in the opt-in Cropwright live write tier (`frontend/e2e/live/test_writes.py`, `pytest.mark.writes`) on a throwaway public project, run by the coordinator. | The plan's "live harness on a public COCO ingest" is not possible as written. |
| D-20 | **Map rendering tests launch Chromium with `--use-gl=angle --use-angle=gl-egl`**, or run headed on display `:11`. The live conftest's SwiftShader args (`frontend/e2e/live/conftest.py:185-200`) do not give WebGL2 on this host (wf-1d finding). Stubbed map tests that need a canvas use a separate `webgl_browser` fixture with those args; every other stubbed test keeps `--disable-gpu`. | wf-1d root-caused the map bug and found which launch works. |
| D-21 | **Exclude reasons are served.** A new `GET /exclude_reasons` serves `[{id, label}]`. `ExcludeReason` becomes a backend `Literal` that adds `not_an_object`. The frontend `EXCLUDE_REASONS` array (`clusterDetailActions.svelte.ts:111-117`) and the `ExcludeReason` type (`api_cropWrites.ts:318`) are deleted. | The frontend is dumb: the labels are a vocabulary, so the backend serves them. |
| D-22 | **Label history stays at `/label_ops`.** ~~Single writes do not become label ops in this release.~~ **Overridden by the owner on 2026-11-12:** a single label write (`PUT /crops/{crop_id}/label`) is recorded as a label op in the same request (no job), so undo is one served history that survives a reload (G-22, lane `wf-g22`). The owner's second ruling the same day: the editor's class change also undoes through the served op (its command stack keeps geometry only); `label/undo` stays for discard and new-class resolve, which are not label ops. Single writes are retained apart from bulk ops (200 bulk, 1,000 single). | The owner reopened it; the single write stays synchronous, so the op record is written in the request through the same recorder the auto-label loop uses. |

---

## 2. One home per concern (reuse, never re-implement)

| Concern | Backend home | Frontend home | Created or extended by |
|---|---|---|---|
| Item filter model | `ItemFilter`, `src/services/curation/item_filter.py:51` | `ItemFilterState`, `frontend/src/lib/itemFilter/itemFilterState.svelte.ts:37`; types `frontend/src/lib/types_itemFilter.ts:40-111` | 2B (geo, `cluster_uids`, `near_dup_group`), 2D (`origin=drawn`), 4C (`viz_box`) |
| Filter to OpenSearch clauses | `item_filter_clauses(f, *, embedding, image_join=None)` (`item_filter.py:349`) | none: the browser never builds a query | 2B |
| Image-predicate join | `resolve_image_join(opensearch, f, *, policy) -> ImageJoin \| None`, new `src/services/curation/image_join.py`. It absorbs, moved verbatim, `map_cells.parse_bbox` (`:61`), `geo_box_query` (`:74`), `item_image_ids` (`:112`), `LOCATION_FIELD` (`:42`), `JOIN_MAX_IMAGES` (`:49`) and the geo, location and near-dup parts of `image_view._image_level_clauses` (`image_view.py:113-134`) | n/a | 2B |
| Filter query parameters | `ItemFilterQuery` (`src/routers/curation/_item_filter_params.py:147`) | `ItemFilterState.toQuery()` (`:75`), `toUrl()` (`:152`), `fromUrl()` (`:124`) | 2B |
| Image-only list parameters | new `ImageViewQuery` dependency, `src/routers/curation/_image_query_params.py`, extracted from `images_view.py` (wf-1b) `list_images` params | `imageViewParams` in `frontend/src/lib/images/imageQuery.ts` | 2C |
| Cursor | `src/services/curation/list_cursor.py`: `encode_cursor`, `decode_cursor`, `CursorError` | `createPager` (`frontend/src/lib/pager.svelte.ts:64`), changed to cursor | 2E |
| Sort vocabulary of a list | `SortOption` in `src/schemas/curation/views.py`; each list keeps its own sort table (`crop_browse.py:13-23`, `image_view.py:49-56` (wf-1b), `cluster_cards.py:50-59`, `review_sorts.py:365`) | `SortSelect.svelte` (new, `frontend/src/lib/components/workspace/SortSelect.svelte`) renders served `sort_options` | 2E (backend), 2A (component) |
| Hidden counts | `hidden_counts(...)` (`src/services/curation/hidden_counts.py`, wf-1a), model `HiddenCounts` in `src/schemas/curation/views.py` | `HiddenItemsNote.svelte` (wf-1a, `frontend/src/lib/components/itemFilter/`) | 2B |
| Empty reason | `EmptyReason` models stay per route (each route has its own reasons), with a served `empty_message` string wherever the UI shows text | `ViewState.svelte` (new, `frontend/src/lib/components/state/ViewState.svelte`): `loading`, `error`, `empty` with served text, `ready` | 2A |
| Selection | `ItemSelection` (`item_selection.py:43`), `SelectionTargets` (`src/routers/curation/_selection.py:22`), `selected_crop_ids` (`_selection.py:38`) | `WorkspaceSelection` + `toItemSelection()` in `frontend/src/lib/workspace/selection.ts` (new); id-click mechanics stay `createSelection` (`frontend/src/lib/selection.svelte.ts:32`) | 2A |
| Selection summary | R-19 service `src/services/curation/selection_summary.py` | `getSelectionSummary` in `frontend/src/lib/api_select.ts` | 2C (backend), 2A (client) |
| Bulk writes | `PUT /crops/batch_label`, `POST /crops/move`, `POST /crops/batch_exclude`, `POST /crops/batch_unexclude` (`crops_write.py:86-170`, `crops_exclude.py:22-85`) | `api_select.ts:140-200` (`bulkLabelSelection`, `batchExcludeSelection`, `batchUnexcludeSelection`, `moveSelectionToCluster`); dialog controller `createSelectionActionController` (`frontend/src/lib/itemFilter/selectionActionController.svelte.ts:52`) | reused as-is |
| URL state | n/a | `frontend/src/lib/workspace/workspaceUrl.ts` (pure parse/write) + `WorkspaceController` (`frontend/src/lib/workspace/workspaceController.svelte.ts`) | 2A |
| API client | one router module per route family | one function per route, in the `api_<area>.ts` named in section 10.1 | the lane that first consumes the route |
| Keymap | `src/config/keymap_actions.json` (served vocabulary), validator `src/services/curation/keymap_validator.py:270-326`, `reserved_hotkeys` (`src/services/curation/keymap.py:154`) | `frontend/src/lib/keymapFallback.ts` (mirrors the JSON), `keymapStore` (`stores/keymap.svelte.ts:206`), `keyboardStore.registerAction` (`stores/keyboard.svelte.ts:128`) | 2A (`workspace`, includes), 3C (`editor`) |
| Box drawing (read-only) | `BoxView` built by `src/services/curation/image_boxes.py` (`:42-96`) | `BoxLayer.svelte` (new, `frontend/src/lib/components/boxes/BoxLayer.svelte`) + `boxTone.ts` (tone to CSS class); `SourceImageOverlay` mounts `BoxLayer` and draws only the served `boxes` | 2C (labels, component and the overlay swap, so wave 3 never draws from `items`) |
| Box geometry math (edit) | `validate_bbox_norm` (`src/services/curation/region_box_edits.py:60`) | `frontend/src/lib/annotations/boxGeometry.ts` (new): `clamp01`, `normalizeXyxy`, `moveBox`, `resizeBox(handle)`, `hitTest`, `nudge`, extracted from `MultiBoxCanvas.svelte:106-260` | 3C (extract first, then use) |
| View-model serializers | `serialize_item` (`wire.py:227`); image cards `image_view._card` (wf-1b); boxes `image_boxes.item_box_views` (wf-1b); table cells `src/services/curation/table_cells.py` (new) | none: render served text | 2C |
| Class picker | `POST /classes` (`classes_write.py:57-76`) + R-22 | `ClassPicker.svelte` + `classPicker.svelte.ts` (new). Replaces `ClassPickerModal.svelte` and `reviewClassPicker.svelte.ts` | 2A (create), 3B (Review switches, deletes old) |
| Next steps and default view | R-12 service `src/services/curation/workflow_steps.py` | `nextStepsStore` (new, `frontend/src/lib/stores/nextSteps.svelte.ts`) | 2C (backend), 2A (store) |
| Errors | `api_error(status, code, message, **fields)` (`src/routers/curation/_api_error.py:20`); codes in `ErrorCode` (`src/schemas/curation/config_common.py:24`) | `ApiError`, `structuredDetail`, `apiErrorText` (`frontend/src/lib/api_errors.ts:92,123,138`) | every lane adds its codes to `ErrorCode` |
| Formatters | n/a | `frontend/src/lib/format/` only (`format.scan.test.ts`) | reuse |
| Dialogs | n/a | `Modal.svelte`, `ConfirmDialog.svelte` | reuse |
| Test builders | `tests/curation/query_fakes.py:399` `QueryFakeOpenSearch`; `tests/_curation_app.py:21,33`; leak tables `tests/curation/cross_project_leak/_route_tables.py` | vitest `frontend/src/lib/test/fixtures/` (`imageDetail.ts` wf-1b; add `imagePage.ts`, `tablePage.ts`, `selectionSummary.ts`); e2e `frontend/e2e/fixtures/wire.py`, `contract.py` `fill(schema_name, **over)` (`:57`) | the lane that adds the shape adds its builder; no copied setup |

### 2.1 Extractions of existing duplicates (each is its own commit, code moved verbatim first)

| Duplicate today | Move to | Lane |
|---|---|---|
| `HiddenCounts` in `src/routers/curation/_item_models.py` (wf-1a) **and** in `src/schemas/curation/images.py` (wf-1b) | one `HiddenCounts` in `src/schemas/curation/views.py`; both importers repointed | coordinator at the wave-1 merge if possible, else 2B commit 1 |
| `SortOption` in `src/schemas/curation/images.py` (wf-1b) and the cluster sort option dicts in `src/schemas/curation/clusters.py` | `SortOption` in `src/schemas/curation/views.py` | 2E |
| Cursor encode and decode, private in `image_view.py:94-110` | `src/services/curation/list_cursor.py` | 2E |
| Join and area helpers public in `map_cells.py` (`parse_bbox :61`, `geo_box_query :74`, `item_image_ids :112`) but used by the gallery (`image_view.py:34`) and soon every list route, plus the geo, location and near-dup parts of `image_view._image_level_clauses` (`:113-134`) | `src/services/curation/image_join.py` (`item_image_ids` keeps its name) | 2B |
| Inline `search_after` loops (`item_selection._scan` `:87-116`, `embedding_viz_read.get_cached_projection` `:117-140`) | **not** in scope: internal scans, not API cursors; leave them | none |
| Box label rule `image_boxes.item_label` (`:28`) vs `wire._display_name` (`wire.py:162`) | `image_boxes` calls `wire._display_name` (renamed public `display_name_of`) | 2C |
| `SourceImageOverlay.itemLabelInfo` (`SourceImageOverlay.svelte:175-199`) computing colours and labels from `items`, the last client-side label workaround | `BoxLayer.svelte` + `boxTone.ts` over served `boxes` | 2C |
| Nine local `_shot(page, name)` screenshot helpers in stubbed tests (`test_label_ops.py:146` and `test_review_ui_polish.py`, `test_map_page.py`, `test_ingest_policy.py`, `test_detector_seed.py`, `test_near_duplicates.py`, `test_image_search_popover_430.py`, `test_embedding_state.py`, `test_source_missing.py`) | one `shot(page, name, *, env_var)` in `frontend/e2e/harness/screens.py` (create it if no lane has; full-page, directory from the env var, no-op when unset) | 2E commit 0 |
| Box math private to `MultiBoxCanvas.svelte:106-260` | `frontend/src/lib/annotations/boxGeometry.ts` | 3C |
| Class pickers: `ClassPickerModal.svelte` + `reviewClassPicker.svelte.ts` (Review) and the toolbar `<select>` (`ClusterDetailToolbar.svelte:128-140`) | `ClassPicker.svelte` + `classPicker.svelte.ts` | 2A (create, use in BulkBar and rail), 3B (Review, delete old) |
| Exclude reason labels in `clusterDetailActions.svelte.ts:111-117` | served `GET /exclude_reasons` (R-16) | 2D (route), 2A (BulkBar uses it, deletes array) |
| Inline "Loading…" and empty paragraphs in the clusters, review and map pages | `ViewState.svelte` | 2A for the pages it moves; each later lane for the pages it touches |

---

## 3. The shared item filter

One schema for `/crops`, `/images`, `/clusters`, `/review/{tab}`, `/map/cells`, `/viz/projection` (4C), `/table`,
`/images/locate`, `/crops/locate`, `/selection/summary`, `/stats/*` and every bulk write (through
`ItemSelection.filter`).

### 3.1 Backend model (`src/services/curation/item_filter.py`)

The existing fields (`item_filter.py:56-108`) stay as they are. Changes:

| Field | Type | Query param (repeatable where a list) | Clause | Lane |
|---|---|---|---|---|
| `cluster_uids` **replaces** `cluster_uid` | `list[uuid.UUID] = []`, `max_length=MAX_SELECTION_IDS` | `cluster_uid` (repeatable, max 200 in the query) | `terms cluster_uid` | 2B |
| `geo_bbox` | `GeoBox \| None` (`west, south, east, north` floats; `west > east` crosses the antimeridian, as `map_cells.parse_bbox` `:61-71`) | `geo_bbox=w,s,e,n` | image predicate (D-3) | 2B |
| `geo_polygon` | `list[tuple[float, float]] \| None`, 3 to 64 `(lon, lat)` vertices, the ring closed server-side | `geo_polygon=lon,lat;lon,lat;...` | image predicate, OpenSearch `geo_polygon` on `capture.location` (https://docs.opensearch.org/latest/query-dsl/geo-and-xy/geopolygon/) | 2B |
| `has_location` | `bool \| None` | `has_location` | image predicate | 2B |
| `near_dup_group` | `uuid.UUID \| None` | `near_dup_group` | `term near_dup_group_id` on items **or** an image predicate on `images.near_dup_group_id` (one `bool.should`) | 2B |
| `image_ids` | max raised to `MAX_SELECTION_IDS` in the model (D-7); the query param stays 200 | `image_id` | unchanged | 2B |
| `origin` | `Origin` gains `'drawn'` | `origin=drawn` | `term geometry_source: human` and `must_not exists detector_class_name` | 2D |
| `viz_box` | `tuple[float,float,float,float] \| None` | `viz_box=x1,y1,x2,y2` | `range viz_x`, `range viz_y` | 4C |

Validation: the model validators raise `ValueError` with a plain sentence. `ItemFilterQuery` turns a parse failure
into the stock FastAPI 422. The routes' existing `except ValueError -> 400` stays (`crops_list.py:227-228`). New
geo parse errors are 422 through pydantic (a `field_validator` on `GeoBox` and on the polygon parser), with the
messages `geo_bbox needs west,south,east,north in degrees` and `geo_polygon needs 3 to 64 lon,lat vertices`.

### 3.2 The image join (`src/services/curation/image_join.py`, 2B)

```python
IMAGE_PREDICATE_FIELDS: Final = ('geo_bbox', 'geo_polygon', 'has_location', 'near_dup_group')

@dataclass(frozen=True)
class ImageJoin:
    image_ids: tuple[str, ...]   # images matching every image predicate (capped)
    truncated: bool              # more than JOIN_MAX_IMAGES matched

class ImageJoinMissing(RuntimeError): ...   # raised by item_filter_clauses (D-4)

def has_image_predicates(f: ItemFilter) -> bool
def image_predicate_clauses(f: ItemFilter, *, policy: LocationPolicy) -> list[dict[str, Any]]  # images-index clauses, snapped (D-5)
async def joined_image_ids(opensearch: Any, *, index: str, clauses: list[dict[str, Any]]) -> tuple[list[str], bool]
async def resolve_image_join(opensearch: Any, f: ItemFilter, *, policy: LocationPolicy) -> ImageJoin | None
```

- `resolve_image_join` returns `None` when `has_image_predicates(f)` is false (no query issued).
- `item_filter_clauses(f, *, embedding, image_join=None)`. With image predicates and `image_join is None` it raises
  `ImageJoinMissing`. With a join it adds `terms image_id: join.image_ids`; for `near_dup_group` it adds the
  `bool.should` of the item term and the joined ids.
- Snapping lives in `src/domain/location_policy.py`: `snap_box(box, policy) -> GeoBox | None` and
  `snap_polygon(vertices, policy) -> list[tuple[float, float]] | None`. `None` means the `drop` policy, so
  `match_none`.
- Callers that must pass the join (2B swaps each one in the same commit; the route-level sweep test in 3.4
  proves it): `crops_list.list_crops` via `build_crop_list_query` (`crop_listing.py:34`), `review_request.build_review_request`
  (`src/services/curation/review_request.py:58`), `cluster_cards.list_cluster_cards` (`:307`), `cluster_facets`,
  `stats_dataset.build_dataset_stats` (`:163`), `map_cells.map_cells` (`:195`), `image_view.image_page`
  (wf-1b), `item_selection.selection_query` (`:74`), and every 2C route.

### 3.3 Query, body and URL spelling (one table, both sides)

| Concept | Query param (GET) | Body field (`ItemFilter`) | Frontend `ItemFilterState` property | URL key in the Workspace |
|---|---|---|---|---|
| classes | `class_name` | `class_names` | `classNames` | `class_name` |
| clusters | `cluster_uid` | `cluster_uids` | `clusterUids` | `cluster_uid` |
| area box | `geo_bbox` | `geo_bbox` (object `{west,south,east,north}`) | `geoBbox: GeoBox \| null` | `geo_bbox` |
| area polygon | `geo_polygon` | `geo_polygon` (array of `[lon, lat]`) | `geoPolygon: [number, number][] \| null` | `geo_polygon` |
| located | `has_location` | `has_location` | `hasLocation: boolean \| null` | `has_location` |
| near-dup group | `near_dup_group` | `near_dup_group` | `nearDupGroup: string \| null` | set by `scope=near_dup:<id>`, not a bar control |
| images | `image_id` | `image_ids` | `imageIds` | set by `scope=image:<id>`, not a bar control |
| queue | `review_tab` | `review_tab` | not on the bar | `queue` (D-17) |
| embedding box | `viz_box` | `viz_box` | `vizBox` | `viz_box` (4C) |
| region text match | `region_text_match` (`exact`, `fuzzy`, `fragment`; `TextMatch`, `region_text_search.py:27`) | `region_text_match` | `regionTextMatch` (**missing today**: the API supports it, the bar has no control) | `region_text_match`; the bar renders it beside region text from a served `filter_specs` entry with labels (2B adds the entry in `src/services/curation/review_filter_specs.py` if absent) |

All other keys keep today's spelling (`URL_KEYS`, `itemFilterState.svelte.ts:365-380`). `ItemFilterState`
(2B adds the properties above, with `toQuery`, `toBody`, `toUrl`, `fromUrl` and `chips()` entries) is the
**only** frontend parser and serializer. No view, controller or page reads `page.url.searchParams` for a
filter key.

### 3.4 Tests that keep it one source of truth

- `tests/curation/test_item_filter_geo.py` (2B): validation, snapping per policy, and one parametrized test over
  `FILTER_ROUTES` (a list in the test of `(method, path)` for every route that takes `ItemFilterQuery`, built by
  walking `app.routes` for the `item_filter_query` dependency). For each route, `geo_bbox` narrows the result
  and `join_truncated` is served. A new route that takes the filter is picked up automatically.
- `frontend/src/lib/itemFilter/itemFilterState.geo.test.ts` (2B): URL round trip and query/body parity for
  the new keys.
- `frontend/src/lib/itemFilter/filterKeys.contract.test.ts` (2B): every `ItemFilterQuery` param in the OpenAPI
  `GET /crops` operation is either in `URL_KEYS` or in a named `NOT_ON_BAR` list in the test.

---

## 4. Cursor paging (2E; R-1, R-17 and R-18 use the same helper)

```python
# src/services/curation/list_cursor.py
class CursorError(ValueError): ...
def encode_cursor(sort_id: str, after: list[Any]) -> str
def decode_cursor(cursor: str, *, sort_id: str) -> list[Any]   # CursorError on garbage or a sort mismatch
def page_body(body: dict, *, sort: list[dict], tiebreak: str, cursor: str | None, sort_id: str, limit: int) -> dict
    # sets size=limit+1, sort=[*sort, {tiebreak: asc}], search_after when a cursor is given
def split_page(hits: list[dict], *, sort_id: str, limit: int) -> tuple[list[dict], str | None]
    # (kept hits, next_cursor or None)
```

- Wire on every cursored list: `next_cursor: str | None`, `total: int`, `limit: int`. No `page`, no `page_size`.
- Routes refuse a bad cursor with 400 `{"detail": {"error": "cursor_invalid", "message": ...}}` through
  `api_error` (add `cursor_invalid` to `ErrorCode`).
- `guard_page_depth` (`_common.py:57-71`) and `MAX_RESULT_WINDOW` are deleted once no caller remains (2E).
- Locate answers (R-17, R-25, the changed `/review/{tab}/locate`) serve `{index, total, cursor_before,
  cursor_after, reason}`. `cursor_before` is the cursor of the page that **starts** at the located item:
  `search_after` = the sort values of the item before it. `cursor_after` is the next page's cursor.
- Frontend: `createPager<T>({ fetchPage: (cursor: string | null) => Promise<CursorPage<T>>, idOf, ... })`
  with `CursorPage<T> = { items: T[]; next_cursor: string | null; total: number }`. `loadPage(page)` is deleted;
  `loadAt(cursor)` replaces it for locate. "N of total" comes from the served `index`, never a page number.

---

## 5. The selection model

### 5.1 Wire

The bulk routes keep `SelectionTargets` (`_selection.py:22`): exactly one of `crop_ids` (max 5,000) or
`selection: ItemSelection`, plus `dry_run`. The Workspace always sends `selection`.

| Workspace selection | `ItemSelection` sent |
|---|---|
| explicit items (Detections, Table detection rows, Review, cluster-scope grid) | `{crop_ids: [...]}` |
| images (Gallery, Table image rows) | `{filter: {...current filter, image_ids: [...]}}` |
| clusters (Clusters view cards) | `{filter: {...current filter, cluster_uids: [...]}}` |
| all matching | `{filter: {...current filter}}` (with the queue as `review_tab`, the scope as its filter field) |

`include_test` is never set by the Workspace. The frozen holdout is never written by a bulk op.

### 5.2 Frontend (`frontend/src/lib/workspace/selection.ts`, 2A)

```ts
export type WorkspaceSelection =
  | { kind: 'none' }
  | { kind: 'items'; cropIds: string[] }
  | { kind: 'images'; imageIds: string[] }
  | { kind: 'clusters'; clusterUids: string[] }
  | { kind: 'matching' };

export function toItemSelection(sel: WorkspaceSelection, filter: ItemFilter): ItemSelection | null; // null for 'none'
export function selectionKindFor(view: ViewId, rows?: TableRows): 'items' | 'images' | 'clusters';
```

- A click in a view whose kind differs from the current selection replaces the selection.
- Filter, queue or scope changes keep `items`, `images` and `clusters` selections (images and clusters are ANDed
  with the new filter at send time). The bulk bar re-requests the summary, so its numbers stay true.
- `Ctrl/Cmd+A` (`cluster.select_all`): first press selects the loaded page (`items`, `images` or `clusters` by
  view). A second press, when the whole page is already selected, opens the confirm "Select all N matching"
  (N = the served `total`) and on Enter sets `{kind: 'matching'}`. In `matching`, every card renders selected. A
  plain click on a card leaves `matching` and sets that one card (no mixed mode).
- `Esc` cascade (`workspace.escape`): clear the selection, then leave Review, then clear the scope, then close the
  details panel.

### 5.3 Summary (R-19)

The bulk bar and the selection details panel render `SelectionSummary` verbatim:
`"183 detections in 41 images · 7 locked, skipped"`. The text is served as `summary_text`. The bar calls R-19
on every selection change, debounced 200 ms and aborted on the next change. The dry run (`dry_run: true`) of
the chosen action gives the count the confirm dialog shows (`selectionActionController.svelte.ts:125-151`).

---

## 6. Workspace URL state and controller

Route: `/p/[project]/workspace`. Section label: `Workspace`.

### 6.1 Parameters

| Param | Values | Default when absent | History |
|---|---|---|---|
| `view` | `gallery`, `clusters`, `detections`, `review`, `table`, `map`, `embedding` (4C) | the served `workspace_default_view` (R-12); the controller then writes it with `replace` | push |
| filter keys | section 3.3 | none | push on Apply or chip removal |
| `queue` | a served queue id (`GET /review/tabs` ids) | none | push |
| `scope` | `cluster:<uuid>`, `near_dup:<uuid>`, `image:<image_id>` | none | push |
| `focus` | `item:<crop_id>`, `image:<image_id>`, `cluster:<uuid>`, `cell:<tile>` | none | replace |
| `sort_gallery`, `sort_clusters`, `sort_detections`, `sort_review`, `sort_table` | a served sort id of that view | the server's default; the view shows the served `sort_applied` | push |
| `rows` | `detections`, `images` (Table) | `detections` | push |
| `columns` | comma list of served column ids (Table) | the served `default` | replace |
| `thumb` | `s`, `m`, `l` | `m` | replace |
| `q`, `similar_to` | existing search modes (`clusterSearchModes.svelte.ts:167,304-308`) | none | push |
| `map` | `<zoom>/<lat>/<lon>` with 2 decimals for lat and lon | fit to the served bounds | replace |

- Push = `goto(resolve(projectHref(`/workspace${search}`)), { reset: false })`. Replace adds `replace: true`.
  SvelteKit 3 `goto` options (https://svelte.dev/docs/kit/$app-navigation): `replace`, `reset`. The deprecated
  `replaceState` and `invalidateAll` are not used, and no shallow routing is needed (the route has no `load`).
- **One source of truth:** `WorkspaceController.url` is `$derived(parseWorkspaceUrl(page.url.searchParams))`.
  Nothing copies URL state into `$state` except the filter bar's **draft** (`ItemFilterState`), which an
  `$effect` re-seeds with `fromUrl` whenever `page.url` changes. That is how Back and Forward restore view,
  filter, queue and scope. Selection is not restored (D-6).
- A deep link opens exactly that view, filter, queue and scope. `focus=item:<id>` with `view=review` locates the
  item (R-25, or `/review/{tab}/locate` with a queue). A focus the server cannot locate shows the served `reason`
  in the view's `ViewState` and lands on the first item.

### 6.2 Controller API (`frontend/src/lib/workspace/workspaceController.svelte.ts`, 2A)

```ts
export type ViewId = 'gallery' | 'clusters' | 'detections' | 'review' | 'table' | 'map' | 'embedding';
export type ScopeRef = { kind: 'cluster' | 'near_dup' | 'image'; id: string };
export type FocusRef = { kind: 'item' | 'image' | 'cluster' | 'cell'; id: string };
export type TableRows = 'detections' | 'images';

export interface WorkspaceUrlState {
  view: ViewId | null;
  queue: string | null;
  scope: ScopeRef | null;
  focus: FocusRef | null;
  sort: Partial<Record<ViewId, string>>;
  rows: TableRows;
  columns: string[] | null;
  thumb: 's' | 'm' | 'l';
  q: string | null;
  similarTo: string | null;
  map: { zoom: number; lat: number; lon: number } | null;
}

// frontend/src/lib/workspace/workspaceUrl.ts (pure, unit-tested)
export function parseWorkspaceUrl(params: URLSearchParams): WorkspaceUrlState;
export function writeWorkspaceUrl(params: URLSearchParams, next: Partial<WorkspaceUrlState>): void;

export class WorkspaceController {
  readonly filter: ItemFilterState;            // the bar's draft
  readonly url: WorkspaceUrlState;             // $derived from page.url
  readonly view: ViewId;                       // url.view ?? nextStepsStore.defaultView ?? 'gallery'
  selection: WorkspaceSelection;               // $state, D-6
  detailsOpen: boolean;                        // $state; default from a (min-width: 1280px) media query

  setView(view: ViewId): void;                 // push
  setQueue(id: string | null): void;           // push; keeps selection
  setScope(scope: ScopeRef | null): void;      // push
  setFocus(focus: FocusRef | null): void;      // replace
  setSort(view: ViewId, sortId: string): void; // push
  setTable(rows: TableRows, columns?: string[] | null): void;
  applyFilter(): void;                         // writes the draft to the URL, push
  removeChip(param: string, value?: string): void;
  itemQuery(): ItemFilterQuery;                // URL filter + queue as review_tab + scope field
  itemBody(): ItemFilter;                      // same, body form
  itemSelection(): ItemSelection | null;       // toItemSelection(selection, itemBody())
  dispose(): void;
}
export function createWorkspaceController(): WorkspaceController; // one per page mount
```

`itemQuery()` and `itemBody()` are the only places that combine filter, queue and scope. Views receive the
controller and call them. No view builds its own filter.

### 6.3 View and details registries

```ts
// frontend/src/lib/workspace/viewRegistry.ts
export interface ViewEntry {
  id: ViewId;
  label: string;                     // UI copy, not data: the view names are fixed product terms
  component: Component<{ ws: WorkspaceController }>;
  keyContext: 'cluster' | 'review' | 'table' | 'map' | 'workspace';
  selectionKind: (ws: WorkspaceController) => 'items' | 'images' | 'clusters';
  available?: (ws: WorkspaceController) => boolean; // e.g. map: mapAvailabilityStore; absent, never disabled
}
export const VIEWS: ViewEntry[];     // each view lane appends one entry

// frontend/src/lib/workspace/detailsRegistry.ts
export type DetailsKind = 'item' | 'image' | 'cluster' | 'cell' | 'selection';
export const DETAILS: Record<DetailsKind, Component<{ ws: WorkspaceController; id: string | null }>>;
```

---

## 7. Keymap

Backend `src/config/keymap_actions.json` and frontend `keymapFallback.ts` change together in the same commit,
followed by `make contracts` (`contracts/json/keymap_actions.json`). Every key is an action. No new window
listener (`keymap.literals.scan.test.ts`).

### 7.1 Contexts

| Context | Includes | Class hotkeys live | Status | Lane |
|---|---|---|---|---|
| `global` | none | yes | unchanged | none |
| `workspace` | `global` | yes | **new**: view-level keys in every view | 2A |
| `cluster` | `workspace` (was `global`) | yes | relabel "Workspace grid" (Gallery, Clusters, Detections, cluster scope) | 2A |
| `review`, `review.queue`, `review.region` | `review` includes `workspace` (was `global`) | yes | relabel "Review view" | 3B |
| `table` | `workspace` | yes | **new** | 3F |
| `map` | `workspace` | **no** (letters would assign classes to an area by accident) | **new** | 3D |
| `editor` | `global` | yes (a class letter sets the selected box's class) | **new** | 3C |
| `box_edit`, `region_gallery`, `clusters_search` | as today | as today | `clusters_search` is deleted with the `/clusters` page if its actions have no caller left (2A checks with `rg`) | 2A |

Dispatch with includes (D-12; `keyboard.svelte.ts:215-247`, 2A): the candidate scopes are the current scope,
then its `includes` transitively in declared order, then `global`. The first registered handler for the
resolved action wins.

### 7.2 New and changed actions (defaults; all `modifiable` unless the key is locked)

| Action id | Context | Default keys | Notes |
|---|---|---|---|
| `workspace.view_prev` / `workspace.view_next` | workspace | `[` / `]` | |
| `workspace.scope_prev` / `workspace.scope_next` | workspace | `,` / `.` | previous or next cluster in a cluster scope (served order of the Clusters list) |
| `workspace.toggle_details` | workspace | `\` | |
| `workspace.open_editor` | workspace | `shift+o` | opens `/p/<slug>/images/<image_id>` for the focused item or image |
| `workspace.class_picker` | workspace | `/` | the same picker as `review.queue.class_picker` |
| `workspace.escape` | workspace | `escape` (locked) | the cascade in 5.2 |
| `workspace.select_toggle` | cluster, table | `space` | |
| `workspace.open_review` | cluster, table | `enter` (locked) | D-13 |
| `cluster.select_all` | cluster | `a`, `ctrl+a`, `meta+a` | change: adds two combos |
| `cluster.confirm` | cluster | `shift+enter` | change (D-13) |
| `cluster.undo` | cluster | `z`, `ctrl+z`, `meta+z` | change |
| `table.prev_row` / `table.next_row` | table | `arrowup` / `arrowdown` (locked) | |
| `editor.*` | editor | see 12c lane 3C | `v`, `b`, `h` and the rest |

**Class-hotkey clashes are refused already.** `reserved_hotkeys` (`keymap.py:154-166`) collects every single
unmodified character bound in a class-hotkeys-live context, and `validated_hotkey`
(`src/routers/curation/_class_hotkeys.py:56-98`) answers 422 `hotkey_reserved`. Adding `editor.tool_select` `v`
therefore refuses a class hotkey `v` with the served message, and the user can rebind either one (owner
answer 4). The lanes add one test per new letter.

---

## 8. Shared view models (`src/schemas/curation/views.py`, new)

```python
class SortOption(BaseModel):
    id: str
    label: str

class HiddenCounts(BaseModel):
    test_holdout: int = Field(description='Items matching the filter that the frozen test holdout hides (0 when included).')
    excluded: int = Field(description='Items matching the filter that are ignored or deleted (0 when included).')
```

Every cursored list response carries, in this order: `items`, `next_cursor`, `total`, `limit`, `sort_applied`,
`sort_options`, `hidden`, `join_truncated`, `empty_reason`, `empty_message`. `empty_reason` is a per-route
`Literal`; `empty_message` is the served sentence, null when not empty.

`BoxView` (`src/schemas/curation/images.py` (wf-1b)) after waves 2 (2C, 2D):

```python
class BoxView(BaseModel):
    crop_id: str
    box_id: str | None = None
    kind: Literal['item', 'region']
    bbox_norm: tuple[float, float, float, float]          # image coordinates
    label: str                                           # served display text (D-16)
    label_kind: Literal['class', 'proposed', 'detector', 'region', 'none']   # 2C
    class_name: str | None = None
    tone: BoxTone                                        # validated|machine|unlabeled|holdout|rejected
    state: str | None = None
    geometry_revision: int | None = None                 # items only, 2D
    locked: bool = False                                 # items: geometry lock; regions: is_locked_box, 2D
    lock_label: str | None = None                        # served text, e.g. 'Set by a person', 2D
```

---

## 9. Routes

All project-scoped paths are under `/curation/projects/{project}` (written `{prefix}`). Every new route:
(a) a thin handler in `src/routers/curation/` registered with one import line in
`src/routers/curation/__init__.py:16-101`; (b) logic in `src/services/curation/` with no FastAPI import
(layering guard); (c) `response_model=` or `responses={200: {'model': ...}}` so
`tests/test_contract_typed_responses.py` passes; (d) a row in
`tests/curation/cross_project_leak/_route_tables.py` (`READ_QUERIES :30`, `SLUG_QUERIES :42`; bodies in
`_route_bodies.py:71`); (e) structured refusals through `api_error` (`_api_error.py:20`) with new codes added to
`ErrorCode` (`config_common.py:24`); (f) `make contracts` and `contracts/` committed in the same commit.
Router tests copy `tests/curation/test_review_capture_date_filter.py:47-59` (a `QueryFakeOpenSearch`, a
dependency override on `_raw_opensearch_dep`).

### R-1 `GET {prefix}/images` (built in wave 1, wf-1b; changed in wave 2)

As built: `src/routers/curation/images_view.py`, `src/services/curation/image_view.py`, models
`ImagePage`, `ImageCard`, `ImageCounts`, `ImageNearDup` in `src/schemas/curation/images.py` (wf-1b). Wave 2 changes:

- 2B: `geo_bbox` and `has_location` move into the shared filter and are removed from the route's own params;
  `hidden` follows D-15; models move per section 2.1.
- 2C: the image-only params (`source`, `in_holdout`, `near_dup`, `label_state`, `sort`, `cursor`, `limit`,
  `include_test`) move into `ImageViewQuery` (`_image_query_params.py`) so R-17 and R-18 reuse them; the cursor
  uses `list_cursor` (2E). `BoxView.label` follows D-16.
- Response gains `limit` and `empty_message`, in the section 8 order.
- `ImageQueryError` stays a 400 with a string; a bad cursor is 400 `cursor_invalid`.

### R-2 `GET {prefix}/images/{image_id}` and R-3 `.../file`, `.../thumbnail` (built, wf-1b)

Unchanged except: `BoxView` fields (D-16, 2C; locks, 2D); `ImageDetail.items` keeps `serialize_item`.

### R-4 `GET {prefix}/crops` sort (2E)

- `CROP_SORT_FIELDS` (`crop_browse.py:13-23`) gains `class_name`, `detector_class_name`, `validated`
  (`class_validated`). Sort ids are `<field>:<asc|desc>`.
- The response serves `sort_applied` and `sort_options: list[SortOption]` with labels (for example `Detector
  confidence (low first)`). A table `CROP_SORT_LABELS` lives beside `CROP_SORT_FIELDS`; a test asserts every
  field has both directions labelled.
- Response model `CropsPageResponse` (`_item_models.py:297-314`): drop `page`, `page_size`; add `next_cursor`,
  `limit`, `sort_options`, `join_truncated`, `empty_reason` (`no_items`, `no_match`), `empty_message`.
- The `ids=` lookup path (`crops_list.py:177-182`) is unchanged and serves no cursor.

### R-5 `GET {prefix}/clusters/{cluster_uid}` (built, wf-1c). R-6 near-duplicate views (built, wf-1c)

No changes. The Clusters view's scope header and Quality page consume them.

### R-7, R-8 (built, wf-1a)

No changes beyond 2B's `join_truncated` on `GET /clusters` (it takes the filter).

### R-9 Map area (2B)

- `MapCellProperties` (`src/schemas/curation/map.py:42`) gains `bounds: MapBounds` (the tile's served box,
  `map_cells.tile_box` `:251`, snapped per D-5).
- `MapMeta` (`:17`) gains `n_without_location: int` (images with no stored position or one the policy drops).
- `GET /map/cells` takes the full shared filter including area (it already takes `ItemFilterQuery`, `map.py:45`).
- `GET /map/images` and `MapTileImages` are **deleted by 3D**, with their only caller.

### R-10 `GET {prefix}/review/tabs`: the queue set (3B)

As built (wf-1a): each `ReviewTab` has `available`, `unavailable_reason`, `count`
(`src/services/curation/review_tab_status.py`, `_review_tab_models.py`). 3B changes:

- The served list is the **queues** only, in this order:

| Queue id | Label | Definition (`review_tab_clauses.py` clause) | Default sort |
|---|---|---|---|
| `to_label` | To label | not excluded, no `class_id` | `detector_confidence:desc` |
| `check_machine` | Check machine labels | `class_id` set, `class_validated` false, `class_source` not human | `confidence:asc` |
| `disagreements` | Disagreements | union of today's `mismatches`, `detector_disagreements`, `model_disagreements` clauses | `disagreement_entropy_asc` |
| `new_class_proposals` | New class proposals | today's clause | today's default |
| `audit` | Audit sample | items of the pending audit sample (`src/routers/curation/audit.py` sample store) | sample order |
| `regions` | served region slot label | today's `regions` tab; present only when a region profile is active | today's |

- A queue that cannot fill is **absent**. Its served reason is a `next_steps` entry (R-12), not a dead tab.
- The old ids (`all`, `mismatches`, `vlm_low_conf`, `detector_disagreements`, `outliers`, `uncertainty`,
  `model_disagreements`, `primary_low_conf`, `classifier_blind_spots`, `imported`) stay valid only as
  `review_tab` **filter values** (`SIMPLE_TABS`) and as served `review_status`/sort options on `/crops`, so a
  saved link with `review_tab=outliers` still filters. They are no longer paths of `/review/{tab}` (404 with the
  served reason `not_a_queue`).
- `GET /review/tabs` accepts `ItemFilterQuery`. `count` is the queue's count under the current filter (one
  `_msearch`, `review_tab_status.tab_counts`).
- `GET /audit/queue` and `AuditQueueList.svelte` are deleted.

### R-11 Cursor paging (2E)

`/crops`, `/review/{tab}`, `/regions`: section 4. `/review/{tab}/locate` returns
`{crop_id, in_queue, index, total, cursor_before, cursor_after, reason, sort_applied, sort_fallback_reason}`
(`ReviewLocateResponse` `_page_models.py:38-52`; `rank`, `page`, `page_size` removed).

### R-12 `GET {prefix}/next_steps` (2C)

- Files: `src/routers/curation/workflow_steps.py`, `src/services/curation/workflow_steps.py`, models in
  `src/schemas/curation/workflow_steps.py`. (`src/services/curation/next_steps.py` already exists for job
  follow-up actions (`next_steps.py:14-38`); do not mix them.)
- Response `WorkflowSteps`:

```python
class WorkflowStep(BaseModel):
    id: Literal['ingest', 'classes', 'cluster', 'activate', 'label', 'hotkeys', 'near_dups', 'export', 'train', 'review_disagreements']
    label: str                 # 'Create classes from the detector'
    detail: str                # served sentence with counts
    done: bool
    section: str               # a ProjectSubPath: 'ingest', 'classes', 'workspace', 'quality/near-duplicates', 'export', 'train'
    query: str | None = None   # optional search string for the link, e.g. 'view=clusters'

class WorkflowSteps(BaseModel):
    steps: list[WorkflowStep]          # the loop order
    next: WorkflowStep | None          # the first undone step
    workspace_default_view: Literal['gallery', 'clusters']
```

- Rules: `ingest` done when the images index has a doc. `classes` done when the registry has an active class.
  `cluster` done when any item partition exists (`list_partitions`, `src/services/clustering/partitions.py:384`).
  `activate` done when `serving_partition_id(..., target='item')` (`src/services/clustering/assign.py:82-93`;
  pass the level explicitly) is not null. `label` done when validated items reach the class floor
  (`ClassListResponse.thresholds`). `hotkeys` done when every active class has a `hotkey_letter`. `near_dups`
  done when the image level has a finished run (`near_duplicates/status`). `export` and `train` from the export
  and train run tables. `review_disagreements` appears only after a promoted model exists.
- `workspace_default_view` is `clusters` when an item partition serves, else `gallery` (owner answer 2).
- Tests `tests/curation/test_workflow_steps.py`: one case per step state; the default view in both states; a
  Postgres-backed case marked `db`.

### R-13 Per-image counts (4A)

See 12d lane 4A.

### R-14 `POST {prefix}/images/{image_id}/items` (2D)

```python
class ItemCreateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    bbox_norm: tuple[float, float, float, float]   # image coordinates, x1<x2, y1<y2, all in [0,1]
    class_name: str = Field(min_length=1, max_length=128)
    request_id: uuid.UUID

class ItemCreated(BaseModel):
    item: ItemDoc                                   # serialize_item
    created: bool                                   # false on an idempotent repeat
    image_boxes: list[BoxView]
    image_counts: ImageCounts
    embedding_state: EmbeddingState                 # 'embedded' or 'deferred' (D-8)
    embedding_message: str | None                   # served text when deferred
```

- 201 on create; 200 with `created: false` on a repeat of `request_id` (D-9).
- Refusals (`api_error`): 404 `image_not_found`; 422 `unknown_class` (unknown or deprecated, with `message` and
  `class_name`); 422 `box_out_of_image`, `box_too_small` (`min_side` in the body; `MIN_BOX_SIDE_NORM = 0.004`, a
  constant in `src/domain/item_geometry.py`), `degenerate_box`.
- Service `src/services/curation/image_item_write.py`:
  `async def create_drawn_item(opensearch, *, image_id, request: ItemCreateRequest, now, embedding, registry) -> ItemCreated`.
  Steps: resolve the class by name (`class_id_for_name`, `src/services/classes/resolve.py:46`; refuse a
  deprecated one); read the image doc and its sibling items; `build_item_doc` (`src/services/curation/item_doc.py:160`)
  with `class_source='human'`, `label_source='human'`, `class_validated=True`, `region_status=region_seed_status(...)`
  (`item_doc.py:37-56`), `test_holdout` and `dataset_split` per D-11, plus extra fields `geometry_source='human'`,
  `geometry_revision=1`, `embedding_state='deferred'`; index with `op_type=create`; re-rank
  `crop_rank_in_image` for the image (machine field, `occ_skip_on_conflict_bulk`, `src/clients/occ.py:183`);
  `enqueue_pending_bound('item', [crop_id], 'added')` (`src/services/near_duplicates/pending.py:68`); inline embed
  (D-8); build `image_boxes` and `image_counts` with `image_counts.item_facts` (wf-1b).
- Files: `src/routers/curation/image_items.py`, `src/services/curation/image_item_write.py`,
  `src/domain/item_geometry.py` (pure validation shared with R-15), models in `src/schemas/curation/images.py`.

### R-15 `PATCH {prefix}/crops/{crop_id}/geometry`, `POST {prefix}/crops/{crop_id}/geometry/undo` (2D)

```python
class GeometryPatchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    bbox_norm: tuple[float, float, float, float]
    expected_geometry_revision: int = Field(ge=0)     # 0 for an item never edited
    drop_region_box_ids: list[str] = []

class GeometryWritten(BaseModel):
    item: ItemDoc
    image_boxes: list[BoxView]
    embedding_state: EmbeddingState
    embedding_message: str | None
```

- Merger (inside `occ_update_one`, `occ.py:105`; a typed exception raised in the merger propagates, the
  same hook `regions_boxes_edit._check_revision` uses at `:196-202`): check the revision, else raise
  `GeometryConflictError(current)`; check region boxes inside the new box (region boxes are stored in the source
  frame, `region_writes.parent_to_source_bbox` `:46-66`), else raise `RegionsOutsideItemError(box_ids)` unless they
  are listed in `drop_region_box_ids` (then they are removed through `boxes_write_fields`,
  `src/domain/region_boxes.py:357`); `record_edit(kind=GEOMETRY)` (`edit_history.py:95-116`); set `bbox_norm`,
  `crop_area_norm`, `geometry_source='human'`, `geometry_revision+1`, `detector_bbox_norm` (only when absent,
  D-10), drop the vector fields and set `embedding_state='deferred'`.
- After the write: evict the crop cache file (`evict_crop_cache(slug, crop_id)`, new in
  `src/services/curation/source_image_cache.py` beside `crop_cache_file` `:68-70`); re-rank the image; inline
  embed (D-8). Near-dup `vector_changed` is enqueued by `reembed_items` already (`reprocess_embed.py:350`).
- 409 `geometry_conflict` body: `{error, message, current_geometry_revision, item}` (the shape of
  `_region_conflict_detail`, `regions_boxes_edit.py:205-215`). 422 `regions_outside_item` body:
  `{error, message, box_ids}`. Refusals of R-14 apply (box rules).
- Undo: `find_edit_undo` pairing (`edit_history.py:119-132`) on kind `geometry`; restores `bbox_norm`,
  `crop_area_norm`, `geometry_source`, `detector_bbox_norm`, bumps the revision, drops the vector and re-embeds.
  409 `nothing_to_undo`.
- Crop thumbnail cache busting: the wire's `thumbnail_url` becomes `{prefix}/crops/{id}/thumbnail?v=<geometry_revision>`
  (`wire.py:399`), and `GET /crops/{crop_id}/thumbnail` (`src/routers/curation/images.py:195` (wf-1b)) declares
  `v: int | None` (documented as a cache key, unused server-side) so the endpoint catalog test accepts it.
- `EditKind` (`edit_history.py:46-48`) gains `GEOMETRY`; snapshot fields `bbox_norm`, `crop_area_norm`,
  `geometry_source`, `detector_bbox_norm`, `crop_rank_in_image`.
- Files: `src/routers/curation/crops_geometry.py`, `src/services/curation/item_geometry.py`.

### R-16 Geometry lock and soft delete (2D)

- Mapping migration (`ensure_items_geometry_fields` in `src/clients/curation_opensearch/ensure_fields.py`, registered
  in `index_bootstrap._migrations()` `:183-208`, fields also in `_items_body()` `bodies_core.py:261+`):
  `geometry_source` keyword, `geometry_revision` integer, `detector_bbox_norm` float (array of 4).
- `src/clients/occ_locks.py`: `is_locked_geometry(source: Mapping[str, Any]) -> bool`: `geometry_source == 'human'`,
  or `import_ids` present, or (`class_excluded` and `excluded_by == 'human'` and `excluded_reason == 'not_an_object'`),
  or `test_holdout`. `lock_includes()` (`:126`) adds `geometry_source`, `import_ids`, `class_excluded`,
  `excluded_by`, `excluded_reason`, `detector_bbox_norm`.
- `reprocess_detect.redetect_image` (`:179-221`) and `proposal_merge.merge_item_proposals` (`:138`): a detection
  matching (IoU at or above `LABEL_IOU_MATCH`) a geometry-locked item's `bbox_norm` or `detector_bbox_norm` is
  dropped; a geometry-locked item is never in `deletable` (`item_delete.py:43-75`) and never re-indexed with a
  new box. `open_vocab_run.py:441,456` and `dataset_import/reconcile.py:98` get the same predicate.
- `ExcludeReason = Literal['ignore', 'blurry', 'unidentifiable', 'not_the_subject', 'partial_crop', 'not_an_object']`
  in `src/schemas/curation/exclusion.py`; `CropExcludeRequest.reason: ExcludeReason = 'ignore'`
  (`_crop_write_models.py:31-41`).
- New `GET {prefix}/exclude_reasons` -> `ExcludeReasonList{reasons: [{id, label}]}` (D-21), served from the
  same module.
- Wire (`serialize_item`, `ItemDoc`): `geometry_revision: int` (0 when absent), `geometry_source:
  Literal['detector', 'human', 'import']`, `geometry_locked: bool`, `lock_label: str | None`.

### R-17 `GET {prefix}/images/locate` (2C)

- Query: `ItemFilterQuery` + `ImageViewQuery` (without `cursor` and `limit`) + `image_id: str` (required).
- Response `ImageLocate{image_id, in_view: bool, index: int | None, total, cursor_before, cursor_after, reason:
  Literal['not_found', 'filtered_out'] | None, reason_message: str | None}`.
- Index = a count of the images sorting before it (the `review_request.before_query` pattern, `:151`).
- Registered **before** `GET /images/{image_id}` in `images_view.py` (literal path first, the
  `clusters.py` note on `/clusters/representatives`).

### R-18 `GET {prefix}/table/columns`, `GET {prefix}/table` (2C)

```python
TableRows = Literal['detections', 'images']

class TableColumn(BaseModel):
    id: str
    label: str
    sortable: bool
    align: Literal['start', 'end', 'center']
    kind: Literal['thumbnail', 'text', 'number', 'date', 'badge']

class TableColumns(BaseModel):
    rows: TableRows
    columns: list[TableColumn]
    default: list[str]

class TableCell(BaseModel):
    text: str
    value: str | float | int | bool | None = None
    tone: Literal['validated', 'machine', 'unlabeled', 'holdout', 'rejected', 'muted'] | None = None

class TableRow(BaseModel):
    crop_id: str | None
    image_id: str
    thumbnail_url: str
    cells: dict[str, TableCell]

class TablePage(BaseModel):
    rows: TableRows
    columns: list[TableColumn]          # the requested ones, in order
    items: list[TableRow]
    next_cursor: str | None
    total: int
    limit: int
    sort_applied: str
    sort_options: list[SortOption]      # one per sortable column and direction
    hidden: HiddenCounts
    join_truncated: bool
    empty_reason: Literal['no_items', 'no_match'] | None
    empty_message: str | None
```

- `GET /table` query: `rows` (default `detections`), `columns` (comma list; unknown id 422 `unknown_column`;
  default the served `default`), `sort`, `cursor`, `limit` (1 to 200, default 50), `ItemFilterQuery`, and
  `ImageViewQuery` when `rows=images`.
- Detection columns: `thumbnail`, `class`, `detector_class`, `detector_confidence`, `label_source`, `validated`,
  `cluster`, `image_file`, `taken_at`, `area`, `rank`, `clarity`, `near_dup`, `embedding_state`, `updated_at`.
  Image columns: `thumbnail`, `file`, `taken_at`, `located`, `items`, `labeled`, `unlabeled`, `near_dup`,
  `split`, `indexed_at`.
- Detection rows reuse `build_crop_list_query` and `serialize_item`; image rows reuse `image_page`. Cells are
  built by `src/services/curation/table_cells.py` from the **wire** dicts, so a cell equals the wire value.
  Formatting of numbers and dates is server-side (`format_confidence`, `format_date` in that module).
- Files: `src/routers/curation/table_view.py`, `src/services/curation/table_view.py`,
  `src/services/curation/table_cells.py`, `src/schemas/curation/table.py`.

### R-19 `POST {prefix}/selection/summary` (2C)

- Body: `ItemSelection`. Response:

```python
class ClassCount(BaseModel):
    class_name: str           # 'other' bucket and 'unlabeled' are served words
    count: int

class SelectionSummary(BaseModel):
    detections: int
    images: int
    clusters: int
    locked: int               # is_locked_item OR is_locked_geometry, counted by a filters aggregation
    in_holdout: int           # always 0 unless include_test; kept so the bar can say so
    by_class: list[ClassCount]   # top 10, then 'other'
    taken_from: str | None
    taken_to: str | None
    located: int              # images with a served position
    join_truncated: bool
    summary_text: str         # '183 detections in 41 images · 7 locked, skipped'
```

- One aggregation on the items index over `selection_query(sel)` (`item_selection.py:74`): `cardinality`
  `image_id` and `cluster_uid`, `terms class_name` size 10 with `sum_other_doc_count`, `min/max capture_taken_local`,
  a `filters` aggregation for the lock fields. `located` from the images index over the selection's image ids
  (capped join). Over `MAX_BULK_SELECTION` (20,000) it refuses 422 like `_selection.py:51-54`.
- Files: `src/routers/curation/selection_summary.py`, `src/services/curation/selection_summary.py`, models in
  `src/schemas/curation/selection.py`.

### R-20 Shared filter additions (2B, `origin=drawn` 2D)

Section 3.1. `ItemFilter.cluster_uid` is removed. Callers swap to `cluster_uids` in the same commit:
`crops_list.py:58` (the route's own `cluster_uid` param is deleted, ItemFilterQuery serves it),
`assist/scope.py:76,86`, `autolabel/selection.py:77,189`, `autolabel/embed_stage.py:56`,
`semantic_search.py:113-114`, `embedding_viz_read.py:105-106`, `embedding_viz.py:197`,
`crop_orders.py:50,59`, `cluster_cards.py:97-99` (its `ClusterListParams.cluster_uid` stays, a card lookup, not a
filter), `cluster_facets.py:47`, `cluster_representatives.py:26`. Routes whose `cluster_uid` is a **scope** of a
job or search (`search.py:65,217`, `viz.py:71,144`, `assist.py:114`, `pipeline_start.py:65`,
`pipeline_public.py:65`, `select.py:43`, `regions.py:159`, `CropMoveRequest.cluster_uid`) keep their own param and
build `ItemFilter(cluster_uids=[uid])` where they build a filter.

### R-21 `GET {prefix}/map/cells` single-photo pins (2B)

`MapCellProperties` gains `image_id: str | None` and `thumbnail_url: str | None`, set when the cell holds
exactly one image (a `top_hits` size 1 sub-aggregation on the `geotile_grid`, only when `precision` equals
`max_precision(policy)`, so a pin never reveals a finer position than the policy).

### R-22 `POST {prefix}/classes` suggestion (2A)

A name that fails `CLASS_NAME_PATTERN` (`_class_models.py:62,65-70`) is refused **by the handler** (the
pattern constraint moves from the field to a validator in the service) with 422
`{error: 'invalid_class_name', message, suggested_name}`. `suggested_class_name(raw)` in
`src/utils/class_names.py`: lower-case, spaces and dashes to `_`, other characters dropped, repeated `_`
collapsed, trimmed; `None` when nothing is left. Tests: `Traffic light` -> `traffic_light`, `Stop-sign!` ->
`stop_sign`, `***` -> no suggestion.

### R-23 `GET {prefix}/viz/projection` filter (4C)

See 12d lane 4C.

### R-24 Served box display names (2C, new)

- Region box elements on the item wire (`region_boxes_to_wire`, `wire.py`) gain `display_name: str`: the
  box's read text when present, else `<slot label> <n>` where the slot label is the active region profile's
  served label and `n` is the box's 1-based order in the list. One function:
  `region_box_display_name(box, *, slot_label, ordinal) -> str` in `src/domain/region_boxes.py`.
- `BoxView` for `kind='region'` uses it as `label` with `label_kind='region'`; for items, D-16.
- Frontend: the region overlay labels ("Object 1/2/3", wf-1d finding) come from `display_name`.

### R-25 `GET {prefix}/crops/locate` (2E, new)

Query: everything `GET /crops` takes except `cursor`, `limit` and `ids`, plus `crop_id` (required). Response
`CropLocate{crop_id, in_view, index, total, cursor_before, cursor_after, reason: Literal['not_found',
'filtered_out'] | None, reason_message}`. Registered before `GET /crops/{crop_id}`.

### R-26 `GET {prefix}/exclude_reasons` (2D, new)

See R-16.

### R-27 `GET {prefix}/review/tabs` filter (3B)

See R-10: the route takes `ItemFilterQuery` for its counts.

---

## 10. Shared frontend modules

### 10.1 API client functions (one per route; the owner lane writes it)

| Function | File | Route | Owner |
|---|---|---|---|
| `listImages(query: ItemFilterQuery & ImageViewParams, cursor: string \| null, signal?)` | `frontend/src/lib/api_images.ts` (new) | R-1 | 3A |
| `getImage`, `getImageFileUrl`, `getImageThumbUrl` | `frontend/src/lib/api_cropRead.ts` (wf-1b) | R-2, R-3 | built; 3C moves them to `api_images.ts` in its first commit |
| `locateImage(query, imageId, signal?)` | `api_images.ts` | R-17 | 3C |
| `createImageItem(imageId, body: ItemCreateRequest)` | `api_images.ts` | R-14 | 3C |
| `patchItemGeometry(cropId, body)`, `undoItemGeometry(cropId)` | `frontend/src/lib/api_cropWrites.ts` | R-15 | 3C |
| `getExcludeReasons(signal?)` | `api_cropWrites.ts` | R-26 | 2A |
| `getCrops(query, cursor, signal?)`, `locateCrop(query, cropId, signal?)` | `api_cropRead.ts` | R-4, R-11, R-25 | 2E |
| `getReviewQueue(tab, query, cursor, signal?)`, `locateInReviewQueue` (changed) | `frontend/src/lib/api_review.ts:193` | R-11 | 2E |
| `getTableColumns(rows)`, `getTable(rows, columns, query, sort, cursor, signal?)` | `frontend/src/lib/api_table.ts` (new) | R-18 | 3F |
| `getSelectionSummary(sel: ItemSelection, signal?)` | `api_select.ts` | R-19 | 2A |
| `getNextSteps(signal?)` | `frontend/src/lib/api_workflow.ts` (new) | R-12 | 2A |
| `getMapCells` (changed), delete `getMapImages` | `frontend/src/lib/api_map.ts:22` | R-9, R-21 | 3D |
| `createClass` (changed to surface `suggested_name`) | `frontend/src/lib/api_classes.ts` | R-22 | 2A |
| `getProjection(query, ...)` (changed) | the existing viz client | R-23 | 4C |

Each call site is written `` `${scoped()}/path${qs({...})}` `` in a file listed in `SCANNED_FILES`
(`frontend/src/lib/contract/endpointCatalog.test.ts:33-73`; a new `api_*.ts` is added to that list in the same
commit). Types for each response are written once in the matching `types_<area>.ts` (`types_images.ts`,
`types_table.ts`, `types_workflow.ts`) and pinned by a `*Contract.test.ts` against
`$contracts/openapi/curation.json` (copy `frontend/src/lib/contract/nearDupContract.test.ts`).

### 10.2 Components

| Component | Path | Props (Svelte 5 `$props()`) | Owner |
|---|---|---|---|
| Workspace page | `frontend/src/routes/p/[project]/workspace/+page.svelte` | none; creates `WorkspaceController`, sets `keyboardStore.setScope` from the view entry | 2A |
| `FilterBar` | `frontend/src/lib/components/workspace/FilterBar.svelte` | `{ ws: WorkspaceController; served?: ReviewFilterSpec[] }`. Wraps `ItemFilterBar` (`components/itemFilter/ItemFilterBar.svelte:17-33`) plus chips, the served "N matching · M hidden" line (`HiddenItemsNote`) and the next-step chip. Under `md` it collapses behind one "Filters" button (G-25) | 2A |
| `ViewBar` | `.../workspace/ViewBar.svelte` | `{ ws }`. Tabs (`role="tablist"`) from `VIEWS` where `available`; a `<select>` under `md`; right side `SortSelect` and the thumb size | 2A |
| `SortSelect` | `.../workspace/SortSelect.svelte` | `{ options: SortOption[]; applied: string; fallbackReason?: string \| null; onchange: (id: string) => void }` | 2A |
| `WorkspaceRail` | `.../workspace/WorkspaceRail.svelte` | `{ ws }`. Queues (served tabs: label, count, `aria-current`), then Classes (served `count`, hotkey glyph, click = filter, drop target via `dropOnClassStore`), "+ New class" opens `ClassPicker` in create mode. A drawer under `lg` | 2A |
| `DetailsPanel` | `.../workspace/DetailsPanel.svelte` | `{ ws }`. Picks `DETAILS[kind]` from `ws.url.focus` or the selection; 360 px; a drawer under 1280 px | 2A |
| `BulkBar` | `.../workspace/BulkBar.svelte` | `{ ws }`. Grows from `ClusterDetailToolbar.svelte`: Assign (`ClassPicker`), Confirm, Move, Ignore (served reasons), Assist, Reprocess, Undo; shows `SelectionSummary.summary_text`; every action dry-runs through `createSelectionActionController` | 2A |
| `ClassPicker` | `frontend/src/lib/components/ClassPicker.svelte` + `frontend/src/lib/classes/classPicker.svelte.ts` | `{ open: boolean; mode: 'assign' \| 'create'; anchor?: HTMLElement \| null; recent?: string[]; onpick: (cls: RegistryClass) => void; onclose: () => void }`. Inside `Modal` (`align="top" layer="top"`), a combobox (`role="combobox"`, `aria-activedescendant`). Filters the bounded served registry by name; deprecated hidden; rows show the hotkey glyph and served count; the last row is `Create "<typed>"`; a 422 `invalid_class_name` turns it into `Create "<suggested_name>"` | 2A |
| `ViewState` | `frontend/src/lib/components/state/ViewState.svelte` | `{ status: 'loading' \| 'error' \| 'empty' \| 'ready'; message?: string \| null; action?: { label: string; href: string } \| null; children: Snippet }`. Loading uses a skeleton grid, never a spinner over content | 2A |
| `BoxLayer` | `frontend/src/lib/components/boxes/BoxLayer.svelte` | `{ boxes: ImageBox[]; selectedCropId?: string \| null; showLabels?: boolean; onselect?: (cropId: string) => void }`. An SVG with `viewBox="0 0 1 1"` and `preserveAspectRatio="none"` over a sized image | 2C |
| `ImageCard`, `ImageGrid` | `frontend/src/lib/components/images/` | `ImageCard { card: ImageCardWire; selected: boolean; focused: boolean; onclick; onopen }` | 3A |
| `DataTable`, `ColumnPicker` | `frontend/src/lib/components/table/` | see 12c lane 3F | 3F |
| `EditorCanvas`, `ObjectList`, `Filmstrip`, `ToolStrip` | `frontend/src/lib/components/editor/` | see 12c lane 3C | 3C |

---

## 11. Existing code to reuse, and what is deleted

### 11.1 Reuse (do not rewrite)

- Cluster triage: `clusterDetailController.svelte.ts` (public API `:366-515`), `clusterDetailActions.svelte.ts`
  (`:20`, keys `:264-330`), `ClusterDetailGrid` (`:15-25`), `ClusterFacetBar`, `ClusterDetailFilters`,
  `MoveClusterPicker`, `ClusterProposalPanel`, `ClustersGrid` (`:21-47`), `ClusterGridCard`, `ClusterRunPanel`,
  `clusterListController.svelte.ts`, `clusterListFilter.ts` (`:25-99`, its URL helpers move into
  `workspaceUrl.ts` callers unchanged).
- Review: every controller under `frontend/src/lib/review/` (list in plan §8), `ReviewCropPane.svelte`
  (`:30-39`), `reviewKeyboard.svelte.ts` (`:50,62,98`), `reviewTabsVocabularyStore`.
- Matching items and bulk: `MatchingItemsView.svelte`, `createMatchingItems` (`matchingItems.svelte.ts:19`),
  `SelectionActionDialog.svelte`, `selectionActionController.svelte.ts`.
- Map: `MapController` (`mapController.svelte.ts:22-140`), `MapView.svelte` (`:24-31`).
- Embedding: `EmbeddingPlot.svelte` (`:55-77`).
- Box editing: `MultiBoxCanvas.svelte` (`:36-56`, `handleKey` `:219`), `annotations/multiBox.ts` (`:16-146`).
- Undo: `undoStore` (`stores/undo.svelte.ts:62-253`).
- Backend: everything in section 2's backend column.

### 11.2 Deleted (in the lane named, no redirects, no aliases)

| Deleted | Lane |
|---|---|
| `frontend/src/routes/p/[project]/clusters/` (both pages and their tests, moved to the Workspace tests) | 2A |
| `ClusterDetailToolbar.svelte` (becomes `BulkBar`), the toolbar `<select>` class picker | 2A |
| `clusters_search` context, if no action remains | 2A |
| `EXCLUDE_REASONS` array and frontend `ExcludeReason` type | 2A (after 2D's route) |
| `ItemFilter.cluster_uid`, the `/crops` own `cluster_uid` param | 2B |
| `page`, `page_size`, `guard_page_depth`, `MAX_RESULT_WINDOW`, `Pager.loadPage` | 2E |
| `frontend/src/routes/p/[project]/review/`, `ClassPickerModal.svelte`, `reviewClassPicker.svelte.ts`, `GET /audit/queue`, `AuditQueueList.svelte`, the non-queue `/review/{tab}` paths | 3B |
| `frontend/src/routes/p/[project]/map/`, `GET /map/images`, `MapTileImages`, `getMapImages`, `MapController.openCell/listQuery/listBody` | 3D |
| `frontend/src/routes/p/[project]/{audit,near-duplicates,label-ops}/` (bodies move to Quality) | 3E |
| `SourceImageOverlay.itemLabelInfo` and its colour rules | 2C |
| `EmbeddingPlot` mount in `ClustersGrid` (moves to the Embedding view) | 4C |

---

## 12. Visual rules (keep the Cropwright look, make it professional)

- Tokens only (`frontend/src/app.css:13-67`): surfaces `--bg-base`, `--bg-elevated`, `--bg-hover`; borders
  `--border-subtle`, `--border-strong`; text `--text-primary/secondary/tertiary`; control heights `--ctl-h-md/sm/xs`
  and radii `--ctl-radius-*`. Classes `.btn`, `.btn-primary`, `.btn-sm`, `.btn-icon`, `.select`, `.input`, `.chip`,
  `.surface`. No new colour literal; tones map to `--system-green` (validated), `--system-orange` (machine),
  `--system-gray-3` (unlabeled), `--system-indigo` (holdout), `--system-red` (rejected) in `boxTone.ts` only.
- Layout grid: app bar, filter bar 48 px, view bar 40 px, rail 240 px, details 360 px, bulk bar 48 px sticky at the
  bottom of the main pane. Spacing on Tailwind's 4 px scale: 8 between controls, 12 inside cards, 16 between
  sections. Every row aligns to the same left edge as the rail's content.
- Focus: the global `:focus-visible` ring (2 px `--system-blue`, offset 2). The focused card also gets
  `aria-current="true"` and a 2 px inset ring so focus and selection are distinct (selection = filled check
  badge top-left plus `--bg-hover` background).
- Keyboard hints: tooltips and button labels print `keymapStore.glyph(id)`; the bulk bar shows the glyph after
  each label in `--text-tertiary`.
- States: every view renders through `ViewState`: skeleton cards on first load, the served `empty_message` and
  one action link when empty, `apiErrorText` plus Retry on error. Counts unknown render through `formatCount`.
- 430 px: rail and details become drawers (non-modal `<aside>` panels over the main pane with a close
  button and an accessible name; Esc closes them through `workspace.escape`; not `Modal`, which traps focus),
  the view bar becomes a `<select>`, the filter bar one "Filters" button, the bulk bar
  sticks to the bottom and wraps to two rows. No horizontal page scroll (`test_narrow_viewport_430.py` probe);
  the Table scrolls inside its own container.

---

## 13. Standards gate (copy into every lane's done-when)

- [ ] `.venv/bin/pre-commit run --files <every file the lane touched>` is clean, with no hook bypassed.
- [ ] Python: ruff check and format, mypy, bandit clean. The ratchets do not grow: file size 700 (split by
  concern, never trim), layering (routers > services > clients; no `fastapi` below routers), complexity C901 15,
  broad-except, `noqa` only with a reason, vulture, process ids (no lane, wave or R-n ids in code comments or
  test names), cited paths.
- [ ] Frontend: `npx prettier --check`, `npx eslint` and `npx knip` clean on the touched files; `npm run check` 0
  errors and 0 warnings.
- [ ] No duplicated logic or test setup. List by name, in the hand-back, every helper, model, component or fixture
  that was reused or extracted, and every duplicate removed. A grep for the new function's core expression finds
  one definition.
- [ ] No dead code, no shims, no commented-out code; replaced code deleted in the same commit; `contracts/`
  regenerated and committed with the route change.
- [ ] Red first: each behaviour test was seen failing on the parent commit (paste the failing line in the commit
  body).

## 14. Conflicts found in the wave-1 worktrees (for the coordinator)

| # | Conflict | Resolution |
|---|---|---|
| C-1 | Two `HiddenCounts` models: wf-1a `src/routers/curation/_item_models.py` (project-wide counts from `hidden_counts`) and wf-1b `src/schemas/curation/images.py` (page sums from `item_facts`). Same name, so the OpenAPI schema names collide, and the meanings differ | D-15. One model in `src/schemas/curation/views.py`; filter-aware counts; 2B, unless fixed at the wave-1 merge |
| C-2 | Box label rule: wf-1b `image_boxes.item_label` gives the class name or "unlabeled"; wf-1d `wire._display_name` gives class, proposed or detector names; wf-1d `SourceImageOverlay.itemLabelInfo` computes labels client-side from items, while wf-1b changed the same file to read by `imageId`. Textual merge conflict in `SourceImageOverlay.svelte` and two rules | D-16, R-24. wf-1b and wf-1d are merged (`1685723a9`); 2C makes `BoxView.label` use the display name, adds `BoxLayer` and deletes `itemLabelInfo` |
| C-3 | wf-1b `GET /images` takes `geo_bbox` and `has_location` as its own params, not through the shared filter (plan R-9) | 2B moves them into `ItemFilter` |
| C-4 | wf-1d `ProjectPickerController` filters `selectable` in the browser after a served `limit`, so `total` and the shown rows disagree | RESOLVED in wf-1d follow-up 66312b08b + 890f70574 (merged bb8e8311f): `GET /projects?selectable=true`, served `total` counts the same set, client drop deleted |
| C-5 | wf-1a removed `RegistryClass.cluster_size` and the rail uses `count`; the plan says `item_count` | the specs use the served `count` |
| C-6 | Plan R-14 enqueues `reprocess`, which has an exclusive group | D-8 |
| C-7 | Plan R-11 and R-18 name a point-in-time | D-1 |
