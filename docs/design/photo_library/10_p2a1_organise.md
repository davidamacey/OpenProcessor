# P2a-1: Organise: timeline, albums, smart albums, Places (phase spec)

| Field | Value |
|---|---|
| Status | plan, 2026-11-13 (replaces the single P2a spec after the owner approved splitting it). Starts after the P1 and P2a-0 exits and the owner's answer to Q2 ([90](90_owner_questions.md)) |
| Master plan | [00_photo_library_plan.md](00_photo_library_plan.md): rules, data model (6), routes (7), UX standards (9), budgets (10), tests and eval sets (11) |
| Builds on | [05_p2a0_metadata.md](05_p2a0_metadata.md) (reader, overrides, keywords, tracks, projection drain, operations) |
| People | not here: [15_p2a2_people.md](15_p2a2_people.md) |

Notation: `{L}` = `/curation/projects/{project}/library`. Frontend paths starting with `lib/` or `routes/`
are under `frontend/src/`.

---

## 1. Scope and non-goals

**In scope.** Timeline (buckets, pages, scrubber, locate, viewer with the served info panel); favourites (a
heart, and a system Favourites album that cannot be deleted), pick/reject flags, archive, hidden, trash with
retention; albums; smart albums and saved searches with nested any/all/none rules (including keywords,
favourites, place, camera, date, objects, text); Places (offline reverse geocoding, place lists, the map,
**tracks on the map** with a time slider and the photos taken along them); combined search with type-ahead
and facets; the Duplicates, Hidden, Archived and Trash utility views; the editing UI for P2a-0's metadata
(info-panel edits, batch capture-time shift dialog, keyword editor, GPX upload and geotag); the 5M scale
replica.

**Non-goals.** People and anything person-based (P2a-2); exclusions, stacks, culling (P2b); events,
calendar, memories (P2c); video playback (P3); the later items of 00 section 2.2.

**Needs.** P1: N3, N4, N5, N8, N9, N10 (N1/N7 are read by P2a-2). P2a-0: everything.

---

## 2. Tables and migrations

Revision `<next>_library_organise.py` (lane L-A1, first commit): `photo_annotations`, `albums`,
`album_members`, `smart_albums`, `geo_places` (00 section 6.1), forced RLS for the project tables;
`library_operations.kind` gains `annotate`, `album_add`, `album_remove`, `trash`, `restore`. OpenSearch field
modules `library_annotation_fields.py` (`lib.*`), `library_place_fields.py` (`place.*`),
`library_object_fields.py` (`objects.*`), and `capture.taken_month_day`/`taken_year` in `capture_fields.py`.

---

## 3. Routes and wire models

Routers: `src/routers/curation/library.py` (summary fields, timeline, photos, neighbours, annotations,
trash), `library_collections.py` (albums, smart albums, rules schema), `library_places_search.py`. Every
router depends on `require_library` (P2a-0). Models in `src/schemas/curation/library_*.py`.

### 3.1 Shared models

| Model | Fields |
|---|---|
| `LibraryFilter` (query params on every list) | `taken_from`, `taken_to`, `on_this_day: bool`, `class_name: list[str]` (all of, by name), `place: list[str]` (`country:PT`, `admin1:PT.14`, `locality:<geonames_id>`), `geo_bbox`, `camera_model: list[str]`, `lens_model: list[str]`, `keyword: list[str]` (paths, with descendants), `favourite: bool`, `pick: Literal['pick','reject','none']`, `album: str`, `not_in_album: bool`, `has_location: bool`, `text: str` (OCR; quoted = exact phrase), `smart_album: str`, `visibility: Literal['default','favourites','archived','hidden','trash']` (default excludes archived, hidden, trashed). P2a-2 adds `person`, `person_any` |
| `LibrarySelection` | exactly one of `image_ids` (max 5,000) or `filter` with optional `buckets: list[str]`; never an exclusion list |
| `LibraryOpResponse` | `dry_run`, `n_photos`, `changes: dict[str, int]`, `op_id`, `job_id` |
| `PhotoTile` | `image_id`, `aspect`, `thumbhash`, `thumbnail_url`, `taken_local`, `favourite`, `pick`, `badges: list[Literal['archived','hidden','trashed','location','edited']]`, `alt` (served: date, place) |

### 3.2 Routes

| Route | Request | Response | Notes |
|---|---|---|---|
| `GET {L}` | | `LibrarySummary` (P2a-0) + `favourites`, `albums`, `places_total` | |
| `GET {L}/timeline/buckets` | `granularity: Literal['year','month','day']`, `LibraryFilter` | `TimelineBuckets {granularity, buckets: list[{key, label, count, first_taken_local, last_taken_local}], undated_count, total, revision}` | `date_histogram` on effective `capture.taken_local` |
| `GET {L}/timeline/photos` | `bucket`, `cursor`, `limit ≤ 500`, `order`, `LibraryFilter` | `TimelinePhotos {bucket, photos: list[PhotoTile], next_cursor, count}` | `search_after` on (`taken_local`, `image_id`) |
| `GET {L}/timeline/locate` | `date` or `image_id`, `granularity`, `LibraryFilter` | `TimelineLocate {bucket, index}` | |
| `GET {L}/photos/{image_id}` | | `PhotoDetail {image_id, file_name, folder, source_state, width, height, capture: CaptureWire (effective) and its sources, place, objects: list[{class_name, count}], text: {ocr, available}, title, caption, keywords, albums, annotations, signals: dict[str, {value, model, version}], track_id, near_dup_group_id, preview_url, file_url}` | positions omitted when the policy forbids serving them |
| `GET {L}/photos/{image_id}/neighbours` | `context` (`timeline`, `album:<id>`, `smart_album:<id>`, `search:<token>`, `track:<id>`), `LibraryFilter` | `PhotoNeighbours {previous_image_id, next_image_id, position, total}` | |
| `GET {L}/photos/{image_id}/preview` | | bytes, immutable cache headers | N3 |
| `POST {L}/photos/annotations` | `{selection, set: {favourite?, pick?, archived?, hidden?}, dry_run}` | `LibraryOpResponse` | write-through with the revision rule of 00 section 3.1 |
| `POST {L}/photos/trash`, `/restore` | `{selection, dry_run}` | `LibraryOpResponse` | |
| `GET {L}/trash` | `cursor` | `TrashPage {photos: list[PhotoTile & {trashed_at, days_left}], retention_days, next_cursor, count}` | |
| `POST {L}/trash/empty` | `{dry_run, confirm_count}` | `TrashEmptyResponse {n_photos, n_items, n_faces, job_id}` | deletes index docs and derivatives through the OCC delete path (`src/services/curation/item_delete.py`); never a source file |
| `GET/POST {L}/albums` | `{title, description?, sort}` | `Album {album_id, kind: Literal['manual','favourites'], title, cover_url, sort, member_count, revision}` | the `favourites` album is served, evaluated from `lib.favourite`, and cannot be edited or deleted |
| `GET/PATCH/DELETE {L}/albums/{album_id}` | `{title?, description?, cover_image_id?, sort?, expected_revision}` | `Album` | |
| `GET {L}/albums/{album_id}/photos` | `cursor` | `TimelinePhotos` | |
| `POST {L}/albums/{album_id}/photos`, `/remove` | `{selection, dry_run}` | `LibraryOpResponse` | fractional positions |
| `POST {L}/albums/{album_id}/photos/reorder` | `{image_ids, after_image_id \| None, expected_revision}` | `Album` | |
| `GET {L}/rules/schema` | | `RuleSchema {version: 'library_rule/1', fields: list[{field, label, value_kind, operators, value_source_url}], max_depth: 3, max_conditions: 50}` | |
| `GET/POST {L}/smart_albums`, `GET/PUT/DELETE {L}/smart_albums/{smart_album_id}`, `GET .../photos`, `POST {L}/smart_albums/preview` | `{title, kind, rule, pinned}` | `SmartAlbum {smart_album_id, title, kind, rule, count, cover_url, revision}`; `RulePreview {count, photos}` | `422 rule_invalid` with the JSON path |
| `GET {L}/places` | `level`, `parent`, `LibraryFilter` | `Places {level, places: list[{key, name, photo_count, cover_url}]}` | |
| `GET {L}/search` | `q`, `example_image_id`, `LibraryFilter`, `cursor` | `LibrarySearch {photos: list[PhotoTile & {score}], next_cursor, query_echo, count_estimate}` | frame-space kNN (N9) with the filter inside `knn` |
| `GET {L}/search/suggest` | `q` | `SearchSuggest {items: list[{kind: Literal['place','keyword','album','class','camera','text'], label, filter}]}` | P2a-2 adds `person` |
| `GET {L}/facets` | `fields`, `LibraryFilter` | `Facets {fields: dict[str, list[{value, label, count}]]}` | `year`, `class_name`, `place_country`, `camera_model`, `lens_model`, `keyword` |
| `GET {L}/photos/by_content/{image_id}` | | `{image_id, present}` | |

### 3.3 Rule grammar `library_rule/1` (`src/domain/library_rules.py`, pure)

```json
{"match": "all", "rules": [
  {"field": "taken_at", "op": "between", "value": ["2019-06-01", "2019-08-31"]},
  {"field": "favourite", "op": "is", "value": true},
  {"group": {"match": "none", "rules": [
    {"field": "class_name", "op": "contains_any", "value": ["car"]},
    {"field": "keyword", "op": "contains_any", "value": ["Work"]}]}}]}
```

| Field | Operators | Image doc field |
|---|---|---|
| `taken_at` | `between`, `before`, `after`, `last_n_days`, `on_this_day` | effective `capture.taken_local`, `capture.taken_month_day` |
| `year` | `in` | `capture.taken_year` |
| `class_name` | `contains_any`, `contains_all`, `contains_none` | `objects.class_names` |
| `text` | `contains` | `text.ocr` |
| `title_caption` | `contains` | `meta.title`, `meta.caption` |
| `keyword` | `contains_any`, `contains_all`, `contains_none` (with descendants) | `meta.keyword_paths` |
| `place` | `in`, `not_in` | `place.*` |
| `geo` | `within_bbox`, `within_polygon` | `capture.location` |
| `camera_model`, `lens_model` | `in`, `not_in` | `capture.*` |
| `favourite`, `archived` | `is` | `lib.*` |
| `pick` | `in` | `lib.pick` |
| `album` | `in_any`, `in_none` | `lib.album_ids` |
| `has_location` | `is` | `capture.has_location` |

Depth 3, 50 conditions; semantic text only in saved searches (a kNN result is not a stable set); unknown
fields or operators raise `RuleError` with the path (fail closed). P2a-2 adds `person`; P2b adds exclusion
visibility; P2c adds `event`, `trip`, `calendar_event`, `holiday`.

### 3.4 Places and tracks

| Concern | Design |
|---|---|
| Reverse geocoding | GeoNames `cities500`, `admin1CodesASCII`, `countryInfo` (CC BY, credited in `ATTRIBUTION.md`) loaded into `geo_places` by `library_geonames_load` from a file fetched by `scripts/library/fetch_geonames.py`; nearest place with SciPy `cKDTree` in the worker (keyed by `dataset_version`, never in the API); written as `place.*` by the drain; `drop` writes none, `store_coarse` keeps admin levels, `location_purge` clears them |
| Map | existing `{P}/map/cells` with the library filter; tracks from P2a-0's `{L}/tracks` drawn as lines (MapLibre GeoJSON source) with a time slider: the served points give the marker position; the photos along the track (served) are pins that open the viewer |
| Places list | country → admin1 → locality drill-down with counts and covers |

---

## 4. Jobs

| Job or tick | Queue | Singleton |
|---|---|---|
| projection drain (P2a-0 tick, extended by L-A1 with `objects`, `place`, `lib`, `capture.taken_*`) | `periodic` | advisory `LIBRARY_PROJECTION_DRAIN` |
| `library_bulk_op` | `library` | none |
| `library_trash_empty`; trash expiry tick | `library`; `periodic` | `exclusive_group='library_trash'`; advisory `LIBRARY_TRASH_EXPIRE` |
| `library_geonames_load`, `library_places_backfill` | `library` | `exclusive_group='library_places'` |

---

## 5. Cropwright components

| Piece | Path (new unless marked) | Reuses |
|---|---|---|
| Library layout and sidebar | `routes/p/[project]/library/+layout.svelte`, `lib/components/library/LibrarySidebar.svelte` (wraps P2a-0's Metadata page unchanged) | `ui/SidebarNav.svelte`, `ui/PageShell.svelte` |
| Nav | `lib/projectPaths.ts` (edit): `library` group first when the served `kind` is `library` | |
| URL grammar | `lib/library/libraryUrl.ts` | pattern of `lib/workspace/workspaceUrl.ts` |
| API and types | `lib/api_library.ts`, `lib/api_libraryCollections.ts`, `lib/types_library*.ts` | `lib/api.ts`; contract test pattern of `lib/contract/` |
| Timeline store, grid, scrubber, tile | `lib/library/timelineStore.svelte.ts`, `lib/components/library/{PhotoGrid,JustifiedSection,TimelineScrubber,PhotoTile}.svelte` | `justified-layout`, `virtua/svelte`, `thumbhash` (exact pins; README and changelog read and cited) |
| Selection, bulk bar | `lib/library/librarySelection.svelte.ts`, `lib/components/library/LibraryBulkBar.svelte` | `lib/selection.svelte.ts` (extended), the `BulkBar.svelte` pattern, `ui/Menu.svelte` |
| Viewer and info panel | `lib/components/library/{PhotoViewer,InfoPanel}.svelte`; the panel shows effective values and their sources, with edit actions for time, position, title, caption and keywords | `ui/Lightbox.svelte`, `components/Filmstrip.svelte` |
| Metadata editing | `lib/components/library/{CaptureTimeDialog,LocationDialog,KeywordEditor,GpxUploadDialog}.svelte`, `lib/api_libraryMetadata.ts` (P2a-0) | `components/Modal.svelte` |
| Collections | `lib/components/library/{AlbumGrid,AlbumPage,SmartAlbumEditor,RuleBuilder,KeywordTree}.svelte` | `ui/Popover.svelte`, `ui/Drawer.svelte` |
| Places and tracks | `lib/components/library/{PlacesView,TrackLayer,TrackTimeSlider}.svelte` | `components/map/**`, `lib/map/mapController.svelte.ts` |
| Search | `lib/components/library/{SearchBar,SearchSuggest,FacetPanel}.svelte` | `lib/searchBox.svelte.ts` |
| Utilities | Duplicates, Hidden, Archived, Trash pages | `components/nearDup/NearDupGroupModal.svelte`, `lib/nearDup/**` |
| Keymap | `contracts/json/keymap_actions.json` source gains `library.*` actions; `lib/keymapFallback.ts` | |

---

## 6. Lanes, order and verify commands

```
wave 1:  L-A1 schema + wire models FIRST COMMIT (contracts generated), then drain extension and annotations/trash
         || L-A3 query, timeline, rules, smart albums, facets || L-A5 albums || L-A6 places and search || L-S1 scale replica
wave 2:  L-F1 shell, timeline, viewer, info panel, metadata dialogs || L-FIX (reds of wave 1)
wave 3:  L-F2 albums, smart albums, keywords, utilities || L-F3 places, tracks, search
then:    sweep, budgets on the 500k real-EXIF set and the 5M replica, owner walk
```

Frontend lanes start only after L-A1's wire models are merged (rule 4). Writes once: L-A3 owns `query.py`
and every list filter; others call `query.build()`. L-F1 owns `PhotoGrid` and `LibraryBulkBar`; L-F2 and
L-F3 add actions through the `more: MenuItem[]` prop only.

| Lane | Model, effort | Owns | Verify (paste the output) |
|---|---|---|---|
| L-A1 | Sonnet, high | migration, `src/db/models/library.py`, repos, `src/schemas/curation/library_*.py` (all P2a-1 models), field modules, drain extension, `src/services/library/{annotations,trash}.py`, their routes | `.venv/bin/python -m pytest tests/library/test_annotations.py tests/library/test_annotation_writer_order.py tests/library/test_trash_empty_sources.py -q --no-cov && .venv/bin/python scripts/codegen/generate_contracts.py --check` |
| L-A3 | Sonnet, high | `src/services/library/{query,timeline,facets,smart_albums}.py`, `src/domain/library_rules.py`, `src/domain/library_visibility.py`, timeline and smart-album routes | `.venv/bin/python -m pytest tests/domain/test_library_rules.py tests/library/test_timeline.py tests/library/test_no_image_join.py -q --no-cov` |
| L-A5 | Sonnet, medium | `src/services/library/albums.py`, album routes, the favourites album | `.venv/bin/python -m pytest tests/library/test_albums.py -q --no-cov` |
| L-A6 | Sonnet, medium | `src/services/library/{places,search}.py`, `scripts/library/fetch_geonames.py`, places and search routes | `.venv/bin/python -m pytest tests/library/test_places.py tests/library/test_search.py -q --no-cov` |
| L-S1 | Sonnet, medium | `scripts/bench/library_build_scale_replica.py`, `scripts/bench/library_bench_budgets.py` (00 sections 10 and 11.6) | `.venv/bin/python -m pytest tests/scripts/test_scale_replica.py -q --no-cov` (unit, on a 1k fixture) |
| L-F1 | Sonnet, high | layout, sidebar, `lib/projectPaths.ts`, timeline, grid, scrubber, tile, selection, bulk bar, viewer, info panel, metadata dialogs, `frontend/package.json` (three deps) | `cd frontend && npm run check && npx vitest run src/lib/library` plus the stubbed e2e files of section 7 |
| L-F2 | Sonnet, medium | collections components and routes, utilities pages | same pattern, its files |
| L-F3 | Sonnet, high | places, tracks, search components and routes | same pattern, its files |

---

## 7. Tests to write red first

| Lane | Tests |
|---|---|
| L-A1 | `tests/library/test_annotations.py` (favourite, pick, archive, hidden, undo keeps a later edit); `test_annotation_writer_order.py` (a stale drain never overwrites a newer write-through; both orders); `test_trash_empty_sources.py` (source tree byte-identical, `confirm_count` required); `tests/db/test_library_organise_schema.py` (RLS, wave run) |
| L-A3 | `tests/domain/test_library_rules.py` (every field and operator, nesting, limits, `RuleError` path, semantic refused in smart albums); `tests/library/test_timeline.py` (bucket counts equal page totals, undated bucket, stable `search_after`, locate; effective time after a P2a-0 override); `test_no_image_join.py` (AST) |
| L-A5 | `tests/library/test_albums.py` (fractional order, favourites album read-only and live) |
| L-A6 | `tests/library/test_places.py` (nearest place, policy behaviour, purge); `test_search.py` (filter inside `knn`, quoted OCR phrase) |
| L-F1 | `e2e/stubbed/library/test_timeline_grid.py` (≤ 600 tiles in the DOM after 20k stub photos); `test_scrubber.py`; `test_no_scroll_jump.py`; `test_library_keys.py`; `test_range_selection.py`; `test_capture_time_dialog.py` (dry-run samples shown, apply, undo toast) |
| L-F2 | `e2e/stubbed/library/test_rule_builder.py`; `test_albums.py`; `test_keywords.py` |
| L-F3 | `e2e/stubbed/library/test_places_tracks.py` (track line drawn, slider moves the marker to the served point, photo pins open the viewer); `test_search_facets.py` |

Pending wave runs: ingest 1k real-EXIF photos into a library on the live harness → timeline, places, tracks;
trash empty with the checksum manifest.

---

## 8. Acceptance walk (owner, own library, 1600 and 430 px)

1. Photos: placeholders then thumbnails; scrub to the oldest and newest years; no jump; open a photo, step
   with arrows; the info panel shows effective time, place, objects, text, keywords and where each came from.
2. Heart ten photos; the Favourites album shows them; flag two as rejects; undo once.
3. Fix a capture time from the info panel; batch-shift a camera's day from the dialog; the timeline moves them.
4. Make an album, reorder; make a smart album "favourites in a country, summer, keyword X, not in album Y";
   ingest more photos; they appear without editing the rule.
5. Places: country → city; a GoPro or drone track on the map; drag the time slider; the marker follows and
   the photos along the track open.
6. Search "beach" + place + year; facets narrow; a quoted OCR phrase matches.
7. Trash 10 photos, empty the trash; the source checksum manifest is unchanged.
8. Budgets of 00 section 10 on the 500k real-EXIF set and the 5M replica, with the bench header.
