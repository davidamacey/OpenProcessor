# 12c Wave 3 lane specs: the views, Review, the image editor, the map and Quality

| Field | Value |
|---|---|
| Status | spec; wave 3 starts after wave 2 merges and its sweep is green |
| Interfaces | [12a_workflow_interfaces.md](12a_workflow_interfaces.md) (`12a §n` below) |
| Rules | the "Rules every lane follows" block of [12b_wave2_lane_specs.md](12b_wave2_lane_specs.md) applies unchanged |
| Lanes | 3A Gallery and image details, 3B Review view and the queue cut, 3C Image annotation editor, 3D Map view, 3E Quality page and next steps, 3F Table view |
| Shared files | every view lane appends one entry to `frontend/src/lib/workspace/viewRegistry.ts` and, where it has one, `detailsRegistry.ts`; keymap lanes append to `keymapFallback.ts` and `src/config/keymap_actions.json`; lanes that delete a page edit the nav in `routes/p/[project]/+layout.svelte` and `projectPaths.ts`; docs lanes add a line to `docs-site/sidebars.ts`. These are additive and the coordinator resolves them at merge |

---

## Lane 3A: Gallery view and image details

**Size:** M, 3 days. **Merge:** first in wave 3 (3C and 3D use its `ImageCard`, `listImages` and `types_images.ts`).

### Goal and outcome

After ingest the user opens the Workspace on Gallery (the served default) and sees whole pictures with every box
drawn from served coordinates, served count chips (items, labelled, unlabelled), a near-duplicate badge and a
"Has unlabelled objects" chip that finds images needing work. Selecting images acts on their matching items. The
crop dialog's siblings open their crop and "Open image" goes to the editor.

### Files

Create: `frontend/src/lib/api_images.ts` (`listImages`), `frontend/src/lib/types_images.ts` (`ImagePage`,
`ImageCardWire`, `ImageViewParams`; move `ImageBox`, `ImageCounts`, `ImageInfo`, `ImageDetailResponse` from
`types.ts` (wf-1b) here), `frontend/src/lib/images/imageQuery.ts`, `frontend/src/lib/images/imageGalleryController.svelte.ts`
+ test, `frontend/src/lib/components/images/{ImageCard,ImageGrid}.svelte` (drawing through 2C's `BoxLayer`) + tests, `frontend/src/lib/workspace/views/GalleryView.svelte`,
`frontend/src/lib/workspace/details/ImageDetails.svelte`, `frontend/src/lib/contract/imagesContract.test.ts`,
`frontend/src/lib/images/imageList.scan.test.ts`, `frontend/src/lib/test/fixtures/imagePage.ts`,
`frontend/e2e/stubbed/test_gallery_view.py`, `docs-site/docs/cropwright/user-guide/gallery.md`.

Change: `frontend/src/lib/components/CropMetaPanel.svelte`,
`frontend/src/lib/components/cropMeta/CropSourceContext.svelte:74-84` (siblings are buttons opening that crop;
"Open image" link; drop the duplicate small source block), `viewRegistry.ts`, `detailsRegistry.ts`,
`frontend/src/lib/contract/endpointCatalog.test.ts` (`SCANNED_FILES`), `test_a11y_axe.py`, `test_narrow_viewport_430.py`,
`frontend/e2e/live/{test_route_sweep,test_data_agreement}.py`.

### Steps and commits

1. `refactor(frontend): image types in one module` (move only).
2. `feat(frontend): image gallery client and controller`. Red: `imageGalleryController.test.ts::keeps the served order`, `::appends by next_cursor`, `::shows the served empty message`, `::selection of images builds a filter with image_ids`.
3. `feat(frontend): Gallery view, image card and image details`. Red: `ImageCard.test.ts::count chips are the served counts`, `::near-dup badge only when served`, `::boxes_truncated says so`.
4. `fix(frontend): crop dialog siblings open their crop and the image opens in the editor` (G-06). Red: `CropMetaPanel.test.ts::a sibling click opens that crop`.
5. `docs(cropwright): gallery page; changelog`.

### Interfaces

Consumes R-1, R-2, R-3, R-24 (labels), 12a §5 (`images` selection), §6, §10.2 (`ViewState`, `SortSelect`), the
`label_state`, `has_location`, `in_holdout`, `near_dup` params (`ImageViewQuery`), R-12 default view, `BoxLayer` and
`boxTone` (2C). Provides `ImageCard`, `ImageGrid`, `listImages`, `types_images.ts`.

### Other tiers

- `imageList.scan.test.ts`: the gallery files contain no `.sort(`, `.toSorted(`, `.reverse(` (copy
  `clusterList.scan.test.ts`).
- Stubbed `test_gallery_view.py`: Gallery is the default without a partition (stub `GET /next_steps`); set a class
  filter in Clusters, switch to Gallery, the chips carry; select 2 images, assign a class, the stub receives
  `selection.filter.image_ids` and the confirm shows the dry-run count; "Has unlabelled objects" sends
  `label_state=has_unlabeled`; 430 px no overflow. a11y rows `workspace-gallery`, `workspace-gallery-details`.
- Live tier (coordinator): Gallery total equals `GET /images` `total` on `coco-e2e` (91 minus hidden, hidden shown);
  every visible card draws exactly the served number of boxes.

### Visual checks

Gallery at 1600 and 430 on the stub and `coco-e2e`: a busy image (the COCO street scene), an image with no
items, the near-dup badge, "Has unlabelled objects" on, the image details panel. "Right": a reviewer can spot
images needing work at a glance (the unlabelled chip stands out in `--system-orange`); boxes align with objects at
every thumb size; labels never cover the whole object (labels hide below 96 px card width; served
`boxes_truncated` shows "+N").

### Docs and changelog

`gallery.md` (what it is for, the chips, select and act, open in the editor). Frontend CHANGELOG Added: "Gallery
view of whole images with served boxes and counts; crop dialog siblings open their crop."

### Risks and do not

- Never fetch full-resolution pixels in the grid: `card.thumbnail_url` only.
- No client join of items to images; the card's counts and boxes are served.

### Done when

- [ ] Red tests seen failing then green; overlay callers' tests pass.
- [ ] Screens viewed and judged against "see what came in, find images that need work".
- [ ] Docs, changelog, docs check.
- [ ] Standards gate (12a §13).

---

## Lane 3B: Review view and the queue cut

**Size:** L, 4 days. **Merge:** second in wave 3.

### Goal and outcome

Review is a view of the Workspace: the two panes of today's Review over the same query and order as the grid.
`Enter` on any grid card opens Review at that card; `Esc` returns to the same card. The rail shows five plain
queues (plus the region queue when a profile is active), each a filter preset for every view with a served count
under the current filter. Review uses the shared class picker. The `/review` page is deleted.

### Files

Backend change: `src/services/curation/review_tabs.py:16-29`, `src/services/curation/review_tab_clauses.py`
(new `to_label`, `check_machine`, `disagreements`, `audit` clauses; `SIMPLE_TABS :467-478` keeps the old ids as
filter values), `src/services/curation/review_sorts.py:376-449` (defaults per queue),
`src/services/curation/review_tab_status.py` (wf-1a; counts under the filter; absent when unavailable),
`src/routers/curation/review.py:100-118` (`/review/tabs` takes `ItemFilterQuery`; non-queue tab 404 `not_a_queue`),
`src/routers/curation/audit.py` (delete `GET /audit/queue`), `src/services/curation/workflow_steps.py` (a missing
queue's reason as a next step), tests `tests/curation/test_review_queues.py`.

Frontend create: `frontend/src/lib/workspace/views/ReviewView.svelte`, `frontend/src/lib/review/reviewSource.svelte.ts`
(chooses `/review/{queue}` or `/crops` with the Detections sort, D-18), `frontend/src/lib/components/review/Filmstrip.svelte`
(served page around the focused item; shared with 3C through props), `frontend/e2e/stubbed/test_review_view.py`.

Frontend change: `frontend/src/lib/review/reviewTabsController.svelte.ts` (reads `ws.url.queue` and `focus`, no
own URL writes), `reviewFilterController.svelte.ts` (the shared filter comes from `ws`; its own `replaceState`
writes `:59-199` are deleted), `reviewQueueLoader.svelte.ts` (source from `reviewSource`), `reviewKeyboard.svelte.ts`
(`review` includes `workspace`), `ReviewCropPane.svelte` (picker swap), `ReviewTabsBar.svelte` (deleted: the rail
shows queues), `WorkspaceRail.svelte` (queue rows), `reviewTabsVocabularyStore` (passes the filter for counts),
`keymapFallback.ts` and `src/config/keymap_actions.json` (`review` includes `workspace`; relabel).

Delete: `frontend/src/routes/p/[project]/review/` (move its 17 tests to `frontend/src/lib/review/` or
`frontend/src/lib/workspace/views/`, adapted), `ClassPickerModal.svelte`, `reviewClassPicker.svelte.ts`,
`AuditQueueList.svelte`, `docs-site/docs/cropwright/user-guide/review.md` content rewritten (kept as the Review
view page).

### Steps and commits

1. `feat(curation): five plain queues as served filter presets` (backend; R-10). Red: `test_review_queues.py::test_served_queues_in_order`, `::test_to_label_is_items_without_a_class`, `::test_check_machine_is_unvalidated_machine_classes_lowest_first`, `::test_disagreements_unions_the_three_sources`, `::test_audit_queue_only_while_a_sample_is_pending`, `::test_unfillable_queue_is_absent_and_named_in_next_steps`, `::test_old_tab_id_still_filters_crops`, `::test_old_tab_path_is_404_not_a_queue`, `::test_counts_follow_the_filter`. `make contracts`.
2. `refactor(curation): delete the audit queue route` (and `AuditQueueList.svelte`; the audit page links to the `audit` queue).
3. `feat(frontend): Review view over the workspace query` (`ReviewView`, `reviewSource`, controller changes). Red: `reviewSource.test.ts::uses the queue route when a queue is set`, `::uses crops with the detections sort otherwise`, `::locates the focused item with one request`.
4. `feat(frontend): Enter opens Review at the focused card and Esc returns to it`. Red: `workspaceController.test.ts::open review keeps the focus and returns to it`.
5. `refactor(frontend): Review uses the shared class picker` (delete the modal and its controller).
6. `refactor(frontend): delete the review page` (tests moved).
7. `docs(cropwright): review view page; changelog`.

### Interfaces

Consumes 12a §5, §6 (`queue`, `focus`, `sort_review`), §7, R-10, R-11, R-25, `ClassPicker` (2A). Provides
`ReviewView`, `Filmstrip`, the queue rail rows.

### Other tiers

- Moved vitest files pass; `keymap.literals.scan.test.ts` passes (the `ALLOWED` exemption for `ClassPickerModal`
  `:36-41` is removed).
- Stubbed: `test_review_view.py`: grid, focus the third card, `Enter`, confirm 3 items (keys), `Esc`, the fourth card
  is focused (the next unhandled), the URL has `view=detections&focus=item:<id>`; deep link
  `?view=review&queue=to_label&focus=item:<id>` lands on it with one locate request. Repoint
  `test_labeling_flow.py`, `test_region_confirm.py`, `test_review_*.py`, `test_visual_audit_review.py`,
  `test_label_confirmation.py`. Confirm the class name shows (`test_review_object_name.py`, wf-1d) still passes.
  a11y rows `workspace-review`, `workspace-review-region`. 430 px rows.
- Live tier (coordinator): `test_deep_link.py` uses the Workspace URL.

### Visual checks

Review at 1600 and 430: a machine-labelled item with a proposal, a region item, an empty queue (served next
step), the class picker. "Right": settle the leftovers one at a time fast; the class name and the reason are
readable without hunting; the filmstrip shows where you are in the served order ("12 of 412"); the rail shows at
most six queues with counts.

### Docs and changelog

`review.md` rewritten for the Review view and the queues; `audit.md` points at the `audit` queue;
`keyboard-shortcuts.md`. Root CHANGELOG Changed: "Review queues are five served filter presets (To label, Check
machine labels, Disagreements, New class proposals, Audit sample) plus the region queue; the other tab ids are
filter values. Removed `GET /audit/queue`." Frontend CHANGELOG: "Review is a Workspace view; `/review` removed."

### Risks and do not

- Do not keep the 12 tabs in the rail "for now". The owner asked for the cut.
- Do not change review controllers' behaviour beyond the source and URL; the keyboard model stays identical.
- Region tab: the slot work (`reviewSlotActions`, `multiBoxRegionController`) moves untouched.

### Dependencies and merge

Needs wave 2. Merges after 3A (fewer shared-file conflicts in `viewRegistry.ts` and the rail).

### Done when

- [ ] Backend and frontend red tests green; moved tests pass; `/review` deleted; no `GET /audit/queue`.
- [ ] Screens viewed against "settle the leftovers one at a time, fast".
- [ ] Docs, changelogs, docs check.
- [ ] Standards gate (12a §13).

---

## Lane 3C: Image annotation editor

**Size:** L, 6 days, the largest lane. **Model:** Sonnet at high effort. **Merge:** last in wave 3.

### Goal and outcome

`/p/<project>/images/<image_id>` shows one picture with all its objects and fixes it fast: draw, move, resize,
delete boxes, change or create a class without leaving the page, undo and redo, autosave through OCC, a filmstrip
of the Workspace query. Target: draw 3 boxes and change 2 classes in under a minute by keyboard.

### Files

Create: `frontend/src/routes/p/[project]/images/[image_id]/+page.svelte`,
`frontend/src/lib/editor/editorController.svelte.ts`, `editorCommands.svelte.ts`, `editorKeys.svelte.ts` + tests,
`frontend/src/lib/annotations/boxGeometry.ts` + test (extracted from `MultiBoxCanvas.svelte:106-260`),
`frontend/src/lib/components/editor/{EditorCanvas,ObjectList,ToolStrip,EditorHeader}.svelte` + tests,
`frontend/e2e/stubbed/test_image_editor.py`, `docs-site/docs/cropwright/user-guide/image-editor.md`.

Change: `frontend/src/lib/api_images.ts` (move `getImage`, `getImageFileUrl`, `getImageThumbUrl` from
`api_cropRead.ts`; add `locateImage`, `createImageItem`), `frontend/src/lib/api_cropWrites.ts`
(`patchItemGeometry`, `undoItemGeometry`), `frontend/src/lib/components/MultiBoxCanvas.svelte` (imports
`boxGeometry.ts`), `frontend/src/lib/projectPaths.ts` (`images` section), `keymapFallback.ts` and
`src/config/keymap_actions.json` (`editor` context), `frontend/src/lib/workspace/workspaceController.svelte.ts`
(`workspace.open_editor` target), `frontend/e2e/live/test_writes.py` (D-19), `test_a11y_axe.py`, 430 px tests.

### Editor keymap (`editor` context, all modifiable; class hotkeys live)

| Action id | Default | Behaviour |
|---|---|---|
| `editor.tool_select` | `v` | select and move tool |
| `editor.tool_draw` | `b` | draw one box, then back to select |
| `editor.tool_draw_sticky` | `shift+b` | draw tool stays on |
| `editor.tool_pan` | `h` | pan tool; holding `space` pans temporarily (pointer state, not an action) |
| `editor.zoom_in` / `editor.zoom_out` | `=`, `+` / `-` | zoom at the cursor |
| `editor.zoom_fit` / `editor.zoom_actual` | `0` / `1` | fit; 100 % (loads `getImageFileUrl(id, max_dim)` above fit) |
| `editor.delete` | `delete`, `backspace` | soft delete (`not_an_object`) |
| `editor.next_box` / `editor.prev_box` | `tab` / `shift+tab` | cycle objects (CVAT) |
| `editor.nudge_up/down/left/right` | arrows (locked keys, as `box_edit.nudge_*`) | 1 px at the current zoom |
| `editor.nudge_*_far` | `shift+arrow*` | 10 px |
| `editor.resize_*` | `alt+arrow*` | move the bottom-right corner |
| `editor.regions` | `shift+r` | edit the selected item's region boxes (`MultiBoxCanvas`, `PUT /crops/{crop_id}/regions`) |
| `editor.labels` | `shift+l` | hide or show box labels |
| `editor.undo` / `editor.redo` | `ctrl+z`, `meta+z` / `ctrl+shift+z`, `meta+shift+z` | session command stack |
| `editor.class_picker` | `/` | `ClassPicker` for the selected box |
| `editor.prev_image` / `editor.next_image` | `,` / `.` | walk the filmstrip |
| `editor.cancel` | `escape` (locked) | cancel the draw, then deselect, then back to the Workspace |

A class hotkey on `v`, `b` or `h` is refused by the served validator (12a §7.2); a test per letter.

### Saves, conflicts, undo (exactly plan §4.7 and 12a R-14, R-15)

`editorCommands.svelte.ts`:

```ts
export interface EditorCommand {
  label: string;                               // 'Draw car', shown in the undo tooltip
  apply(): Promise<CommandResult>;             // forward write
  revert(): Promise<CommandResult>;            // inverse write
}
export type CommandResult = { ok: true; image: ImageDetailResponse } | { ok: false; message: string; conflict: boolean };
export class CommandStack {                    // per image id, session only
  readonly canUndo: boolean; readonly canRedo: boolean; readonly saving: boolean;
  run(cmd: EditorCommand): Promise<void>;      // apply; on conflict drop the entry and show the served message
  undo(): Promise<void>; redo(): Promise<void>; // redo re-sends with the fresh revision
}
export function drawCommand(imageId: string, box: Xyxy, className: string): EditorCommand;      // R-14 / batch_exclude not_an_object
export function moveCommand(item: Crop, box: Xyxy): EditorCommand;                              // R-15 / geometry undo
export function deleteCommand(item: Crop): EditorCommand;                                       // batch_exclude / batch_unexclude
export function classCommand(item: Crop, cls: RegistryClass): EditorCommand;                    // PUT label / label undo
export function regionCommand(item: Crop, boxes: RegionBoxInput[]): EditorCommand;              // PUT regions / region undo
```

Every write redraws from the served response (`image_boxes`, `item`). `EditorHeader` shows the save state:
"Saved", "Saving…", "Changed by someone else, reloaded". `request_id` is `crypto.randomUUID()` per draw command
and reused on retry.

### Steps and commits

1. `refactor(frontend): box geometry helpers in one module` (extract from `MultiBoxCanvas`; behaviour identical, its tests pass).
2. `refactor(frontend): image API client in one module` (move `getImage` and the URL builders).
3. `feat(keymap): editor context and actions` (backend JSON, fallback, `make contracts`). Red: `tests/curation/test_keymap_vocabulary.py::test_editor_letters_refuse_a_class_hotkey[v|b|h]`.
4. `feat(frontend): editor command stack with undo and redo`. Red: `editorCommands.test.ts::each command's revert inverts apply` (per command), `::a 409 drops the entry and keeps the served item`, `::redo sends the fresh geometry revision`, `::draw retry reuses the request id`.
5. `feat(frontend): editor canvas with select, draw, move, resize and pan` (Pointer Events: https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events, `setPointerCapture`). Red: `EditorCanvas.test.ts::a drag draws a clamped box`, `::two clicks draw a box`, `::a handle drag resizes`, `::wheel zooms at the cursor`.
6. `feat(frontend): object list, class picker on draw, filmstrip`. Red: `editorController.test.ts::draws served boxes only`, `::after a draw the picker opens with the last class`, `::Enter in the picker saves`, `::next image follows the served locate cursor`.
7. `feat(frontend): image editor page and the open-editor action`.
8. `test(frontend): live write tier draws, moves and deletes on a throwaway project`.
9. `docs(cropwright): image editor page; changelog`.

### Other tiers

- Stubbed `test_image_editor.py`: draw (drag) then the picker then `Enter` saves (stub records R-14 with
  `class_name`); inline create of "Traffic light" accepted as `traffic_light` after the served 422; move with a 409
  stub shows the served message and the server's box; delete then `ctrl+z` restores (stub records
  `batch_exclude` then `batch_unexclude`); `Tab` cycles; `.` walks to the next image via `GET /images/locate`; 430 px:
  the object list is a drawer, a touch drag draws (Playwright `hasTouch` context). a11y row `image-editor`.
- Live write tier (coordinator, D-19): draw, move, delete and undo on a throwaway public project, reload and
  every change is there.
- Key scan: `keymap.literals.scan.test.ts` passes; component tests assert no window keydown listener
  (`EditorCanvas`, `ObjectList`).

### Visual checks

Editor at 1600 and 430 on a busy COCO image and an empty one: select state, draw in progress, picker open,
conflict banner, the object list with lock glyphs, zoom at 100 %. "Right": fix one image fast; handles are 8 px
hit targets with a visible focus ring; the selected box is obvious; labels never hide the object being edited;
the save state is always visible.

### Docs and changelog

`image-editor.md` (tools, keys by glyph, saving and undo, what a re-detect does to your boxes),
`keyboard-shortcuts.md` (`editor` context). Frontend CHANGELOG Added: "Image editor
(`/images/<image_id>`): draw, move, resize and delete boxes, class picker with inline create, undo and redo,
autosave."

### Risks and do not

- Do not write a second box-math helper; extend `boxGeometry.ts`.
- Do not compute a class from an index; the picker returns the registry entry and the draw sends its name.
- Do not autosave mid-drag: one write per finished gesture.
- Do not hold `pointermove` handlers on `window`; use pointer capture on the canvas.

### Dependencies and merge

Needs 2D (writes), 2C (R-17, labels, `BoxLayer`, `boxTone`), 2A (`ClassPicker`), 3A (`types_images.ts`, `api_images.ts`). Merges last
in wave 3, after rebasing on 3A.

### Done when

- [ ] Red tests seen failing then green; `MultiBoxCanvas` tests unchanged and green.
- [ ] The one-minute keyboard target met on the stub (record the time in the hand-back).
- [ ] Screens viewed against "fix one image fast".
- [ ] Docs, changelog, docs check.
- [ ] Standards gate (12a §13).

---

## Lane 3D: Map view

**Size:** M, 3 days. **Merge:** after 3A.

### Goal and outcome

The map is a Workspace view: the same filter and queue, served counts ("1,920 located · 88 without location"),
cells and single-photo pins, a cell's served summary with its first 12 image cards, and area selection
(`Shift`+drag rectangle, polygon lasso) that becomes a filter chip for every view. The `/map` page, its crop
side panel and `GET /map/images` are deleted.

### Files

Create: `frontend/src/lib/workspace/views/MapView.svelte` (view wrapper), `frontend/src/lib/workspace/details/CellDetails.svelte`,
`frontend/src/lib/map/areaTools.svelte.ts` + test, `frontend/e2e/stubbed/test_map_view.py` (replaces
`test_map_page.py`), `docs-site/docs/cropwright/user-guide/map.md`.

Change: `frontend/src/lib/map/mapController.svelte.ts` (filter from `ws`; delete `openCell :110`, `closeCell`,
`listQuery :126`, `listBody :131`, the `images` dep), `frontend/src/lib/components/map/MapView.svelte` (pins, area
tools; `map.boxZoom.disable()` per https://maplibre.org/maplibre-gl-js/docs/API/classes/BoxZoomHandler/; cell
click through `queryRenderedFeatures`), `frontend/src/lib/api_map.ts` (delete `getMapImages`),
`frontend/src/lib/types_map.ts`, `keymapFallback.ts` and the JSON (`map` context, no letters), nav and
`projectPaths.ts` (delete `map`), `frontend/e2e/stubbed/conftest.py` (`webgl_browser` fixture, D-20),
`frontend/e2e/live/conftest.py:185-200` (launch args `--use-gl=angle`, `--use-angle=gl-egl`),
`frontend/e2e/live/test_map_renders.py`. Backend delete: `GET /map/images` (`src/routers/curation/map.py:81`),
`MapTileImages` (`src/schemas/curation/map.py:65`), `map_cells.tile_images :273`, their tests and leak rows.

Delete: `frontend/src/routes/p/[project]/map/`.

### Steps and commits

1. `test(frontend): map tests launch Chromium with ANGLE on EGL` (D-20; the live conftest and a stubbed `webgl_browser` fixture).
2. `feat(frontend): map as a workspace view with served counts, pins and the cell summary`. Red: `mapController.test.ts::sends the workspace filter and queue`, `::a cell click focuses the cell and asks for its summary with the cell bounds`, `::a pin opens the image details`.
3. `feat(frontend): area selection becomes a filter for every view`. Red: `areaTools.test.ts::a rectangle sets geo_bbox`, `::a lasso sets geo_polygon`, `::panning never changes the filter`.
4. `refactor(curation): delete the map tile image list with its only caller` (backend + frontend in one commit; `make contracts`).
5. `docs(cropwright): map view page; changelog`.

### Other tiers

- Stubbed `test_map_view.py`: cells render (WebGL fixture); a cell click shows the served summary (R-19 with the
  cell's `bounds` in `geo_bbox`) and 12 cards (R-1); "Show in Gallery" switches view with `geo_bbox` equal to the
  served bounds; `Shift`+drag adds an area chip and Gallery is filtered; no-GPS and drop-policy states; 430 px;
  without WebGL2 the served fallback message (kept from `test_map_page.py:168`). a11y row `workspace-map`.
- Live (coordinator): `test_map_renders.py` with the ANGLE args, or headed on display `:11`: the "3 items" case of
  `coco-e2e` shows its points (the wf-1d fix stays fixed).

### Visual checks

Map at 1600 and 430: zoomed out with cells, zoomed in with pins, a cell selected with its panel, an area chip, the
no-location count link. "Right": find and work data by place; counts in the header match `GET /map`; selection
and chips are obvious; the map never steals the page scroll.

### Docs and changelog

`map.md`. Frontend CHANGELOG: "Map is a Workspace view with area filters; `/map` removed." Root CHANGELOG
Removed: "`GET /map/images`."

### Risks and do not

- Do not filter by the viewport on pan.
- Do not request one `/images` call per cell; the panel asks once for the focused cell.

### Dependencies and merge

Needs 2B (R-9, R-21), 2C (R-19), 3A (`ImageCard`, `listImages`). Merges after 3A.

### Done when

- [ ] Red tests green; `/map` and `GET /map/images` gone; the endpoint catalog passes.
- [ ] Screens viewed against "find and work data by place"; the "3 items, no points" case shows points.
- [ ] Standards gate (12a §13).

---

## Lane 3E: Quality page and next steps

**Size:** M, 3 days. **Merge:** after 3B (the audit queue).

### Goal and outcome

`/quality` holds Accuracy audit, Near-duplicates and Label history as tabs. The Dashboard shows a served Next
steps card and the empty Workspace shows the same served steps. Nav goes from 14 to 10 entries (plan §3).

### Files

Create: `frontend/src/routes/p/[project]/quality/+layout.svelte` (tab strip), `quality/audit/+page.svelte`,
`quality/near-duplicates/+page.svelte`, `quality/history/+page.svelte` (bodies moved from the old pages verbatim),
`frontend/src/lib/components/NextStepsCard.svelte` + test, `frontend/e2e/stubbed/test_quality_page.py`,
`frontend/e2e/stubbed/test_next_steps.py`, `docs-site/docs/cropwright/user-guide/quality.md`.

Change: `frontend/src/routes/p/[project]/dashboard/+page.svelte` (Next steps card), the empty state of the
Workspace (`ViewState` action from `nextStepsStore.next`), `frontend/src/lib/components/export/*` stale near-dup
panel link and the holdout freeze dialog link to `/quality/near-duplicates`, near-dup group "Open in Workspace"
(`/workspace?view=gallery&scope=near_dup:<id>`), nav, `projectPaths.ts`.

Delete: `frontend/src/routes/p/[project]/{audit,near-duplicates,label-ops}/`; `audit.md`, `near-duplicates.md`
folded into `quality.md`.

### Steps and commits

1. `feat(frontend): quality page with audit, near-duplicates and label history` (moves and deletes in one commit).
2. `feat(frontend): served next steps on the dashboard and the empty workspace`. Red: `NextStepsCard.test.ts::renders served steps in order`, `::each link is the served section`, `::done steps are marked, not hidden`.
3. `feat(frontend): open a near-duplicate group in the workspace`.
4. `docs(cropwright): quality page; changelog`.

### Other tiers

Stubbed: `test_near_duplicates.py`, `test_label_ops.py` repointed to `/quality/...`; `test_next_steps.py` (fresh
project: "Create classes" then "Cluster", each link lands on its section); a11y rows `quality-audit`,
`quality-near-duplicates`, `quality-history`, `dashboard-next-steps`. Live sweep rows (coordinator).

### Visual checks

The three tabs at 1600 and 430; the Dashboard with next steps on a fresh stub project and on `coco-e2e`.
"Right": the next step is obvious from the Dashboard; Quality reads as one place for dataset hygiene.

### Done when

- [ ] Old pages deleted, links repointed (`rg "/label-ops|/near-duplicates|/audit" frontend/src` finds only Quality and API paths).
- [ ] Red tests green; screens viewed.
- [ ] Standards gate (12a §13).

---

## Lane 3F: Table view

**Size:** M, 3 days. **Merge:** any time after 3A.

### Goal and outcome

Rows of detections or images with served columns and cells; a header click asks for a served sort; the column
choice is in the URL; multi-select feeds the bulk bar; `Enter` opens Review, `Shift+O` the editor. "Sort by
anything and act on the rows."

### Files

Create: `frontend/src/lib/api_table.ts`, `frontend/src/lib/types_table.ts`, `frontend/src/lib/table/tableController.svelte.ts`
+ test, `frontend/src/lib/components/table/{DataTable,ColumnPicker}.svelte` + tests,
`frontend/src/lib/workspace/views/TableView.svelte`, `frontend/src/lib/contract/tableContract.test.ts`,
`frontend/e2e/stubbed/test_table_view.py`, `docs-site/docs/cropwright/user-guide/table.md`.

Change: `viewRegistry.ts`, `keymapFallback.ts` and the JSON (`table` context), `endpointCatalog.test.ts`
(`SCANNED_FILES`), axe and 430 px tests.

`DataTable` props: `{ columns: TableColumn[]; rows: TableRow[]; sortApplied: string; sortOptions: SortOption[];
selectedIds: Set<string>; focusedId: string | null; onsort: (id: string) => void; onselect; onfocus; onopen }`.
A real `<table>` with `<th scope="col">` and `aria-sort` from the served `sort_applied`; the thumbnail column's
header carries a visually hidden label (`tableHeaders.scan.test.ts`); sticky header; the table scrolls inside its
own container.

### Steps and commits

1. `feat(frontend): table client and controller`. Red: `tableController.test.ts::cells render verbatim`, `::header click sends the served sort id`, `::no client sort` (scan), `::appends by cursor`, `::column choice writes columns to the URL`.
2. `feat(frontend): table view with column picker and row selection`.
3. `docs(cropwright): table page; changelog`.

### Other tiers

Stubbed `test_table_view.py`: sort by detector confidence (the stub records `sort=detector_confidence:asc`),
choose columns, select rows with `Space` and `Shift`+arrows, bulk assign, 430 px horizontal scroll inside the table
only. a11y row `workspace-table`. Live (coordinator): the first page equals `GET /table` on `coco-e2e`.

### Visual checks

Detections and images rows at 1600 and 430; a sorted column; selected rows. "Right": numbers right-aligned
(served `align`), the sort is obvious, rows are dense but readable, the thumbnail does not dominate.

### Done when

- [ ] Red tests green; screens viewed.
- [ ] Standards gate (12a §13).
