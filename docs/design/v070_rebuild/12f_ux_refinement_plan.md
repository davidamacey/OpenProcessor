# 12f: Cropwright UX refinement plan (buttons, layout, type, flow)

Status: plan, 2026-11-12. Written from the live app (`op-gpu`, `int/b4-stateless` at 48fd4a034), the code in
`frontend/src`, the owner's feedback F1-F8, the functional walk report and the owner's direction on
near-duplicates. Nothing here is built yet. Coding lanes implement it in two waves (section 8).

Read first: [VISION_AND_GOALS.md](../../VISION_AND_GOALS.md) "The user's job". Earlier plans this builds on:
[12d_wave4_lane_specs.md](12d_wave4_lane_specs.md) and [12e_workflow_execution_order.md](12e_workflow_execution_order.md).

Evidence (not committed, local only):
- Review screenshots: `artifacts_local/ux_review_1112/screens/` (desktop 1600x1000 `desktop_*`, 430x900 `mobile_*`),
  scripts in `artifacts_local/ux_review_1112/scripts/r01..r05`.
- Walk report: `artifacts_local/briefs/reports/walk_1112.md`, screens in `artifacts_local/screens/accept_1112/`.
- Earlier spec: `artifacts_local/ux_audit_1112/REPORT.md` (the design system, owner issues 1-12).

Out of scope here, already owned by lane fix-walk: walk P1-1 (class count drift), P1-2 (class create by display
name), P1-3 (editor geometry 409), P1-4 (Anthropic probe), the backend of P2-1 (interactive label-op queue) and P2-2
(assist-loop `min_root_size`). This plan covers only the UI side of P2-1 (section 4.6.3).

---

## 1. Principles (a)

Every rule below is applied in sections 2-6. Each one cites a page that was fetched for this review.

| # | Principle | How we apply it | Source |
|---|---|---|---|
| P1 | Show the state and the one next step. | Every view says what it shows and what to do next. An empty view states why and offers one action. | NN/g heuristic 1: "keep users informed about what is going on, through appropriate feedback" ([10 heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/)). Empty states: "Communicate System Status", "Provide Learning Cues", "Provide Direct Pathways for Key Tasks" ([NN/g empty states](https://www.nngroup.com/articles/empty-state-interface-design/)). |
| P2 | Progressive disclosure: few controls first, advanced ones one obvious click away. | Each toolbar shows the loop's controls. A labelled `Refine` or `More` control opens the rest. Nothing is deleted. | "Initially, show users only a few of the most important options." "It must be obvious how users progress from the primary to the secondary disclosure levels." "place the advanced features button in a clearly visible spot" ([NN/g progressive disclosure](https://www.nngroup.com/articles/progressive-disclosure/)). |
| P3 | One primary action per view. | Exactly one filled accent button is visible in each view state. Other actions are secondary (outlined) or ghost. | "Primary buttons have the most visual emphasis" ([NN/g button states](https://www.nngroup.com/articles/button-states-communicate-interaction/)). "a separate color should be reserved for primary calls-to-action" ([NN/g similarity](https://www.nngroup.com/articles/gestalt-similarity/)). Filled buttons are for "important, final actions that complete a flow" ([Material web button docs](https://github.com/material-components/material-web/blob/main/docs/components/button.md)). NN/g states no literal one-per-screen rule; this house rule follows from the emphasis guidance. |
| P4 | Undo instead of confirm for routine writes. | Labelling, moving and ignoring apply and then show a toast with Undo. A bulk write over many items shows its served dry-run count first, then applies with Undo. | "Do not use confirmation dialogs for routine actions." Confirmation is for "actions that cannot be undone" ([NN/g confirmation dialogs](https://www.nngroup.com/articles/confirmation-dialog/)). Heuristic 3: users "need a clearly marked 'emergency exit'". |
| P5 | Keyboard accelerators for the loop. | `L`, a class, `Enter` labels a cluster and opens the next one. Arrow keys move between cards. Number keys pick recent classes. | Heuristic 7: "Shortcuts — hidden from novice users — may speed up the interaction for the expert user". CVAT's review mode uses Tab for the next object, digit keys to set a value and F2 for the key list ([CVAT attribute mode](https://docs.cvat.ai/docs/annotation/manual-annotation/modes/attribute-annotation-mode-basics/)). |
| P6 | Minimal cards. | Cards show a name, a count and at most one status. Rank, clarity, confidence and source go behind hover or the details panel. | Heuristic 8: "Interfaces should not contain information that is irrelevant or rarely needed." |
| P7 | Recognition over recall. | Use served names (class, cluster, file name), never raw ids. Ids appear in tooltips and a copy button. | Heuristic 6: "making elements, actions, and options visible". |
| P8 | One type scale, a real UI font, aligned numbers. | Inter for UI text and JetBrains Mono only for code, ids and paths, both bundled. Counts use tabular figures in the sans face. | MD3 type scale: body 14/20, label small 11/16, title medium 16/24 ([Compose M3 type table](https://developer.android.com/develop/ui/compose/designsystems/material3)). HIG macOS: body 13 pt, minimum 10 pt ([HIG typography, mirror](https://apple-docs.everest.mt/docs/design/human-interface-guidelines/typography/)). `tabular-nums` gives "figures where numbers are all of the same size" ([MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/font-variant-numeric)). Inter `tnum`: "useful for tabular data" ([rsms.me/inter](https://rsms.me/inter/)). |
| P9 | WCAG 2.2 AA. | Text contrast 4.5:1, controls and states 3:1. A visible focus ring that sticky bars never cover. Pointer targets at least 24x24 CSS px or spaced. | [1.4.3](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum), [1.4.11](https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast), [2.4.7](https://www.w3.org/WAI/WCAG22/Understanding/focus-visible), [2.4.11](https://www.w3.org/WAI/WCAG22/Understanding/focus-not-obscured-minimum), [2.5.8](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum). |
| P10 | Bulk actions appear with a selection. | The bulk bar shows the count and the actions only once something is selected. With nothing selected the row shows the view's own primary action. | Linear: select issues, then "use Cmd/Ctrl K ... and select the preferred action"; "Common bulk actions will show up at the bottom" ([Linear select issues](https://linear.app/docs/select-issues)). Label Studio: check rows, then a dropdown that names the number of tasks ([data manager](https://labelstud.io/guide/manage_data)). |
| P11 | Settings sub-navigation pattern. | Keep the Settings sidebar as is. Reuse its look for any second level that has more than five sections. | Primer NavList "is typically used as a sidebar that changes what is rendered in the main content area"; NavList.Heading is "recommended for settings-style pages" ([Primer NavList](https://primer.style/product/components/nav-list/)). |

### 1.1 What the established tools do, and where we go further

The baseline is professional tooling. Our goal is to make the user's own class scheme fast to label: the loop is
ingest, cluster, label, review, train. The advanced clustering and training controls stay in the app, one click
away.

| Screen | Established tools (fetched docs) | Ours: faster or clearer | Advanced capability kept or added |
|---|---|---|---|
| Clusters | FiftyOne: "lasso points in the plot to show only the corresponding samples/patches" ([FiftyOne App](https://docs.voxel51.com/user_guide/app.html)). Label Studio: a data manager to "view all your labeling tasks, sort and filter", plus tabs that split the dataset ([LS](https://labelstud.io/guide/manage_data)). The fetched docs describe selecting a group, then acting on it as a separate step. | The cluster is the unit of work. Open a cluster, press `L`, pick a class (or create one by name), `Enter`: the whole cluster is labelled and the next cluster opens. Outliers sort first. A mixed cluster warns before it is labelled. | Facets, subject scope, clarity, text, image and semantic search, the ignored bucket, matching-items mode, region galleries, VLM assist on a cluster, Move, Flag, Reprocess, Embedding lasso. |
| Gallery | FiftyOne: a sample grid, "Click a sample to open an expanded view", sidebar filters. | The same grid. Details open in a side panel, so the grid stays in view. Per-image counts. The duplicates control sits in place, with no separate page. | One per near-duplicate group, lightbox, image editor, diverse sample. |
| Review | Label Studio: "select 'Reject' or 'Accept' at the bottom" ([LS review](https://docs.humansignal.com/guide/onboarding_reviewer)). CVAT: keyboard next-object and digit-key values. Roboflow Auto Label: per-class "Confidence" slider, approve or reject in Review, and "Approved images do not go into the dataset until you ask for that as a separate step" ([Roboflow](https://docs.roboflow.com/datasets/annotate/annotate/ai-labeling/auto-label)). Box prompting "will convert all predictions to annotations" ([Roboflow](https://docs.roboflow.com/datasets/annotate/annotate/ai-labeling/box-prompting-ai-labeling)). | Review always runs on a served queue, and nothing already validated asks to be confirmed. Confident machine labels are accepted in one undoable bulk operation (a threshold, a per-class dry run, then Undo). Letter keys label. | Comparison panes (Context/Tight, siblings, highlight), More like this, Dismissed, sort, semantic search in a queue, region-slot multi-box review. |
| Data quality | Label Studio reviewer analytics: "accepted", "fixed+accepted", "rejected". Encord: "Delete labels below a specified confidence level" ([Encord](https://docs.encord.com/platform-documentation/Annotate/annotate-label-editor/annotate-label-editor-annotate)). | One "Ready to train" checklist. Each line links to the exact items that need work in the Workspace. Nothing is deleted, and duplicates are never treated as errors. | Accuracy audit with confidence intervals and a confusion matrix, label history with undo, leak containment, holdout checks. |
| Export / Train | Roboflow: approval is a separate step before items reach the dataset. | Export and Train sit in one menu, in loop order. "Collapse near-duplicates" is a per-run option, off by default. | Every current Train control (sizes, profiles, GPUs, campaign, augmentation, cohorts) stays. Advanced groups are collapsed. |

---

## 2. Owner feedback F1-F8 (b)

Each item gives the root cause with file and line, the decision, and the exact change. Paths are relative to
`frontend/src` unless they start with `src/` (backend) or `scripts/`.

### F1. The Data pages are not one set (Import dataset looks like another app)

**Root cause.**
- `routes/p/[project]/ingest/+page.svelte:65` reads "Import dataset is its own page". The tab at `:84` links to
  `/datasets/import`, a separate route.
- `routes/p/[project]/datasets/import/+page.svelte:107` uses the `PageShell` default width (max-w-6xl). Ingest uses
  `width="wide"` (`ingest/+page.svelte:104`, max-w-7xl).
- The import page has no tab strip and no Pipeline rail. Its sections are hand-rolled
  `rounded-lg border border-line p-4` with `text-sm`/`text-xs` (`:127-131, :189-191, :211-212, :249-250`) instead of
  `ui/Card` and the type tokens.
- `lib/projectPaths.ts:36` keeps a hidden `datasets` section, so the breadcrumb reads "Data › Datasets › Import" on one
  tab and "Data › Ingest" on the others.
- Walk P2-12 confirms this.

**Decision.** Import dataset becomes a real Ingest tab (`?tab=import`). The import job page moves under the Ingest
layout, and the `datasets` section is deleted. There is no redirect shim: the old URLs fall to the app's 404,
as the project's no-shim rule requires.

**Exact change (lane D).**
1. New `routes/p/[project]/ingest/+layout.svelte`. It owns `PageShell width="wide"`, the `PageHeader` ("Ingest", the
   detector chip), the URL-bound `Tabs` and the `IngestPipelineRail` column. `ingest/+page.svelte` keeps only the tab
   bodies. Tab ids: `upload | folder | sources | import | activity`.
2. Move the wizard body from `datasets/import/+page.svelte` into a new `lib/components/datasets/ImportWizardPanel.svelte`,
   rendered when `tab=import`. Each step (Source, Preview, Class mapping, Options, Start) is a `ui/Card` with a
   `ui/Section` title and a step number. "Start import" stays the only primary button in the tab.
3. Move the list from `datasets/imports/+page.svelte` into `lib/components/datasets/ImportHistoryTable.svelte`
   (`ui/DataTable`), shown under the wizard as the section "Past imports" in the same tab. The loose "All imports"
   link is removed.
4. Move `datasets/imports/[id]/+page.svelte` to `routes/p/[project]/ingest/imports/[id]/+page.svelte`. It renders inside
   the Ingest layout with the `import` tab marked current. A back link "Import dataset" replaces the "All imports"
   link.
5. Delete `routes/p/[project]/datasets/` (`+page.ts`, `+page.svelte`, `import/`, `imports/`) and the `datasets` row of
   `SECTION_TABLE` and `SUBPAGE_LABELS` in `lib/projectPaths.ts`. Add `imports: 'Import'` under `ingest` in
   `SUBPAGE_LABELS`.
6. Update every in-app link to the new paths:
   - `lib/components/provenance/ImportProvenanceRows.svelte:42`;
   - the wizard's own `goto` calls, formerly at `datasets/import/+page.svelte:78, 89, 274, 294`.
7. Update the stubbed e2e tests to the new paths: `e2e/stubbed/test_dataset_import.py`,
   `e2e/stubbed/test_import_leftovers.py:282` and `e2e/stubbed/test_narrow_viewport_430.py:85-86`.

### F2. Data quality has no gallery, and near-duplicates are unexplained

**Root cause.**
- `routes/p/[project]/quality/+layout.svelte:16-20` holds three tabs.
- `quality/near-duplicates/+page.svelte:57-83` stacks five panels with no purpose line: status, groups, decisions,
  runs and the threshold profile.
- The copy frames duplicates as waste: `lib/components/nearDup/NearDupGroupList.svelte:31` says "keep one, drop the
  rest", and `workspace/views/GalleryView.svelte:28` labels members "Extra copies".
- The threshold panel `lib/components/nearDup/NearDupProfilePanel.svelte` duplicates Settings › Near-duplicate
  profiles (`lib/settings/settingsNav.ts:81`).
- The audit page (`quality/audit/+page.svelte`) shows only numbers. The sample is reachable only as a Review list,
  and no served filter selects audited items by outcome.
- Walk P2-11 and P2-10 confirm this (audit counters disagree between calls).

**Decision (folds in the owner's near-duplicate direction, which overrides REPORT.md §10).**
- Duplicates are never something to delete. Working with them happens in the Workspace.
- Using them is a per-run choice on Export and Train, off by default.
- Data quality becomes the final dataset-level review before training, "Ready to train". Its tabs: **Ready to
  train** (new, default) | **Accuracy audit** | **Label history**.
- The Near-duplicates tab is removed. Its parts move:
  - status and run → the Gallery's Duplicates control (lane C);
  - group browsing and decisions → the Gallery and the details panel (lane C);
  - thresholds, run history and pinned pairs → Settings › Near-duplicate profiles (lane D).

Full layout in section 4.9. Backend data in section 7 (B3, B4).

### F3. Settings looks great

Keep it: `routes/p/[project]/settings/+layout.svelte`, `ui/SidebarNav.svelte` and the search. The only Settings change
is new content on the Near-duplicate profiles page (section 4.9.3). The walk P3-6 console 404 on
`/settings/presets` is fixed in lane D, which must root-cause it first.

### F4. Workspace: Clusters says "no data", unlabelled items, too many buttons

**Root cause.**
1. **The "no data" confusion is a bug.**
   - `lib/clusters/clusterListController.svelte.ts:163-173` pins the client-built slot inventory card
     ("Objects · inventory · 170 listed", from `lib/clusters/slotInventory.svelte.ts`) in front of the served list.
   - So `lib/components/clusters/ClustersGrid.svelte:237` (`list.gridItems.length === 0`) is never true. The served
     empty state never renders: `GET /clusters` returns `total 0`, `empty_reason: "pending_activation"`,
     `pending_partition {46 clusters, 279 items}`, and the UI would show "Activate it" (`:250-264`).
   - The header says "0 matching", the card "170 listed" and the rail "Objects 166": three numbers for one thing.
   - After Activate, the list does not reload until a page reload (walk P2-3).
2. **Unlabelled items in Gallery.**
   - On `coco-e2e` they are 217 items that carry only a detector class from before the 2026-11-12 ingest change.
     The served next step `classes` has `action: adopt_detector_classes` (`GET /next_steps`).
   - The Workspace shows this only as a small "Next:" chip (`lib/components/workspace/FilterBar.svelte:90-99`).
   - On fresh projects the detector class is assigned at ingest, so "To label" is 0 and the work sits in "Check
     machine labels". The rail and the copy still point at "To label" (walk P2-7).
3. **Too many buttons** (walk P2-14 counts, confirmed in `desktop_ws_clusters.png` and `desktop_wfa_cluster_open.png`).
   - Cluster list: `ClustersToolbar.svelte` has 11 controls and `ClusterListFilterBar.svelte` 9, so about 22 controls
     come before the first card.
   - Open cluster (`workspace/views/ClusterScope.svelte:106-195`): BulkBar's 7 buttons plus Flag, Reprocess, Accept
     VLM, Assist, then `ClusterDetailFilters` and `ClusterFacetBar`. That makes two sort controls
     (`ClusterFacetBar.svelte:142-152` and the sort chip) and two "needs review" controls. The first crop sits at
     y=640 of 1000.
   - `BulkBar.svelte:113-121` keeps a disabled primary "Assign class…" visible with nothing selected. The real
     cluster action, "Label all 24 in this cluster…", lives only in the details panel
     (`workspace/details/ClusterDetails.svelte:40`).

**Decision.**
- The served empty state wins. The inventory card leaves the cluster grid.
- The cluster is the unit of work. It gets one header with one primary, "Label all N as…" (`L`); see section 4.2.
- Every other control moves behind `Refine` (filters and ordering) or `More` (rare actions). All of them are kept;
  see the keep-list in section 4.12.
- Detector-only items get a Workspace banner with the served adopt action.

Exact changes in sections 4.1-4.3 (lane C) and B9/B10 in section 7 (lane A).

### F5. Review: why confirm what is 100% confident?

**Root cause.**
- `lib/workspace/views/ReviewView.svelte:3-5`: with no queue, Review walks "the Detections order". Opening the Review
  tab loads all 477 items, and the first are import-validated labels at confidence 1.0, each offering "Confirm"
  (`desktop_ws_review.png`, walk P2-5).
- There is no way to accept many confident machine labels at once. `check_machine` lists "least confident first", so
  the confident ones come last, one keypress each.

**Decision: the rule.**
1. Review always runs on a served queue. With no queue in the URL, Review opens the served `default_tab`. That is the
   first queue with work, in served order (B2), so after a detector ingest it is "Check machine labels". When every
   queue is empty, the served `empty_state` shows.
2. An item that is already validated (by a person or by an import) never asks to be confirmed. It can come up from
   `Enter` on a grid card, which keeps Review on the grid's list. Such an item shows "Validated by a person ·
   2026-11-12" or "Imported label". The actions become **Change class** and **Next** (`N`, the skip key), and Confirm
   is not rendered.
3. "Check machine labels" gets one bulk action: **Accept confident labels…**
   - It opens a dialog with a threshold (default 0.95, range 0.50-1.00) and the source (detector, VLM or both).
   - The dialog shows the served dry run: total, per-class counts, and what is held back (test holdout, the open audit
     sample, excluded, locked).
   - When an accuracy audit exists, each class row also shows its audited precision with its interval
     (`AuditReport.detector[].precision`, `ci_low`, `ci_high`), so the user sees whether the detector is trustworthy
     for that class.
   - Apply runs one label op (B1). The toast says "Accepted 412 labels · Undo". The op shows in Label history and
     undoes like any label op.
4. No automatic acceptance. There is no setting that validates on its own, for three reasons:
   - The detector's confidence is about its own class ("car"), not the user's scheme ("make and model").
   - Auto-validation would bias the accuracy audit.
   - The lock rule reserves validation for people and imports.
   The bulk accept is a person's decision, recorded as such (label source `human_bulk_accept`, B1).

**Exact change.**
- Lane A: B1 and B2.
- Lane C: `ReviewView.svelte` (drop the Detections fallback), `lib/review/reviewTabsController.svelte.ts`
  (served default), `components/review/ReviewCropPane.svelte` (validated state),
  `components/review/ReviewHeader.svelte` (Accept button on `check_machine`), new
  `lib/components/review/AcceptConfidentDialog.svelte`, `lib/api_labelOps.ts`, `lib/types_labelOps.ts`.

### F6. The map needs more GPS images (3 of 91)

**Root cause.**
- `coco-e2e` has 3 located photos. COCO images carry no GPS: COCO's image record has `license`, `flickr_url` and
  `date_captured` but no GPS or EXIF field ([COCO format, AWS docs](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html)).
- With a class filter that excludes the 3 photos, the map shows an empty world with no explanation
  (`lib/workspace/views/MapView.svelte:126-197`). `MapCellCollection` (`src/schemas/curation/map.py:59`) serves no
  count of matching located photos.

**Decision: real, licence-recorded, geotagged public photos.**
- Use photos from Wikimedia Commons; synthetic EXIF is not used for the test project. Commons is searchable by
  coordinates (`list=geosearch`, `gscoord=lat|lon`, `gsradius` up to 10000 m, `gslimit` up to 500;
  [GeoData](https://www.mediawiki.org/wiki/Extension:GeoData)). Licences are per file through
  `prop=imageinfo&iiprop=extmetadata` (`LicenseShortName`, `Artist`, `Credit`;
  [CommonsMetadata](https://www.mediawiki.org/wiki/Extension:CommonsMetadata)).
- The repo already fetches Commons with a licence allow-list (`scripts/datasets/fetch_hires_set.py` `ALLOWED_LICENSE`,
  `api_get`; `scripts/datasets/proxy_commons.py`), and the public-data rule allows licence-checked public sets fetched
  into a gitignored directory.
- Synthetic EXIF stays acceptable only where a test needs exact known coordinates. `tests/live/test_live_capture_map.py`
  already writes it for the capture tests. It is not acceptable for the map the owner evaluates: clumps, outliers, real
  timestamps and real street scenes are what the map UX must handle.

**Exact change.**
- Lane A, B8: a new fetcher, a pinned manifest, and an ingest of the set into `coco-e2e`.
- Lane A, B5: served matching counts.
- Lane C: the map's filtered empty state, "None of the 3 located photos match the filter. [Clear filters]", and the
  header "12 of 63 located photos match".

### F7. "Open image" vs "View full size" vs "Show its items", and the vanishing thumbnail

**Root cause.**
- **The vanishing thumbnail** (reproduced on the Map, `desktop_f7map_1_cell.png` → `desktop_f7map_2_card_clicked.png`;
  the walk could not reproduce it in Review).
  - A map cell's details (`workspace/details/CellDetails.svelte`) show the cell's photo thumbnails. Clicking one runs
    `ws.setFocus({kind:'image'})` (`CellDetails.svelte:56`).
  - `setFocus` writes the URL with `replace: true` (`lib/workspace/workspaceController.svelte.ts:304-306`), so the cell
    focus is overwritten and Back cannot return to it.
  - `ImageDetails.svelte` renders no image at all, so the thumbnails vanish. They come back only when the user picks
    another cell.
- **Three unclear buttons** (`workspace/details/ImageDetails.svelte:80-100`).
  - "Open image" opens the editor.
  - "View full size" opens the lightbox.
  - "Show its items" sets `scope=image:<id>` on the current view. In Gallery that shows the same one image again
    (`desktop_f7_2_after_show_items.png`), and on the Map one dot.
- **Two links to the same editor with different names**: `ItemDetails.svelte:40-47` "Open in the image editor" (keeps
  the Workspace query) and `cropMeta/CropSourceContext.svelte:66-71` "Open image" (drops it).
- **Inconsistent counts.**
  - The image card counts exclude the test holdout ("3 items"), while the details include it ("items 7 … in test
    holdout 4").
  - The scope chip shows a raw id (`image · b7f1338c…`), not the file name.

**Decision and exact change (lane C).**
1. `ImageDetails.svelte` gets a preview at the top: the thumbnail with its boxes, using the existing
   `SourceImageOverlay` with the served `ImageDetailResponse`. Clicking the preview opens the lightbox.
2. The actions become:
   - **Edit boxes** (primary, `Shift+O`, opens the editor with the Workspace query);
   - an icon button **Full size** (`maximize` icon, `aria-label="View full size"`, `Shift+Space`);
   - **Show its N detections** (secondary). It switches to the Detections view and scopes it to the image in one URL
     write. A new controller method `ws.showImageItems(id)` writes `{view:'detections', scope:{kind:'image', id}}`.
     Inside Detections with that scope, the button becomes the scope chip.
3. Counts read "7 detections · 4 in the test holdout (not listed)", from the served `ImageCounts`.
4. One editor link everywhere. `CropSourceContext.svelte`'s "Open image" link is deleted, and `ItemDetails.svelte`'s
   button is renamed **Edit boxes** and keeps the query.
5. Map cell to image pushes history. `CellDetails.svelte` calls a new `ws.setFocus(focus, {push: true})`.
   `ImageDetails` shows **Back to the cell (N photos)** when the previous focus was a cell. The controller keeps a
   one-step `#previousFocus`, and Back in the browser also works.
6. The scope chip (`FilterBar.svelte:66-79`) shows the served name: the image `file_name` or the cluster
   `display_name`. The id moves to the chip's `title`.
7. Clicking the "view" overlay on a detection card no longer clears the selection (walk P2-8).

### F8. Fonts: teletype look, inconsistent type

**Root cause.**
- `app.css:128-130` names only macOS and Windows faces (`-apple-system, BlinkMacSystemFont, 'SF Pro Display',
  'Segoe UI', system-ui`). On the owner's Linux host, Chrome renders UI text in **Noto Sans**.
- `font-mono` uses Tailwind's default stack (`ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
  'Liberation Mono', 'Courier New', monospace`, [Tailwind font-family](https://tailwindcss.com/docs/font-family)).
  On the host this resolves to **Liberation Mono**, a Courier New metric clone: the teletype look. Measured with
  Chrome DevTools Protocol `CSS.getPlatformFontsForNode` on `detections-footer`, `api-health-chip` and the rail counts.
- `font-mono` appears 618 times in 171 files. Many uses are prose or labels, not ids, numbers or paths:
  - `ClustersStatusBar.svelte:35-61` ("matching items", "loading more…", "scroll for more", "ignored bucket");
  - `ImageDetails.svelte:45` and `SelectionDetails.svelte:22` (key/value lists);
  - `ClusterDetails.svelte:47,55` (sentences);
  - `ReviewHeader.svelte:77`;
  - CropCard badges (`CropCard.svelte:281,310,318,327,428,455`);
  - the heaviest files: `RunResults.svelte` 30, `DatasetStats.svelte` 26, `ExportControls.svelte` 14.
- The token scale is bypassed:
  - 1,021 `text-xs`, 510 `text-sm`, 58 `text-base`, 12 `text-lg`, 1 `text-xl` and 4 `text-2xl`, against 356
    `text-caption` and 13 `text-h1`;
  - hand-rolled page titles: `export/+page.svelte` (`text-2xl`), `models/+page.svelte` (`text-xl`),
    `routes/projects/+page.svelte` and both `projects/combine` pages (`text-sm`).
- 90 `uppercase` uses with tracking at 11-12 px make section labels harder to read and inconsistent: some tables use
  uppercase headers, others sentence case.

**Decision and exact change (lane B).** See section 5. In short:
- Bundle Inter Variable (UI) and JetBrains Mono Variable (code), self-hosted, OFL-1.1.
- Mono only inside `<code>`, `kbd` and code editors. Numbers in the sans face with `tabular-nums`.
- Only the seven size tokens; no `uppercase`.
- Scan tests enforce all of it.

---

## 3. Information architecture

### 3.1 Top nav: order = the loop

Current (`lib/projectPaths.ts:15-43`): Dashboard | Data ▸ (Ingest, Workspace, Classes, Export) | Model ▸ (Models,
Train, Bake-off) | Data quality | Settings.

Target (lane D):

```
Dashboard | Data v (Ingest, Classes) | Workspace | Data quality | Model v (Export, Train, Bake-off, Models) | Settings
```

- **Workspace becomes a top-level link.** It is where most of the time is spent. Burying it in a menu costs a click on
  every return (heuristic 7).
- **Data quality sits between Workspace and Model.** It is the final review before training (owner direction).
- **Export moves to Model.** It produces the frozen dataset that Train consumes ("Next: train on this export"). The
  export page stays usable on its own for downloading a YOLO dataset.
- In `SECTION_TABLE`, the `group` values become: `ingest`, `classes` → `data`; `workspace` → a new group `work`;
  `export`, `train`, `bakeoff`, `models` → `model`.
- `NAV_GROUP_LABELS` gains `work: 'Workspace'`. The label stays "Model"; the Model menu's items read in loop order.
- **Breadcrumb.**
  - Section and tab pages show no breadcrumb: the nav marks the section, and the tab strip marks the tab. This fixes
    the doubled "Data quality" (walk P3-5).
  - The breadcrumb shows only on object pages: the import job, the image editor, a settings item editor.
  - The last crumb uses the served name (file name, import name), never a raw id (walk P3-2).
  - Change `breadcrumbOf` in `projectPaths.ts`. The image editor passes its served `file_name`.

### 3.2 Where each feature lives, and what moves

| Feature | Now | Target | Why |
|---|---|---|---|
| Import labelled dataset | `/datasets/import` (separate shell) | Ingest › Import dataset tab | F1 |
| Import history and job page | `/datasets/imports`, `/datasets/imports/[id]` | Ingest › Import dataset › Past imports; `/ingest/imports/[id]` | F1 |
| Near-duplicate status and run | Data quality › Near-duplicates | Workspace › Gallery › Duplicates control | owner: work with duplicates in the Workspace |
| Near-duplicate groups browser | Data quality › Near-duplicates › Groups | Gallery "One per group" cards plus the group scope; item level via a Detections scope | one place to see photos |
| Same / different decisions | group modal from Quality and from ImageDetails | the group modal from ImageDetails (kept) | already in the Workspace |
| Pinned pairs list, run history | Data quality › Near-duplicates | Settings › Near-duplicate profiles › Pinned pairs, Runs | rarely needed; they shape future runs |
| Near-duplicate thresholds | twice (Quality "Advanced: thresholds" and Settings) | Settings only (the existing preset page) | duplicate UI deleted |
| Collapse near-duplicates | Export checkbox; Train objects-dataset checkbox | unchanged, off by default, new help copy | owner: per-run choice |
| Readiness before training | scattered (Train preflight, Export chips, Dashboard) | Data quality › Ready to train (new, served) | owner direction |
| Auto-label run | Ingest › Activity card | unchanged location, collapsed under "Advanced: auto-label run" | rarely used, not on the main path |
| "Export Dataset (YOLO)" quick action | Dashboard | deleted (Model › Export does it) | duplicate |
| Clustering run and Activate | Dashboard card, Ingest rail, Clusters empty state | unchanged in all three; the Clusters view now really shows it (F4) | the next step shows where the user is |

### 3.3 Data menu pages (wireframe, desktop)

```
+-------------------------------------------------------------------------------------------+
| Ingest  [Detector yolov11_small v1]                                                        |
| Bring images into the project.                                                             |
| Upload | Server folder | Sources | Import dataset | Activity          +-- Pipeline --------+ |
| -------------------------------------------------------------------- | Images      91     | |
| (Import dataset tab)                                                  |  by source table   | |
| +-- 1 Source -------------------------------------------------------+ | Pending detection 0| |
| | Server path [/data/source/my_dataset       ]  Format [Detect v]   | | Clusters           | |
| | or Upload an archive [Choose file]  up to 2.0 GiB, 200,000 files  | |  46 ready [Activate]| |
| +-------------------------------------------------------------------+ +--------------------+ |
| +-- 2 Preview ------------------------------------------------------+                        |
| +-- 3 Class mapping (by name) ---------------------------------------+                        |
| +-- 4 Options -------------------------------------------------------+                        |
| [Start import]  (the only primary in this tab)                                               |
| Past imports  ------------------------------------------------------------------------------ |
| Name | Format | Images | Boxes | Status | Started            (row opens /ingest/imports/[id]) |
+-------------------------------------------------------------------------------------------+
```

Upload tab: the drop zone comes first and Start is the primary. The detector card collapses into the header chip
plus a "Detector details" disclosure, because its facts (model, input size, class policy) are reference information.

---

## 4. Pages and views: current → target (c)

The tables use these actions: **Keep**, **Move** (where to), **Merge** (with what), **Hide** (behind `Refine`, `More`
or a disclosure; still one click away), **Rename**, **Delete** (only duplicates; each one names its surviving
twin).

### 4.1 Workspace shell (lane C)

Files: `routes/p/[project]/workspace/+page.svelte`, `components/workspace/{FilterBar,ViewBar,WorkspaceRail,RailClasses,BulkBar,DetailsPanel,VlmMenu}.svelte`,
`components/itemFilter/ItemFilterBar.svelte`.

```
+ Filter row ------------------------------------------------------------------------------------------+
| Class [any v]  Not class  [More filters]  203 matching . 1 ignored not listed   [chips x]   Next: ...> |
+ View bar --------------------------------------------------------------------------------------------+
| Clusters  Gallery  Detections  Review  |  Table  Map  Embedding                 [AI v]  [Details] |
+-------------+-------------------------------------------------------------------+-------------------+
| Queues      | (view header: one line, the view's primary on the right)          | Details (360 px)  |
|  Check ...  |                                                                   |                   |
|  To label   |  grid / list                                                      |                   |
| Classes     |                                                                   |                   |
|  legend (i) |                                                                   |                   |
| [+ New class]                                                                   |                   |
+-------------+-------------------------------------------------------------------+-------------------+
  Bulk bar: appears only with a selection, sticky at the bottom of the view area:
  | 12 selected . 9 images . 1 locked   [Label as... L] [Move... M] [Ignore v I] [More v]   [Clear Esc] |
```

| Control (now) | Action | Target |
|---|---|---|
| View tabs (Gallery, Clusters, Detections, Table, Review, Map, Embedding) | Keep, reorder | Clusters, Gallery, Detections, Review, then a divider, then Table, Map, Embedding. The served default view stays (`workflow_steps.py` `workspace_default_view`); B10 makes it Clusters once a clustering serves (walk P3-10). |
| Embedding tab missing while `?view=embedding` works (walk P3-5) | Fix | The tab shows when the view can render. Lane C root-causes `viewRegistry.ts:56` (`isEmbeddingVizAvailable(strategiesStore.methods.overlays)` read before the store loads), then re-evaluates availability when the store loads. |
| "Queues & classes" button (below lg) | Keep | an icon button (`menu` icon, label kept for screen readers) |
| "Details" button | Keep | `ghost`, with `\` in the tooltip |
| AI menu (`VlmMenu.svelte`) | Keep | no change |
| Next-step chip (`FilterBar.svelte:90-99`) | Keep, restyle | `chip-warn` link. When the served step has `action: adopt_detector_classes`, a banner also shows above Gallery, Clusters and Detections: "217 detections carry only the detector's guess. [Adopt detector classes…]". It opens the existing `lib/components/detector/AdoptDetectorDialog.svelte` (used on the Dashboard today; lane C imports it, lane D owns it). |
| Rail class legend ("Count colour: …", `RailClasses.svelte:125`) | Hide | an `info` icon button next to the "Classes" heading with the same text in a `ui/Popover` |
| Rail counts | Restyle | sans `tabular-nums`, right-aligned |
| Rail queue order | Keep served order | the first queue with work is marked by B2 `default_tab` (bold) |
| `+ New class` | Keep | `ui/Button` secondary, `plus` icon |
| Bulk bar with nothing selected (`BulkBar.svelte`: Deselect, Assign class…, Move…, Ignore▾, Restore…, Undo, Diverse sample…) | Hide | Not rendered with no selection (P10). "Diverse sample…" moves to the view header's `More` menu as "Select a diverse sample…". "Undo" moves to the toast plus the `Z` key, and also a `More` › "Undo last change" entry. |
| Bulk bar with a selection | Restructure | count summary · **Label as…** (primary, `L` or `/`) · Move… · Ignore ▾ (reasons) · More ▾ (Restore…, Label whole duplicate group…, Diverse sample from selection) · Clear (`Esc`) |
| "Select all N matching" (`BulkBar.svelte:96-103`) | Keep | inline after the count |
| Filter "More filters" (opens inline and pushes content; walk P3-7) | Change | opens as a `ui/Popover` anchored to the button, so the map and grid do not jump |

`BulkBar.svelte` becomes one contextual toolbar. Its `extra` snippet gives way to a `more: MenuItem[]` prop rendered
with `ui/Menu.svelte` (the `ellipsis` icon, label "More actions").

### 4.2 Clusters (lane C)

Files: `workspace/views/{ClustersView,ClusterListView,ClusterScope}.svelte`,
`components/clusters/{ClustersToolbar,ClusterListFilterBar,ClustersGrid,ClustersStatusBar,ClusterGridCard,ClusterDetailFilters,ClusterFacetBar,ClusterDetailGrid,ClusterRunPanel}.svelte`,
`lib/clusters/{clusterListController,slotInventory,clusterDetailActions}.svelte.ts`.

#### 4.2.1 Cluster list

```
| Clusters  [search: clusters and crops, e.g. "blurry closeup"  ] [Refine (2) v]   Sort [Largest first v] |
| chips of active refinements:  [Validation: unvalidated x] [Subject: largest x]                          |
| Region boxes: 166 Objects to confirm  ->  (one line, only when a region slot exists; link to the queue) |
| +--------+ +--------+ +--------+ +--------+                                                              |
| |4 thumbs| |        | |        | |        |   card: name . size; one meta line; at most one status     |
| |cup  24 | |        | |        | |        |   e.g. "100% labelled . detector: cup 50%"  [Mixed]        |
| +--------+ +--------+ +--------+ +--------+                                                              |
| footer: 24 of 203 . scroll for more     (sans, tabular-nums)                                             |
```

Empty states, all served (`empty_reason`, `empty_message`, `pending_partition`):

| `empty_reason` | Title | Body | Primary |
|---|---|---|---|
| `pending_activation` | "Clusters are ready to activate" | served `empty_message`, plus "46 clusters over 279 items" | **Activate it** (`ClusterRunPanel`) |
| `no_partition` | "No clusters yet" | served `empty_message` | **Cluster items** |
| filters match none | "No clusters match these filters" | the active chips | **Clear filters** |
| other | served message | `EmptyNextSteps` | the served next step |

| Control (now) | Action | Target |
|---|---|---|
| h1 "Clusters" + shortcuts button (`ClustersToolbar.svelte`) | Merge | sr-only h1. Shortcuts stay in the global `?` overlay (`ShortcutOverlay.svelte`, already in the layout). |
| Semantic search box | Keep | the only visible search |
| Crops / Regions target | Hide | Refine › Search in: Crops / Regions |
| Search by image (`ImageSearchPicker`) | Hide | an icon button (`scan`) inside the search field, tooltip "Search by image" |
| Item text + "Search text" | Hide | Refine › Text in the item (OCR) |
| Ignored | Hide | Refine › Show: Ignored bucket |
| Matching items | Hide | Refine › Show: Every matching item (list mode). Label "Matching items" → "Every matching item" |
| Subject All / Largest / +2nd | Hide | Refine › Subject |
| Clarity slider | Hide | Refine › Minimum clarity |
| Sort select | Keep | visible, the one ordering control |
| dominant class, detector class | Hide | Refine › Class / Detector class |
| Size min–max | Hide | Refine › Cluster size |
| Validation select + "Needs review" checkbox + "Needs review" chip | Merge | one Refine › Status group: Unvalidated · Needs review · Label disagreement · Mixed detector classes (served `flag_vocabulary`) |
| Pinned "Objects · inventory" card (`slotInventory`) | Delete from the grid | replaced by the one-line "Region boxes" link to the served `regions` queue count (the rail's "Objects"). A class filter on a slot-bound class still routes to `SlotGallery` (kept). |
| Card "agreement 100%", "detector cup 50% · conf 0.53", "validated", "Mixed detector classes" chips (`ClusterGridCard.svelte:103-199`) | Reduce | title = served `display_name` and size. One meta line: labelled share and the top detector class. At most one status chip, the highest of needs-review, mixed or unvalidated, in a warn or info tone, never danger. Agreement and confidence go to the details panel and the card `title`. |
| Status bar ("1 / 1", "all shown") | Restyle | sans, tabular-nums |

Behaviour fixes:
- The empty state uses the served list: `ClustersGrid.svelte` tests `list.clusterPager.items.length === 0`, not
  `gridItems`.
- After `ClusterRunPanel` reports a newly serving partition, `list.loadFirst()` runs. This fixes walk P2-3: the
  `onserving` callback exists at `ClustersGrid.svelte:259`, but the grid keeps the inventory card. Lane C confirms with
  the stubbed e2e test in section 8.
- **Keyboard into the grid.** Arrow keys move the focus between cluster cards (roving tabindex). `Enter` opens the
  focused cluster, and `.` from the list opens the first one (walk: "opening the first cluster needs a mouse click").

#### 4.2.2 Open cluster

```
| [<- All clusters]  cup  #1   24 items . 100% labelled . agreement 100%           [< ,]  [. >]          |
| [Label all 24 as... L]   [Assist v]   [More v]                     [Refine (0) v]  Sort [Outliers first v] |
| (with a selection the bulk bar replaces this row:)                                                       |
| 3 selected  [Label as... L] [Move... M] [Ignore v I] [More v]                       [Clear Esc]          |
| crop grid ...                                                                                            |
```

| Control (now) | Action | Target |
|---|---|---|
| "All clusters" back, title, number | Keep | the number in sans, tabular-nums; the uid in a tooltip and a copy button in details |
| Details "Label all 24 in this cluster…" (`ClusterDetails.svelte:40`) | Move | the header's primary, **Label all N as…** (`L`). Details keeps a secondary copy. |
| BulkBar "Assign class…" | Rename | **Label as…** (selection only). With a selection it is the primary and "Label all" turns secondary. |
| Label-then-next (`ws.labelThenNext`, `ClusterScope.svelte:150-154`) | Default on | After labelling a whole cluster the next cluster opens, so the loop is `L`, class, `Enter`. A setting in the dialog, "Open the next cluster after labelling" (checked), lets the user turn it off. |
| `,` / `.` | Keep | visible as Prev / Next buttons with key hints |
| Flag, Reprocess…, Accept VLM | Hide | More ▾: "Flag selected as a new class", "Reprocess…", "Accept the VLM's suggestions on this page" |
| Assist… | Keep | its own menu button (the VLM entry point) |
| Restore…, Undo, Diverse sample… | Hide | More ▾ |
| `ClusterDetailFilters` (source select, search, subject, clarity) | Hide | Refine › Label source, Search in this cluster, Subject, Minimum clarity |
| "Outliers first" toggle | Merge | Sort options: Cluster order, Outliers first, Detector confidence high first, low first |
| `ClusterFacetBar` "Sort by" (`:142-152`) | Delete | merged into the one Sort above |
| Facets (detector class, validation, review, confidence histogram) | Hide | Refine › Facets (opened by default when the cluster is mixed). The review facet shows served labels (B6), not ids like `vlm_low_conf`. |
| `ClusterProposalPanel` (VLM proposal) | Keep | as is, under the header, when a proposal exists |
| Crop details modal (info button) | Keep | |
| Mixed cluster | Add | When the served card has the `mixed_detector_classes` flag, the label dialog (`SelectionActionDialog`) says: "This cluster is mixed: the detector named cup in 50%, bottle in 38%. Labelling all 24 as cup also labels the others. Tip: sort by Outliers first and label or move those first." Walk P2-9. |

### 4.3 Gallery (lane C)

Files: `workspace/views/GalleryView.svelte`, `components/images/{ImageCard,ImageGrid}.svelte`,
`lib/images/imageGalleryController.svelte.ts`. New file `components/workspace/DuplicatesControl.svelte`; it uses
`lib/nearDup/nearDupPageController.svelte.ts` and does not import `NearDupStatusCard`.

```
| 91 photos . 25 test holdout not listed  [Has unlabelled objects]  Duplicates [All v (i)]  Sort [Newest v]  [More v] |
| grid of image cards: 4:3 thumb with boxes; chips: 7 detections . 3 labelled . [2 unlabelled]; file name; Details |
```

| Control (now) | Action | Target |
|---|---|---|
| "Has unlabelled objects" chip | Keep | |
| Segmented All images / One per group / Extra copies / Not duplicated | Rename and move into the Duplicates control | A `ui/Popover` button "Duplicates: All". Options: **All photos · One per group · In a group · Not in a group** (served `near_dup` values `collapsed`, `member`, `none` unchanged). Label "Extra copies" → "In a group". |
| (missing) near-dup purpose and run | Add | The same popover holds the copy in 4.3.1, the served status from `GET /near_duplicates/status` and one run button: "Find near-duplicates" with no run yet, or "Group N new photos" when `pending_count > 0`. A link "Thresholds and history" goes to Settings › Near-duplicate profiles. |
| Card "Near-dup · N" badge | Keep | rename "In a group of N", info tone |
| Card chips | Restyle | sans tabular-nums; "items" → "detections" (one word everywhere) |
| Card "Details" text button | Keep | |
| Lightbox hover button | Keep | |
| Footer "91 / 91" | Restyle | "91 of 91", sans |

#### 4.3.1 Near-duplicate copy (exact wording)

- Popover lead:
  > "Near-duplicates are photos that look almost the same, such as burst shots, re-uploads or video frames. Nothing is
  > deleted. You decide per export or training run whether to keep one photo per group. One rule is fixed: a group
  > never sits on both sides of the train/test split, so the test score is not inflated."
- Status, one of:
  - "Grouped 2026-11-11 04:47 · 0 groups among 91 photos · up to date";
  - "3 new photos are not grouped yet [Group new photos]";
  - "Not grouped yet [Find near-duplicates]".
- Group decision buttons in `NearDupGroupModal.svelte` (lane D owns the file):
  - "Keep together" → **Same scene**;
  - "Not a duplicate" → **Different photo**.
  - Lead: "Say whether these photos are near-duplicates. This changes how photos are grouped from the next run. It
    never deletes a photo."
- Export and Train option copy (lane D):
  > "Keep one photo per near-duplicate group (off: keep every photo). Useful when bursts or video frames would
  > over-weight a scene. Off by default, because duplicates of good examples are often worth keeping."

### 4.4 Detections and CropCard (lane C)

File: `components/CropCard.svelte` (6 mono badges, about 4 controls per card, 1,970 controls in the DOM at 477 cards,
walk section d).

| Element (now) | Action | Target |
|---|---|---|
| Rank · clarity badge ("3rd · soft") | Hide | shown on hover or focus only, and in details |
| Confidence badge ("100%") | Keep | sans tabular-nums, bottom-right |
| "object" region badge (top-left) | Keep | sans, holdout or info tone per slot |
| Lock icon, "H" source badge, class name | Merge | one footer line: class name plus a small source chip (H, I, D, V with a `title`) |
| Card buttons (details ⓘ, edit ✎, view) | Keep, show on hover/focus | icon buttons with aria-labels; the "view" overlay no longer clears the selection |
| Plain click selects (replace) but details keep the old focus (`desktop_f7map_5_det_other_card.png`) | Fix | A plain click sets the focus to the clicked card and selects it alone; Ctrl or Shift click adds to the selection. Details follow the last plain-clicked card; with 2 or more selected, details show `SelectionDetails`. |
| Tiny or black thumbnails (walk P2-6) | Fix (with B11) | the card uses the served `thumbnail_url`; on error, one retry, then a neutral "No preview" tile |

### 4.5 Review (lane C)

Files: `workspace/views/ReviewView.svelte`, `components/review/*`, `lib/review/*`.

```
| Check machine labels  #12 of 1,131  [Sort v] [search]       [Accept confident labels...]  [More v] [?] |
+---------------------------------------------+---------------------------------------------------------+
| Source (fits the pane height)                | Crop  [Context|Tight]                         id (copy) |
|  [hide boxes] [All regions]  (toolbar above, |  big crop with the yellow highlight                     |
|   not on the image)                          |  Proposed: car (detector 0.97)   Current: none          |
|                                              |  [Confirm Enter]  [Skip N]  [Discard D]                 |
|                                              |  class keys: [c car] [p person] ... [/ search all]      |
+---------------------------------------------+---------------------------------------------------------+
| filmstrip                                                                                              |
```

| Control (now) | Action | Target |
|---|---|---|
| Queue label + counter "#1 · 30 loaded · 477 total" (mono) | Restyle | "#12 of 1,131", sans tabular-nums; "loaded" goes to the title |
| No queue → all detections | Change | served `default_tab` (F5) |
| Sort, semantic search | Keep | |
| Key-hint text line (`ReviewHeader.svelte:120-144`) | Hide | moved into the `?` overlay; the buttons carry their keys as `Kbd` |
| "Dismissed" | Hide | More ▾ "Dismissed items…" |
| Live "N new" pill | Keep | |
| hide boxes / All regions (overlapping the image; walk) | Move | a slim toolbar row above the source image |
| Source pane black void (walk P2-13) | Fix | the source pane's height follows the image's aspect, capped at the pane; the space goes to the crop pane |
| Confirm (primary), Skip, Discard, Undo | Keep, and Undo → toast | on validated items Confirm is not rendered (F5 rule 2); Undo stays on `Z` and in the toast |
| Class buttons + "/ search all classes" | Keep | the recent classes first, numbered 1-9 (B12 keymap actions) |
| "More like this" | Keep | |
| Accept confident labels… | Add | only on `check_machine` (F5 rule 3), secondary button |
| Details panel (ItemDetails): Open in the image editor, Assist…, Reprocess…, history, source image, Open image, siblings | Reduce | **Edit boxes** (one link, F7); Assist… and Reprocess… under a More ▾; history collapsed to the last 3 entries with "Show all"; siblings kept |

### 4.6 Table, Map, Embedding, and the label dialog (lane C)

#### 4.6.1 Table

Keep: the Detections and Images toggle, column sort, the Columns chooser, row select. Restyle numeric columns as
right-aligned tabular-nums (done in `ui/DataTable.svelte` by lane B).

#### 4.6.2 Map

| Control (now) | Action | Target |
|---|---|---|
| "3 located · 88 without location" | Keep, extend | "12 of 63 located photos match · 28 without location", from B5. When 0 match: an overlay "None of the 63 located photos match the filter. [Clear filters]" (one primary). |
| Location policy chip | Keep | |
| "Shift+drag selects an area, or [draw a lasso]" | Keep | "Draw an area" button (ghost, `square-dashed` icon) with the Shift+drag hint in its tooltip |
| Cells in view (keyboard list) | Keep | |
| Cell details → photo click | Fix | F7 item 5 |
| "Show in Gallery" | Keep | |

#### 4.6.3 Label dialog: queued state (UI side of walk P2-1)

- **Files:** `lib/itemFilter/selectionActionController.svelte.ts`, `components/itemFilter/SelectionActionDialog.svelte`,
  `lib/labelOps/runLabelOp.ts`.
- **Change.** `runLabelOp` passes `onUpdate` to `lib/jobs/followJob.ts` (the option exists). The controller exposes
  `job: JobView | null`.
- **The dialog shows the served state:**
  - `queued`: "Waiting for a worker since 18:05 (queue `jobs`). You can close this; the change runs in the background
    and shows in Label history." (from `JobView.status`, `queued_at`, `queue`);
  - `running`: "Labelling… 120 of 412" (from `JobView.progress`).
- **Buttons.** A secondary **Run in background** closes the dialog, keeps the follow alive and shows a toast when it
  finishes. No queue position is shown, because the API does not serve one.

#### 4.6.4 Embedding

Keep every `EmbeddingPlot` control (12d lane 4C keep-list). Only the tab visibility fix above.

### 4.7 Details panel (lane C)

| Kind | Target content (top to bottom) |
|---|---|
| image (`ImageDetails.svelte`) | preview with boxes (click = lightbox) · file name (title, sans, truncate, full name in `title`) · facts as a two-column `dl` in sans with tabular-nums (size, source, taken, indexed) · "7 detections · 3 labelled · 4 in the test holdout (not listed)" · **Edit boxes** · Full size icon · Show its 7 detections · near-dup line "In a group of 4 · [Review this group]" (opens `NearDupGroupModal`) · "Back to the cell" when it came from a map cell |
| item (`ItemDetails.svelte`) | **Edit boxes** · `CropMetaPanel` (class, source, confidence) · More ▾ (Assist…, Reprocess…) · history (last 3) · source image with siblings |
| cluster (`ClusterDetails.svelte`) | name, size, labelled share, agreement · Label all N as… (secondary here) · class-wide counts as a small table · uid with a copy button |
| selection (`SelectionDetails.svelte`) | count sentence · per-class counts table (sans, tabular-nums, right-aligned) |
| cell (`CellDetails.svelte`) | summary · photo grid (click pushes history, F7 item 5) · Show in Gallery · Close |

### 4.8 Ingest (lane D)

Layout in 3.3.

| Tab | Now | Target |
|---|---|---|
| Upload | Detector card, drop zone, source tag, skip check, Advanced, Start (primary) | Drop zone first. Start is the primary. Source tag and "Skip the already-indexed check" move behind Advanced. The detector card becomes a "Detector details" disclosure under the header chip. |
| Server folder | Bulk ingest + Server-path ingest | Keep both. Each is a `Card` with a one-line purpose. |
| Sources | Sources panel | Keep |
| Import dataset | separate page | F1 |
| Activity | Region worklog + region stage + clustering handoff + auto-label | Keep. Auto-label is collapsed under "Advanced: auto-label run". The source-tag tables in the rail use sans labels: the served source names stay text but lose mono. |
| Pipeline rail | Ingest status, pending detection, clusters | Keep. "Cluster items" is a secondary button in the rail, so Start is the only primary on Upload (walk section d: two primaries). |

### 4.9 Data quality (lane D)

Files: `routes/p/[project]/quality/{+layout.svelte,+page.ts,audit/+page.svelte,history/+page.svelte}`; new
`quality/ready/+page.svelte`; `components/audit/*`, `components/labelOps/*`, `components/dashboard/DataQualityCard.svelte`;
new `lib/api_quality.ts`, `lib/types_quality.ts`.

`quality/+page.ts` redirects to `/quality/ready`. Delete `quality/near-duplicates/+page.svelte`, the `near-duplicates`
entry in `SUBPAGE_LABELS`, and the tab entry.

#### 4.9.1 Ready to train (new default tab)

```
| Data quality                                                                                  |
| Check the dataset before you train: enough checked examples per class, a fair test set,       |
| and machine labels you can trust. Each line links to the items that need work.               |
| Ready to train | Accuracy audit | Label history                                               |
| ------------------------------------------------------------------------------------------- |
| Verdict: [Not ready: 2 blocking, 3 warnings]                       checked 18:20 [Refresh]   |
| BLOCK  Test set: 2 classes have under 5 test examples (bottle 0, book 0)     [Open Export]    |
| WARN   Class balance: 4 classes under 500 checked examples (car 74 ...)     [Label more ->]  |
| WARN   Detector labels not checked: 1,131 (car 412, person 380 ...)          [Review ->]      |
| WARN   Accuracy audit: none yet                                             [Draw a sample]  |
| OK     Near-duplicate groups are up to date; no group spans train and test                   |
| OK     Every exported photo is fully labelled                                                 |
| Coverage per class (table: class | validated | machine only | detector only | test)          |
```

- Rows come from B3 `GET /quality/readiness`, rendered as served (label, severity, message, count, link).
- The link opens the Workspace with the served query, or another section.
- Severity uses an icon (`circle-check`, `triangle-alert`, `circle-x`) and a word, never colour alone (WCAG 1.4.1).

#### 4.9.2 Accuracy audit

| Element (now) | Action | Target |
|---|---|---|
| Empty state (icon, text, Draw sample) | Keep | (good per walk) |
| `AuditStartForm` (sample size, per class) | Keep | |
| 78 per-class chips (walk P2-11) | Replace | `AuditClassTable` sorted by precision, with interval bars; "insufficient sample" rows greyed |
| Pending sample (only a Review link) | Add a gallery | "300 drawn · 112 labelled · 188 waiting · [Label them in Review]" (primary) plus a strip of the first 24 waiting crops (`CropResultGrid`, read-only, from `GET /crops` with `review_tab=audit`). "See all" opens Detections with `queue=audit`. |
| Outcomes (agree, detector wrong, VLM wrong, both wrong) | Link | each count opens Detections with `audit_outcome=<value>` (B4) |
| `ConfusionMatrix` | Link | each off-diagonal cell opens Detections with `audit_outcome` plus `detector_class_names` and `class_name` |
| Counter mismatch "192 waiting" vs "300 waiting" (walk P2-10) | Fix | B13 |

#### 4.9.3 Near-duplicates: where the pieces go (lane D unless noted)

| Piece | Target |
|---|---|
| `NearDupStatusCard` (status, Build, Full rebuild, Group new items) | Workspace Gallery Duplicates control (lane C, new `DuplicatesControl.svelte`). Full rebuild sits under Settings › Near-duplicate profiles › Runs. Then delete `NearDupStatusCard.svelte`. |
| `NearDupGroupList` (Images/Items × Duplicates/Leak groups) | Image groups → Gallery "One per group" (lane C). Item-level leak groups → Settings › Near-duplicate profiles › Groups (advanced, a `DataTable` linking each group to Detections with the `near_dup` scope). Then delete `NearDupGroupList.svelte`. |
| `NearDupDecisionsPanel` (pinned pairs) | Settings › Near-duplicate profiles › Pinned pairs (in `routes/p/[project]/settings/presets/[kind]/+page.svelte` when `kind` is the near-dup preset kind) |
| `NearDupRunsTable` | Settings › Near-duplicate profiles › Runs |
| `NearDupProfilePanel` (thresholds) | Delete: the preset editor already edits the profile |
| `NearDupGroupModal` | Keep, with the copy in 4.3.1 |
| `lib/nearDup/nearDupPageController.svelte.ts` | Keep (used by lane C's control). Delete only what loses its last consumer, and prove it with `rg`. |
| Dashboard `DataQualityCard` "Open near-duplicates" | → "Open duplicates in the Gallery" (`/workspace?view=gallery&near_dup=collapsed`) |

#### 4.9.4 Label history

- **Keep.** Row title: "Labelled 24 items as bottle". Subtitle: "from cluster #3 · by a person · 2026-11-12 23:06"
  (served `count`, `class_name`, `target_cluster_uid`, `label_source`, `at`; walk P3-3).
- **Fix "1 items".** Use a plural helper.

### 4.10 Dashboard (lane D)

Files: `routes/p/[project]/dashboard/+page.svelte`, `components/NextStepsCard.svelte`, `components/DatasetStats.svelte`,
`components/dashboard/DataQualityCard.svelte`. The page is 5,000 px tall at 430 px wide (`mobile_dashboard.png`).

```
| Dashboard                                                                             |
| Next steps (stepper in loop order; the first not-done step is expanded with its one action) |
| [KPI] Photos 91  [KPI] Validated 285  [KPI] Machine, unchecked 0  [KPI] No class 217  |
| Clusters card (ClusterRunPanel)          | Data quality card (verdict from B3 + links) |
| Class balance (validated)                                                             |
| v Pipeline details (collapsed: Labeled by source, Embedding, Last clustering, Objects, |
|   Objects coverage, Unlabeled, In-flight pipeline, Detections table)                 |
| v Recent labels (collapsed)                                                          |
```

| Element | Action | Target |
|---|---|---|
| Next steps (11 rows, 2 primaries; walk section d) | Keep, restyle | A stepper. Done steps are collapsed to one line, and only the first open step shows its button, as the primary. |
| Quick actions: "Activate one" (a link styled like a button), Export Dataset (YOLO), Refresh | Delete row | VLM activation stays in Next steps and Settings. Export is on Model › Export. Refresh → the page's `PageHeader` action. |
| Pipeline stats sections | Hide | one "Pipeline details" disclosure |
| KPI tiles | Add | from `GET /stats/dataset` (`validated`, `unlabeled`, and B14's machine-unchecked row) |
| Recent labels | Hide | disclosure |

### 4.11 Other pages (lane D, headers and controls only)

| Page | Change |
|---|---|
| Classes | Title "Class management" → **Classes** (match the nav). Validated chips in sans tabular-nums. "incl. 10 test" → "10 in test". Per-row Rename / Deprecate → one More ▾ per row (Rename, Merge into…, Deprecate). Keep: Merge classes, Add class (primary), group select, hotkey field, the detector disclosure. |
| Export | The hand-rolled `<h1 class="text-2xl">` becomes `PageHeader` "Export dataset". Banner numbers in sans. Raw parameter chips ("require_fully_labeled_images: false", "skipped items: 0 no image id …") → a `dl` with plain labels ("Only fully labelled photos: off"). Class table: tabular-nums. Mobile: horizontal scroll with a sticky Class column. Re-export stays the one primary. |
| Train | `PageHeader` "Train model". The monitoring links (Grafana, Prometheus, OpenSearch Dashboards, MLflow) move to the header `actions` as one "Dashboards ▾" menu. The dataset version "current (current symlink — latest)" → "Latest export · 2026-11-11 04:47" (walk P3-9). Preflight rows show the served label (B7), not the check id ("WARN class_balance" → "Class balance"); severity uses an icon and a word. Past runs: the job id goes to a tooltip and copy button, with the run name or start time as the label. Model size n/s/m/l/x and Profile stay; Classes, Augmentation and Hyperparameters stay collapsed. "Start anyway (bypass blocking checks)" goes under Advanced. Start training stays the one primary. |
| Models | `<h1 class="text-xl">` → `PageHeader`. Model ids stay `<code>`. Card stat labels lose uppercase. |
| Bake-off | Restyle only. |
| Projects (`routes/projects/+page.svelte`) | Use the app `TopNav` shell (it has its own header now) and `PageHeader` "Projects". Row actions (Open, Edit, Copy settings, Pause pipeline, Archive, Delete): **Open** plus More ▾ (Edit, Copy settings, Pause pipeline, Archive, Delete in danger tone). A row click opens the project. The capacity card becomes a footer line with a details disclosure. A `building` project shows "Building…" and its Open is disabled until the served status is active (walk P3-4). |
| Combine (`routes/projects/combine/**`) | `PageHeader`; no other change |
| 404 (`+error.svelte`, `shell/ErrorPage.svelte`) | Keep the nav (desktop has it). Mobile shows the hamburger too. |

### 4.12 Advanced-feature keep-list (none lost)

Every feature below exists today and keeps a home. "Refine" means the view's Refine popover, and "More" the view's
or row's More menu.

| Feature | Today | Ends up |
|---|---|---|
| Semantic text search (clusters, review) | toolbar | visible search box |
| Search target Crops / Regions | toolbar segmented | Refine › Search in |
| Search by image (crop region) | toolbar button | icon in the search field |
| Item text (OCR) search | toolbar input | Refine › Text in the item |
| Ignored bucket | toolbar toggle | Refine › Show |
| Matching items mode | toolbar toggle | Refine › Show › Every matching item |
| Subject All / Largest / +2nd | toolbar | Refine › Subject (list and cluster) |
| Clarity minimum | slider | Refine |
| Cluster sort | select | visible Sort |
| Dominant class, detector class filters | selects | Refine › Class |
| Cluster size min/max | inputs | Refine › Cluster size |
| Validation, needs review, flags | select, checkbox, 4 chips | Refine › Status (one group) |
| Region galleries (slot inventory) | pinned card | rail queue "Objects" plus the "Region boxes" line; class filter → SlotGallery |
| Open-cluster label source filter | select | Refine › Label source |
| Outliers first, confidence order | toggle, facet sort | Sort options |
| Facets (detector class, validation, review, confidence) | facet block | Refine › Facets |
| Flag as new class | button | More |
| Reprocess (selection, item, image) | buttons | More (cluster, item details) |
| Accept VLM suggestions on page | button | More (cluster) |
| Assist (VLM) and proposals | button, panel | Assist menu button, proposal panel |
| Move to cluster/class | button | bulk bar |
| Ignore with reasons | split button | bulk bar |
| Restore ignored | button | More |
| Undo | button | toast, `Z`, More › Undo last change |
| Diverse sample | button | More › Select a diverse sample |
| Label whole duplicate group | button | More (when in a group scope) |
| Drag crops onto rail classes | drag and drop | kept |
| Class hotkey letters | rail, Classes | kept; Classes page edits them |
| Queues | rail | kept, served order, default marked |
| AI menu (VLM) | view bar | kept |
| Lightbox | card button, details | card button, details preview, `Shift+Space` |
| Image editor | details buttons | **Edit boxes** (one name), `Shift+O` |
| Review Context / Tight, highlight, siblings | panes | kept |
| hide boxes, All regions | over the image | toolbar above the source |
| More like this | button | kept |
| Dismissed panel | header button | Review More |
| Live "N new" refresh | pill | kept |
| Review sort and search | header | kept |
| Region slot review and multi-box edit | Review | kept |
| Reject reason prompt | dialog | kept |
| Table Detections/Images, columns, sort | Table | kept |
| Map area (Shift+drag, lasso), cell list, without-location list, policy chip | Map | kept |
| Embedding plot controls | Embedding | kept (12d keep-list) |
| Near-dup run (incremental, full) | Quality | Gallery Duplicates control; full rebuild in Settings › Runs |
| Near-dup thresholds (profile) | Quality and Settings | Settings preset page only |
| Near-dup pinned pairs, run history, item-level groups | Quality | Settings › Near-duplicate profiles sections |
| Near-dup same/different decisions | modal | modal (from image details) |
| Accuracy audit: draw sample, per-class, confusion | Quality | Quality › Accuracy audit (plus galleries) |
| Label history, op details, undo | Quality | kept |
| Upload: source tag, skip indexed check, advanced | Upload | Upload › Advanced |
| Server folder bulk + paths; Sources; region worklog; region stage pause; auto-label run; clustering handoff | Ingest | Ingest tabs (auto-label under Advanced in Activity) |
| Import: mapping table, accept suggestions, reuse a previous mapping, options, resume, cancel, undo | `/datasets/*` | Ingest › Import dataset, `/ingest/imports/[id]` |
| Export: version tag, fully labelled only, collapse near-dups, filter, artefact files, per-class counts, freeze holdout | Export | kept |
| Train: every current control | Train | kept (advanced groups collapsed) |
| Training cohorts table | Train | kept |
| Models: share, unload, class mapping | Models | kept |
| Bake-off | Bake-off | kept |
| Settings: every section and search | Settings | kept |
| Projects: edit, copy settings, pause, archive, delete, combine, capacity | Projects | Open + More; combine kept; capacity disclosure |

---

## 5. Visual system: tokens, type and the rules that enforce them (d)

Lane B owns this section. It runs first, in wave 1, so that wave 2 builds on the final classes.

### 5.1 Fonts

- Add `@fontsource-variable/inter` and `@fontsource-variable/jetbrains-mono` to `frontend/package.json` dependencies.
  Fontsource's documented pattern is `npm install @fontsource-variable/<name>`, `import "@fontsource-variable/<name>";`
  and the family `"<Name> Variable"` ([Fontsource variable fonts](https://fontsource.org/docs/getting-started/variable)).
  - Both fonts are licensed OFL-1.1 ([Fontsource Inter](https://fontsource.org/fonts/inter/install),
    [rsms.me/inter](https://rsms.me/inter/)).
  - The files are bundled by Vite into the static build, so a LAN-only deployment needs no CDN.
  - Add both to `ATTRIBUTION.md`, with the licence and the source URL.
  - Pin exact versions, per the project's package rule. The lane reads each package's README and changelog for the
    pinned version and cites them in its hand-back.
- Import both once in `routes/+layout.svelte`.
- In `app.css` `@theme`, replace the stack at `:128-130`:

```css
--font-sans: 'Inter Variable', system-ui, 'Segoe UI', 'Noto Sans', sans-serif;
--font-sans--font-feature-settings: 'cv11', 'ss01';
--font-mono: 'JetBrains Mono Variable', ui-monospace, 'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace;
```

- `html, body` use `font-family: var(--font-sans)`. The `font-feature-settings` line at `:130` moves into the token:
  `cv11` and `ss01` are Inter's single-storey a and open digits, and today they do nothing with system fonts.
- The `kbd` rule (`:425`) uses `var(--font-mono)`.
- Add to `@layer base`:

```css
code { font-family: var(--font-mono); font-size: 0.92em; }
```

### 5.2 Scale (unchanged values, enforced)

| Token | px / line | Use |
|---|---|---|
| `text-caption` | 11/16 | captions of 12 px or more importance only; never body copy (MD3 labelSmall 11/16) |
| `text-meta` | 12/16 | secondary text, chips, table meta (MD3 bodySmall 12/16) |
| `text-control` | 13/20 | buttons, inputs, tabs (HIG macOS body 13) |
| `text-body` | 14/22 | body copy, table cells (MD3 bodyMedium 14/20) |
| `text-title` | 16/24 | card and section titles (MD3 titleMedium 16/24) |
| `text-h1` | 20/28 | every page title, through `PageHeader` |
| `text-kpi` | 28/32 | KPI numbers only |

Weights: 400 text, 500 controls, tabs and labels, 600 titles and h1. No 700.

### 5.3 Rules

1. **Mono is only for code, ids, hashes and paths**, written as `<code>` (inline) or in a code editor (JSON/YAML
   textareas, `LogTail.svelte`). The `font-mono` class is not used anywhere else.
2. **Numbers use the sans face.** Counts in tables, chips, rails, KPIs and footers get `tabular-nums` (Inter `tnum`).
   This satisfies the owner's "mono only for ids, numbers and paths": mono is allowed there, and sans tabular figures
   read better inside sentences.
3. **Sentence case.** No `uppercase` class. Section labels use `text-meta font-semibold text-fg-2`, and table headers
   `text-meta font-medium text-fg-2`.
4. **Raw ids and API parameter names are not labels.**
   - Show the served `label` / `display_name` / `file_name`, and put the id in `title` plus a copy button.
   - Where no label is served, the lane adds it to the backend (B6, B7). `lib/humanizeId` is not the answer for new
     surfaces.
5. **Every page title uses `PageHeader`.** Workspace views use an `sr-only` h1.
6. **One primary per view state** (P3), checked at runtime by e2e (section 8).

### 5.4 Codemod (lane B, mechanical, one commit per rule)

| From | To | Note |
|---|---|---|
| `text-xs` | `text-meta` | identical 12/16 |
| `text-sm` | `text-body` | 14/20 → 14/22 |
| `text-base` | `text-title` | identical 16/24 |
| `text-lg` | `text-title` | 12 sites; the cluster scope title becomes `text-title font-semibold` |
| `text-xl`, `text-2xl` | `text-h1`, and only in page h1s that then move to `PageHeader` | 5 sites |
| `uppercase` (+ `tracking-wide(r)`) | removed; the text becomes sentence case | 90 sites; keycap legends stay in `kbd` |
| `font-mono` on a number-only node (`{x.toLocaleString()}`, `formatMetric`, `toFixed`, `{n} / {m}`) | `tabular-nums` | e.g. `RunResults.svelte:170-248`, `WorkspaceRail.svelte:55`, `ClusterFacetBar.svelte:66-131` |
| `font-mono` on prose or labels | removed | e.g. `ClustersStatusBar.svelte:35-61`, `ImageDetails.svelte:45`, `SelectionDetails.svelte:22`, `ClusterDetails.svelte:47,55`, `ReviewHeader.svelte:77`, CropCard badges |
| `font-mono` on an id, path or hash | wrap the value in `<code>`, drop the class | e.g. `CropSourceContext.svelte` id, `ClusterDetails.svelte:23` uid, the import path input |
| `font-mono` on an input that edits a path or JSON | keep, as an allow-listed code editor | each file listed with its reason |

- Each rule's codemod runs from `artifacts_local/` and is not committed: the repo keeps no dead tooling.
- The lane reviews every `font-mono` site by hand. There are 618; the table gives the rule, and the site decides.

### 5.5 Scan tests (extend `lib/designRules.scan.test.ts`)

Each rule comes with a negative control in the existing "the patterns catch each spelling" block:
- **`TW_SIZE`**: `/\btext-(xs|sm|base|lg|xl|[2-9]xl)\b/` forbidden in product source.
- **`MONO_CLASS`**: `/\bfont-mono\b/` forbidden, except an `ALLOWED` entry per code-editor file with its reason.
- **`CODE_PROSE`**: `/<code[^>]*>[^<{]*\b[A-Za-z]{3,}\s+[A-Za-z]{3,}\s+[A-Za-z]{3,}/` forbidden, so a `<code>` that
  wraps a sentence fails.
- **`UPPERCASE`**: `/\buppercase\b/` forbidden.
- **`RAW_H1`**: `/<h1(?![^>]*\bsr-only\b)[^>]*class=/` forbidden outside `lib/components/ui/PageHeader.svelte`.
- **`GLYPH_ICON`**: extend it with U+2039/U+203A (single guillemets), per the wave-1 review note. Fix the two
  `ScrollStrip.svelte:118,137` uses first.

### 5.6 Components (lane B)

- `ui/DataTable.svelte`: numeric columns (`align: 'end'`) get `tabular-nums`; headers are sentence case.
- `ui/PageHeader.svelte`: unchanged API. Every page uses it.
- `ui/Tabs.svelte`: below `md`, the strip scrolls horizontally with an edge fade and keeps the current tab in view.
  Today "Activity" is clipped at 430 px (`mobile_ingest_upload.png`).
- Interactive chips (the 20 px tier) keep at least 4 px spacing between neighbours. This meets the WCAG 2.5.8 spacing
  exception: "a 24 CSS pixel diameter circle ... do not intersect another target".
- Buttons touched by lanes C and D use `ui/Button.svelte`. Untouched `.btn` sites are not mass-migrated (section 9).

---

## 6. Mobile at 430 px

Evidence: `mobile_ws_gallery.png`, `mobile_ws_clusters.png`, `mobile_ws_review.png`, `mobile_export.png`,
`mobile_dashboard.png`, `mobile_ingest_upload.png`.

| Area | Now | Target (lane) |
|---|---|---|
| Nav | hamburger drawer, works | keep (D: the new order) |
| Workspace chrome | Filters row, then view select + AI + "Queues & classes" + Details, then 2 bulk-bar rows (≈260 px before content) | One row: view select, Filters icon, Queues icon, Details icon, AI. The bulk bar appears only with a selection, as a bottom bar. (C) |
| Clusters | 9 control rows before the first card (≈570 px) | search plus a Refine button, then the cards (C) |
| Review | the source pane first, the crop below the fold | the crop pane first, the source collapsed to a 120 px strip that expands on tap (C) |
| Tables (Export class table, cohorts) | columns cut off | horizontal scroll, sticky first column (B in `DataTable`, D on Export) |
| Ingest tabs | "Activity" clipped | scrollable tabs with a fade (B) |
| Dashboard | 5,000 px | about 1,600 px after 4.10 (D) |
| Settings | good | keep |

---

## 7. Backend data and routes (e)

Owner rule: missing data means a new route or field, with its contract and tests, never a client workaround. Every
item regenerates `contracts/` (`make contracts`) in the same commit and adds a contract or service test. Lane A owns
all of them.

| # | What | Where (files) | Tests |
|---|---|---|---|
| B1 | `POST /curation/projects/{project}/label_ops/accept_machine` (F5). See the B1 details after the table. | `src/routers/curation/label_ops.py`, `src/schemas/curation/label_ops.py`, a new `src/services/curation/label_ops/accept_machine.py` (modelled on `src/services/curation/detector_adoption.py` and `label_ops/adopt_merge.py`), `src/domain/ingest_class_sources.py` (new source), `src/domain/label_ops.py` | dry-run counts by class; each lock case untouched; audit-sample items held back; undo restores the machine class; contract |
| B2 | `GET /review/tabs` adds `default_tab: str \| None`: the first tab in served order with `count > 0`, else null. | `src/routers/curation/_review_tab_models.py` (`ReviewTabsResponse`), `src/services/curation/review_tabs.py` | unit: order and empty |
| B3 | `GET /curation/projects/{project}/quality/readiness` (F2). See the B3 details after the table. | a new `src/routers/curation/quality.py` and `src/services/curation/dataset_readiness.py`; refactor `src/services/training/preflight_run.py` and `preflight_checks.py` so the dataset-level checks run without a train request, with Train's preflight calling the same functions (one implementation); `src/services/curation/cohort_summary.py`, `audit.py`, `holdout.py` | each check on a seeded fixture; link query parses as a Workspace URL; Train preflight unchanged (existing tests) |
| B4 | `ItemFilter.audit_outcome: list[Literal['agree','detector_wrong','vlm_wrong','both_wrong']]`, as a query param too, so audit outcomes and confusion cells list their crops. | `src/services/curation/item_filter.py`, `src/routers/curation/_item_filter_params.py` | filter clause test; `GET /crops?audit_outcome=` contract |
| B5 | `MapCellCollection.n_located_matching: int` and `n_unlocated_matching: int` under the request's filter (F6). | `src/schemas/curation/map.py:59`, `src/services/curation/map_cells.py` | counts with and without a class filter |
| B6 | `FacetCount.label: str \| None`, served for the `review_tabs` facet of `GET /clusters/{cluster_uid}/facets`. | `src/schemas/curation/clusters.py:141`, `src/services/curation/cluster_facets.py` | label present for every served value |
| B7 | `PreflightCheck.label: str` (e.g. "Class balance") for every check. | `src/services/training/preflight_checks.py:43` and each check | every check has a label |
| B8 | Geotagged public photos for the test project (F6). See the B8 details after the table. | new `scripts/datasets/fetch_geotagged_commons.py` (reuses `fetch_hires_set.api_get`, `ALLOWED_LICENSE`), new `scripts/datasets/manifests/commons_geotagged_60.json`, a short section in `docs/CURATION.md` on the test data | offline unit test of the licence and GPS filters over a recorded API fixture (no network in CI) |
| B9 | Inventory card fix needs no new route. The "Region boxes" line uses the served `regions` queue count from `GET /review/tabs`. | none | none |
| B10 | `workspace_default_view` becomes `clusters` once a partition is serving (walk P3-10). | `src/services/curation/workflow_steps.py:196` | unit |
| B11 | Thumbnails: root-cause the broken and black cluster/detection thumbnails (walk P2-6) with a repro first; then the served `thumbnail_url` of a crop under 32 px on its long side points at the padded preview (`GET /crops/{crop_id}/preview` with context padding), so the server, not the card, decides. | `src/services/curation/wire.py` (thumbnail url), `src/services/curation/image_serving.py` | the repro as a test; small-crop url rule |
| B12 | Keymap actions `class_picker.recent_1` … `recent_9` (keys `1`-`9` in the picker), so a recent class is one key. | `contracts/json/keymap_actions.json` source, the keymap service; frontend `lib/keymapFallback.ts` is lane C | keymap validate test |
| B13 | Audit counters disagree between calls (walk P2-10: "192 still waiting" vs "300"). Root-cause first. | `src/services/curation/audit.py`, `src/routers/curation/audit.py` | a test reproducing the mismatch |
| B14 | `GET /stats/dataset` `labeled.rows` gains a served row for detector machine labels ("Detector, unchecked"), so they are not counted under "Other" (walk P2-7). The `label` workflow step copy names the queue the work is in ("1,131 machine labels to check in Check machine labels"). | `src/services/curation/stats_dataset.py`, `src/routers/curation/_stats_models.py`, `src/services/curation/workflow_steps.py` | stats row test; step copy test |
| B15 | Gallery list boxes miss region boxes compared with the image detail (walk P3-8). | `src/services/curation/image_counts.py` (`top_hits` includes) | equality test list vs detail |

**B1: `POST /curation/projects/{project}/label_ops/accept_machine` (F5).**
- **Request** `AcceptMachineRequest`:
  - `dry_run: bool = true`;
  - `min_confidence: float` (0.5-1.0);
  - `sources: list[Literal['detector','vlm']] = ['detector','vlm']`;
  - `filter: ItemFilter | None`.
- **Response** `AcceptMachineResponse`:
  - `dry_run`, `items`;
  - `by_class: list[{class_name, count}]`;
  - `held_back: {holdout, audit_sample, excluded, locked}`;
  - `op_id`, `job_id`, `remaining`.
- **Writes.** Each item's current machine class becomes validated, with the new label source `human_bulk_accept`.
- **Never writes:**
  - human, import-validated, holdout or excluded items (`src/clients/occ_locks.py`);
  - items in an open audit batch;
  - items whose class is deprecated.
- **Label op.** Kind `accept_machine`, listed by `GET /label_ops` and undone by `POST /label_ops/{op_id}/undo`. Cap
  20,000 items, like adopt.

**B3: `GET /curation/projects/{project}/quality/readiness` (F2).**
- **Response** `DatasetReadiness`:
  - `verdict: 'ready'|'warn'|'blocked'`;
  - `checks: list[ReadinessCheck]`, each with `id`, `label`, `severity: 'ok'|'warn'|'block'`, `message`,
    `count: int | None` and `link: {section, query} | None`;
  - `coverage: list[{class_name, validated, machine_only, detector_only, test}]`;
  - `computed_at`.
- **Checks:**
  - class minimums (`dataset_thresholds`);
  - test holdout frozen, and the per-class test floor;
  - machine labels not checked, per class;
  - accuracy audit present, pending, and per-class precision with its interval;
  - near-dup groups current, and leak containment (no group spans train/test);
  - unlabelled objects on exportable photos;
  - export current (`export_readiness`).

**B8: geotagged public photos for the test project (F6).**
- **Selection:**
  - about 12 sites × 5 photos;
  - `list=geosearch&gsnamespace=6` (the File namespace) around fixed coordinates, including two sites a few
    kilometres apart, so the map shows cell merging at low zoom;
  - JPEG only, licence in `ALLOWED_LICENSE`;
  - EXIF `GPSLatitude` present in `iiprop=metadata`, so the position is real camera GPS;
  - street scenes, so the COCO detector finds objects.
- **Manifest per file:** title, author, `LicenseShortName`, licence URL, description URL, SHA-256, lat/lon,
  `DateTimeOriginal`.
- **Output.** The fetch writes into a gitignored directory together with a `CREDITS.csv`.
- **Use.** The set is ingested into `coco-e2e` with the existing bulk ingest; that ingest is a coordinator step on the
  live stack, not part of the lane.

---

## 8. Lanes, order and acceptance (f)

The coordinator runs the waves as two parallel lanes per wave:
- Each lane works in its own worktree and branch and commits one concern at a time, with hooks on, no push and no
  merge.
- Lanes run light checks only.
- The coordinator merges with `--no-ff` and runs **one** sweep per wave.
- Every brief includes the 8-line corner-cut checklist (`artifacts_local/briefs/wave3_lanes_1112.md`).
- Every brief asks the lane to read the official docs for each package it touches and to cite them: Svelte 5 runes,
  SvelteKit routing and layouts, Tailwind v4 `@theme`, Fontsource, Playwright locators, FastAPI and Pydantic v2.

```
wave 1:  Lane A (backend served data)  ||  Lane B (design system, fonts, codemod, scan rules)
           -> coordinator merges A, B; one sweep; contracts and types available
wave 2:  Lane C (Workspace, Review, details, map)  ||  Lane D (IA, Ingest/Import, Data quality, pages)
           -> coordinator merges C, then D; one sweep; live screenshots on op-gpu; owner review
```

Wave 2 waits for wave 1 because lane B's codemod touches every `.svelte` file. Starting C and D on the new classes
avoids a write-then-rewrite of the same lines.

### Lane A: backend served data (Sonnet, high effort: write paths and the lock rule)

- **Owns:** every backend file named in section 7, `contracts/`, `scripts/datasets/fetch_geotagged_commons.py` and its
  manifest, the `docs/CURATION.md` test-data section, and the tests under `tests/`.
- **Does not touch:** `frontend/`.
- **Order:** B13 and B11 (root-cause bugs) → B1 → B2 → B4, B5, B6, B7 → B3 (needs B7) → B10, B14, B15 → B12 → B8.
- **Acceptance:**
  - the offline suite for the touched modules is green;
  - `generate_contracts.py --check` is clean;
  - each B item has a test that failed before its change (red, then green);
  - lock-rule tests for B1 cover human, import-validated, holdout, excluded and audit sample;
  - B8's fetcher passes its recorded-fixture test, and the manifest holds at least 50 entries, each with a licence in
    `ALLOWED_LICENSE` and real GPS.
- **Integration tests** (pending the wave run): accept_machine on the live harness; readiness on a seeded project.

### Lane B: visual system (Sonnet, high effort: about 1,700 judged edits)

- **Owns:**
  - `frontend/src/app.css`, `frontend/package.json`, `frontend/package-lock.json`, `ATTRIBUTION.md`;
  - `frontend/src/routes/+layout.svelte`, `frontend/src/lib/components/ui/**`;
  - `frontend/src/lib/designRules.scan.test.ts`, `frontend/src/lib/components/ScrollStrip.svelte`;
  - class-only swaps in every other `.svelte` file: no markup restructuring and no behaviour change.
- **Order:** fonts and tokens → scan rules (failing) → codemod rules one commit each (sizes, uppercase, mono/numbers,
  `<code>` ids) → scan rules green → `DataTable` and `Tabs` changes.
- **Acceptance:**
  - `npm run check`, `npm run build` and vitest for `lib/designRules.scan.test.ts` and `ui/*.test.ts` pass;
  - each new scan rule's negative control fails on a seeded string;
  - a stubbed e2e (`e2e/stubbed/test_typography.py`, new) asserts:
    - the computed `font-family` of body text starts with `"Inter Variable"`;
    - `<code>` starts with `"JetBrains Mono Variable"`;
    - `document.fonts.check('14px "Inter Variable"')` is true;
    - no element outside `code, kbd, textarea, pre` resolves a monospace family (walk the DOM on the Workspace
      Gallery and Detections stubs).
- **Screenshots** (1600x1000 and 430x900): Dashboard, Workspace Gallery, Detections, Review, Export, Train, Settings,
  each viewed against the `desktop_*` baselines in `artifacts_local/ux_review_1112/screens/`.

### Lane C: Workspace, Review, details, map (Sonnet, high effort)

- **Owns:**
  - `frontend/src/lib/workspace/**`, `lib/components/workspace/**`;
  - `lib/components/clusters/**`, `lib/clusters/**`;
  - `lib/components/review/**`, `lib/review/**`;
  - `lib/components/{CropCard,CropMetaPanel,CropResultGrid,SourceImageOverlay}.svelte`, `lib/components/cropMeta/**`;
  - `lib/components/images/**`, `lib/images/**`, `lib/components/map/**`, `lib/map/**`;
  - `lib/components/itemFilter/**`, `lib/itemFilter/**`, `lib/labelOps/runLabelOp.ts`, `lib/jobs/followJob.ts`;
  - `lib/api_labelOps.ts`, `lib/types_labelOps.ts`, `lib/types_review.ts`, `lib/types_map.ts`,
    `lib/types_itemFilter.ts`, `lib/types_clusterList.ts`, `lib/keymapFallback.ts`;
  - `routes/p/[project]/workspace/**`;
  - the stubbed e2e tests named below.
- **May import, must not edit:** `lib/components/nearDup/NearDupGroupModal.svelte`, `lib/components/detector/AdoptDetectorDialog.svelte`
  (lane D).
- **Delivers:** sections 4.1-4.7, F4, F5, F7, the map part of F6, and the UI side of walk P2-1.
- **Acceptance:** `npm run check`, `npm run build`, vitest for touched units, and these stubbed e2e tests (new unless
  named, under `e2e/stubbed/`; each was seen failing on the old code):
  - `test_clusters_empty_states.py`:
    - with `empty_reason: pending_activation` and a region slot, the Activate panel shows and no inventory card does;
    - after a stubbed activation the grid lists clusters without a reload;
    - arrow keys move between cards and `Enter` opens one.
  - `test_cluster_scope_toolbar.py`:
    - with no selection, one primary ("Label all N as…") and no bulk bar;
    - with a selection, one primary ("Label as…");
    - the More and Refine menus list every keep-list item from 4.12;
    - `L`, a class, `Enter` labels and opens the next cluster.
  - `test_review_default_queue.py`:
    - no queue opens the served `default_tab`;
    - a validated item has no Confirm;
    - the Accept confident labels dialog shows the dry-run by-class table, applies one op and offers Undo.
  - `test_details_image_actions.py` (extends `test_ux_d_workspace.py`):
    - image details render a preview;
    - Show its N detections switches to Detections with the image scope;
    - from a map cell, Back to the cell restores the cell's thumbnails;
    - only one editor link per panel.
  - `test_map_filtered_empty.py`: with B5 counts at 0 matching, the overlay and Clear filters show.
  - `test_label_dialog_queued.py`: a stubbed `JobView` with `status: queued` shows the waiting text and Run in
    background.
  - `test_one_primary_per_view.py`: in each of the seven views (no selection and with a selection), exactly one
    visible `.btn-primary` or primary `Button`.
  - Update `test_gallery_view.py`, `test_review_view.py`, `test_map_view.py` and `test_narrow_viewport_430.py` for the
    new labels.
- **Screenshots** (1600x1000 and 430x900), each viewed and listed in the hand-back: cluster list (served and
  pending-activation stubs), open cluster (no selection, with a selection, Refine open, More open), Gallery with the
  Duplicates popover, Detections, Review on `check_machine` with the accept dialog, Review on a validated item, image
  details, map cell → image → Back, map filtered empty.

### Lane D: IA, Ingest/Import, Data quality, pages (Sonnet, medium effort)

- **Owns:**
  - navigation: `frontend/src/lib/projectPaths.ts`, `lib/components/shell/**`;
  - Ingest and import: `routes/p/[project]/ingest/**` (new layout and `imports/[id]`), `routes/p/[project]/datasets/**`
    (deleted), `lib/components/datasets/**`, `lib/datasets/**`, `lib/components/ingest/**`;
  - Data quality: `routes/p/[project]/quality/**`, `lib/components/audit/**`, `lib/components/labelOps/**`;
  - near-dups: `lib/components/nearDup/**` (except the file C creates), `lib/nearDup/**` (no deletion of
    `nearDupPageController`);
  - Settings: `routes/p/[project]/settings/presets/**`, `lib/settings/**`;
  - Dashboard: `routes/p/[project]/dashboard/**`, `lib/components/dashboard/**`, `NextStepsCard.svelte`,
    `DatasetStats.svelte`, `lib/components/detector/AdoptDetectorDialog.svelte`;
  - other pages: the `+page.svelte` of `export`, `train`, `models`, `classes`, `bakeoff`, `routes/projects/**`,
    `lib/components/export/**`, `lib/components/train/**`, `lib/components/models/**`, `routes/+error.svelte`,
    `routes/p/[project]/+error.svelte`;
  - new files: `lib/api_quality.ts`, `lib/types_quality.ts`;
  - the stubbed e2e tests named below.
- **Delivers:** sections 3, 4.8-4.11, F1, F2 and the Settings part of the near-dup move.
- **Acceptance:** `npm run check`, `npm run build`, vitest, `scripts/docs/check_docs_vs_code.py`, and these stubbed e2e
  tests (each seen failing first):
  - `test_ingest_import_tab.py`:
    - the Import dataset tab renders inside the Ingest layout (same width, tab strip, rail);
    - the import job page is under `/ingest/imports/{id}` with the tab current;
    - `/p/default/datasets/import` shows the app 404.
  - Update `test_dataset_import.py`, `test_import_leftovers.py` and `test_narrow_viewport_430.py` for the new paths.
  - `test_quality_ready.py`:
    - readiness rows render served labels and severities, and each link opens the served Workspace URL;
    - `/quality` lands on Ready to train;
    - the near-duplicates route 404s.
  - Update `test_quality_page.py`, `test_near_duplicates.py` (moved to the Settings page sections) and
    `test_next_steps.py`.
  - `test_audit_gallery.py`: the pending strip renders the `review_tab=audit` crops; an outcome count links to
    `view=detections&audit_outcome=…`.
  - `test_nav_order.py`: the header order matches section 3.1; Workspace is a top-level link; Export is in Model; no
    breadcrumb on section or tab pages; the editor breadcrumb shows the file name.
  - `test_dashboard_pages.py` and `test_pages_ux_e.py`: one primary per page; Pipeline details collapsed.
- **Screenshots** (1600x1000 and 430x900), each viewed: every Ingest tab including Import (empty, preview, job page),
  Data quality (Ready to train with warn/block stubs, Accuracy audit pending and done, Label history), Settings ›
  Near-duplicate profiles, Dashboard, Classes, Export, Train, Models, Projects, the 404 page.

### Wave sweeps and the live check

- After each wave the coordinator runs the full set once:
  - the offline suite, vitest, `npm run check`, `npm run build`;
  - stubbed e2e, `pre-commit run --all-files` and the pre-push stage;
  - after wave 2, the live Cropwright tier on `op-gpu`.
- After wave 2 the coordinator fetches B8's photos, ingests them into `coco-e2e`, and re-takes this review's
  screenshots with `artifacts_local/ux_review_1112/scripts/r01..r05`, desktop and 430 px, for the owner.

---

## 9. What we will not change (g)

| Not changing | Reason |
|---|---|
| Settings layout, sidebar and search (F3) | The owner approved them. They already follow the settings-sidebar pattern (P11). |
| The seven Workspace views | Each serves a real task. Gallery, Clusters, Detections and Review are the loop; Table, Map and Embedding are advanced but cost only a tab. |
| The dark theme and the colour tokens in `app.css` | Already contrast-checked (`--color-fg-3` raised to 4.39:1 on `surface-3`). A light theme is a separate project. |
| The 20/26/32 px control tiers | Shared by `.btn`, `.select` and `.chip`. Changing them would reflow every page for no user gain. |
| Automatic validation of confident machine labels | It would bias the accuracy audit and break the lock rule's intent. Detector confidence is not confidence in the user's class scheme. The bulk accept (F5) gives the speed with a person's decision and an Undo. |
| Deleting or dropping near-duplicates | Owner: duplicates can be valuable. Only "collapse for this export or run", off by default, plus the fixed rule that a group never spans train/test. |
| A command palette (Cmd/Ctrl+K) | Linear uses one for actions on a selection, but it needs a served action registry. The keymap and the `?` overlay cover the loop now. Revisit after this pass. |
| Mass migration of the 392 raw `.btn` buttons to `ui/Button` | Only buttons in files lanes C and D touch move. The classes render identically, so a mass rewrite is churn. |
| Clustering defaults (walk P2-9: 203 small mixed clusters for 1,223 items) | A model and profile question for the clustering research track ([09b_research_clustering.md](09b_research_clustering.md)), not UI. The UI side, the mixed-cluster warning, is in 4.2.2. |
| The Keyboard settings page with its 200+ remove buttons (walk P3-6) | Part of the approved Settings. Only its console 404 is fixed. |
| The backend items owned by lane fix-walk | P1-1..P1-4, the backend of P2-1 and P2-2 are already in that lane. |
