# Review workflow audit and consolidation plan (Cropwright and the curation API)

| Field | Value |
|---|---|
| Status | **plan only**: wave 1 lanes in progress, waves 2 to 4 not started. Written 2026-11-11 on `int/b4-stateless` at `634019ded`; revised the same day with the owner's Workspace direction (sections 4.6 to 4.9, routes R-14 to R-23, waves 2 to 4 rewritten) |
| Goal (owner, 2026-11-11) | "easy user workflow and setup such that they can easily sort, review, label, and cluster the detections together" |
| Owner framing applied | the frontend is dumb: it shows what the backend serves. Sort, filter, group, counts, labels and box coordinates come ready from the API. No client-side joins, no derived business logic, no second source of truth. Routes are shaped as view models for pages. Missing routes are ours to add, never worked around. Reuse the Clusters page patterns; consolidate rather than add surfaces. Target is a production app for thousands to hundreds of thousands of items |
| Evidence | screenshots under `artifacts_local/screens/ux_audit_1111/` in the main checkout (gitignored), named in the tables below; `file:line` references are at `634019ded` |
| Data used | public project `coco-e2e` (COCO, 91 images, 502 items, 3 GPS photos) read-only, and a scratch project with 16 COCO val images (created, ingested, clustered, deleted) |
| Reader | a Sonnet implementation lane with no memory of this session. Section 9 is the work order; sections 2 to 8 are the reasons |

---

## 0. Summary in plain words

**What works and should be kept.** The cluster detail page is the best surface in the app: a grid of crops
with served facet chips (detector class, validation, review reason, confidence band) whose counts are the
server's, a served sort, select-page, bulk assign, move, ignore, Assist and Accept VLM, drag onto a class,
per-class hotkeys and undo. The Matching items mode (filter, dry run, apply to all N matching) is the right
bulk model. Review's two-pane layout (whole source image with every box drawn by the browser, the crop,
reason, confidence, proposal hint, Confirm, Skip, Discard, Undo, class buttons) is right for one-at-a-time
work. Counts on Export and Classes agree with the API. The near-duplicate engine, its stale guard on freeze
and export, and its revisioned thresholds are sound.

**What is wrong.** The pieces do not join up into one workflow:

1. There is no way to look at **images**. The app is crop-first everywhere. A user cannot browse whole
   pictures with their boxes, cannot open "this image and all its crops", and from a crop the sibling crops
   are dead thumbnails. The reason is in the API: no route lists images or serves one by id, so the
   frontend reaches images through a crop (`/crops/{id}/context`, `/crops/{id}/image`) and the
   near-duplicate page guesses a picture for an image id by fetching the first crop that stands on it.
2. **Four places to review the same crops**: Clusters, Cluster detail, Review (12 served tabs) and Audit,
   plus Label ops, Map and Near-duplicates as separate pages. 14 top-level nav entries. A first-time user
   cannot tell where to start.
3. **Some numbers mislead** (P0): "Labeled by source" on the dashboard leaves out imported labels (shows
   100 total, 100 % human, on a project with 285 validated); the class sidebar and Classes page show 0 per
   class when no clustering partition is serving; the Clusters page says "run the auto-label pipeline" when
   the actual step is "Activate" a partition that is already built.
4. **First run is a maze.** A new project has no classes, so Review shows no class buttons; the way out is a
   collapsed panel on Classes. Clustering produces a partition that is not serving until "Activate it" on
   the Dashboard is clicked; the Clusters page itself says nothing about it.
5. **Scale limits**: every list pages with `from/size` and refuses past 10,000 results (reproduced on
   `coco-e2e`: `GET /crops?page=10001&page_size=1` answers 422 "exceeds the 10000 result-window depth
   limit"). Projects list and switcher load every project with no search.

**What to build** (section 9 has the lanes):

- Backend view-model routes: an **image gallery page** (`GET /images`), an **image with its detections**
  (`GET /images/{image_id}`), the **image file and thumbnail by id**, a **cluster header**
  (`GET /clusters/{cluster_uid}`), a **near-duplicate decision list**, near-duplicate members with
  pictures and status, a served **labeled-by-source** breakdown, served **review-tab availability and
  counts**, a served **"why is the Clusters grid empty"**, a map **area filter** (`geo_bbox`) in the shared
  item filter, served **sort options** on `/crops`, then cursor paging and per-image label counts for scale.
  Deleted in the same changes: `/crops/{id}/context`, `/crops/{id}/image`, `/map/images`, `/audit/queue`.
- One **Workspace** page that grows out of the Clusters page: views **Images | Detections | Clusters | Map**,
  a left rail with classes and served review queues, the shared filter and served sort on top, the cluster
  detail grid and bulk bar for every view, and a **Focus** mode (today's Review panes) over the same query.
  An **image page** for one picture with all its boxes and crops. A **Quality** page that holds Accuracy
  audit, Near-duplicates and Label history. Old pages are deleted as each step lands; nothing runs in
  parallel.
- **Revised after the owner's direction (2026-11-11, sections 4.6 to 4.8):** the Workspace is one
  professional page with six switchable views of the same filtered set (**Gallery, Clusters, Detections,
  Review, Table, Map**); the filter, the queue, the selection and the focused item carry across views.
  The core flow is **batch labelling by cluster or group**, which no mainstream tool offers (section 4.9),
  then per-image work in a real **image annotation editor** (draw, move, resize, delete boxes; class picker
  with inline create; undo and redo; autosave through OCC). The **Map** becomes a full view with area
  selection that filters every other view. New backend routes R-14 to R-23 cover human-drawn items, box
  geometry edits, a table view model, a selection summary, image locate, and map and filter additions.

---

## 1. How this was checked

- Read: root and frontend `CLAUDE.md`, `docs/VISION_AND_GOALS.md`, the Cropwright user guide
  (`docs-site/docs/cropwright/user-guide/`), `contracts/openapi/curation.json`, every route under
  `frontend/src/routes/p/[project]/`, the controllers and components they use, and the backend routers
  behind the gaps.
- Walked the live stack (`op-gpu`, Cropwright on 5184, API on 4603) with headless Chromium at 1600 px and
  430 px. Every request other than GET under `/curation` was aborted on `coco-e2e`, so nothing was written
  there. The running Cropwright image predates the near-duplicates merge, so a build of `634019ded` was
  served from disk to the browser while every API call still went to the live API (files prefixed `i_` and
  `f_`).
- First run: created a scratch public project, uploaded 16 COCO val images through the Ingest page, ran
  "Cluster items" and "Activate it" from the Dashboard and a near-duplicate build from its page, then
  deleted the project (`GET /curation/projects/<scratch>` now answers 404).
- Caveats. (a) The map needs WebGL2, which this headless browser could not start (`i_map-d.png` shows the
  page's own fallback message), so map rendering was not seen; the map findings come from code and the API.
  (b) The host network churned (`net::ERR_NETWORK_CHANGED`) during the walk, so some cluster-card
  thumbnails show as broken in `f_look-clusters.png`; the same thumbnails load in 20 to 40 ms through nginx,
  so this is environment, not a product bug (but see G-27: a failed tile never retries). (c) `coco-e2e` has
  no serving partition and no near-duplicate groups at the 0.98 cut, so cluster cards and a group dialog
  were seen only on the scratch project, and the group dialog not at all.

---

## 2. The journey today

Plain judgment per step: **Can** a user do it, is it **Obvious**, is it **Correct** (UI agrees with the
API), does it **Scale** to many thousands of items.

| # | Step | Where today | Can | Obvious | Correct | Scale | Evidence |
|---|---|---|---|---|---|---|---|
| 1 | Create a project | `/projects` → New project | yes | mostly: slug first, a served rule check | yes | list is one unpaged table, no search | `f_create-dialog.png`, `projects-d.png` |
| 1b | Open the new project | `/p/<slug>/...` | only after it leaves `building` | no: "Project not available … is building" with no auto-refresh | yes | n/a | `ProjectUnavailable.svelte` |
| 1c | Choose detector / region profile / VLM | Ingest card (read-only), Settings sub-pages, Models | yes | split across 3 pages; detector is shown, not chosen, on Ingest | yes | n/a | `ingest-d.png`, `settings-d.png` |
| 2 | Upload images | `/ingest` Upload | yes | yes | yes (16 files, 191 crops, per-file rows) | chunked, fine | `f_upload-done.png` |
| 2b | Server folder / manifest import | `/ingest` Bulk ingest; `/datasets` import | yes | yes | yes | runs server side | `ingest-d.png` |
| 3a | See whole images with boxes | nowhere | **no** | - | - | - | no route lists images |
| 3b | See crops | `/clusters` → Matching items | yes | hidden behind a toggle named "Matching items" | yes (477 = items minus 25 holdout, but the 25 are not mentioned) | pages of 50 | `clusters__mode-matching-d.png` |
| 3c | Crop → its source image | crop "i" dialog, Review left pane | yes | yes | yes | n/a | `i_crop-detail-d.png`, `review-d.png` |
| 3d | Image → its crops | crop dialog "Siblings" | view only | thumbnails are not links; no image page | yes | capped at 500 | `CropSourceContext.svelte:74-84` |
| 3e | Sort / group by image, class, score, status, date | `/crops` sorts: updated, created, confidence, rank, area, clarity, cluster score, mistakenness, uniqueness | partly | sort is only on cluster detail and review; no group by image; no sort by class or detector score on crops | yes | server side | contract `GET /crops` `sort` |
| 3f | Filters (class, status, region, text, holdout, near-dup) | shared item filter on clusters, review, export, map | yes | yes, "More filters" | yes | server side | `item-filter.md` |
| 3g | Paging | review 30 per page, clusters and crops pages | yes | yes | yes | **stops at 10,000 results** | 422 on `coco-e2e` (section 0) |
| 4 | Review queues | `/review`, 12 served tabs | yes | 3 tabs are empty features on a fresh project (Uncertainty, Model disagreements, Classifier blind spots); a technical sort-fallback banner | yes | pages | `review-d.png`, `f_look-review.png` |
| 5a | Assign / confirm a class | Review buttons, letters; cluster grid bulk | yes | only if classes exist; a new project has none and the seed panel is collapsed on Classes | yes | bulk ops are jobs | `f_look-review.png`, `f_look-classes.png` |
| 5b | VLM assist, auto-label | Assist… on cluster/selection/box; Dashboard Auto-label; Auto-label loop on Clusters | yes (when a VLM is active) | three entry points | yes | jobs | `dashboard-d.png` |
| 5c | Bulk relabel | Matching items "Label all matching…", cluster bulk, Label ops undo | yes | yes | yes, served dry runs | jobs | `clusters__mode-matching-d.png`, `label-ops-d.png` |
| 6a | Cluster | Dashboard "Cluster items", then "Activate it" | yes | the Clusters page does not show the ready partition; copy says "run the auto-label pipeline" | **misleading** | jobs | `f_look-clusters` empty state, `f_click-dashboard-0-Cluster_items.png` |
| 6b | Work a cluster | `/clusters/<uid>` | yes | yes (best page) | yes | pages | `f_look-clusters_<uid>.png` |
| 6c | More like this / text / image search | crop card, Review link, toolbar | yes | yes | yes | top-k | `i_more-like-this-d.png` |
| 7 | Map | `/map` (only when GPS photos exist) | view cells, list crops of a cell | a cell lists crops, not photos; cannot filter the review set by an area | yes | server-side grid cells | `map/+page.svelte:83-140` |
| 8 | Near-duplicates | `/near-duplicates` | yes | wording ("Full rebuild" on a never-built level, "crops inside leak groups"), toggles show no selected state | yes | cursor paged | `i_near-duplicates-*.png`, `f_look-near-duplicates.png` |
| 9 | Audit accuracy | `/audit` | yes | sampled crops are labelled on another page (Review) | yes | small | `audit-d.png` |
| 10 | Export | `/export` | yes | yes; the "every object labeled" option is off by default while 203 unlabeled objects ride along on the exported images | yes | server side | `export-d.png` |
| 11 | Train | `/train` | yes | yes | yes | n/a | `train-d.png` |

---

## 3. Page inventory and recommendation

| Page | What it is for | Overlap and problems | Recommendation |
|---|---|---|---|
| Dashboard | pipeline status, Cluster items / Activate, Auto-label, stats | wrong labeled-by-source total (G-01); clustering controls only here | **Keep**, add a served "Next steps" card (G-08); keep the clustering panel and also show it in Workspace when no partition serves |
| Ingest | upload, bulk, sources, region drain | none | **Keep** |
| Datasets import | labelled dataset import wizard | reached from Ingest | **Keep** (out of scope) |
| Clusters (`/clusters`) | cluster grid, sidebar classes, search, Matching items, region gallery, auto-label loop | is also the crop browser and the region browser; the owner likes it | **Rename and grow** into **Workspace** (`/workspace`) |
| Cluster detail (`/clusters/[id]`) | one cluster's crops, facets, bulk, Assist | the best triage grid; a separate page | **Merge** into Workspace as cluster scope (`/workspace?cluster=<uid>`); delete the route |
| Review (`/review`) | one-at-a-time queue, 12 tabs, region tab | same crops as clusters, different filters, different page | **Merge** into Workspace: tabs become the **Queues** rail, the two panes become **Focus** mode; delete the route |
| Accuracy audit (`/audit`) | draw sample, precision report | its queue links to Review | **Move** into **Quality**; the sample becomes a served review queue (`audit`) labelled in Workspace |
| Classes (`/classes`) | registry: add, rename, merge, deprecate, hotkeys, seed from detector | "TOTAL (IN CLUSTER)" 0 when no partition | **Keep**; serve the item count, not the cluster count (G-04) |
| Label ops (`/label-ops`) | bulk op history and undo | none, but a top-level entry for an admin view | **Move** into **Quality** as "Label history" |
| Near-duplicates (`/near-duplicates`) | status, runs, groups, decisions, thresholds | hygiene, not labelling | **Move** into **Quality**; groups open in Workspace Images view |
| Map (`/map`) | where photos were taken | lists crops of a cell; cannot drive review | **Merge** into Workspace as the **Map** view; a cell filters the Images view by area; delete the route and `/map/images` |
| Export, Train, Bake-off, Models, Settings, Projects | hand-off and admin | out of this audit's scope except G-05, G-21, G-24 | **Keep** |

Target nav: Dashboard · Ingest · **Workspace** · Classes · **Quality** · Export · Train · Bake-off · Models ·
Settings (10, from 14). "Workspace" is a working name; the owner may choose the label (the URL changes once).

---

## 4. Answers to the owner's questions

### 4.1 What the near-duplicates page does, and whether it belongs in review

**In plain words.** It finds pictures that are almost the same picture (a burst of shots, a re-saved or
resized copy) and crops that are almost the same crop, by comparing the image vectors the project already
stores (cosine at or above 0.98 by default). It keeps two kinds of groups:

- **Duplicates**: keep one, drop the rest, so an export does not train on ten copies of one frame (the
  export's "Collapse near-duplicates" option uses them).
- **Leak groups**: members that must stay on the same side of the train/test split. Freezing the test
  holdout moves a whole group together, and freeze and export refuse with `409 near_dup_stale` while the
  groups are out of date, so a near copy of a test image cannot sit in the training set.

A user can pin a pair apart ("Not a duplicate") or together ("Keep together"); the pin applies from the next
run. The page also shows runs and lets an admin change the thresholds.

**Where it fits.** It is **dataset hygiene before export and training**, not a labelling step. It belongs
under **Quality**, linked from Export (the stale panel already exists) and from the holdout freeze dialog.
Inside review it helps in two ways: a "Hide near-duplicates" filter already exists on the review tabs, and a
near-duplicate badge on image cards in the Workspace Images view lets a reviewer see and open a group. It
should not be a step every user must visit; the groups update themselves incrementally after ingest (the
scratch run showed incremental runs following the full build).

**What is missing**: a list of past decisions (G-12), pictures and status for group members served by id
(G-11), the member's class and holdout state in the group view, a way to open a group as a grid in the
Workspace instead of a small dialog, "label this whole duplicate group" from a representative (P2), a served
empty-state line that names the cut and the count ("0 groups at cosine 0.98 among 91 images"), wording
("Build" for a never-built level, a plain line for "item level"), and visible selected state on the
Images/Items and Duplicates/Leak toggles (G-26).

### 4.2 Why an earlier lane wrote "no route serves an image by id" and "no list-decisions route"

Both sentences are true of the API at `634019ded`, and the lane was frontend-only, so it worked around them
instead of adding routes:

- Images: the images index already holds one document per source image (`image_id`, `image_path`,
  `width`, `height`, `source`, `indexed_at`, `capture`, `near_dup_*`, `dataset_split`, `pe_embedding`),
  but the only routes that read it are crop-anchored (`GET /crops/{crop_id}/context`,
  `GET /crops/{crop_id}/image`) or path-anchored (`GET /images/serve?path=`, `GET /images/root/{alias}`).
  So `api_nearDup.ts:166-185` (`previewItemsForImages`) fetches `GET /crops?image_id=...` and uses the
  first crop standing on each image to show its frame; an image with no detections shows "no preview".
- Decisions: `src/routers/curation/near_duplicates.py:90-110` has POST and DELETE for decisions but no
  GET, so `nearDupGroupController.svelte.ts:5` limits undo to decisions made in the open dialog.

We write this backend, so both become routes (section 7: R-1, R-2, R-3, R-6) and the workarounds are
deleted in the same change.

### 4.3 One workspace instead of many pages

The owner likes the Clusters page for sorting detections; the cluster detail grid is the strongest part.
One integrated page does the job better than four, because the four already show the same crops through
the same item filter and the same card, and differ only in grouping and in one-at-a-time vs grid. The
target:

```
/p/<project>/workspace
  top:    [Images | Detections | Clusters | Map]  shared item filter  served sort  search (text | similar | picture)
  left:   Classes (served counts, click = filter, drop target, hotkeys)
          Queues  (served review tabs with counts and availability: All, VLM low confidence, Objects, Audit, ...)
  centre: grid of ImageCard | CropCard | ClusterGridCard | MapView
          bulk bar from cluster detail (assign, confirm, move, ignore, Assist, reprocess, "all N matching")
  Focus:  Enter on a card or "Start" on a queue → today's Review two-pane view over the same query and order;
          Esc returns to the grid at the same place
  scope:  ?cluster=<uid> → cluster header (served), facet chips, proposal panel; the grid is Detections in that cluster
/p/<project>/images/<image_id>
          the whole picture, every box drawn by the browser from served coordinates, per-box class and box edit,
          the strip of its crops, near-duplicate group and location, previous/next in the Workspace query
```

The view list and layout in this sketch are superseded by section 4.6 (Gallery replaces "Images"; Review
and Table become views), and the image page becomes the annotation editor of section 4.7. The moves and
deletions below still hold.

What moves where, and what is deleted:

| From | To | Deleted in that step |
|---|---|---|
| `/clusters` grid, sidebar, search, Matching items, region gallery, auto-label loop | Workspace (Clusters and Detections views; "Matching items" becomes the Detections view) | `/clusters` route |
| `/clusters/[id]` | Workspace `?cluster=<uid>` | `/clusters/[id]` route |
| `/review` tabs, filters, panes, keyboard | Workspace Queues rail and Focus mode (same controllers under `$lib/review`) | `/review` route |
| `/map` | Workspace Map view; a cell → Images view with `geo_bbox` | `/map` route, `GET /map/images` |
| `/audit` queue list | Workspace queue `audit` | `GET /audit/queue`, `AuditQueueList.svelte` |
| `/audit` report and draw form, `/near-duplicates`, `/label-ops` | Quality (`/quality/audit`, `/quality/near-duplicates`, `/quality/history`) | the three routes |
| crop dialog source pane, Review left pane | `GET /images/{image_id}` | `GET /crops/{crop_id}/context`, `GET /crops/{crop_id}/image` |

No redirects or aliases are kept (the project standard is no shims). Every in-app link and the docs-site
pages change in the step that deletes a route.

Migration order, each step shippable on its own (the original order; section 9 now builds the Workspace
shell directly in wave 2 instead of growing views inside `/clusters` first, so no view is written twice):

1. Honest numbers and empty states (no layout change).
2. Image routes, cluster header, near-dup decisions (backend, with the call-site swaps).
3. Images view and image page added **inside** `/clusters`; crop dialog links to the image; siblings clickable.
4. `/clusters` becomes `/workspace`; cluster detail folds in as `?cluster=`.
5. Review folds in as Queues and Focus; audit sample becomes a queue.
6. Quality page; Map view with `geo_bbox`.
7. Scale: cursor paging, per-image label counts, project search.

### 4.4 The map as part of review

Today (`map/+page.svelte`, `src/services/curation/map_cells.py`): the server aggregates located photos into
grid cells for the viewport (GeoJSON, `truncated` and `join_truncated` flags), so it does not ship every
point; that is the right design for many points and is kept. A click on a large cell zooms; a click on a
small cell calls `GET /map/images?tile=` for up to N image ids and then lists **crops** of those images with
`GET /crops?image_id=<id>&image_id=...` (`mapController.svelte.ts:125-135`). The filter bar is the shared
item filter, so class and status filters already apply. The header shows "3 located photos" but not how many
have no position.

Plan: add `geo_bbox=west,south,east,north` to the shared item filter (every list route, including the new
`/images`), serve each cell's `bounds` in `MapCellProperties`, and serve `n_without_location` in `MapMeta`.
The Map view in Workspace keeps MapView; a cell click switches to the Images view with `geo_bbox` set to the
served cell bounds and the same class and status filters, where the user sees whole photos with boxes and
can Focus-review them. A "Has location: yes/no" filter (`has_location`) covers photos without GPS. This
deletes `GET /map/images` and the crop-list side panel.

### 4.5 Many projects and scale

- Moving between projects: the switcher lists every project with no search; `/projects` is one unpaged
  table (`projects-d.png` shows dozens of rows and the shard-budget banner). Add `q`, `status`, `sort` and
  cursor paging to `GET /curation/projects`, a search box in the switcher, and served counts already exist
  in `ProjectSummary.counts`. There are no user accounts, so "my work" is out of scope; recent projects can be
  kept in `sessionStorage` (view state only, allowed by frontend rules). P2.
- Per-project scale: every list is server-sorted and server-filtered, which is right, but pages with
  `from/size` (`src/services/curation/crop_listing.py:74`, `src/routers/curation/review.py:177`,
  `src/services/curation/region_rows.py:206`) and refuses past 10,000. Switch to cursors (`search_after`
  with a point-in-time) for `/crops`, `/review/{tab}`, `/regions` and the new `/images` (R-11).
- Pages that load everything client-side: the projects list and switcher only. Classes, keymap and the
  detector vocabulary are bounded registries. Client-side sorts of served lists exist in
  `ClassSidebar.svelte:121`, `ExportClassTable.svelte:34-45` and `ClassSubsetPicker.svelte:66` (bounded,
  P2, see section 5).

### 4.6 The Workspace: views, shared filter and selection, differentiator

**In plain words.** One page, `/p/<project>/workspace`, where all sorting, labelling and checking happens.
The user picks *what set* of objects to work on once (filter, queue, area, cluster), then looks at that same
set in whichever view suits the job, and acts on a selection that survives a view switch. Everything shown
comes from the server: lists, sort orders, counts, labels, box coordinates and empty-state text.

**What sets Cropwright apart.** Label Studio, CVAT and Kili label one image or one object at a time; their
bulk tools stop at whole-image classification (section 4.9). FiftyOne and Encord can select a cluster in an
embedding plot, but only to tag it or save a collection, then hand off to another tool. Cropwright labels
**many objects at once by cluster or group**, in review-style batches, then drops to the single image to fix
what the batch could not. The Workspace is built around that order.

#### Views and their place in the loop

The loop is: ingest, sort, label, build a dataset, train, then repeat with the improved model.

| View | Shows | Step of the loop it serves | Primary actions | Data (route) |
|---|---|---|---|---|
| **Gallery** | whole images, boxes drawn from served coordinates, served count chips (items, labelled, unlabelled), near-dup badge | after ingest: see what came in; find images with missed or unlabelled objects; open one to annotate | open the image editor; select images (acts on their matching items) | R-1 |
| **Clusters** | cluster cards (served size, dominant class, purity, unlabelled share, representatives); a card opens the cluster scope (today's cluster detail grid with facet chips) | **sort and label in batches** (the core) | assign a class to a whole cluster or to a facet slice of it; move, ignore, Assist | `GET /clusters`, R-5, `.../facets` |
| **Detections** | flat grid of crops (today's Matching items) | label "all N matching" a filter; work a class or a detector-class slice across clusters | bulk assign, confirm, ignore, move; dry run first | `GET /crops` (R-4) |
| **Review** | one object at a time: the whole image with every box, the crop, reason, confidence, proposal; a filmstrip of neighbours | the leftovers a batch could not settle, and the served queues | confirm, assign, skip, discard, undo (today's keys) | `GET /review/{tab}`, `/locate`, R-2 |
| **Table** | rows of detections or images, served columns, sort by any served field | auditing and power sorting: "lowest detector confidence first", "largest unlabelled first" | sort, choose columns, multi-select, open | R-18 |
| **Map** | where photos were taken, server grid cells, single-photo pins at high zoom | location datasets: find thin or noisy areas, label or review by area | select an area (filters every view), open a cell's images | `GET /map`, `/map/cells`, R-1, R-19 (section 4.8) |

Default view after opening the Workspace (owner answer 2): **Gallery** until a partition serves, then
**Clusters**. An **Embedding** view (the existing UMAP scatter in `EmbeddingPlot.svelte`, with box select
as a filter, R-23) is planned for wave 4: it earns its place for spotting mislabels that sit inside the
wrong cloud (the FiftyOne and Encord pattern), but it is not needed for the first full loop.

#### What carries across views

All of it lives in the URL (`$app/state` page URL; view state only, never business data), so a link
reproduces the screen and Back works.

| State | Carries | Notes |
|---|---|---|
| Filter | yes, to every view | the shared item filter (`item_filter.py`), plus `geo_bbox` / `geo_polygon` (R-9, R-20), `cluster_uids` (R-20), `image_id` |
| Queue | yes | a queue is a **served filter preset** (`review_tab`) with a served count; picking one in the rail filters every view, not only Review |
| Scope | yes | one cluster, one near-dup group, or one image; shown as a removable chip; the Clusters view in a scope shows that cluster's grid |
| Sort | per view | each view has its own served `sort_options`; on a switch the server's `sort_applied` says what it used (the client never maps sorts between views) |
| Selection | yes | always an `ItemSelection` (`src/services/curation/item_selection.py`): explicit crop ids (Detections, Table rows, Review), or a filter: Gallery selection = current filter + `image_id=[...]`; Clusters selection = current filter + `cluster_uids=[...]`; "all matching" = the filter alone. The bulk bar shows the served summary (R-19): "183 detections in 41 images, 7 locked and skipped" |
| Focused item | yes | the item, image, cluster or map cell under the cursor; drives the details panel; Review opens at it |

#### Layout

```
┌ app bar: Cropwright · project switcher · nav (Dashboard Ingest Workspace Classes Quality Export Train …) ┐
├ filter bar: [search: text | similar | picture] [Class ▾] [Status ▾] [Source ▾] [More ▾]  chips…        ┤
│             "1,240 matching · 25 test-holdout hidden" (served)                         next step chip   │
├ view bar:   Gallery  Clusters  Detections  Review  Table  Map          sort ▾ (served)   size ▾   ⓘ    ┤
├ rail 240px ─────────┬ main pane ───────────────────────────────────┬ details 360px (collapsible) ────┤
│ QUEUES (served)     │ grid | cluster cards | two-pane review |      │ focused image: picture + boxes, │
│  To label     412   │ table | map                                   │ its items, class, provenance,   │
│  Check machine 88   │                                               │ history, near-dup, location;    │
│  Disagreements 12   │                                               │ or focused cluster header; or   │
│  Audit sample  50   │                                               │ map cell summary; or selection  │
│ CLASSES (served)    │                                               │ summary (R-19)                  │
│  car  a  74 ▮▮▮     │                                               │ [Open in editor] [Review these] │
│  person p 133       │                                               │                                 │
│  + New class        │                                               │                                 │
├─────────────────────┴─ bulk bar (when a selection exists) ─────────┴─────────────────────────────────┤
│ 183 selected (served) · Assign ▾ · Confirm · Move · Ignore · Assist… · Undo                         │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- Rail: **Queues** first (owner feedback: a few plain-named queues; section 7, R-10) then **Classes**
  (served `item_count`, hotkey letter, click = filter, drop target, "+ New class" inline).
- Details panel: one component per focused kind (image, detection, cluster, map cell, selection), each
  rendering one served view model. Closed by default under 1280 px.
- At 430 px: the rail and the details panel become drawers, the view bar becomes a select, the filter bar
  collapses behind one "Filters" button (G-25), the bulk bar sticks to the bottom.
- Empty states are served (`empty_reason`, R-8, R-12): an empty Workspace shows the served next steps.

#### Keyboard model

One registry: every key is a keymap action (`lib/keymapFallback.ts` defaults, the served project keymap,
validated by `POST /keymap/validate`); no new window listener (`keymap.literals.scan.test.ts`). Class
hotkeys are plain lower-case letters; the new workspace actions use non-letter keys so they never collide;
the existing letters (Review N, D, Z; cluster grid M move, X ignore, Shift+N flag) and the editor tool
letters (section 4.7) are reserved in their contexts, and the validator refuses a class letter that
collides (fail closed, served message).

| Key | Action | Context |
|---|---|---|
| `[` `]` | previous / next view | workspace |
| `,` `.` | previous / next scope item (cluster in a cluster scope, image in the editor) | workspace, editor |
| arrows | move the focus card or row | grid, table |
| `Space` | select or unselect the focused card | grid, table |
| `Shift`+click or `Shift`+arrows | extend the selection | grid, table |
| `Ctrl/Cmd+A` | select the loaded page; a second press selects all matching (served count, confirm) | grid, table |
| class letter | assign that class to the selection, else to the focused item | grid, table, Review |
| `/` | class picker (search, inline create) | everywhere |
| `Enter` | grid: open Review at the focused item over the same query and order; Review: confirm | grid, Review |
| `Shift+O` | open the image editor for the focused item's image | grid, table, Review |
| `Ctrl/Cmd+Z`, `Ctrl/Cmd+Shift+Z` | undo, redo (Review also keeps `Z`) | everywhere |
| `\` | show or hide the details panel | workspace |
| `Esc` | clear selection, then leave Review or the scope, then close the panel | everywhere |
| `?` | shortcut overlay (served keymap) | everywhere |

#### The primary flow: batch labelling, then per image

1. **Clusters view**, served sort "most unlabelled first". Each card shows the served dominant class and
   purity, so pure clusters are obvious.
2. Open a cluster (scope). The grid is sorted nearest-to-centre first, outliers last (served sort). Facet
   chips show the detector-class mix with served counts.
3. **Pure cluster:** press a class letter with nothing selected; the bar asks "Assign *car* to all 21 in
   this cluster? 2 locked items are skipped" (served dry run); `Enter` applies; one label op, undoable from
   Label history.
4. **Mixed cluster:** click a facet chip (for example detector class *person*), `Ctrl+A`, press `p`; clear
   the chip, select the rest, press the *motorcycle* letter; outliers go to Ignore (`X`) or Move (`M`).
5. `.` moves to the next cluster. Several look-alike clusters can be selected in the Clusters view and
   labelled in one step (`cluster_uids`, R-20).
6. **Review** the leftovers: the "Check machine labels" queue in Review, one at a time, with the same keys
   as today.
7. **Per image:** Gallery with "Has unlabelled objects" (`label_state`, R-1); open an image (`Shift+O`) in
   the editor (section 4.7); draw missed objects, fix boxes, delete false positives; `.` for the next image.
8. Then the rest of the loop: Export, Train, promote; the promoted model's re-run feeds the
   "Disagreements" queue and the next round starts. The served next-step chip (R-12) names the next step.

### 4.7 Image annotation editor

**In plain words.** The page `/p/<project>/images/<image_id>` shows one picture with all its objects and
lets the user fix it: draw a box around a missed object, move or resize a wrong box, delete a false one,
change a class, create a new class without leaving the page. Every change is saved at once, can be undone,
and never overwrites someone else's newer edit. This page is the image page of section 4.3; there is no
separate view-only version.

**Layout.** Tool strip on the left; the canvas in the centre (zoom and pan); on the right the **object
list** (every item with thumbnail, class, state, lock glyph, origin; its region boxes nested under it) and
the class picker; a **filmstrip** of the Workspace query along the bottom (served page around the current
image, R-17), with `,` and `.` for previous and next. The header shows the served image facts (file, size,
taken at, location, split, holdout, near-dup group) and the save state ("Saved", "Saving…", "Conflict,
reloaded").

**Tools** (defaults; every key is a keymap action in a new `image_edit` context):

| Tool or action | Key | Behaviour |
|---|---|---|
| Select / move | `V` | click selects; drag the body moves; 8 handles resize; `Tab` / `Shift+Tab` cycle boxes (CVAT); arrows nudge 1 px, `Shift`+arrows 10 px; `Alt`+arrows move the bottom-right corner (resize) |
| Draw box | `B` | drag, or click two corners (Label Studio, Kili); the draw is clamped to the image; on release the class picker opens at the box with the last-used class pre-selected, so `Enter` repeats it (the CVAT "repeat last" idea, one meaning only); `Shift+B` keeps the draw tool on for several boxes |
| Pan | `H`, or hold `Space` and drag, or middle drag | |
| Zoom | wheel at the cursor, `+` `-`, `0` fit, `1` 100 % | the canvas draws the served full-resolution file (`/images/{id}/file?max_dim=`, R-3) at zoom above fit |
| Delete box | `Delete` / `Backspace` | soft delete (R-16); undoable |
| Set class | class letter, or `/` | applies to the selected box |
| Region boxes | `Shift+R` | edits the selected item's region boxes (wheels in a car) with the existing `MultiBoxCanvas` logic, written through `PUT /crops/{crop_id}/regions` with `frame: "source"` |
| Labels on/off | `Shift+L` | hides box labels on busy images (G-19) |
| Undo / redo | `Ctrl/Cmd+Z`, `Ctrl/Cmd+Shift+Z` | see below |
| Cancel / deselect | `Esc` | |

`V`, `B` and `H` are reserved in the `image_edit` context; the keymap validator refuses a class letter on
them. A minimap for very large photos is P2 (wave 4 polish): no surveyed tool documents one, and zoom plus
fit covers the first loop.

**Class picker with inline create.** One component (`ClassPicker.svelte`, replacing `ClassPickerModal` and
the editor's need) used by the editor, Review and the bulk bar. Type to filter the served registry (a
bounded registry, so filtering the served list in the browser is allowed, section 4.5); recent classes from
this session first; each row shows its hotkey letter and served count. When no class matches, the last row
is **Create "<typed>"**: it calls `POST /classes`; if the name is not a valid slug the server refuses with
the reason and a `suggested_name` (R-22, for example `traffic_light` for "Traffic light") and the row turns
into **Create "traffic_light"**. After creation the class is assigned to the selected box in the same
gesture. Deprecated classes are not offered (served `deprecated`). "Flag for a new class" stays for an
object the user cannot name yet (`POST /crops/flag_new_class`).

**How saves reach the backend.** Autosave, one small write per finished gesture (box drawn and class
chosen, drag released, delete, class change). No Save button. The UI updates at once and shows the box as
"saving" until the server answers; a refusal puts the box back and shows the served message.

| Gesture | Route | Concurrency |
|---|---|---|
| draw a box + class | `POST /images/{image_id}/items` (R-14), with a client `request_id` so a retry is idempotent | creates a new item; nothing to conflict with |
| move or resize an item box | `PATCH /crops/{crop_id}/geometry` (R-15) with `expected_geometry_revision` | 409 `geometry_conflict` carries the current item; the editor replaces its copy, says "changed by someone else", drops that undo entry |
| delete an item box | `POST /crops/batch_exclude` with `reason: "not_an_object"` (R-16) | OCC skip-on-conflict, as today |
| change class | `PUT /crops/{crop_id}/label` (existing; `class_id` from the served registry entry the picker chose) | existing OCC and lock rule |
| region boxes | `PUT /crops/{crop_id}/regions`, `PATCH .../regions/{box_id}` (existing) | `expected_region_revision`, 409 `region_conflict` |

**The lock rule for geometry.** Today the lock rule protects classes and region boxes. It gains geometry
(R-16): a new predicate `is_locked_geometry(source)` in `src/clients/occ_locks.py` is true when
`geometry_source` is `human` or import, the item was deleted by a person (`not_an_object`), or it is in the
frozen test holdout. Automated writers (the `detect` reprocess scope in `reprocess_detect.py` and
`item_delete.py`, combine, import re-runs) never move, delete or recreate such an item: a re-detect that
finds a box matching a human-deleted item leaves it deleted, and one matching a human-drawn item does not
add a duplicate. Humans can always edit; a frozen test-holdout item shows a "Test set" badge and the
server's answer stands.

**Undo and redo.** The editor keeps a session stack of commands (view state only). Each entry holds the
server write it made and its inverse: create ↔ soft delete / un-exclude; move or resize ↔
`POST /crops/{crop_id}/geometry/undo` (R-15, a new `geometry` kind in `edit_history.py`); class change ↔
`POST /crops/{crop_id}/label/undo`; region edit ↔ `POST /crops/{crop_id}/region/undo`. Redo re-sends the
forward write with the fresh revision. Every write is recorded server-side, so history survives a reload
(`GET /crops/{crop_id}/history`), even though the session stack does not. Moving to another image keeps the
stack per image for the session.

**What happens after a new box (backend, R-14).**

1. The route validates the box (inside the image, not degenerate, at least the served minimum size),
   resolves the class **by name** to an active registry entry, and writes one item through the same item
   builder ingest uses (`build_item_doc` in `src/services/curation/item_doc.py`): `class_source` and
   `label_source` `human`, `class_validated: true`, `geometry_source: human`, `geometry_revision: 1`,
   `embedding_state: deferred`, `crop_area_norm`, and the region seed status when a region profile is
   active (`region_seed_status`), so the region worker picks it up like any new item.
2. **No crop file is made up front.** The crop and its thumbnail are cut from the source on first request
   (`crop_bytes.py`, `THUMBNAIL_CACHE`). The served `thumbnail_url` carries `?v=<geometry_revision>`; a
   geometry change evicts the crop-cache file and the cached thumbnail for that crop id.
3. **Embedding:** the route enqueues the existing `reprocess` Celery job (`src/tasks/jobs_a.py`) for that
   image with scope `embed`, which embeds only items without a vector (`embed_image_chunk`), so many boxes
   drawn in a row cost one real run and the rest are no-ops. A moved item drops its vector and returns to
   `deferred` (its old vector describes the old crop) and is re-embedded the same way. The response carries
   the job id; the object list shows the served `embedding_state` until it reads "embedded".
4. **Near-duplicates:** the item is marked pending like any new item (`near_duplicates.pending`); the
   periodic `near_dup_rescore` task scores it.
5. **Clusters:** a new item has no cluster until the next clustering run (the periodic `cluster_refresh`
   starts one when the pool grows past the profile's rule). It is already a validated label, so it needs
   none to count in the dataset; the object list shows the served "joins at the next clustering run".
6. `crop_rank_in_image` is set by area among the image's items at write time; the follow-up re-ranks the
   image's other items (a machine field, not locked).
7. The response returns the new item (`serialize_item`) and the image's refreshed `boxes` and `counts`, so
   the editor and the Gallery card redraw from served data.

Moving an item box that has region boxes now outside it is refused with 422 `regions_outside_item` and the
box ids; the editor asks "Delete these 2 region boxes too?" and re-sends with `drop_region_box_ids`.

### 4.8 Map as a real view

**What is wrong today.** The map draws cells and lists crops of a small cell; it cannot sort, cannot
narrow the review set, shows no picture details, and on `coco-e2e` it says "3 items" but draws no points
(wave 1 bug fix). For datasets taken from vehicles or in the field, location is how users think about
their data, so the map must be a full Workspace view, not a side page.

**Design.**

- **Same filter, same counts.** `GET /map/cells` already takes the shared item filter; the Map view sends
  the Workspace filter and queue, so "unlabelled *person* in the To label queue" shows only where those
  are. The header shows served counts: "1,920 located · 88 without location" (`n_located`,
  `n_without_location`, R-9); "without location" links to Gallery with `has_location=false`.
- **Cells and pins.** Server grid cells stay (no shipping of every point). At high zoom a cell holding one
  photo is served with its `image_id` and `thumbnail_url` (R-21) and is drawn as a photo pin.
- **Click a cell** (small enough to list, the existing `opensList` rule): the details panel shows the
  served summary of that cell (R-19 over the filter plus the cell's served `bounds`: images, detections,
  labelled share, top classes, date range) and the first 12 image cards with boxes (R-1 with `geo_bbox`),
  in the Gallery's served sort. Buttons: **Show in Gallery** (sets `geo_bbox`, switches view), **Review
  these** (Review view over that filter). A large cell zooms, as today.
- **Click a pin:** the details panel shows that image with its boxes (R-2) and **Open in editor**.
- **Select an area:** `Shift`+drag draws a rectangle (MapLibre's `boxZoom` handler is disabled for this,
  per its API docs); the lasso tool draws a polygon. The area becomes a filter chip (`geo_bbox` or
  `geo_polygon`, R-20) that applies to **every** view, until removed. Panning never filters by itself.
- **Time:** the existing capture-day filter (`taken_from`, `taken_to`) sits in the map toolbar.
- **Privacy, fail closed.** The location policy limits precision (`max_precision` in `map_cells.py`). The
  server snaps `geo_bbox` and `geo_polygon` to the policy's grid before querying, so a drawn area can never
  reveal positions finer than the policy serves.
- **No external key.** Tiles come from the served `tile_url` (location policy); nothing needs a Mapbox
  token (a FiftyOne requirement we avoid).
- **WebGL2 missing:** the existing served fallback message; the live map test runs headed on `DISPLAY=:11`.

### 4.9 Research notes

Checked 2026-11-11 against official docs and product pages (fetched; "unverified" where a page did not
confirm a detail).

| Tool | What it does | Source |
|---|---|---|
| Label Studio | rectangle drawn by drag, double-click or two clicks; Outliner (region list, grouped by label or tool, ordered by score or time) and Details panel, both collapsible and movable; Backspace deletes, Ctrl+D duplicates, Ctrl-click multi-selects | https://labelstud.io/guide/labeling |
| Label Studio | hotkeys remappable per user and via `EDITOR_KEYMAP` | https://labelstud.io/guide/hotkeys |
| Label Studio | Data Manager: list and grid views, columns, filters and sorts saved as tabs, "Label Tasks As Displayed" walks the current filter and order | https://labelstud.io/guide/manage_data |
| Label Studio | bulk labelling supports **only global classification**; a control tag with `perRegion` is not supported, so boxes cannot be bulk labelled | https://docs.humansignal.com/guide/labeling_bulk |
| Label Studio Enterprise | review stream: Accept, Fix + Accept, Reject; rejected work requeued or dropped by setting; acceptance score | https://docs.humansignal.com/guide/quality |
| CVAT | rectangles by 2 points or 4 extreme points; label and method chosen before drawing | https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/annotation-with-rectangles/ |
| CVAT | Attribute Annotation Mode: one object at a time, Tab / Shift+Tab between objects, Up / Down between attributes, number keys set a value, Left / Right to the next frame with annotations, a `label=="Person"` filter | https://docs.cvat.ai/docs/annotation/manual-annotation/modes/attribute-annotation-mode-basics/ |
| CVAT | navigation: D / F previous and next frame; skip to the next frame with or without objects | https://docs.cvat.ai/docs/annotation/annotation-editor/navbar/ |
| CVAT | review: job stages annotation, validation, acceptance; issues pinned to a point with quick presets, Resolve / Reopen | https://docs.cvat.ai/docs/qa-analytics/manual-qa/ |
| CVAT | shortcuts rebindable per scope with conflict warnings | https://docs.cvat.ai/docs/getting_started/shortcuts/ |
| Kili | boxes by drag or two clicks; per-class shortcuts auto-assigned and overridable; Alt+S lists shortcuts | https://docs.kili-technology.com/docs/bounding-box-tool, https://docs.kili-technology.com/docs/customizing-the-interface-in-kili-ui |
| Kili | review queue: "Start reviewing" pulls the next asset by priority; reviewers edit then Review, or Send back with issues; no plain Accept | https://docs.kili-technology.com/docs/reviewing-labeled-assets |
| Encord | Curate: embedding plot, similarity search, Collections sent to Annotate as a batch | https://docs.encord.com/solutions-documentation/applied-ai/annotation-and-curation |
| Encord | editor keys: Ctrl+Z undo, Ctrl+Shift+Z redo, Ctrl+S save, Backspace delete, N / B approve / reject | https://docs.encord.com/platform-documentation/Annotate/annotate-label-editor/annotate-label-editor-settings-shortcuts |
| FiftyOne | Embeddings panel: lasso filters the sample grid, two-way sync, legend filters by class; Map panel: lasso filters samples (needs a Mapbox token); Spaces: panels side by side over one view | https://docs.voxel51.com/user_guide/app.html |
| FiftyOne | annotation is delegated to CVAT or Label Studio through `annotate()`; in-app tagging only (a newer in-app Annotate tab exists, version-gated, unverified for our use) | https://docs.voxel51.com/integrations/cvat.html |
| OpenSearch | `geo_polygon` query works on `geo_point` fields (our `capture.location`) | https://docs.opensearch.org/latest/query-dsl/geo-and-xy/geopolygon/ |
| MapLibre GL JS | `BoxZoomHandler` is Shift+drag; `map.boxZoom.disable()` frees the gesture for area selection | https://maplibre.org/maplibre-gl-js/docs/API/classes/BoxZoomHandler/ |

No surveyed tool assigns classes to a whole embedding cluster of detections in one step: FiftyOne and
Encord select a cluster to tag it or make a collection; Label Studio's bulk labelling is whole-task
classification only. That is the gap Cropwright fills.

**Adopted.**
- Box drawing by drag or two clicks, then select / move / resize; label chosen as the box is finished, with
  the last class pre-selected (Label Studio, Kili, CVAT).
- Tab / Shift+Tab to cycle objects and one-object-at-a-time keyboard work (CVAT AAM) for the editor's object
  list and for Review.
- Undo Ctrl+Z, redo Ctrl+Shift+Z (Encord), alongside Review's existing `Z`.
- Remappable, scoped keys with conflict refusal and an in-app list (CVAT, Kili, Label Studio); ours are
  already served and validated per project.
- "Work in the order displayed" (Label Studio's Data Manager): Review and the filmstrip walk the Workspace
  query in its served order.
- Area and embedding selection that filters the grid (FiftyOne Map and Embeddings panels), as filters that
  carry to every view.
- Next / previous with "skip to the next image with unlabelled objects" (CVAT navigation) as a served
  Gallery filter (`label_state`).
- Accept, Fix + Accept, Reject in review (Label Studio): Confirm, edit-then-confirm, Discard in Review.

**Rejected, with the reason.**
- Bulk labelling limited to whole-image classification (Label Studio): our core is per-object batches.
- Hand-off to a separate annotation tool by export and import (FiftyOne): breaks the loop and class
  identity at a boundary.
- Review that only edits and re-submits with no explicit accept (Kili): no cheap "looked, fine" signal.
- Hotkeys documented only in a repository file (Label Studio `keymap.json`): ours are served and shown with
  `?`.
- A key whose meaning changed across versions (CVAT `N`): one meaning per key.
- A map that needs a third-party token (FiftyOne Mapbox): tiles come from the served policy.
- Job and assignment workflows (CVAT jobs, Kili steps): the project has no user accounts; queues and the
  lock rule cover the multi-user case. Revisit only with accounts.
- Minimap and filmstrip have no documented precedent in the surveyed pages; the filmstrip is adopted for
  our own reason (walking a sorted query), the minimap waits for wave 4 polish.

---

## 5. Existing workarounds in `frontend/src` and the route that replaces each

| # | Where | What it does | Replace with | Priority |
|---|---|---|---|---|
| W-1 | `lib/api_nearDup.ts:166-185` (`previewItemsForImages`), used by `nearDupPageController.svelte.ts:71-74`, `nearDupGroupController.svelte.ts:39` | image-level member picture = first crop found by `GET /crops?image_id=` | `thumbnail_url` on `NearDupGroupSummary` and `NearDupMemberView` (R-6) built from `GET /images/{id}/thumbnail` (R-3) | P1 |
| W-2 | `lib/nearDup/nearDupGroupController.svelte.ts:5,27,64,78` | undo only for decisions made in the open dialog | `GET /near_duplicates/decisions` (R-6) and a served `decision` on each member | P1 |
| W-3 | `lib/components/nearDup/NearDupThumb.svelte:22-26` | image frame via `GET /crops/{item}/image` of a guessed item | served `thumbnail_url` (R-6) | P1 |
| W-4 | `lib/api_clusters.ts:274-300` | cluster header from `GET /clusters?cluster_uid=&max_clusters=1`, with a client-built fallback card | `GET /clusters/{cluster_uid}` (R-5) | P1 |
| W-5 | `lib/datasetStats.ts:70-94` (`labeledTotalOf`, `labeledRowsOf`, `withPct`) | sums `by_human + by_vlm + by_classifier + other`, **omits `by_import`**, computes percentages | served `labeled.rows` with label, count and share (R-7) | **P0** |
| W-6 | `lib/components/DatasetStats.svelte:305` | coverage % computed in the browser | served share in the `regions` block (R-7) | P2 |
| W-7 | `lib/map/mapController.svelte.ts:125-135` | tile → image ids → crops list | `geo_bbox` filter (R-9) + Images view; delete `GET /map/images` | P1 |
| W-8 | `lib/cropMeta/cropMetaLoader.svelte.ts:61`, `SourceImageOverlay.svelte:37,281`, `ReviewCropPane.svelte:101` | image and its items reached through a crop | `GET /images/{image_id}` and `/images/{image_id}/file` (R-2, R-3) | P1 |
| W-9 | `lib/review/reviewDiverseController.svelte.ts:198-213` | `POST /select/diverse` returns ids, then `getCropsByIds` hydrates each page | serve item pages for a finished selection (`GET /select/diverse/{job_id}/items?cursor=`) | P2 |
| W-10 | `lib/train/trainCohortsController.svelte.ts:244-262` | one `/crops` count query per cohort (N requests) | served counts on `GET /training_cohorts` | P2 |
| W-11 | `lib/components/ClassSidebar.svelte:119-121` | sorts classes by `cluster_size` in the browser; shows 0 with no partition | `GET /classes?sort=` and a served sidebar count (`item_count`, G-04) | P1 (the 0) / P2 (the sort) |
| W-12 | `lib/components/export/ExportClassTable.svelte:34-45` | sorts the served class table in the browser | `sort` on `GET /stats/classes` | P2 |
| W-13 | `lib/stores/undo.svelte.ts:190-215` | a client-side undo stack for single writes, lost on reload; bulk writes use server label ops | record single writes as label ops too, so undo is one served history | P2 |

Not workarounds (keep): the shared item filter serialising to query params; rank and clarity chips that
format served numbers; `SourceImageOverlay` drawing served boxes.

---

## 6. Ranked gap list

Size: S (under a day), M (1 to 3 days), L (over 3 days). Priority: P0 broken or misleading, P1 core workflow
friction, P2 polish. "Now" = essential for a smooth workflow; "Later" = polish or scale beyond thousands.

| ID | User problem | Evidence | Fix | Backend route | Size | Pri | When |
|---|---|---|---|---|---|---|---|
| G-01 | Dashboard says 100 labels, 100 % human; the project has 285 validated (185 imported) | `dashboard-d.png`; W-5; `/stats/dataset` `labeled.by_import=185` | serve the breakdown rows; render as served | CHANGED `GET /stats/dataset` (R-7) | S | P0 | Now |
| G-02 | Clusters page empty and tells the user to "run the auto-label pipeline" while a built partition waits for "Activate it" on another page | `clusters-d.png`, `f_look-clusters` empty, `ClustersGrid.svelte:258-263`, `dashboard-d.png` | serve `empty_reason` and `pending_partition`; show the ClusterRunPanel activate banner on the page | CHANGED `GET /clusters` (R-8) | S | P0 | Now |
| G-03 | No way to see whole images with their boxes, or one image and all its crops | section 4.2 | Images view + image page | NEW R-1, R-2, R-3 | L | P1 | Now |
| G-04 | Class sidebar and Classes "TOTAL (IN CLUSTER)" show 0 for every class with no serving partition (74 car, 133 person exist) | `clusters-d.png`, `classes-d.png`, W-11 | serve `item_count` per class and the cluster count separately; sidebar shows the item count | CHANGED `GET /classes` | S | P0 | Now |
| G-05 | Four review surfaces, 14 nav entries | section 3 | Workspace + Quality | none beyond R-x | L | P1 | Now |
| G-06 | Crop dialog: siblings are dead thumbnails; no "open image"; the source is drawn twice | `i_crop-detail-d.png`, `CropSourceContext.svelte:74-84` | siblings open that crop; "Open image" → image page; drop the duplicate small source | uses R-2 | S | P1 | Now |
| G-07 | Review shows tabs for features that cannot fill them (Uncertainty without a probe, Model disagreements without a promoted model) and no counts | `f_look-review.png`; `GET /review/uncertainty` `empty_reason` "no probe predictions" | serve `available`, `unavailable_reason` and `count` per tab; Queues rail hides or explains | CHANGED `GET /review/tabs` (R-10) | M | P1 | Now |
| G-08 | A new project has no classes; Review shows no class buttons; the seed panel is collapsed on Classes | `f_look-review.png`, `f_look-classes.png` | served "Next steps" (create classes from detector, cluster, activate, set hotkeys, build near-dups) with section links | NEW `GET /next_steps` (R-12) | S-M | P1 | Now |
| G-09 | Sort-fallback banner reads "default sort 'atypicality' orders by 'cluster_score', which no item has yet; using 'rece…" | `review-d.png`, `src/services/curation/review_sorts.py:433` | plain served text: "Sorted by most recent. 'Most atypical' needs a clustering run." | CHANGED message only | S | P1 | Now |
| G-10 | Cannot sort crops by class, detector class or detector score; no served sort labels on `/crops` | contract `GET /crops` `sort` | add fields, serve `sort_options` like `/clusters` | CHANGED `GET /crops` (R-4) | S | P1 | Now |
| G-11 | Near-dup members have no picture when an image has no crop; member class and holdout unknown | W-1, W-3 | served member view | CHANGED R-6 | S | P1 | Now |
| G-12 | Near-dup decisions cannot be listed or undone after the dialog closes | W-2 | list route; served `decision` per member | NEW R-6 | S | P1 | Now |
| G-13 | Map cell lists crops, not photos; the map cannot narrow the review set | W-7 | `geo_bbox` filter, cell `bounds`, Map view → Images | CHANGED shared filter, `GET /map/cells`, `GET /map`; DELETE `GET /map/images` (R-9) | M | P1 | Now |
| G-14 | "Activate it" copy says "review it, then activate it" but nothing shows an unserved partition | `f_click-dashboard-0-Cluster_items.png` | change the copy now; a preview of a pending partition is Later | none | S | P2 | Now |
| G-15 | Matching items says 477 while the dashboard says 502; the 25 test-holdout items are hidden without saying so | `clusters__mode-matching-d.png`, `dashboard-d.png` | serve `hidden: {test_holdout: 25, excluded: n}` on list pages | CHANGED `GET /crops`, `/review/{tab}`, `/images` | S | P2 | Now |
| G-16 | Lists stop at 10,000 results | 422 on `coco-e2e` | cursor paging | CHANGED R-11 | L | P1 | Later (before 100k) |
| G-17 | Cannot sort or filter images by "has unlabeled objects" at scale; export ships 203 unlabeled objects on 29 images | `export-d.png` | Now: `label_state` filter by item join (capped, like the map); Later: per-image counts on the image doc for sorting | R-1 now; R-13 later | M / L | P1 | Now / Later |
| G-18 | Project in `building` shows "not available" with no refresh | `ProjectUnavailable.svelte` | poll the project until active or failed, show its status | none | S | P2 | Now |
| G-19 | Crop dialog source overlay labels every region box "Object N" over the item boxes; unreadable on busy images | `i_crop-detail-d.png` | item boxes labelled by served label; region boxes only for the selected item unless "all regions" is on | serve `label` per box in R-2 | S | P2 | Later |
| G-20 | Card chips "#7 b0.14" and "★1" are cryptic | `clusters__mode-matching-d.png` | serve `clarity_label` (the threshold is a backend decision); show rank as "largest", "2nd" | CHANGED item wire | S | P2 | Later |
| G-21 | Projects list and switcher load every project, no search | `projects-d.png` | `q`, `status`, `sort`, cursor | CHANGED `GET /curation/projects` | M | P2 | Later |
| G-22 | Undo for single writes is a client stack lost on reload | W-13 | single writes become label ops | CHANGED label writes | M | P2 | Later |
| G-23 | Three places start VLM work (Dashboard Auto-label, Clusters auto-label loop, Assist…) | `dashboard-d.png`, `clusters-d.png` | one "VLM" menu in Workspace (Assist on selection or scope; loop status) | none | M | P2 | Later |
| G-24 | Docs drift: `dashboard.md` "Recluster now"; `review.md` lists 6 of 12 tabs | docs-site | rewrite with the Workspace and Quality pages | none | S | P1 | with each lane |
| G-25 | Review filter row fills the first 430 px screen | `review-m.png` | collapse filters behind one "Filters" button under `md` | none | S | P2 | with Workspace |
| G-26 | Near-dup Images/Items and Duplicates/Leak toggles show no selected state; "Full rebuild" on a never-built level | `i_near-duplicates-advanced-d.png`, `f_look-near-duplicates.png` | `aria-pressed` styling; "Build" when no run | none | S | P2 | Now |
| G-27 | A failed thumbnail stays broken | `f_look-clusters.png` | one retry with backoff, then the served source-missing badge | none | S | P2 | Later |
| G-28 | Diverse selection, cohort counts, export table sort done in the browser | W-9, W-10, W-12 | served pages, counts and sort | CHANGED | S each | P2 | Later |

---

## 7. Backend routes to add or change

Patterns to follow: routers are thin adapters under `src/routers/curation/`; logic in `src/services/curation/`
(no FastAPI import); wire models in `src/schemas/curation/` or the router's `_*_models.py`; every item goes
through `serialize_item` in `src/services/curation/wire.py`, region boxes through `region_boxes_to_wire`
(and `_source_to_parent` for image coordinates); OpenSearch only through the factory; the bound project
names the indexes (`items_index()`, `images_index()`); every new scoped route is added to
`tests/curation/cross_project_leak/_route_tables.py`; `make contracts` and commit `contracts/` in the same
commit. "View-ready" means the response carries what the page draws: labels, counts, normalized box
coordinates, served sort options, an `empty_reason`; the frontend computes none of them.

### R-1 NEW `GET {prefix}/images` (image gallery page)

- Query: the shared item filter (`src/routers/curation/_item_filter_params.py`; an image matches when at
  least one of its items matches, joined through the items index with the capped-join pattern of
  `map_cells.py` (`JOIN_MAX_IMAGES`, `join_truncated`)), plus `source`, `has_location`, `geo_bbox` (R-9),
  `in_holdout`, `near_dup` (`member|representative|none`), `label_state` (`has_unlabeled|all_labeled|no_items`,
  by join now, by stored counts after R-13), `sort` (served ids: `indexed_at:desc` default,
  `indexed_at:asc`, `taken_at:desc`, `taken_at:asc`, `file_name:asc`; `unlabeled_count:desc` after R-13),
  `cursor`, `limit` (1-200, default 60), `include_test` (default false, like `/crops`).
- Response `ImagePage`: `items: ImageCard[]`, `next_cursor`, `total`, `join_truncated`, `sort_applied`,
  `sort_options: [{id, label}]`, `hidden: {test_holdout, excluded}`, `empty_reason`.
- `ImageCard`: `image_id`, `file_name`, `thumbnail_url`, `width`, `height`, `source`, `indexed_at`,
  `taken_at`, `has_location`, `counts: {items, labeled, validated, unlabeled, in_holdout}`,
  `near_dup: {group_id, purpose, size, is_representative} | null`, `boxes: BoxView[]`, `boxes_truncated`.
- `BoxView` (shared with R-2): `crop_id`, `box_id | null`, `kind` (`item|region`), `bbox_norm` `[x1,y1,x2,y2]`
  in image coordinates, `label` (served display text: class name, or the served "unlabeled" word),
  `class_name | null`, `tone` (served: `validated|machine|unlabeled|holdout|rejected`), `state | null`.
- Counts for a page: one aggregation on the items index over the page's image ids (terms on `image_id`
  with filter sub-aggregations), so a page is two queries regardless of size.
- Files: `src/routers/curation/images_view.py` (new, router), `src/services/curation/image_view.py` (new),
  `src/schemas/curation/images.py` (new). Tests: `tests/curation/test_images_view.py` (counts equal an
  items-index aggregate, holdout hidden by default, filter join and `join_truncated`, sort options served,
  cursor stable), the leak-sweep row.

### R-2 NEW `GET {prefix}/images/{image_id}` (one image with its detections)

- Response `ImageDetail`: `image: {image_id, file_name, file_url, width, height, source, indexed_at,
  taken_at, location | null (per the project's location policy), dataset_split, in_holdout, near_dup}`,
  `counts` (as the card), `items: ItemDoc[]` via `serialize_item`, ordered by `crop_rank_in_image`,
  `boxes: BoxView[]` (all items and their region boxes), `items_truncated`.
- Width and height: reuse `_pixel_size_from_header` from `crop_context.py` when the doc lacks them.
- **Delete `GET /crops/{crop_id}/context`** (`src/routers/curation/crop_context.py`, its model,
  `tests/curation/test_crop_context.py` folded into the new test) in the same change, and switch
  `cropMetaLoader`, `SourceImageOverlay`, `ReviewCropPane` to `crop.image_id`.

### R-3 NEW `GET {prefix}/images/{image_id}/file?max_dim=` and `GET {prefix}/images/{image_id}/thumbnail?size=`

- Clean source pixels by id (EXIF orientation and resize only, no overlay), through `render_source_image`
  and `project_owns_image_path` in `src/services/curation/image_serving.py`; the thumbnail uses
  `THUMBNAIL_CACHE` like the crop thumbnail; `source_missing` 404 body as today.
- **Delete `GET /crops/{crop_id}/image`**. Check `GET /images/serve` and `GET /images/root/{alias}`: no
  Cropwright caller; the lane greps `tests/`, `scripts/` and `docs-site/` and deletes them if nothing but
  their own tests and the API reference uses them (they expose paths; ids replace them).

### R-4 CHANGED `GET {prefix}/crops`

- Add sort fields `class_name`, `detector_class_name`, `detector_confidence`, `validated`; serve
  `sort_applied` and `sort_options: [{id, label}]`; serve `hidden` (G-15).

### R-5 NEW `GET {prefix}/clusters/{cluster_uid}`

- The served cluster card (the item shape of `ClusterListResponse` plus `representatives`), 404 when the
  uid is not in the serving partition, with the served reason. Delete the `max_clusters=1` call and the
  fallback card in `api_clusters.ts:274-300`.

### R-6 Near-duplicates (`src/routers/curation/near_duplicates.py`, `src/schemas/curation/near_duplicates.py`, `src/services/near_duplicates/`)

- NEW `GET /near_duplicates/decisions?level=&member_id=&cursor=&limit=` → `NearDupDecisionPage {items:
  NearDupDecisionView[], next_cursor}`, newest first.
- CHANGED `NearDupGroupSummary` + `representative_thumbnail_url`; `NearDupMemberView` + `thumbnail_url`,
  `image_id`, `crop_id | null`, `class_name | null`, `in_holdout`, `decision: {id, kind} | null` (the pin
  between this member and the representative). Built from R-3 and the items index in one query per page.
- CHANGED empty answers carry `empty_reason` with the cut and the scored count.

### R-7 CHANGED `GET {prefix}/stats/dataset`

- Add `labeled.rows: [{key, label, count, share}]` covering human, import, VLM, classifier, other, with
  `labeled.total`; `regions.coverage_share`. Remove the frontend sums (W-5, W-6). Test: rows sum to
  `labeled.total`; on `coco-e2e` today that is 100 + 185 = 285 = `validated`.

### R-8 CHANGED `GET {prefix}/clusters`

- Add `empty_reason` (`no_partition`, `pending_activation`, `filtered_empty`, served text) and
  `pending_partition: {partition_id, clusters, items} | null` from the partitions table, so the grid can
  show the activate banner (`ClusterRunPanel`) in place.

### R-9 Map and area filter

- CHANGED shared item filter: `geo_bbox` (`west,south,east,north`, validated server-side, refused with the
  usual 422 body) and `has_location`; applied on `/crops`, `/review/{tab}`, `/clusters`, `/images`,
  `/map/cells` (`src/services/curation/item_filter.py`, `_item_filter_params.py`, the capture fields).
- CHANGED `MapCellProperties` + `bounds`; `MapMeta` + `n_without_location`.
- DELETE `GET /map/images` and `MapTileImages`.

### R-10 CHANGED `GET {prefix}/review/tabs`

- Per tab: `available`, `unavailable_reason`, `count` (one `msearch` server-side), and a new `audit` tab
  (the pending audit sample) so the sample is labelled in the Workspace. DELETE `GET /audit/queue`.
- **Queue set cut (owner feedback, lane 3B).** The served queues become five plain-named presets, each a
  `review_tab` filter usable by every Workspace view: **To label** (no class yet), **Check machine labels**
  (machine class, not validated, lowest confidence first), **Disagreements** (detector, label, VLM or
  promoted model disagree), **New class proposals**, **Audit sample** (only while a sample is pending).
  Uncertainty, Classifier blind spots, Imported labels, Objects and the rest stop being tabs and become
  served filter fields and sorts on `GET /crops` (`review_tab` values stay valid filter values where
  the filter needs them). A queue that cannot fill is absent, with its served reason in the Next steps.

### R-11 CHANGED paging (Later, before 100k)

- `/crops`, `/review/{tab}`, `/regions`, `/images`: `cursor` (opaque: point-in-time id plus `search_after`
  sort values, with `crop_id`/`image_id` as tiebreak) and `next_cursor`; `/review/{tab}/locate` returns a
  cursor and an index. Remove `page` in the same change (one mechanism). The UI shows "N of total" from the
  served index, never a page number.

### R-12 NEW `GET {prefix}/next_steps`

- `steps: [{id, label, detail, done, section}]` decided by the backend from counts it has (no classes but
  detector proposals; embedded but no partition; partition pending activation; no hotkeys; near-dups never
  built; validated below the class floor). The Dashboard and an empty Workspace render it.
- Also serves `workspace_default_view` (`gallery` until a partition serves, then `clusters`; owner
  answer 2), so the rule lives in the backend, and `next` (the first undone step) for the Workspace's
  next-step chip. Built in lane 2C.

### R-13 Per-image label counts (Later)

- Store `counts` on the image doc, refreshed by a coalescing Celery task keyed by `image_id` after label,
  region and exclude writes (enqueued from the OCC write helpers), so `/images` can sort by
  `unlabeled_count` and filter `label_state` without a join at 100k+ images.

### R-14 NEW `POST {prefix}/images/{image_id}/items` (a human-drawn box becomes an item)

- Body `ItemCreateRequest`: `bbox_norm` `[x1,y1,x2,y2]` in image coordinates, `class_name` (identity is the
  name; resolved to an active registry entry, deprecated or unknown is 422 `unknown_class` with the served
  text), `request_id` (client UUID; a repeat with the same id returns the first result, so a retry never
  makes two items).
- Refusals: 404 image not in the bound project; 422 `box_out_of_image`, `box_too_small` (served minimum
  from the active region profile limits or a curation constant, named in the body), `degenerate_box`.
- Writes through `build_item_doc` (section 4.7 lists the fields: human class and label source,
  `class_validated`, new `geometry_source: human`, `geometry_revision: 1`, `embedding_state: deferred`,
  region seed). Marks near-dup pending. Enqueues the `reprocess` job, scope `embed`, for the image.
- Response `ItemCreated`: `item` (`serialize_item`), `image_boxes: BoxView[]`, `image_counts`,
  `followup_job_id`.
- New `origin` value `drawn` in the shared filter (`item_filter.py`: `geometry_source` is `human` and the
  item has no detector fields), so drawn boxes can be found and audited.
- Files: `src/routers/curation/image_items.py` (new), `src/services/curation/image_item_write.py` (new),
  `src/schemas/curation/images.py`. Tests: `tests/curation/test_image_items.py` (fields written, class by
  name, idempotent retry, refusals, cross-project 404, follow-up enqueued), the leak-sweep row.

### R-15 NEW `PATCH {prefix}/crops/{crop_id}/geometry` and `POST {prefix}/crops/{crop_id}/geometry/undo`

- PATCH body: `bbox_norm` (image coordinates), `expected_geometry_revision`, optional
  `drop_region_box_ids`. Writes through the OCC read/merge/write helper: sets `bbox_norm`,
  `crop_area_norm`, `geometry_source: human`, `geometry_revision + 1`, snapshots the prior geometry into
  `edit_history` (new kind `geometry`), drops the item vector and sets `embedding_state: deferred`, evicts
  the crop-cache file and the cached thumbnail, enqueues the `embed` follow-up.
- 409 `geometry_conflict` with the current item; 422 `regions_outside_item` with the box ids (unless they
  are listed in `drop_region_box_ids`); refusals as R-14.
- Undo restores the last restorable `geometry` snapshot (same pairing rule as `region/undo`); 409 when
  none is left.
- CHANGED wire: items and `BoxView` (R-1, R-2) gain `geometry_revision`, `geometry_source`, `locked`
  (served from the lock predicates) and `lock_label` (served text, for example "Set by a person"); item
  `thumbnail_url` carries `?v=<geometry_revision>`.
- Files: `src/routers/curation/crops_geometry.py` (new), `src/services/curation/item_geometry.py` (new),
  `src/services/curation/edit_history.py`, `src/services/curation/wire.py`. Tests:
  `tests/curation/test_item_geometry.py` (OCC conflict, undo pairing, vector dropped and state deferred,
  cache evicted, regions outside refused).

### R-16 CHANGED geometry lock rule and soft delete

- `src/clients/occ_locks.py`: `is_locked_geometry(source)` (human or import geometry, a human
  `not_an_object` exclusion, or the frozen holdout). Consulted by `reprocess_detect.py` (never moves,
  deletes or duplicates a locked item; a new detection that matches a locked or human-deleted item by IoU
  is dropped) and by `item_delete.py` callers.
- `POST /crops/batch_exclude`: `reason` becomes a served enum (`ignore`, `not_an_object`); the editor's
  delete is `not_an_object`; un-exclude is the undo. Excluded items stay hidden by default everywhere
  (`include_excluded`).
- Tests: `tests/curation/test_geometry_lock.py` (re-detect on an image with a drawn item, a moved item and
  a human-deleted item: none moved, recreated or duplicated); live harness case in
  `tests/live/test_live_annotation.py`.

### R-17 NEW `GET {prefix}/images/locate`

- Query: the shared item filter, the `/images` sort and `image_id`. Response: `{index, total,
  cursor_before, cursor_after}` (like `/review/{tab}/locate`), so the editor's filmstrip and `,` `.` walk
  the Workspace query from a deep link. 404 when the image does not match the filter (served reason).

### R-18 NEW `GET {prefix}/table` and `GET {prefix}/table/columns`

- `columns?rows=detections|images` → `{columns: [{id, label, sortable, align, kind}], default: [ids]}`.
- `table?rows=&columns=&sort=&cursor=&limit=` plus the shared item filter → `{columns, rows: [{crop_id |
  null, image_id, thumbnail_url, cells: {<col id>: {text, value, tone | null}}}], next_cursor, total,
  sort_applied, sort_options, hidden, empty_reason}`. Cells carry the served display text (class names,
  formatted confidence, dates); `value` is for alignment and copy only, never re-sorted in the browser.
- Cursor paging from day one (`search_after` plus a point-in-time), so R-11 does not touch it.
- Detection columns: thumbnail, class, detector class, detector confidence, label source, validated,
  cluster, image file, taken at, area, rank, clarity, near-dup, embedding state, updated. Image columns:
  thumbnail, file, taken at, located, items, labelled, unlabelled, near-dup, split, indexed at.
- Files: `src/routers/curation/table_view.py`, `src/services/curation/table_view.py`,
  `src/schemas/curation/table.py` (new), reusing `crop_listing.py` and `image_view.py` queries. Tests:
  `tests/curation/test_table_view.py` (every served sortable column sorts server-side, cursor stable,
  cells equal the item wire).

### R-19 NEW `POST {prefix}/selection/summary`

- Body: an `ItemSelection`. Response: `{detections, images, clusters, locked, in_holdout, by_class:
  [{class_name, count}] (top 10 and "other"), taken_range | null, located}`. One aggregation; refuses like
  `item_selection.py` above `MAX_BULK_SELECTION`. Feeds the bulk bar, the selection details panel and the
  Map cell panel (a filter selection with the cell `bounds`).
- Tests: counts equal a `/crops` count with the same filter; image selection expands to matching items.

### R-20 CHANGED shared item filter

- `cluster_uids` (repeatable) replaces the single `cluster_uid` on the filter and every list route that
  takes it (one mechanism; call sites swapped in the same change).
- `geo_polygon=lon,lat;lon,lat;...` (3 to 64 vertices) on the images' `capture.location` (`geo_polygon`
  query on `geo_point`), joined like `geo_bbox`; both are snapped to the location policy precision.
- `origin=drawn` (R-14).
- Tests: `tests/curation/test_item_filter_geo.py` (polygon, snapping, refusal bodies).

### R-21 CHANGED `GET {prefix}/map/cells`

- `MapCellProperties` + `image_id | null` and `thumbnail_url | null` when the cell holds one photo (plus
  `bounds` from R-9). Test: a single-photo cell at max zoom serves its image.

### R-22 CHANGED `POST {prefix}/classes` refusal

- A 422 for a non-slug name carries `suggested_name` (lower-case, spaces and dashes to `_`, other
  characters dropped) and the served text, so the inline create row can offer it. The name rule itself is
  unchanged. Test: "Traffic light" → `traffic_light`.

### R-23 CHANGED `GET {prefix}/viz/projection` (wave 4, Embedding view)

- Accepts the shared item filter (points not matching are served with `matched: false`, drawn dimmed);
  the shared filter gains `viz_box=x1,y1,x2,y2` on `viz_x` / `viz_y`, so a box drawn on the scatter
  filters every view. Tests: filter applied; `viz_box` equals a range query count.

---

## 8. Frontend changes

All under `frontend/src`; components and controllers are reused, not rewritten.

**File map for the Workspace, editor and Map (sections 4.6 to 4.8; section 9 assigns them to lanes):**

| Path | What |
|---|---|
| `routes/p/[project]/workspace/+page.svelte` | the shell: filter bar, view bar, rail, main pane, details panel, bulk bar |
| `lib/workspace/workspaceState.svelte.ts` | URL state: filter, queue, scope, view, per-view sort, selection, focus (no business data) |
| `lib/workspace/viewRegistry.ts` | one entry per view (id, served label, component, keymap scope); each view lane adds one line |
| `lib/workspace/detailsRegistry.ts` | one details component per focused kind (image, detection, cluster, map cell, selection) |
| `lib/workspace/views/{Gallery,Clusters,Detections,Review,Table,Map}View.svelte` | the six views |
| `lib/components/workspace/{WorkspaceRail,ViewBar,DetailsPanel,BulkBar}.svelte` | shell parts; `BulkBar` grows from `ClusterDetailToolbar` |
| `lib/components/images/{ImageCard,ImageGrid}.svelte`, `lib/images/imageGalleryController.svelte.ts` | Gallery |
| `lib/components/table/{DataTable,ColumnPicker}.svelte`, `lib/table/tableController.svelte.ts` | Table (served columns and cells) |
| `lib/components/ClassPicker.svelte`, `lib/classes/classPicker.svelte.ts` | one picker with inline create; replaces `review/ClassPickerModal.svelte` and `lib/review/reviewClassPicker.svelte.ts` |
| `routes/p/[project]/images/[image_id]/+page.svelte` | the image annotation editor |
| `lib/editor/{editorController,editorCommands,editorKeys}.svelte.ts` | editor state, command stack (undo/redo), keymap context `image_edit` |
| `lib/components/editor/{EditorCanvas,ObjectList,Filmstrip,ToolStrip}.svelte` | editor parts; `EditorCanvas` (zoom, pan, handles) reuses the pure box helpers in `lib/annotations/multiBox.ts` and mounts `MultiBoxCanvas` for region boxes; `SourceImageOverlay` stays the read-only drawing for Review and the details panel; no box math is duplicated |
| `lib/map/mapController.svelte.ts`, `lib/components/map/MapView.svelte` | grown into the Map view: area tools, pins, cell panel |

- **Image page** `routes/p/[project]/images/[image_id]/+page.svelte` + `lib/images/imagePageController.svelte.ts`:
  `SourceImageOverlay` at full width drawing served `boxes`; `MultiBoxCanvas` for box edit; a strip of
  `CropCard` for its items with the cluster-detail bulk bar; near-dup and location rows; previous/next from
  the Workspace query (the controller keeps the served cursor and the current list in URL state, never a
  client sort).
- **ImageCard** `lib/components/images/ImageCard.svelte`: served thumbnail, boxes drawn as SVG over it from
  `bbox_norm`, served `label` and `tone`, served counts as chips, near-dup badge. Grid in
  `lib/components/images/ImageGrid.svelte`, controller `lib/images/imageGalleryController.svelte.ts`.
- **Crop dialog**: siblings become buttons that open that crop; "Open image" link; drop the duplicate small
  source block (`CropMetaPanel.svelte`, `cropMeta/CropSourceContext.svelte`).
- **Workspace** `routes/p/[project]/workspace/+page.svelte` grows from `routes/p/[project]/clusters/+page.svelte`:
  view switch, Queues rail (served tabs with counts, from `reviewTabsVocabularyStore`), cluster scope
  (`ClusterDetailToolbar`, `ClusterFacetBar`, `ClusterProposalPanel`, `ClusterDetailGrid`), Focus mode
  mounting `ReviewCropPane` and the `$lib/review` controllers over the selected queue or the current
  Detections query. Keyboard: Focus registers the existing `review.*` actions; grid registers the existing
  `cluster.*` actions; new actions (`workspace.focus`, `workspace.view_next`, `workspace.view_prev`) go
  into `lib/keymapFallback.ts`, no new window listener.
- **Quality** `routes/p/[project]/quality/{audit,near-duplicates,history}/+page.svelte` reusing the current
  page bodies; a tab strip; Near-dup groups get "Open in Workspace" (`/workspace?view=images&near_dup_group=<id>`).
- **Map view** in Workspace mounting `MapView`; cell click sets `geo_bbox` from served `bounds` and
  switches to Images.
- **Nav** in `routes/p/[project]/+layout.svelte`: the target list of section 3; `SECTION_LABELS` and
  `lib/projectPaths.ts` updated; `showSidebar` keyed on `workspace`.
- **Rules kept**: links through `projectHref()` + `resolve()`; served text verbatim; absent, not disabled,
  when a feature is not served; thumbnails only in grids (`/crops/{id}/thumbnail`, `/images/{id}/thumbnail`).
- **Docs**: `docs-site/docs/cropwright/user-guide/` gains `workspace.md`, `image-view.md`, `quality.md`;
  `clusters.md`, `review.md`, `audit.md`, `near-duplicates.md` are rewritten or folded in the step that
  deletes their page; `frontend/CLAUDE.md` route table and `frontend/CHANGELOG.md` `[Unreleased]` in each
  lane.

---

## 9. Lanes

One lane = one worktree on its own branch, targeted tests only (red then green for its own changes, static
checks on touched files), commits with hooks on. The coordinator merges a wave with `--no-ff` and runs one
full sweep. Each lane reads the official docs for what it touches and cites them in its commit messages:
OpenSearch `search_after` and point-in-time, terms and composite aggregations; FastAPI response models and
`Query` validation; SvelteKit routing and `$app/state`; Svelte 5 runes; MapLibre GL JS `fitBounds` and
`queryRenderedFeatures` for the map lane.

Test tiers per lane: **unit** (pytest offline, `tests/curation/`; vitest in `frontend/src/**`, including
`src/lib/contract/endpointCatalog.test.ts`, which fails until call sites match the regenerated contract),
**stubbed e2e** (`frontend/e2e/stubbed/`, plus a row in `test_a11y_axe.py` `PAGES` for every new page or
state, and the 430 px tests), **live Cropwright tier** (`frontend/e2e/live/`, read-only against
`coco-e2e`: `test_route_sweep.py` and `test_data_agreement.py` gain the new pages), **live harness**
(`tests/live/`, write paths under its own compose project name).

### Wave 1 (do now, three lanes in parallel)

**Lane 1A Honest numbers and empty states** (backend + frontend, M; G-01, G-02, G-04, G-07, G-09, G-14,
G-15, G-18, G-26)
- Backend: `src/routers/curation/_stats_models.py`, `src/routers/curation/stats.py` (R-7);
  `src/schemas/curation/clusters.py`, `src/routers/curation/clusters.py` (R-8); `src/routers/curation/classes.py`
  and its model (`item_count`); `src/routers/curation/_review_tab_models.py` and the tabs route (R-10 without
  the audit tab); `src/services/curation/review_sorts.py:433` message; `hidden` counts in
  `src/services/curation/crop_listing.py` and `src/routers/curation/review.py`.
- Frontend: `lib/datasetStats.ts` (delete the sums), `lib/components/DatasetStats.svelte`,
  `lib/components/clusters/ClustersGrid.svelte` (served empty reason + `ClusterRunPanel` banner),
  `lib/components/ClassSidebar.svelte`, `lib/components/classes/ClassRegistryTable.svelte`,
  `lib/components/review/ReviewTabsBar.svelte`, `lib/components/ProjectUnavailable.svelte` (poll),
  `lib/components/nearDup/NearDupStatusCard.svelte`, `NearDupGroupList.svelte` (pressed state, "Build").
- Tests: pytest rows sum to total, empty reasons per state, tab availability; vitest for each component;
  stubbed `test_dashboard_stats.py` (labeled rows include import), `test_cluster_first_load.py` (pending
  partition banner), `test_near_duplicates.py` (pressed state); live tier: labeled total equals
  `validated` on `coco-e2e`.

**Lane 1B Image view-model routes** (backend + call-site swaps, L; G-03 backend, W-8)
- R-1, R-2, R-3, `BoxView`; delete `/crops/{id}/context` and `/crops/{id}/image` (and the path routes if
  unused); swap `lib/api_cropRead.ts` (`getCropContext` → `getImage`, `getSourceImageScaled` → by image id),
  `lib/cropMeta/cropMetaLoader.svelte.ts`, `lib/components/SourceImageOverlay.svelte`,
  `lib/components/review/ReviewCropPane.svelte`; `make contracts`.
- Tests: `tests/curation/test_images_view.py` (watch fail first), leak-sweep rows, list-uniqueness test,
  `tests/test_contract_typed_responses.py`; vitest for the swapped loaders; stubbed e2e for the crop dialog
  and Review source pane on the new stubs; live harness `tests/live/test_live_images.py` on a public COCO
  ingest (gallery `counts.items` summed equals the items index count; detail items equal `/crops?image_id=`).

**Lane 1C Cluster header and near-duplicate views** (backend + frontend, M; G-11, G-12, W-1 to W-4)
- R-5, R-6; frontend `lib/api_clusters.ts`, `lib/clusters/clusterDetailController.svelte.ts`,
  `lib/api_nearDup.ts` (delete `previewItemsForImages`), `lib/nearDup/*Controller.svelte.ts`,
  `lib/components/nearDup/NearDupThumb.svelte`, `NearDupGroupModal.svelte` (served decision, undo any).
- Depends on 1B for `thumbnail_url` targets: start after 1B's routes are merged, or stub the URL builder
  against the R-3 path and merge after 1B.
- Tests: pytest decisions list order and cursor, member view fields, 404 for a foreign cluster; vitest
  controllers; stubbed `test_near_duplicates.py` (decision made, dialog closed, reopened, undo works),
  `test_cluster_header_refresh.py` on the new route.

### Rules for waves 2 to 4

- Waves 1 to 4 all ship before the speed review (owner). One full sweep per wave on the merged tree.
- **Visual check in every frontend lane:** full-page screenshots at 1600 x 2000 and 430 px of every page
  and state the lane touches, on `coco-e2e` (read-only) and on the lane's own scratch public COCO project
  for writes, looked at and judged against the page's purpose (the views table in section 4.6, the editor
  in 4.7, the map in 4.8). A screen that works but does not serve its purpose is a lane bug.
- **Shared files:** each view lane adds one line to `lib/workspace/viewRegistry.ts` and one to
  `lib/workspace/detailsRegistry.ts`; the coordinator resolves these additive conflicts at merge.
- **Docs read per lane** (cite in commits): Svelte 5 runes and SvelteKit `$app/state` (all frontend
  lanes); OpenSearch optimistic concurrency (`if_seq_no`), `search_after` with point-in-time, `geo_polygon`
  and `geo_bounding_box` queries (backend lanes); Celery task idempotency (2D); MapLibre GL JS
  `BoxZoomHandler` and `queryRenderedFeatures` (3D); Pointer Events (3C).

### Wave 2 (after wave 1 merges): the Workspace shell and the backend for every view, four lanes

**Lane 2A Workspace shell, Clusters and Detections views, class picker** (frontend + R-22, L, 4 to 5
days; G-05 part, G-25)
- Shell and state per the section 8 file map: `workspace/+page.svelte`, `workspaceState.svelte.ts`,
  `viewRegistry.ts` (Clusters, Detections), `detailsRegistry.ts` (cluster, detection), `WorkspaceRail`
  (Classes with served `item_count`; Queues rendered from the served tabs as they are, 3B cuts them),
  `ViewBar`, `DetailsPanel`, `BulkBar` (from `ClusterDetailToolbar`, acting on an `ItemSelection`).
- Move `/clusters` into the Clusters view and `/clusters/[id]` into the cluster scope (`?cluster=`);
  Matching items becomes the Detections view. Delete both old routes; update every link
  (`rg -l "projectHref\(.*/clusters"`), the nav, `SECTION_LABELS`, `projectPaths.ts`; docs-site
  `clusters.md` becomes `workspace.md`.
- `ClassPicker.svelte` with inline create, used by the rail ("+ New class") and the bulk bar; backend R-22
  (`suggested_name`) in `src/routers/curation/classes_write.py`. Review keeps its modal until 3B.
- `workspace.*` keymap actions (`[`, `]`, `,`, `.`, `\`, `Shift+O`, `Space`, `Ctrl+A`) in
  `keymapFallback.ts`; no new letters.
- Tests: move and adapt `routes/p/[project]/clusters/**/*.test.ts`; vitest for `workspaceState` (URL round
  trip, selection survives a view switch, per-view sort taken from served `sort_applied`) and
  `ClassPicker` (create row, served refusal and suggestion); stubbed: repoint `test_cluster_*.py`,
  `test_item_filter.py`, `test_selection_actions.py`, `test_keymap.py`; new `test_workspace_shell.py`
  (switch views keeps filter and selection; drawers at 430 px), a11y rows; pytest for R-22;
  `projectRouting.test.ts` (old paths are the app's 404); live route sweep.
- Visual purpose: "label a cluster in one batch": a pure cluster takes one key plus the confirm.

**Lane 2B Filters, sorts and map data** (backend, M, 2 to 3 days; G-10, G-13)
- R-4, R-9 (`geo_bbox`, `has_location`, cell `bounds`, `n_without_location`), R-20 (`cluster_uids`,
  `geo_polygon`, `origin=drawn` clause; snapping to the policy precision), R-21. `GET /map/images` stays
  until 3D deletes it with its caller.
- Files: `src/services/curation/item_filter.py`, `src/routers/curation/_item_filter_params.py`,
  `src/services/curation/map_cells.py`, `src/routers/curation/map.py`, `src/services/curation/crop_listing.py`,
  every `cluster_uid` caller (swap in the same change); `make contracts`.
- Tests: pytest `geo_bbox` / `geo_polygon` validation, snapping and filtering on `/crops`, `/images`,
  `/clusters`, `/map/cells`; `cluster_uids` on every list; sort options served; leak-sweep rows; live
  harness `test_live_capture_map.py` extended for both area filters on the e2e GPS photos.

**Lane 2C View models: table, selection summary, image locate, next steps** (backend, M, 3 days; G-08
backend)
- R-17, R-18, R-19, R-12 (with `workspace_default_view`). Builds on 1B's `image_view.py`.
- Tests: `tests/curation/test_table_view.py`, `test_selection_summary.py`, `test_images_locate.py`,
  `test_next_steps.py` (one case per state, default view Gallery without a serving partition and Clusters
  with one); contract tests; leak-sweep rows.

**Lane 2D Annotation writes** (backend, L, 4 days; **high risk**: data integrity and the lock rule, so
Sonnet at high effort and an independent review before merge)
- R-14, R-15, R-16, the wire fields (`geometry_revision`, `geometry_source`, `locked`, `lock_label`,
  versioned `thumbnail_url`) on items and `BoxView`; `edit_history` kind `geometry`; follow-up `reprocess`
  enqueue (`embed` scope); crop-cache and thumbnail eviction; `origin=drawn` field write.
- Tests (watch each fail first): `test_image_items.py`, `test_item_geometry.py`, `test_geometry_lock.py`;
  live harness `tests/live/test_live_annotation.py` on a public COCO ingest under its own compose project
  name: draw a box, the item exists with a human class, the follow-up job embeds it, `/images` counts move;
  move it, the thumbnail bytes change and the vector is rebuilt; re-run detect: drawn and moved boxes are
  unchanged and a deleted one is not recreated; undo restores the geometry; two writers on one item, the
  second gets 409.

### Wave 3 (after wave 2 merges): the views and the editor, six lanes

**Lane 3A Gallery view and image details** (frontend, M-L, 3 to 4 days; G-03, G-06, G-17 now part)
- `ImageCard`, `ImageGrid`, `imageGalleryController`, `GalleryView`, the image details component; chips
  for `label_state` and `has_location`; Gallery selection as an `ItemSelection` with `image_id`; default
  view from the served `workspace_default_view`; crop dialog: siblings open their crop, "Open image" goes
  to the editor route (3C), the duplicate small source is dropped.
- Tests: vitest (served order kept, cursor appended, no client sort; a scan test like
  `clusterList.scan.test.ts` for the image list); stubbed `test_gallery_view.py` (filter carries from
  Clusters, select images, bulk assign uses the served summary, 430 px), a11y rows; live tier: Gallery
  total equals `GET /images` total on `coco-e2e`, every card draws as many boxes as served.
- Visual purpose: "see what came in, find images that need work".

**Lane 3B Review view and the queue cut** (frontend + small backend, L, 4 days)
- Backend: R-10 (audit queue, the five queues, the rest as filters and sorts); delete `GET /audit/queue`
  and `AuditQueueList.svelte`.
- Frontend: `ReviewView` mounting `ReviewCropPane` and the `$lib/review` controllers over the Workspace
  query; queues in the rail are filter presets for every view; filmstrip; `Enter` from any grid opens
  Review at the focused item and `Esc` returns to the same card; deep link `?view=review&focus=<crop_id>`
  via `/review/{tab}/locate`; Review switches to `ClassPicker` and `ClassPickerModal.svelte` and
  `reviewClassPicker.svelte.ts` are deleted; delete `routes/p/[project]/review/` and move its 17 tests.
- Tests: the moved tests pass against the view; stubbed `test_labeling_flow.py`, `test_region_confirm.py`,
  `test_review_*.py` repointed; new stubbed test: grid, `Enter`, confirm 3, `Esc`, same card focused;
  pytest for the queue set and availability; live tier deep link test.
- Visual purpose: "settle the leftovers one at a time, fast"; also confirm the class name shows (the wave 1
  bug stays fixed).

**Lane 3C Image annotation editor** (frontend + keymap context, L, 5 to 6 days, the largest lane)
- The route and files in the section 8 file map; tools, picker, saves, conflicts, undo and redo exactly as
  section 4.7; filmstrip via R-17; region layer via `MultiBoxCanvas`; keymap context `image_edit` with the
  reserved `V`, `B`, `H` (backend: `src/services/curation/keymap_validator.py` and the keymap vocabulary,
  small).
- At 430 px: the object list is a drawer; select, class change, delete and touch-drag draw work (Pointer
  Events).
- Tests: vitest `editorCommands` (each command and its inverse, a 409 drops the entry, redo sends the fresh
  revision), `editorController` (draws served boxes only, no class lookup in the browser), key scan;
  stubbed `test_image_editor.py` (draw, picker, `Enter` saves; inline create with the suggested name;
  move with a 409 stub; delete then undo; `Tab` cycles; `.` walks the filmstrip; 430 px), a11y row; live
  tier `test_writes.py` gains draw, move, delete and undo on a scratch public project (never `coco-e2e`).
- Visual purpose: "fix one image fast": draw 3 boxes and change 2 classes in under a minute by keyboard.

**Lane 3D Map view** (frontend + one deletion, M, 3 days; G-13)
- Section 4.8: shared filter and queue, served counts, cell panel (R-19 with the cell bounds, R-1 cards),
  pins (R-21), Shift-drag rectangle with `boxZoom` disabled, polygon lasso, area chips that carry to every
  view. Delete the `/map` route, the crop side panel, `GET /map/images` and `MapTileImages` (backend, same
  change).
- Tests: stubbed `test_map_page.py` repointed (cell click opens the panel; an area becomes a chip; Gallery
  is filtered); live `test_map_renders.py` headed on `DISPLAY=:11`; `test_live_capture_map.py` polygon.
- Visual purpose: "find and work data by place"; the "3 items, no points" case must show the points.

**Lane 3E Quality page and Next steps** (frontend, M, 2 to 3 days; G-08 frontend)
- `routes/p/[project]/quality/{audit,near-duplicates,history}` from the current page bodies; delete
  `/audit`, `/near-duplicates`, `/label-ops`; Export stale panel and holdout freeze link to Quality ›
  Near-duplicates; "Open in Workspace" from a group (`?view=gallery&near_dup_group=<id>`). Dashboard Next
  steps card and the empty Workspace state from R-12.
- Tests: stubbed `test_near_duplicates.py`, `test_label_ops.py` repointed; fresh-project stubbed test for
  Next steps; a11y rows.

**Lane 3F Table view** (frontend, M, 2 to 3 days)
- `TableView`, `DataTable`, `ColumnPicker`, `tableController` on R-18: rows of detections or images, served
  columns, header click requests a served sort, column choice in the URL, multi-select feeds the bulk bar,
  `Enter` opens Review, `Shift+O` the editor.
- Tests: vitest (cells rendered verbatim, no client sort, cursor appended); stubbed `test_table_view.py`
  (sort by detector confidence, choose columns, select rows, 430 px horizontal scroll), a11y row; live
  tier: first page equals `GET /table` on `coco-e2e`.
- Visual purpose: "sort by anything and act on the rows".

### Wave 4 (before the speed review): scale, the Embedding view and polish

- **4A Cursor paging** (L, 4 days): R-11 on `/crops`, `/review/{tab}`, `/regions` (R-1, R-17 and R-18
  already use cursors); one shared pager helper in `lib/`. Live harness: a public bench project with more
  than 10,000 items reads past item 10,000 in every view and in Review.
- **4B Per-image counts** (L, 4 days): R-13; Gallery and Table sort by `unlabeled_count`; `label_state`
  stops joining.
- **4C Projects at scale** (M): G-21.
- **4D Embedding view** (M, 3 days): R-23 and an `EmbeddingView` from `EmbeddingPlot.svelte` (box select
  becomes a `viz_box` chip, colour by class from the served legend); moved out of `ClustersGrid`.
- **4E Polish** (S each): G-19, G-20, G-22, G-23, G-27, G-28; the editor minimap for large photos; "label
  this whole duplicate group" from a near-dup representative (owner answer 3).
- Then the full-loop acceptance run (section 10, checks 13 to 18) on the merged tree.

---

## 10. Acceptance checks

Run on `coco-e2e` (read-only) and on a fresh public COCO project for the write steps.

1. Dashboard "Labeled by source" total equals `GET /stats/dataset` `validated` when no VLM labels exist
   (285 on `coco-e2e` today), and lists an "Imported" row (185).
2. With a built but unserved partition, Workspace (Clusters view) shows the served pending-partition banner
   with Activate; after Activate, cards appear without visiting the Dashboard.
3. Class rail counts equal `GET /classes` `item_count` (car 74, person 133 on `coco-e2e` today, holdout
   included as the Classes page shows it).
4. Images view total equals `GET /images` `total` (91 on `coco-e2e`, minus any hidden, with the hidden count
   shown); summing `counts.items` over all pages equals the items count the page states.
5. Image page → a crop → Focus → back to the image keeps the same image; a sibling in the crop dialog opens
   that sibling; every box on the image page matches `GET /images/{id}` `boxes`.
6. Near-dup: make a decision, close the dialog, reload, the decision is listed and can be undone; a group
   member with no crops still shows its picture.
7. No request to `GET /crops/{id}/context`, `/crops/{id}/image`, `/map/images` or `/audit/queue` remains
   (contract and `endpointCatalog.test.ts` enforce it); the old page URLs answer the app's 404 page.
8. A fresh project's Dashboard shows the served next steps in order and each link lands on the step.
9. Review queue in Focus: Enter confirms, a class letter assigns, N skips, Z undoes, Esc returns to the grid
   at the same card; no new window keydown listener (`keymap.literals.scan.test.ts` and the component
   listener tests pass).
10. Map (WebGL browser): a cell's "Show in Gallery" opens the Gallery view with `geo_bbox` equal to the
    served bounds; the header shows located and not-located counts. (Checks 4, 5 and 9 read "Images view"
    as Gallery and "Focus" as the Review view, per section 4.6.)
11. Full-page screenshots of every new or changed page at 1600 and 430 px, looked at, with no overlap,
    clipping, `NaN`, `undefined` or console errors; Cropwright's inner scroll area means the screenshot
    viewport must be tall (1600 x 2000) or the capture taken of the scroll container.
12. Scale (Wave 4): Workspace and Focus read past item 10,000 on a public bench project without a 422.

**The whole loop on a public project (after wave 3, again after wave 4).** One person, one fresh project
of 200 COCO val images plus the e2e GPS photos, run in a browser on `DISPLAY=:11`, with screenshots of each
step at 1600 and 430 px kept under `artifacts_local/`. Every number on screen is compared with the API.

13. **Ingest and first look:** create the project, upload, wait for ingest; the Workspace opens on Gallery
    (served default); Gallery total equals `GET /images` total; Next steps names "create classes" then
    "cluster".
14. **Batch labelling:** seed classes from the detector, cluster and activate from the served banner; the
    default view becomes Clusters. Label five clusters: a pure cluster of 20 or more items takes **at most
    two keys** (class letter, `Enter`); a mixed cluster is split with facet chips; the labelled count on
    the Dashboard rises by exactly the served dry-run totals; Label history lists each batch and one undo
    restores it.
15. **Views agree:** set a filter (class *person*, To label queue), switch through Gallery, Clusters,
    Detections, Review, Table and Map; the filter chips, the queue and a 12-item selection stay; each
    view's served total agrees with its route for the same filter; the bulk bar's summary equals
    `POST /selection/summary`.
16. **Per-image annotation:** Gallery with "Has unlabelled objects", open an image; draw 3 missed boxes
    (one with a new class created inline from "Traffic light", accepted as `traffic_light`), move 2 boxes,
    delete 1 false positive, undo and redo each once, all by keyboard; reload: every change is there; the
    new items reach `embedding_state` "embedded" through the follow-up job; a second browser editing the
    same box gets the conflict message, not a lost write; a `detect` reprocess of the image leaves the
    drawn, moved and deleted boxes as the person left them.
17. **Map:** the header counts match `GET /map`; a rectangle drawn on the map adds an area chip and Gallery
    shows only those photos; a cell click shows its served summary and images; a pin opens the image in the
    editor.
18. **Dataset, train, repeat:** Table sorted by detector confidence shows the lowest first; export with
    "every object labelled" passes; a short training run (public preset, GPUs 0 or 2) completes and is
    promoted; reprocessing with the promoted model fills the Disagreements queue; working that queue in
    Review closes the loop. Next steps reflects each stage.

---

## 11. Keep (what is good)

- The cluster detail grid: served facets with served counts, served sort, bulk bar, drag to class, hotkeys,
  undo, Assist and the proposal panel.
- Matching items: served dry run, limit, sample and seed, apply to the filter as it was.
- The shared item filter: one definition, server-side, by class name, absent-not-disabled controls.
- Review's two panes and keyboard model, the multi-box editor and its OCC conflict handling.
- Served refusals shown verbatim; `formatCount` em dash for unknown counts; the stale-near-dup panel on
  Export.
- The map's server-side grid cells (no point shipping).
- Contract tests that pin every API call to `contracts/`.

## 12. Questions for the owner (small batch, answered with defaults if silent)

1. Name of the merged page: "Workspace" (default), "Review" or "Label"?
2. Should the Images view be the default Workspace view after ingest (default: yes when no partition serves,
   Clusters when one does)?
3. "Label this whole duplicate group" from a near-dup representative: in Wave 3 (default: later, P2)?

**Owner answers (2026-11-11):**
1. The merged page is named **Workspace**.
2. The default view is smart: Images until a partition serves, then Clusters.
3. **Waves 1 to 4 all ship before the speed review.** "Label this whole duplicate group" goes in with wave 4 polish.

**Owner feedback from a hands-on look (2026-11-11), binding for all waves:**
- **The goal is the loop:** ingest, sort, label, build a dataset, train a model, then repeat by improving the model
  (or by other methods). Every page must serve a step of that loop, and the next step must be obvious.
- **The project picker** (top left) lists every project in one long dropdown that runs off the page. It needs to
  scroll and be searchable (bug now; full project search stays in 4C).
- **The map** says "3 items" but shows no points. This is a bug; root-cause it.
- **Review overwhelms:** too many tabs and "types" of review items. Wave 3B must cut the queue set to a few
  plain-named queues (the rest become filters on the Detections query), not just move the 12 tabs into a rail.
- **Review pages do not show the proper object (class) name.** This is a bug; root-cause it (class identity is the
  name).
- **Label ops:** it is the history of bulk label operations (bulk relabel, merge, reclassify) with undo. It moves
  into Quality as "Label history", as planned.
- **Every lane checks its pages visually** against what the page is for, at desktop and 430 px.

**Questions from the Workspace revision (defaults apply if silent):**
4. Editor tool keys `V` (select), `B` (draw), `H` (pan) are reserved, so classes cannot use those letters
   (default: yes; the alternative is Shift+letter tools and no reserved letters).
5. Deleting a box in the editor is a soft delete (`not_an_object`, undoable, kept so a re-detect does not
   bring it back) rather than removing the item (default: soft).
6. Embedding view in wave 4 (default: yes, after the six core views).
