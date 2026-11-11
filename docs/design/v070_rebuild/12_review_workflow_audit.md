# Review workflow audit and consolidation plan (Cropwright and the curation API)

| Field | Value |
|---|---|
| Status | **plan only**: nothing implemented. Written 2026-11-11 on `int/b4-stateless` at `634019ded` |
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

Migration order, each step shippable on its own (section 9 has the lanes):

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

### R-11 CHANGED paging (Later, before 100k)

- `/crops`, `/review/{tab}`, `/regions`, `/images`: `cursor` (opaque: point-in-time id plus `search_after`
  sort values, with `crop_id`/`image_id` as tiebreak) and `next_cursor`; `/review/{tab}/locate` returns a
  cursor and an index. Remove `page` in the same change (one mechanism). The UI shows "N of total" from the
  served index, never a page number.

### R-12 NEW `GET {prefix}/next_steps`

- `steps: [{id, label, detail, done, section}]` decided by the backend from counts it has (no classes but
  detector proposals; embedded but no partition; partition pending activation; no hotkeys; near-dups never
  built; validated below the class floor). The Dashboard and an empty Workspace render it.

### R-13 Per-image label counts (Later)

- Store `counts` on the image doc, refreshed by a coalescing Celery task keyed by `image_id` after label,
  region and exclude writes (enqueued from the OCC write helpers), so `/images` can sort by
  `unlabeled_count` and filter `label_state` without a join at 100k+ images.

---

## 8. Frontend changes

All under `frontend/src`; components and controllers are reused, not rewritten.

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

### Wave 2 (now; after wave 1 merges)

**Lane 2A Images view and image page inside Clusters** (frontend, M-L; G-03, G-06, G-17 now part)
- New `ImageCard`, `ImageGrid`, `imageGalleryController`, image page; view switch on `/clusters`; crop
  dialog links; `label_state` filter chip.
- Tests: vitest for the controller (served order kept, cursor appended, no client sort; add a scan test
  like `clusterList.scan.test.ts` for the images list); stubbed `test_images_view.py` and
  `test_image_page.py` (image → crop → image round trip, 430 px), a11y rows; live tier: Images view total
  equals `GET /images` total on `coco-e2e`, the image page shows as many boxes as served.

**Lane 2B Sort options, area filter and map data** (backend, M; G-10, G-13 backend)
- R-4, R-9 (filter, `bounds`, `n_without_location`, delete `/map/images` and the map side panel's call:
  the map page keeps working by linking a cell to `/clusters?view=images&geo_bbox=` until 3B).
- Tests: pytest `geo_bbox` validation and filtering on `/crops`, `/images`, `/map/cells`; sort options
  served; live harness `test_live_capture_map.py` extended for `geo_bbox` on the e2e GPS photos.

### Wave 3 (now; the consolidation)

**Lane 3A Workspace replaces Clusters and Cluster detail** (frontend, L)
- Move `routes/p/[project]/clusters/+page.svelte` to `routes/p/[project]/workspace/+page.svelte`; cluster
  scope from `routes/p/[project]/clusters/[id]/+page.svelte` folds in as `?cluster=`; delete both old
  routes and their test files' routes (move the tests); update the 14 link sites
  (`rg -l "projectHref\(.*/(clusters|review|label-ops|audit|near-duplicates|map)"` lists them), the nav,
  `SECTION_LABELS`, `projectPaths.ts`; docs-site `clusters.md` → `workspace.md`.
- Tests: move and adapt `routes/p/[project]/clusters/**/*.test.ts`; stubbed cluster tests repointed;
  `projectRouting.test.ts` (old paths are 404 under the catch-all, no redirect); live route sweep.

**Lane 3B Review folds into Workspace (Queues + Focus), audit becomes a queue** (frontend + small backend, L)
- Backend: `audit` tab in R-10, delete `GET /audit/queue` and `AuditQueueList.svelte`.
- Frontend: Queues rail; Focus mode mounting `ReviewCropPane` and the `$lib/review` controllers with the
  queue or Detections query; deep link `?focus=<crop_id>` via `/review/{tab}/locate`; delete
  `routes/p/[project]/review/`; move its tests to `lib/review/` or the workspace; keymap scopes unchanged
  (`review.*`), new `workspace.*` actions in `keymapFallback.ts`.
- Tests: the 17 `routes/p/[project]/review/*.test.ts` move and still pass against Focus mode; stubbed
  `test_labeling_flow.py`, `test_region_confirm.py`, `test_keymap.py` repointed; a new stubbed test: grid →
  Enter → Focus → confirm 3 → Esc back to the same grid position; live tier deep link test.

**Lane 3C Quality page and Map view** (frontend, M)
- `routes/p/[project]/quality/{audit,near-duplicates,history}`; delete `/audit`, `/near-duplicates`,
  `/label-ops`, `/map` routes; Map view in Workspace (cell → Images with served `bounds`); Export stale
  panel and holdout freeze link to Quality › Near-duplicates; "Open in Workspace" from a group.
- Tests: stubbed `test_near_duplicates.py`, `test_label_ops.py`, `test_map_page.py` repointed; live
  `test_map_renders.py` repointed (needs a WebGL-capable browser; run headed on `DISPLAY=:11` if headless
  cannot start WebGL2).

**Lane 3D Next steps and first run** (backend + frontend, S-M; G-08)
- R-12; Dashboard card; empty Workspace shows it. Tests: pytest per state; stubbed fresh-project test.

### Wave 4 (later: scale and polish)

- **4A Cursor paging** (L): R-11 across `/crops`, `/review/{tab}`, `/regions`, `/images`; pagers in
  `lib/` switch to `next_cursor`. Live harness: a public bench project with more than 10,000 items reads
  past item 10,000 in Workspace and Focus.
- **4B Per-image counts** (L): R-13 and `unlabeled_count` sort.
- **4C Projects at scale** (M): G-21.
- **4D Polish** (S each): G-19, G-20, G-22, G-23, G-25, G-27, G-28.

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
10. Map (WebGL browser): a cell click opens the Images view with `geo_bbox` equal to the served bounds; the
    header shows located and not-located counts.
11. Full-page screenshots of every new or changed page at 1600 and 430 px, looked at, with no overlap,
    clipping, `NaN`, `undefined` or console errors; Cropwright's inner scroll area means the screenshot
    viewport must be tall (1600 x 2000) or the capture taken of the scroll container.
12. Scale (Wave 4): Workspace and Focus read past item 10,000 on a public bench project without a 422.

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
