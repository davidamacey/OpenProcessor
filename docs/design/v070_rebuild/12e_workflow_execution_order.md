# 12e Workflow build: execution order, merges, sweeps and the acceptance walk

Specs: [12a interfaces](12a_workflow_interfaces.md), [12b wave 2](12b_wave2_lane_specs.md),
[12c wave 3](12c_wave3_lane_specs.md), [12d wave 4](12d_wave4_lane_specs.md). Plan:
[12_review_workflow_audit.md](12_review_workflow_audit.md).

## Waves and lanes

| Wave | Lane | Size | Can code in parallel with | Merge order | Shared files (resolve at merge) |
|---|---|---|---|---|---|
| 1 | wf-1a, wf-1c (wf-1b, wf-1d merged at `1685723a9`) | | each other | 1a, 1c | 12a §14 C-1 (`HiddenCounts`) |
| 2 | 2B Shared filter and map data | M-L 4d | all of wave 2 | **1** | `item_filter.py`, `crops_list.py`, `image_view.py` |
| 2 | 2E Cursor paging and crop sorts | L 4d | all | **2** (rebase on 2B) | `crops_list.py`, `review.py`, `image_view.py`, stubbed tests |
| 2 | 2C View models and served boxes | L 5d | all | **3** (rebase on 2B, 2E) | `image_boxes.py`, `images_view.py`, `SourceImageOverlay.svelte` |
| 2 | 2D Annotation writes (high risk, independent review) | L 5d | all | **4** (rebase on 2B, 2C) | `wire.py`, `image_boxes.py`, `item_filter.py`, `selection_summary.py` |
| 2 | 2A Workspace shell | L 5d | all (never edits `fetchPage` or the pager) | **5** (rebase on all) | keymap JSON and fallback, nav, stubbed tests |
| 3 | 3A Gallery | M 3d | all of wave 3 | **1** | `viewRegistry.ts`, `detailsRegistry.ts` |
| 3 | 3B Review view and queues | L 4d | all | **2** | rail, keymap, nav, `workflow_steps.py` |
| 3 | 3F Table | M 3d | all | **3** | registry, keymap |
| 3 | 3E Quality and next steps | M 3d | all | **4** (after 3B: the audit queue) | nav, `projectPaths.ts`, sidebars |
| 3 | 3D Map view | M 3d | all | **5** (after 3A) | registry, nav, e2e conftests |
| 3 | 3C Image editor | L 6d | all | **6** (after 3A) | keymap, `api_images.ts`, `projectPaths.ts` |
| 4 | 4A Per-image counts | L 4d | 4B, 4C, 4D, 4E | 1 | `image_view.py` |
| 4 | 4B Projects at scale | M 2d | all | any | none |
| 4 | 4C Embedding view | M 3d | all | any | registry, `ClustersGrid.svelte` |
| 4 | 4D Polish | S each, about 4d | all (one commit per item) | last | various; one item at a time |
| 4 | 4E Live fakes | M 3d | all | any | `docker/test/` only |

No wave-2 or wave-3 lane is file-disjoint from all others. Parallel coding is safe because every lane codes
against the names in 12a and **rebases onto the integration branch after the lanes before it merge**, then reruns
its targeted tests before hand-back. Lanes never merge or push.

## Coordinator merge and sweep points

1. Wave 1: merge wf-1a, wf-1c `--no-ff`; fix C-1 at the merge if trivial; one full sweep (offline pytest with `-n 24`,
   db tier, vitest, `npm run check`, `npm run build`, stubbed e2e, `--all-files` pre-commit and pre-push, docs check);
   tag the wave.
2. Wave 2: merge in the order above, one at a time, each `--no-ff`. After each merge: `make contracts-check` and the
   merging lane's targeted tests only. After 2A: the one full sweep plus the live Cropwright read-only tier on
   `coco-e2e` and the live harness. 2D's independent review report must be in before 2D merges.
3. Wave 3: same pattern; after 3C, one full sweep, the live read-only tier, and the live write tier on a throwaway
   project (`frontend/e2e/live/test_writes.py`, `-m writes`). Then the acceptance walk below (after wave 3).
4. Wave 4: same; then the acceptance walk again, with check 12 (scale).
5. While one wave's sweep runs, the next wave's lanes may start coding against 12a.

## Live acceptance walk (plan §10, checks 1 to 18)

On `coco-e2e` read-only for reads (checks 1 to 12); on a fresh public project `wf-accept-<date>` for writes
(checks 13 to 18). Browser on display `:11`, Chromium launched with `--use-gl=angle --use-angle=gl-egl` for the
map. Screens at 1600 x 2000 and 430 x 900 of every step, saved under `artifacts_local/screens/workflow_accept/`, and
looked at. Every number on screen is compared with the API (`curl -s localhost:4603/curation/projects/<slug>/...`).

1. Dashboard: the "Labeled by source" total equals `GET /stats/dataset` `labeled.total` (285 on `coco-e2e`), and it has an Imported row of 185.
2. On the fresh project after "Cluster items": the Workspace Clusters view shows the pending-partition banner. Activate there; cards appear.
3. The rail's class counts equal `GET /classes` `count` (car 74, person 133 on `coco-e2e`).
4. Gallery total equals `GET /images` `total` (91 minus hidden, with the hidden note); summing `counts.items` over the pages equals the served items count.
5. Gallery, then an image in the editor, then a crop, then the Review view, then `Esc`: the image is unchanged. A sibling in the crop dialog opens that sibling. The editor's boxes equal `GET /images/{id}` `boxes`.
6. Quality, Near-duplicates: make a decision, close the dialog and reload. The decision is listed and can be undone. A member with no crops shows its picture.
7. Network log for the session: no request to `/crops/{id}/context`, `/crops/{id}/image`, `/map/images` or `/audit/queue`. `/p/<slug>/clusters`, `/review`, `/map`, `/audit`, `/near-duplicates` and `/label-ops` show the app's 404.
8. Fresh project Dashboard: the served next steps appear in order, and each link lands on its step.
9. Review view on the To label queue: `Enter` confirms, a class letter assigns, `N` skips, `Z` undoes, and `Esc` returns to the grid at the same card.
10. Map: a cell's "Show in Gallery" sets `geo_bbox` to the served bounds, and the header shows the located and not-located counts.
11. Every screen viewed: no overlap, clipping, `NaN`, `undefined` or console errors.
12. (After wave 4.) On a public bench project with more than 10,000 items, Detections, Review and Table read past item 10,000 without a 422.
13. Create `wf-accept-<date>` and upload 200 COCO val images plus the e2e GPS photos. The Workspace opens on Gallery, its total equals `GET /images`, and Next steps names "Create classes" then "Cluster".
14. Seed classes from the detector, cluster, and activate from the banner. The default view becomes Clusters. Label five clusters: a pure cluster of 20 or more takes at most two keys (letter, `Enter`), and a mixed one is split with facet chips. The Dashboard labelled count rises by exactly the dry-run totals. Quality, Label history lists each batch, and one undo restores it.
15. Set class `person` and the To label queue, select 12 items, then switch through Gallery, Clusters, Detections, Review, Table, Map (and Embedding after wave 4). Chips, queue and selection stay. Each view's total equals its route for the same filter, and the bulk bar text equals `POST /selection/summary`.
16. Gallery with "Has unlabelled objects", open an image. Draw 3 boxes (one with "Traffic light", accepted as `traffic_light`), move 2, delete 1, and undo and redo each once, all by keyboard; then reload, and everything is there. New items reach `embedding_state` embedded. A second browser editing the same box gets the conflict message. `POST /images/{image_id}/reprocess` with scope `detect` leaves the drawn, moved and deleted boxes as left.
17. Map: the header counts match `GET /map`. A `Shift`+drag rectangle adds an area chip and Gallery shows only those photos. A cell click shows its served summary, and a pin opens the image in the editor.
18. Table sorted by detector confidence shows the lowest first. Export with "every object labelled" passes. A short public-preset training run on GPU 0 or 2 completes and is promoted. Reprocess with the promoted model fills Disagreements, and working that queue closes the loop. Next steps reflects each stage.

19. The user's job on a real vehicle photo set (a fresh project on an NVMe copy of the private set P sample; its
    screenshots and data stay out of the repository). Ingest gives every box the detector's class. Embed, cluster and
    activate. A stand-in user (Claude Haiku through the Anthropic provider) looks at each cluster's representative
    crops and labels make and model through the same UI and API actions a person uses, part of them driven in the
    browser: label a whole cluster with a new class created by name, move or exclude outliers, fix boxes, review.
    Then run the assist loop with Haiku as the VLM on the remaining clusters and correct its output. A human label
    always wins over the detector class. Export, train on the make-and-model classes on GPU 0, promote, serve, and
    detect with the new model. Any step that fails or is awkward gets a root-cause fix and the step is repeated.

Note (2026-11-12): since the UX wave, ingest assigns the detector's class itself, so on a fresh project "Create
classes" (checks 13 and 14) is satisfied at ingest; on older items it is the "Adopt detector classes" step.

Afterwards, delete `wf-accept-<date>` through `DELETE /curation/projects/{project}` and confirm the 404.
