# UI0: #380, #381, #382 and the axe gate

Branch `feat/v070-cluster-ui`, merge base `0e2d3fe7`. Issues: #380, #381, #382. Head: the commit that adds this file.

## What changed

- 7abd2ace fix(frontend): stop probing a combine job id on every /projects load (#382)
- 7bb86352 fix(frontend): stop nesting controls in crop cards and label empty table headers, add an axe e2e check (#380, axe gate)
- 09a5dfbf fix(frontend): re-read the cluster page after a bulk assign, hide empty Proposed, one timestamp format (#381)

## Acceptance

| Bar | Measured | Met | Raw data |
|---|---|---|---|
| #382 no probe of `projects/combine/__probe__`, zero console errors on `/projects` | vitest asserts zero probe requests; stubbed e2e `test_projects_page_offers_combine_without_a_probe`; live `/projects` 0 console errors, 0 4xx/5xx | yes | `frontend/artifacts_local/UI0/live/report.json` |
| #380 no nested interactive, no `role` on a button, labelled empty `<th>` | axe: 0 violations on 8 pages (stub) and 11 live pages | yes | same |
| #381 footer count, placeholder, `Proposed` row, one timestamp format | see below | partly (footer not reproduced live) | |
| axe gate committed | `frontend/e2e/stubbed/test_a11y_axe.py`, axe-core devDependency injected from `node_modules` | yes | |
| check, lint, knip, build, vitest, e2e | all green; vitest 468 files / 4889 tests; e2e 305 passed | yes | |

axe before (stub, same test, old code): cluster detail `nested-interactive` (3 cards) and `aria-allowed-role` (class sidebar button given `listitem` by the dnd action); settings `empty-table-header`; the clusters page `aria-allowed-role`. After: 0 on projects, clusters, cluster detail, review, classes, settings, prompt-packs, open-vocab.

## How it was tested

Red first (seen failing, then green): `projectsPage.test.ts` "offers Combine projects without probing a combine job id" (failed against the old page); `CropCard.selection.test.ts` (4 of 5 failed); `tableHeaders.scan.test.ts` (8 empty headers listed); `clusterController.assign.test.ts` "re-reads the served first page after a successful assign"; `ReviewItemMeta.proposed.test.ts` "is absent when there is no proposal"; `DatasetStats.test.ts` timestamp test and `sourcesFormat.test.ts`; `SemanticSearchBox.placeholder.test.ts`; `test_a11y_axe.py` failed on cluster detail, clusters and settings before the fixes.

Live: build with `PUBLIC_TRITON_API_URL` set to the optest1 API (`http://127.0.0.1:5603`, /health checked first, 38 ms), served by `vite preview` on port 5820 (stopped afterwards). Pages read on `cocotest` read-only; a throwaway `ui0walk` project was made by a combine job from `cocotest` (public data) for the combine job page, then deleted (it is gone from the project list). No GPU used.

Screenshots (all opened and viewed unless marked), under `frontend/artifacts_local/UI0/live/`: `d_projects`, `d_cluster_detail`, `m_cluster_detail`, `d_clusters_top`, `m_review`, `d_combine_job`, `m_combine_job`. Also captured and checked only through the axe, console and overflow report: `{d,m}_combine`, `{d,m}_clusters`, `{d,m}_classes`, `{d,m}_settings`, `{d,m}_settings_prompt_packs`, `{d,m}_settings_open_vocab`, `{d,m}_dashboard`, `{d,m}_export`, `m_projects`, `d_review`, `{d,m}_walk_dashboard`. The 1600 px `d_clusters` full-height shot is 17853 px tall and was not viewable at size; its top was captured as `d_clusters_top`.

## Risks and what was not done

- #382 decision: the plan allowed a boolean on the global health payload, but the combine router is mounted unconditionally (`src/routers/curation/_mounting.py`), so a flag would always be true. The entry point is always offered and the gate and its "not available" states were deleted. No backend change, no `make contracts`.
- Crop selection is now the thumbnail button only (one select button); a click on the card text no longer selects. Drag-and-drop is unaffected.
- #381 footer: root cause traced in code (a later page load returns the shrunken served total while the locally relabeled crops stay listed), fixed by re-reading the page after a successful assign. Not reproduced live: the combined project has no clusters, and creating them needs a recluster job on the owner's GPU stack, so it was not run.
- Thumbnails on `cocotest` return 404 on this stack (source images missing under `/data/source/samples`), so live cards show empty tiles; card layout was checked with the empty tiles and in the stub.
- `m_settings_prompt_packs` shows horizontal overflow at 430 px (the other 21 shots do not). Not investigated; not part of #380 to #382.
- The stub default 404 for the combine probe in `e2e/harness/stub.py` is still registered (harmless, job reads use that route); the host was heavily loaded (load average above 100 at times), which made the first two full e2e runs fail on click timeouts that passed when rerun.

## Decisions needed from the owner

None.

## Five-minute check for the owner

`cd frontend && npx vitest run && node scripts/run-e2e.mjs e2e/stubbed/test_a11y_axe.py` (8 passed). Open `/projects` with the browser console open: no 404. Open a cluster: each card has one select button (Tab reaches it, Space toggles).

## Backend requests

None.
