# OpenProcessor — detector-class roots with a hold-back guard as the auto-label loop partition (issue #398, Z5 and Z5-I)

Status: **PLAN ONLY, not implemented.** Written 2026-11-04 against branch `int/b1-correctness` at
`81c4e4ef` (B1 job leases and the class registry document merged). Every `path:line` below is at
that commit. Nothing in the repository was changed by this planning pass.

Reader: an implementer or research agent with no memory of the planning session. This gist is the
execution-ready, per-issue version of RQ-1 / E-1 in
`docs/design/research_clustering_plan.md` (sections 2, 4.6, 5, 6.5, 7.1). Where this gist and
that plan differ, the difference is stated and argued here (section 9, "Corrections").

---

## 0. TL;DR — what I recommend, and what I am NOT deciding

**The work has two stages, and the second only happens if the first passes.**

1. **Z5 (research, issue result is a doc).** Re-run the Z4 comparison (detector-class roots, the
   same plus the label-free hold-back guard, IVF) on a new, pre-registered Open Images V7
   **train-split** set, sized so that the open-set bar B-c can actually be read for **both** key
   arms. Output: `docs/research/cluster_config_z5.md` plus a backing JSON folder under
   `docs/benchmarks/cluster_config_z5/` with `SHA256SUMS`. No product code.
2. **Z5-I (product, only on a Z5 PASS + visual check + the owner's yes).** Add a second
   implementation of the loop's existing `PartitionReader` seam
   (`src/services/curation/assist_loop/partition.py:108`) that makes **one root per detector class
   name**, and add the hold-back guard to the loop's propagate step
   (`src/services/curation/assist_loop/node_step.py:289`). No new Celery task: the loop already
   runs as its own job family (`op.assist_loop.run` after B2 track 2E). No new lock of any kind.

**Stop rule.** If Z5 is FAIL, or INCONCLUSIVE again, IVF stays, nothing ships, and #398 closes with
the evidence note. That is the registered rule from Z4 and the research plan (7.1 RQ-1); this plan
does not soften it.

**The headline correction** (section 9, C1): the issue and the research plan both say "at least
1,200 open-set crops". That number was derived from the **unguarded** `key` arm (19.9 open-set
items propagated per run on 395 held-out open-set crops). The guarded arm, which is the one the
evidence points at, propagated only **9.2 per run** on the same crops, so reading its B-c bar
(at least 30 per run) needs about **1,300 held-out, about 2,500 total** open-set crops at Z4's
open-set share, before any margin. A set of 1,200 would very likely end INCONCLUSIVE again for
`key_guard`, which under the stop rule would close the line for a sizing reason, not a method
reason.

**What I am NOT deciding** (section 10): the set size and composition (J1), whether B-c keeps
its "at least 30 propagated" readability floor or gains a pre-registered exposure-rate bar (J2),
whether `within_key` stays in the registration (J3), the root-key grammar on the wire (J4), the
default partition kind after adoption (J5), the pool definition (J6), a blast-radius cap on very
large roots (J7), whether held-back members get a review surface of their own (J8), detector-mix
handling (J9), and the judge for the visual check (J10).

---

## 1. Context — why the obvious approach is wrong here

- **"Just group by `detector_class_name` and call the loop" skips the only open question.** Z4
  already showed the effort and precision win (section 2.3). What it could not show is whether a
  large detector-class root that *passes* the gate hides unknown animals that then get the wrong
  class. The guard exists for that risk. The experiment must be sized to read the guard arm, not
  the unguarded arm.
- **Over-sampling open-set images biases the very risk being measured.** The risk is minority
  exposure: a few open-set crops hidden inside a big, mostly pure `dog` or `sheep` root. If the
  set is made open-set dense (cheaper to fetch), those crops stop being a minority, the roots
  fail the gate and split, the open-set crops land in small pure children, and B-c passes for
  the wrong reason. Section 4.2 keeps the open-set share near Z4's.
- **The loop already has the seam.** `PartitionReader` (`partition.py:108-117`) was designed for a
  second reader ("A versioned partition ... implements the same two calls, so the driver, the
  gate and the writer do not change", `partition.py:4-6`; `docs/design/auto_label_loop_plan.md`
  line 403, "#343 seam"). Z5-I adds a reader; it must not fork the driver.
- **Path grammar is integer-only today.** Node paths are `r<cluster_id>` (`src/domain/loop_tree.py:15`,
  `_PATH = re.compile(r'r(\d+)((?:\.\d+)*)')`), and the integer is used for ordering
  (`best_first_key`, `loop_tree.py:79`; `views._order`, `views.py:38`) and exposed on the wire
  (`LoopNode.path`, `src/schemas/curation/assist_loop.py:258`, and the members route
  `/curation/projects/{project}/assist/loops/{loop_id}/nodes/{node_path}/members`). A class name
  is not an integer. Mapping names to an ordinal would put a raw class index on the wire, which
  the project forbids (class identity by name). Section 6.2 and J4.
- **Key roots are far larger than IVF roots.** IVF roots are at most a few thousand crops
  (`K = min(512, n // 6)`). A `person` or `car` key in a 1M-crop project can hold 100,000+ crops.
  The current splitter and propagate read vectors with `mget` into Python lists
  (`splitter.fetch_embeddings`, `src/services/curation/assist_loop/splitter.py:47`), which is fine at 2,000 members and not at 100,000 (about 3 GB of Python floats for
  1024-d vectors). Z5-I must stream vectors into a float32 array. Section 6.4.
- **The research simulator is not the loop.** `scripts/research/cluster_diag/loop_sim.py` follows
  the loop's algorithm with its own seeded draws (its docstring, lines 17-19). The product guard
  must be tested for parity at the function level against the simulator's `held_back`
  (`loop_sim.py:150`), not by hoping a live run matches a simulated number.

---

## 2. What is actually there today (measured, not estimated)

### 2.1 The loop and its partition

| Fact | Where |
|---|---|
| Roots are candidate clusters only: `cluster_id >= RESIDUAL_CLUSTER_ID_OFFSET` (10000) | `partition.py:146-149`; offset `src/services/curation/cluster_ids.py:18` |
| `PartitionReader` protocol: `survey(scope)`, `roots(scope)`, `root_of(ids)` | `partition.py:108-117` |
| `ClusterFieldPartition` is the only implementation; the API (`prepare_loop`) and the worker (`run_job`) both construct it directly | `prepare.py:68`, `runtime.py:181` |
| `PartitionScope(cluster_ids, min_root_size)`; wire `LoopScope(cluster_ids, min_root_size=40)` | `partition.py:59-64`; `schemas/curation/assist_loop.py:15-24` |
| `RootRef.cluster_id: int`, `RootSize.cluster_id: int`; snapshot file stores `cluster_id` | `partition.py:66-105`, `write_snapshot` `partition.py:228-249` |
| The job state records `partition.kind = 'cluster_field'` (a kind field already exists) | `driver.py:126`; wire `LoopPartition.kind: str` `schemas/curation/assist_loop.py` (class `LoopPartition`) |
| Eligibility filter for the survey (not holdout, not excluded, not validated, not human) | `partition.py:42-53`; `member_state` `partition.py:120-136` |
| Start refuses `clustering_running` when an auto-label job is busy, because a retrain would move the clusters under the loop. `is_busy()` is a heartbeat-file mtime check, not a lease | `prepare.py:79-85`; `src/services/curation/autolabel/job.py:171` |
| Drift: before writing, `propagate_node` asks `reader.root_of(rest)` and treats members no longer in the node's root as drifted; more than `1 - SAME_SCOPE_JACCARD` drifted makes the node `stale` | `propagate.py:39`, `propagate.py:82-89` |
| The propagate step resolves the class through the registry document (B1) by name; a class not in the registry restates the node `stale` (`class_not_in_registry`) | `node_step.py:289-293` |
| Splitter: complete-linkage cosine AHC into `branching` children up to `MAX_REFINE_MEMBERS` (8000, `OP_MAX_REFINE_MEMBERS`), seeded k-means above it; at most `MAX_CONCURRENT_SPLITS = 2` at once | `splitter.py` `split_node`; `src/services/curation/clustering/refine.py:59`; `driver.py:76` |
| Gate: `top in registry and not vetoed and dissent <= max_dissent and answered >= need` | `src/services/curation/assist_loop/evaluate.py` `judge` (lines 98-136) |
| Skip reasons recorded per propagated node: `dissent, drifted, locked, token_changed, conflicts`; totals add `locked, holdout, excluded` | `propagate.py:37`; `progress.py:66-83` |
| The loop takes the `assist_loop` lease (B1) for its whole run | `driver.py:378-386` (`acquire_heartbeat(..., LeaseKind.ASSIST_LOOP, ...)`) |
| A generic lease liveness read exists: `is_live(client, kind, scope)` | `src/clients/job_lease.py:236` |

### 2.2 The detector key in the data

| Fact | Where |
|---|---|
| `detector_class_name` is a `keyword` field on items, written by NAME through `normalize_class_name` | mapping `src/clients/curation_opensearch/items_extra.py:24`; writer `src/services/curation/ingest_detect.py:251` (non-assigning proposer) and `:274` (assigning detector) |
| With the default ingest profile (`assigns_class` false) every detection is an unlabeled proposal that keeps the proposer's own label as `detector_class_name`; that is why the whole residual pool carries a detector name | `src/config/ingest_profiles.py:1-18`; `ingest_detect.py:242-256` |
| Confident classes keep an item out of the residual pool; the pool is what IVF clusters | `src/domain/ingest_class_sources.py:76-92`; `src/services/curation/clustering/residual_fetch.py:44-60` |
| The research simulator keys a crop by `normalize_class_name(detector_class_name) or 'none'` | `scripts/research/cluster_diag/dataset.py:50-54` |
| The residual vector field is `pe_embedding` (PE-Core-L14-336, 1024-d), overridable by `OP_RESIDUAL_EMBEDDING_FIELD` | `residual_fetch.py:44` |

### 2.3 What Z4 measured (quoted from `docs/research/cluster_config_z4.md`)

Held-out half, fold 1, 7,209 labelled crops of which 395 open-set, 20 seeds, clean oracle:

| Arm | U1 decisions/item | U2 | Open-set propagated per run (wrong) | Held back per run |
|---|---|---|---|---|
| `key` | 0.293 | 0.982 | 19.9 (9.05) | 0 |
| `key_guard` | 0.377 | 0.988 | 9.2 (0.25) | 606.7 |
| `within_key` | 0.526 | 0.990 | 8.0 (0.95) | 0 |
| `ivf_loop` | 0.834 | 0.979 | 83.8 (2.6) | 0 |

Verdict: INCONCLUSIVE (B-c unreadable: at least 30 open-set propagated per run reached in 2 of 20
seeds for `key`). IVF stays. The open-set quota of 1,200 images could not be met on the test
split (644 images, 758 labelled open-set crops, no tiger).

Derived figures this plan relies on (arithmetic on the table, not new measurements):

- Open-set share of labelled crops in Z4: 758 / 14,475 = **5.2 %**; labelled crops per image
  14,475 / 8,000 = **1.81**; open-set images 644 / 8,000 = **8.05 %** of the set.
- Open-set propagated per held-out open-set crop: `key` 19.9 / 395 = 0.050; `key_guard`
  9.2 / 395 = **0.0233**.
- Held-out open-set crops needed for a mean of 30 propagated per run, if that rate holds:
  `key` about 600; `key_guard` about **1,290**. With Z4's held-out/total ratio (395/758), that is
  about 1,150 and about **2,470** open-set crops in total.

### 2.4 The research tooling that exists (reuse, do not rewrite)

| Tool | Where | Notes |
|---|---|---|
| Loop simulator with the guard | `scripts/research/cluster_diag/loop_sim.py` (`GUARD_PERCENTILE = 95.0` at :56, `held_back` at :150) | 12 mutations killed by its tests (Z4 note, "Tests") |
| Z4 bars and adoption rule | `scripts/research/cluster_diag/z4_bars.py` (`U2_BAR`, `IVF_FACTOR`, `OPEN_BAR`, `MIN_OPEN_PROPAGATED`) | Z5 gets its own `z5_bars.py` only if the bars change (J2); otherwise it reuses `z4_bars.py` unchanged |
| Driver of one experiment | `scripts/research/cluster_diag/z4_run.py` | Z5 reuses it with new inputs; rename is not needed |
| Vector export, labels | `fetch_vectors.py`, `build_labels.py` (same folder) | |
| Split pin | `scripts/bench/eval_split.py make-pin` | |
| Fetcher | `scripts/datasets/fetch_vehicle_eval.py` (`--split` choices = keys of `BBOX_URL`, :184) with selection `scripts/datasets/vehicle_eval_selection.py` (`QUOTAS`, `TARGET_IMAGES` at :29-30) | **No train split**: `BBOX_URL` / `IMAGES_URL` (imported from `scripts/datasets/fetch_openimages_plates.py:74-81`) list only `validation` and `test` |
| Tests | `tests/research_scripts/test_loop_sim.py`, `tests/research_scripts/test_z4_bars.py`, `tests/test_vehicle_eval_selection.py` | |
| Contact sheets and judge | `src/services/labeling/contact_sheet.py`, `scripts/research/visual_judge.py`, `judge_*.py` | Face and plate sheets refuse a hosted model (`tests/test_contact_sheet.py`) |

### 2.5 Where the work sits in the program

- Research is outside the 0.7.0 program (`docs/design/program/00_index.md` section 10, lines
  356-358) and starts after the v0.7.0 tag. Its GPU work runs in research window RW-1 on an agreed
  GPU (GPU 2 or GPU 0; never GPU 1), announced to peer sessions, `-p op-rs-rw1`
  (`research_clustering_plan.md` 5 and 5.1).
- Writing and committing the registration (phase R0 / E-0) is docs and research scripts only and
  may happen any time (`research_clustering_plan.md` header).
- Z5-I is product work and lands after B2 track 2E, which turns the assist loop into a Celery
  consumer (`docs/design/program/03c_b2_w3_beat_reaper_workers.md` section 2, 2E-2). It also lands
  after B5c ("crops at model size") if the experiment ran on the B5 stack, because the vectors
  must be the vectors the product makes (section 7.4, trap T14).

---

## 3. Requirement R1 — a set that can read B-c for the guarded arm

✅ **Recommended: a train-split set at Z4's composition, scaled up about 4.4x.**

| Group | Z4 images | Z5 images (recommended) | Rule |
|---|---|---|---|
| `openset` | 644 | 2,850 | the image holds one of the 12 open-set animals (`OI_RARE`, `scripts/datasets/cluster_eval_selection.py:81-97`) |
| `animal` | 1,500 | 6,600 | largest box cat, dog, horse, sheep or cow, balanced |
| `vehicle` | 2,500 | 11,000 | largest box car, truck, bus, motorcycle, bicycle or train, balanced |
| `rest` | 3,356 | 14,750 | filler to the target, balanced over the largest-box class |
| **total** | 8,000 | **35,200** | |

Projected at Z4 rates (to be replaced by the fetcher's real counts before registration is
committed): about 63,700 labelled crops, about 3,300 open-set crops, about 1,700 in the held-out
half, so `key_guard` should propagate about 40 open-set items per run on average (above the floor
of 30 with a margin for the heavy per-seed tail Z4 saw: 0 to 72 for `key`).

Fetcher rule (registered): **count before downloading.** The fetcher reads the train boxes CSV,
applies the same filters as Z4 (CC BY 2.0 licence, no stored rotation, not group-of, not
depiction, area at least 0.5 % of the frame), counts qualifying open-set boxes and images per
class, and writes the counts to the registration **before** any image is downloaded. If the
projected held-out open-set count is below the registered minimum, the experiment is recorded as
INCONCLUSIVE by registration and stops (no partial run, no moved bar).

❌ **Rejected: the issue's "at least 1,200 open-set crops".** Readable for `key` only (about 600
held-out at a rate of 0.05 per crop is exactly the floor). The arm the evidence points at would
very likely be INCONCLUSIVE again (section 2.3).

❌ **Rejected: an open-set-dense set** (draw every open-set image the train split has and little
else). It changes the risk being measured (section 1, second bullet; trap T3).

❌ **Rejected: re-reading the Z4 set or adding train images to it.** Z4's registration names the
test split and its held-out half has been read; a repeat on train is a new registration (Z4 note,
"What follows from the registered rule").

🟡 **Phase 2 (optional, unbarred): an open-set-dense stress row** on the calibration half only, to
show how B-c moves with open-set share. Reported beside, never used by the adoption rule.

### 3.1 Train-split fetcher support (E-0 work)

Official locations (Open Images V7 download page,
<https://storage.googleapis.com/openimages/web/download_v7.html>, consulted 2026-11-04):

- boxes: `https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv`
- images with rotation and licence: `https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv`
- image files: the same `open-images-dataset/{split}/{image_id}.jpg` pattern the fetchers already use
  (`scripts/datasets/fetch_openimages_plates.py:82`, `IMAGE_DOWNLOAD_TMPL`), with `split=train`.

Changes:

1. Move `BBOX_URL`, `IMAGES_URL` and `IMAGE_DOWNLOAD_TMPL` out of
   `scripts/datasets/fetch_openimages_plates.py` (a plate fetcher is the wrong owner; three other
   fetchers import them) into a small `scripts/datasets/openimages_urls.py`, add a `train` entry,
   and update every importer in the same commit (`fetch_vehicle_eval.py:41-46`, `fetch_cluster_eval.py:47`,
   and any other `rg -n "from scripts.datasets.fetch_openimages_plates import"` hit). No re-export shim is kept
   in the plate fetcher.
2. The train boxes CSV is large (multi-GB). Parse it **streaming** (`csv.DictReader` over the
   opened file, keep only rows whose `LabelName` is in the wanted MIDs); never `read_text()` the
   whole file. `fetch_csv_text` (used today) reads the whole text, which is fine for validation and
   test but not for train: add a streaming reader and use it for every split, so there is one path.
3. **Check the header before using a column.** The train image CSV is a different file name and
   version from the 2018_04 validation/test files; if `License`, `Rotation` or `OriginalURL` is
   missing or renamed, the fetcher stops with the header it found (fail closed). The same check
   applies to the boxes CSV columns the selection reads (`IsGroupOf`, `IsDepiction`, `XMin`...).
4. Record the box `Source` column distribution (`xclick` manual, `activemil` machine-assisted and
   human-verified) of the selected boxes in the pin, so the note can state the label provenance.
5. `--exclude-manifest` for every earlier pin, **including** `openset_test_z4.json` and its split
   (file-name and SHA-256 disjointness, as Z4 did, written to `disjointness.json`).
6. Pin size: the Z4 pin is 3.8 MB for 8,000 images (`scripts/datasets/manifests/openset_test_z4.json`),
   so a 35,200-image pin is about 17 MB in git. J1 asks whether that is acceptable or whether the pin
   stores ids and hashes only (the selection is re-derivable from the official CSVs and the seed).

Red/green tests (E-0, no network): a fixture boxes CSV streamed in chunks gives the same selection
as the in-memory reader (red: the streaming reader drops the last partial chunk); a header missing
`License` raises (red: today the reader would `KeyError` deep inside, or silently skip); the train
URL is chosen for `--split train` (red: argparse refuses `train` today, `fetch_vehicle_eval.py:184`).

### 3.2 Class-name map (E-0, shared with other RQs)

Open-set is defined **by name**: an Open Images class whose normalized name is not in the
detector's label space. Commit the COCO-to-Open-Images name map as data (the research plan rule
"Class identity by name", section 0) and have `build_labels.py` and the selection read it, instead
of the hard-coded `DETECTABLE_ANIMALS` / `VEHICLES` / `OI_RARE` sets drifting apart. A test asserts
that no name is both "detectable" and "open-set" (red: add `sheep` to `OI_RARE` in a fixture).

---

## 4. Requirement R2 — the registration (written and committed before any score)

`docs/research/cluster_config_z5.md`, registration section only, committed before the set is
scored (the Z4 note's structure: "Registration (fixed before any Z5 score was computed)").

### 4.1 Contents

- **Set:** split (train), licence and filters, groups and quotas (section 3), seed (a new value,
  never one used by an earlier pin), the fetcher's pre-download counts, disjointness result.
- **Project:** a throwaway project on a research stack (`-p op-rs-rw1`), the API `/health` shown
  `ready` before ingest, the GPU id agreed with peers, the detector model and version from
  `class_detector` / `class_detector_version` on the items (section 2.2) and the encoder (`pe_embedding`
  model id). Any image the product's decoder rejects is listed and kept in the pin (Z4 precedent).
- **Split:** by source image with `eval_split.py make-pin`, a new seed, 50/50. Fold 1 chooses any
  cut on the calibration half; every bar is read on the held-out half, once.
- **Arms:** `key`, `key_guard`, `ivf_loop`, `ivf_gate` (comparator), and `within_key` only if J3
  keeps it. Definitions exactly as the Z4 registration's "Arms" table, with the simulator pinned to
  the loop commit it follows (record the sha of `src/services/curation/assist_loop/` at the time).
- **Simulator settings:** theta 0.9, alpha 0.05 (each look at alpha / 2), max dissent 1, branching 4,
  max depth 3, min node 40, guard percentile 95, 20 seeds (0-19).
- **Bars** (J2 decides between the two options):
  - Option A (Z4 unchanged): B-a U2 >= 0.95; B-b U1 <= 0.5 x IVF's (IVF = the cheaper of `ivf_loop`
    and `ivf_gate`); B-c open-set U2 >= 0.90, INCONCLUSIVE below a mean of 30 open-set propagated
    per run.
  - Option B (Z4 plus an exposure bar): as A, plus B-d: wrong open-set propagations per 1,000 pool
    items <= a registered limit with its bootstrap 95 % upper bound below the limit. B-d is readable
    whatever the guard suppresses, which removes the structural trap T2.
- **Adoption rule:** the cheaper (lower U1) of `key` and `key_guard` among the arms that PASS, ties
  to `key`; else IVF stays. (`within_key` is removed from the adoption order: it FAILED B-b in Z4
  and was killed, `docs/research/cluster_program_outcome.md` line 58. J3.)
- **Unbarred rows (reported, never used by the rule):** 10 % noisy labeler; registry-closed reading;
  per-key purity and open-set counts; held-back counts and how many would have been wrong; the
  open-set stress row (section 3, phase 2); per-class open-set exposure.
- **What this cannot show:** oracle labeler hides VLM errors; pool is labelled crops only; class
  level only (nothing about make and model or faces); train-split boxes are not exhaustively
  annotated for every class in an image (more unlabelled crops, never scored).

### 4.2 Why the bars stay where they are

The research plan (4.6 RQ-1) and the Z4 note fix them. Moving a bar after Z4's read would need a
new registration and a stated reason (research plan section 0, "Pre-register"). Option B **adds** a
bar; it does not move one. Either way the registration commit precedes the scoring commit and the
reviewer checks the order with `git log --follow docs/research/cluster_config_z5.md`.

---

## 5. Requirement R3 — run, score and visually check Z5 (E-1)

### 5.1 Steps

1. **Preflight** (`docs/research/CLAUDE.md` and the live-test memory rules): `uptime` (a 1-minute
   load above 16 at start marks the run noisy and it is repeated), ask the peer sessions for the
   GPU, `-p op-rs-rw1`, `/health` through the proxy, GPU 2 or GPU 0 only.
2. **Fetch** with the counts check (section 3). Images and CSVs go to a scratch directory outside
   the repo (never `/tmp` for anything that backs a number: move validated artefacts to
   `artifacts_local/`).
3. **Ingest** into a new throwaway project with `scripts/curation/ingest_upload.py --project <slug>
   --force-rehash` (Z4 "Reproduce"). Estimated 2 to 4 GPU hours on the B5 stack for 35,200 images
   (unmeasured; record the real wall time). Time box: 6 hours; past it, stop and report.
4. **Export** vectors read-only (`fetch_vectors.py`), build labels (`build_labels.py`, IoU 0.5).
5. **Fit IVF** for the comparator with the production configuration (`K = min(512, n // 6)`) on all
   vectors of the project, as Z4 did.
6. **Simulate** every arm, 20 seeds, write `z5.json` (raw) to `artifacts_local/cluster_search/z5/`.
   CPU only, capped at 8 threads and `nice`, in a CPU slot (research plan section 5). Key roots above
   8,000 members take the k-means split path (`MAX_REFINE = 8000`, `loop_sim.py`), as the loop does.
7. **Read the bars once**, apply the adoption rule, write the results sections of the note.
8. **Visual check** (section 5.2) only if some arm passes the machine bars. A machine PASS with a
   failed visual check is reported as "PASS, not adopted: visual check failed" and the line stops.
9. **Publish** the small JSON files that back the published numbers to
   `docs/benchmarks/cluster_config_z5/` with `SHA256SUMS`; never prune existing benchmark folders.
10. **Delete the throwaway project** through the project delete route with the confirm parameter;
    remove nothing from `docs/benchmarks/` or `artifacts_local/`.
11. Update #398 and the #346 tracker with the verdict and the numbers; add the note to
    `docs/research/README.md` and the research table in `docs/design/README.md`
    (`docs/research/CLAUDE.md`, "Adding a note").

### 5.2 Visual verification (owner rule: VLM check plus a human re-judge)

What it checks, and why the oracle does not make it redundant: the oracle trusts Open Images
labels. The sheets check (a) that what a passing key root propagates is visually that class (label
noise in the ground truth would flatter U2), and (b) that what the guard holds back is visually the
odd member out (pose, truncation, a different animal) rather than arbitrary.

- **Sheets:** the existing builder (`src/services/labeling/contact_sheet.py`): seeded member
  selection, letterboxed cells, shuffled order, manifest with hashes.
- **Sample** (owner decision O3, small subsets): at most 20 cluster sheets. Half targeted: the
  largest passing roots of the keys that hold open-set crops (`dog`, `cat`, `cow`, `horse`, `sheep`,
  `bird`, `bear` in Z4), and sheets of held-back members. Half a seeded uniform control over all
  passing nodes. Targeted sheets are never pooled into the purity estimate.
- **Judge agreement first** (research plan 4.4 step 1): the hosted judge passed on COCO crops (pair
  kappa 0.821, flag precision 0.827) but **failed on Open Images vehicles** (0.611, 0.454). Open
  Images animals have never been scored. Before its verdicts count, score the judge on 40 planted
  sheets from the **calibration** half with known ground truth (half clean, half contaminated:
  the balanced-sheet lesson, research plan 1.3 item 8). Bars: pair kappa >= 0.7 and flag precision
  >= 0.8. Below them the judge only ranks sheets for a person (J10).
- **Controls:** about 10 % planted known-same and known-different sheets; a batch whose controls
  fail is discarded; 10 % re-rendered in a second cell order to measure position bias.
- **Human re-judge:** a blind, seeded 5 % of verdicts, at least 20, by the owner or a delegate.
  Judge-vs-human kappa below 0.7 suspends the judge for this kind.
- **Reporting:** judge purity with a Wilson 95 % interval (`statsmodels.stats.proportion.proportion_confint`,
  method `wilson`) beside the ground-truth U2, never instead of it. A half-width above 0.10 is
  reported as "insufficient sample", not a pass.
- **Privacy:** Open Images animal crops are public, non-face, non-plate: a hosted judge is allowed.
  Any sheet the builder flags as containing faces or plates goes to the local VLM or a person only.

### 5.3 Stop rule and time box

| Outcome | Then |
|---|---|
| Fetcher counts below the registered minimum | INCONCLUSIVE by registration; close #398 with the counts; IVF stays |
| Any arm FAILS and no arm PASSES | FAIL; close #398 with the note; IVF stays |
| No arm FAILS, some arm INCONCLUSIVE, none PASSES | INCONCLUSIVE again; **the line stops** (research plan 7.1) and #398 closes |
| An arm PASSES, the visual check fails or is insufficient | not adopted; close #398 with the note |
| An arm PASSES and the visual check passes | owner decision on Z5-I (section 6) |

Time box for Z5 as a whole: one RW-1 slot (6 GPU hours) plus two CPU days. Overrun is reported, not
silently extended.

---

## 6. Requirement R4 — Z5-I: the loop on detector-class roots (only on a PASS)

### 6.1 Shape

- A new reader `DetectorClassPartition` in `src/services/curation/assist_loop/partition.py` (or a
  sibling module `partition_detector.py` if the file-size guide asks; same package), implementing
  `survey`, `roots`, `root_of`.
- A partition **kind** on the request: `LoopScope.partition: Literal['clusters', 'detector_class']`
  plus `LoopScope.detector_class_names: list[str] | None` (by name, normalized with
  `normalize_class_name` at validation; `cluster_ids` and `detector_class_names` are mutually
  exclusive and each only valid with its kind, 422 otherwise). The worker builds the reader from the
  stored request (`runtime.py:181`) and the API from the same factory (`prepare.py:68`): one
  function `partition_reader(opensearch, scope) -> PartitionReader`, no second construction site.
- `job.partition.kind` records `'detector_class'` (`driver.py:126` already writes a kind).
- The hold-back guard as a step inside `_propagate` (`node_step.py:289`), gated by
  `LoopGate.hold_back: bool`. Default per J5.
- No new Celery task, no new lease, no in-process or file lock. The loop's existing `assist_loop`
  lease (`driver.py:378`) and job directory carry everything.

❌ **Rejected: writing key roots into `cluster_id`.** The loop never writes cluster fields
(`auto_label_loop_plan.md` D2, line 51). Class clusters already own `0..9999`; an encoding of names
into cluster ids is a class-index-on-the-wire by another name.

❌ **Rejected: a separate "key loop" driver.** The seam exists so that the driver, gate and writer
do not change (`partition.py:4-6`).

### 6.2 Root keys and node paths (J4)

✅ **Recommended:** a root key `k<12 lowercase hex>` where the hex is the first 12 characters of
`sha256(normalized class name)`, with the name stored in the root snapshot and served on every node
(`LoopNode.root_name`, `LoopNode.root_kind`). Grammar becomes
`(r\d+|k[0-9a-f]{12})((?:\.\d+)*)`; `parse_path` returns the root key as a string plus the indices.

- The hash is derived from the name, so identity stays the name; no ordinal is invented; the path is
  URL-safe whatever the class name contains.
- Collisions: 48 bits over at most a few hundred detector names; still, `survey` refuses (fail
  closed, 409 `partition_key_collision`) if two names in one survey hash to the same key.
- Ordering: `best_first_key` and `views._order` sort ties by the root key string instead of the
  integer. Existing `r<int>` ordering must not change for cluster roots: compare a `(kind, int or
  str)` tuple so `r10002` still sorts before `r10010` (trap T8).
- Wire: `LoopNode.path` description changes; `root_kind` and `root_name` are new fields; run
  `make contracts` and commit `contracts/` in the same commit; update `frontend/src/lib/types_assistLoop.ts`
  and the node list component to show `root_name` for key roots.

❌ **Rejected: the class name itself in the path** (`kdog.2`). Names can hold spaces, slashes,
dots and non-ASCII; the dot is the child separator.

❌ **Rejected: an ordinal (`k3`).** A raw index that means nothing outside one job directory, served
on the wire.

### 6.3 `DetectorClassPartition` semantics

| Call | Behaviour |
|---|---|
| `survey(scope)` | Composite aggregation over `detector_class_name` with `missing_bucket: true` (the `none` key, J6) on the pool query, with the same `_ELIGIBLE` sub-aggregation (`partition.py:42-53`). Pool = J6 (recommended: exactly the IVF loop's pool, `cluster_id >= RESIDUAL_CLUSTER_ID_OFFSET`). Roots under `min_root_size` dropped, as today |
| `roots(scope)` | Snapshot per key: `scan_items` with `{'term': {'detector_class_name': name}}` (or `must_not exists` for `none`) intersected with the pool query; members sorted by id; `version = membership_digest(ids)`; the snapshot stores `kind`, `root_name`, and no `cluster_id` |
| `root_of(ids)` | `mget` `detector_class_name` (plus the pool fields) and return the key each id has **now**; an id that left the pool or whose detector name changed (a re-detect) is absent or under another key, so `propagate_node` counts it as drifted (`propagate.py:82-83`) |

Snapshot format: `write_snapshot` / `read_snapshot` (`partition.py:228-271`) gain `kind` and
`root_name` and lose the hard `cluster_id` requirement for key roots. An old snapshot without
`kind` reads as `clusters` (it is the only kind that existed): this is reading existing job
directories correctly, not a compatibility shim, and a test pins it. A snapshot with an unknown
kind is an error (fail closed).

Start refusal (`prepare.py:79-85`): `clustering_running` stays for the `clusters` kind (an IVF
retrain moves clusters). For `detector_class` the moving part is the detector name, which a
re-detect rewrites: refuse with 409 `reprocess_running` while the `reprocess` lease is live
(`is_live(client, LeaseKind.REPROCESS, None)`, `src/clients/job_lease.py:236`; the reprocess family
takes that lease with scope `None`, `src/services/curation/reprocess_job.py:139-146`). This is a courtesy check; the
fail-closed backstop is the drift check at write time (a reprocess that starts after the loop makes
its nodes `stale`, never mislabelled).

### 6.4 Vectors at key-root scale

- Replace `splitter.fetch_embeddings` (`mget` into a dict of Python lists) with a paged reader that
  fills a preallocated `numpy.float32` array (search_after pages, `_source` limited to the vector
  field, or `docvalue_fields` if the vector is not in `_source`; check the items mapping first).
  The splitter and the guard both use it. One reader, not two.
- Keep `MAX_CONCURRENT_SPLITS = 2` (`driver.py:76`); the guard's vector read runs under the same
  semaphore (`LoopRun.split_slots`), so at most two large vector sets are in memory per worker.
- Measure peak RSS on a 100,000-member synthetic root (runtime and memory may use synthetic vectors;
  quality may not: `scripts/bench/scorecard.py` already refuses quality fields on synthetic data).
  Bar: under 2 GB per root at 1024-d.

### 6.5 The hold-back guard in the product

Algorithm (must equal `loop_sim.held_back`, `loop_sim.py:150-160`):

1. `sampled` = the node's members with an answer on record (reused from ancestors plus its own),
   exactly the set the simulator calls "sampled" (`loop_sim.py` `sampled_of`). Decide whether
   failed answers count (the simulator has none): **they do not**; a member with no usable answer
   is not evidence (trap T6).
2. centroid = mean of the sampled unit vectors, renormalized; zero norm => hold back nothing and
   log `guard: degenerate` (the simulator returns empty; keep parity, but record it).
3. limit = `numpy.percentile(1 - V_sampled @ c, 95)` with the default linear interpolation
   (NumPy docs, `numpy.percentile`, `method='linear'`).
4. held back = unasked eligible members with `1 - v @ c > limit`. A member **without a vector**
   is held back (fail closed; the simulator never has one, so parity tests use full vectors).

Placement: in `_propagate` (`node_step.py:289`), after the registry check and before
`propagate_begin`. Log a `hold_back` event **before** `propagate_begin` with the held-back ids,
the limit and the sample size. On resume, read the ids from the log; never recompute (vectors can
change between sessions after a re-embed; trap T7). Pass `eligible - held_back` to `propagate_node`.

Accounting: add `held_back` to `SKIP_REASONS` (`propagate.py:37`) so `totals` and the level rows
count it (`progress.py:72-76`); add `items_held_back` to the totals. Held-back members are never
written, so rollback (`rollback.py` reads `written_ids`) is unaffected; a test pins that.

Review surface (J8): held-back members must be findable. Recommended: the node members route
gains `set=held_back` (default `all`), reading the ids from the `hold_back` event; Cropwright shows
a "Held back for review" count on the node and opens that set.

Constant: `GUARD_PERCENTILE = 95.0` as a module constant with a comment naming the Z4/Z5 notes. Not
a request field: no other value has been measured, and an option no one can calibrate is a trap.

### 6.6 Plan (dry run) and docs

- `build_plan` (`plan.py:23`) works unchanged on `RootSize.eligible`; add `root_name` per row of a
  new `roots_preview` (largest 25) so a user sees "dog 12,400 eligible" before starting.
- Docs: `docs/CURATION.md` (loop section), `docs-site/docs/guides/auto-label-loop.mdx`,
  `docs/design/curation_api_contract.md` (scope fields, node fields, `set` parameter),
  `docs/ARCHITECTURE.md` if it names the partition, CHANGELOG entry. Routes named in docs must
  exist (`scripts/docs/check_docs_vs_code.py`).

---

## 7. How we PROVE it — verification plan

### 7.1 Z5 (research)

| Claim | Proof | Red looks like |
|---|---|---|
| The fetcher supports train and streams | unit tests in section 3.1 | `--split train` refused; a chunk boundary drops a row |
| Bars fixed before scores | `git log` shows the registration commit before the results commit; `z5.json` stamps the registration sha | results commit precedes or edits the registration |
| B-c is readable for `key_guard` | `z5.json` `open_set_propagated.mean >= 30` for the arm (or B-d readable under option B) | INCONCLUSIVE with a mean below 30 |
| Simulator unchanged since Z4 or changes disclosed | `tests/research_scripts/test_loop_sim.py` and `test_z4_bars.py` green; any simulator change carries a mutation test killed before scoring | a simulator edit after registration with no disclosure |
| Data kept | `docs/benchmarks/cluster_config_z5/SHA256SUMS` verifies; no file under `docs/benchmarks/` deleted (`git diff --stat` on the folder shows additions only) | a pruned or rewritten benchmark file |
| Visual check honest | judge agreement table on planted sheets, Wilson intervals, the 5 % human re-judge kappa | a pass reported with a half-width above 0.10 |

### 7.2 Z5-I (product), static

- **Whole-enum test:** every `LoopScope.partition` value has a reader (`partition_reader` raises for
  an unknown kind; a test iterates the `Literal` args with `typing.get_args`). Red: add a value to
  the Literal without a reader.
- **Grammar test:** `parse_path` accepts `r10042`, `r10042.2.0`, `k0123456789ab`, `k0123456789ab.3`;
  refuses `k0123`, `kDOG`, `r`, `k0123456789ab.`, `r1.k0123456789ab`. Red: today `k...` is refused
  (`loop_tree.py:15`).
- **Contracts:** `make contracts-check` clean after `make contracts` in the same commit.

### 7.3 Z5-I, dynamic (offline suite, fakes)

| Test | Proves | Red today / red on a mutation |
|---|---|---|
| `tests/curation/test_assist_loop_partition_detector.py::test_survey_one_root_per_name` | survey groups by name, `none` bucket included per J6, `min_root_size` applied | no reader exists |
| `...::test_root_of_detects_redetect` | a member whose `detector_class_name` changed is absent from its old key; propagate marks drift | mutate `root_of` to return the snapshot key |
| `...::test_snapshot_kind_roundtrip` and `test_old_snapshot_reads_as_clusters` and `test_unknown_kind_refused` | snapshot format | |
| `tests/curation/test_assist_loop_guard.py::test_parity_with_simulator` | `held_back_ids` equals `loop_sim.held_back` on 200 random seeded unit-vector nodes (sizes 40 to 5,000, sample sizes 20 to 60) | mutate percentile to 50, invert the comparison, include failed answers in the sample |
| `...::test_no_vector_is_held_back` | fail closed on a missing vector | mutate to skip |
| `...::test_hold_back_logged_before_propagate_begin` and `test_resume_reads_ids_not_vectors` | resume determinism | change a vector between sessions: the held-back set must not change |
| `...::test_held_back_never_written_never_rolled_back` | accounting and rollback | |
| `tests/curation/test_assist_loop_prepare.py::test_reprocess_running_refused_for_detector_kind` and `test_clustering_running_only_for_clusters_kind` | kind-specific refusals | |
| `tests/curation/test_assist_loop_vectors.py::test_paged_reader_float32` | vectors are read into one float32 array; peak allocation bounded | |
| `tests/curation/cross_project_leak/test_assist_loop_isolation.py` (extend) | the detector reader only touches the bound project's items index (OpenSearch guard) | construct a raw client in the reader |
| Lock rule (extend `test_assist_loop_lock_rule.py`) | a human-validated, import-validated or holdout member in a key root is never written | |

Every test is watched failing first; the PR description lists each red run.

### 7.4 Z5-I, empirical (one run, not per commit)

- Inside the next live harness or research window (never a new GPU window of the program): one loop
  on a throwaway project ingested from the Z5 **calibration** half (never the held-out half), with
  the real VLM, `partition='detector_class'`, `hold_back=true`, a small budget. Check: decisions
  per item and propagated precision within the Z5 confidence band of the simulator for the same
  arm; held-back share within 0.5x to 2x of the simulated share; no lock-rule violation
  (`rg`-able in the event log: zero writes to locked ids). This is a plausibility check of the
  integration, not a new acceptance read.
- Cropwright: open the loop panel in a browser, full-page screenshots at desktop and 430 px, key
  roots show their class name, the held-back set opens (`DISPLAY=:11` for headed runs).

---

## 8. Edge cases and traps (the failure register)

| # | Trap | How it silently looks done | Guard |
|---|---|---|---|
| T1 | Sizing from the wrong arm | Z5 runs on 1,200 open-set crops, `key` reads, `key_guard` is INCONCLUSIVE, the stop rule closes the line | size from `key_guard`'s rate (section 3) |
| T2 | A working guard makes its own bar unreadable | the better the guard, the fewer open-set items propagate, the more likely "fewer than 30" | count check before download; option B adds a bar that is readable regardless (J2) |
| T3 | Open-set-dense set | B-c passes because open-set crops form their own pure children | keep Z4's composition; dense set only as an unbarred stress row |
| T4 | Reading the held-out half twice | a "quick look" before registration | the split pin is created after the registration commit; the results JSON stamps the registration sha |
| T5 | Train split not exhaustively annotated | many more unlabelled crops; U1 comparisons drift if the pool definition changes | pool = labelled crops of the held-out half, as Z4; the unlabelled count is reported |
| T6 | Failed VLM answers in the guard sample | product and simulator disagree only on outage nodes; parity tests with no failures pass | exclude failed answers by rule; a parity test with failures injected |
| T7 | Recomputing the guard on resume | a re-embed between sessions changes the held-back set; the log and the writes disagree | log ids before `propagate_begin`; resume reads the log |
| T8 | Path ordering change | `r10010` starts sorting before `r10002`, or key roots interleave with cluster roots oddly; existing order tests still pass if they only use one digit count | ordering test with mixed lengths and both kinds |
| T9 | Ordinal keys | `k3` on the wire is a class index | hash-of-name keys (J4) |
| T10 | Two construction sites | the API surveys with one reader, the worker runs another (`prepare.py:68` vs `runtime.py:181`) | one `partition_reader` factory; a test that both call it |
| T11 | `none` key quietly dropped or quietly huge | composite aggregation without `missing_bucket` drops items with no detector name; with it, a heterogeneous `none` root may be the largest | J6 decides; the survey reports the `none` count either way |
| T12 | Memory at key scale | tests use 50-member roots; production `person` root OOMs the worker | paged float32 reader and a 100k synthetic memory test |
| T13 | Courtesy check mistaken for safety | the `reprocess_running` refusal is a check-then-act; a reprocess starting after it is not blocked | drift at write time is the real guard; test a re-detect after start => `stale` |
| T14 | Vectors of another pipeline | Z5 measured on vectors from one preprocessing, product runs on B5c "crops at model size" | run Z5 on the merged B5 stack, or run a 1,000-crop parity check (cosine >= 0.999) before trusting the numbers (research plan 7.2) |
| T15 | Registry-closed projects | the barred reading assumes the registry names the open-set animals; in a project whose registry lacks them, every open-set item propagated is wrong by construction | report the registry-closed row; the guard's held-back count is the protection there |
| T16 | Detector mix | a project ingested by two detectors with overlapping names merges them into one root silently | the survey reports the `class_detector` set per key; J9 |
| T17 | Gate blast radius | one pass of a 100,000-member root writes 100,000 labels from about 35 answers; average precision bars hide the tail | the audit (`audit.py`) and rollback exist; J7 asks about a cap |
| T18 | Benchmark folder edits | an "update" rewrites `docs/benchmarks/cluster_eval/` | new folder only; `SHA256SUMS`; diff shows additions only |
| T19 | Private data | the owner's set used for a published number | owner-run option only, aggregates only, never named (`docs/research/CLAUDE.md`) |

---

## 9. Corrections to the issue's premises

| # | Issue or plan says | What is true | Consequence |
|---|---|---|---|
| C1 | "at least 1,200 open-set crops" (issue comment of 2026-11-02; research plan RQ-1 and 4.1) | 1,200 total reads B-c for `key` only. `key_guard` propagated 9.2 per run on 395 held-out open-set crops; reading it needs about 1,290 held-out, about 2,470 total, at Z4's share | size the set from `key_guard` (section 3, J1) |
| C2 | "cut decisions per item ~2.2x vs IVF" (issue body) | 2.2x is the guarded arm (0.834 / 0.377). The unguarded arm is 2.85x (0.834 / 0.293). The guard costs 29 % more effort than `key` | state both; the adoption rule picks the cheaper passing arm |
| C3 | "members beyond the 95th percentile distance" (issue body) | the percentile is of the **node's sampled members'** cosine distances to **their own** centroid, per node, label-free (`loop_sim.py:150-160`); not a project-wide or per-class percentile | the product implementation follows the simulator exactly (section 6.5) |
| C4 | "See `docs/design/improvement_backlog.md` (branch `docs/v070-user-docs`)" | that file was deleted in the R-1 removal wave; its rows live in `docs/design/program/12_prior_plans_reconciliation.md` section 2.14 and the research plan (12_prior_plans_reconciliation.md:794). `docs/handoff/0.7.0-next-steps.md` (cited in the triage comment) is removed too | read the research plan and the Z4 note instead |
| C5 | Milestone | the issue sits on v0.8.0 (matching `docs/design/program/09_github_actions.md:642`); the triage comment says v0.10.0; `12_prior_plans_reconciliation.md:226,349,424` says v0.9.0 | one place should win; the coordinator aligns the docs and the comment (no plan impact) |
| C6 | "the fetcher needs train-split support" | true: `--split` choices are the keys of `BBOX_URL`, which hold only validation and test (`fetch_openimages_plates.py:74-81`, `fetch_vehicle_eval.py:184`). The URL tables also live in the wrong module | section 3.1 |
| C7 | (implicit) the loop roots on "clusters" generically | roots are **candidate** clusters only (`cluster_id >= 10000`, `partition.py:147`); class clusters are never roots | J6: the key reader's pool should be the same residual pool for a like-for-like swap |
| C8 | (implicit) the loop's start refusal guards the partition | the refusal reads a heartbeat file (`autolabel/job.py:171`), not a lease, and only covers IVF retrain; nothing covers a re-detect | kind-specific refusal plus drift at write time (section 6.3) |

---

## 10. Judgment calls (J1 … J10)

| J | Question | Recommendation | Trade-off |
|---|---|---|---|
| J1 | Set size and composition | about 35,200 train images at Z4's composition (section 3), registered minimum 1,300 held-out open-set crops; pin stores ids and hashes only if 17 MB is too much in git | about 4 to 5x Z4's ingest (2 to 4 GPU hours) and about 1 to 2 CPU hours of simulation; the alternative (1,200 total) is cheap but likely ends the line on sizing |
| J2 | Bars: Z4 unchanged (A) or Z4 plus an exposure-rate bar B-d (B) | **A**, with the size of J1, because the research plan fixes "Z4 bars unchanged"; choose **B** only if the owner prefers a bar the guard cannot make unreadable. B-d's limit must be set before scoring (suggestion for the owner to weigh: at most 1 wrong open-set label per 1,000 pool items, upper bound) | A keeps comparability with Z4; B removes trap T2 but adds a registered number with no prior evidence behind its value |
| J3 | Keep `within_key` in Z5 | drop it from the adoption order; keep it as an unbarred row only if CPU time allows | it was killed in Z4 (effort bar); re-including it as adoptable reopens killed research |
| J4 | Root-key grammar | `k<12 hex of sha256(name)>` plus `root_name` on the wire | longer paths; avoids names in URLs and ordinals |
| J5 | Default after adoption | `partition='detector_class'` and `hold_back=true` become the defaults; `clusters` stays a selectable kind (both are tested readers, not a shim) | users with tuned IVF habits see a change; flipping defaults is the point of adoption |
| J6 | Pool and the `none` key | pool = exactly the IVF loop's pool (`cluster_id >= 10000`); include the `none` key as Z4 did | same items as today, only regrouped; a project that never ran clustering has no roots (fails closed with `no_roots`); an alternative pool ("all residual items, clustered or not") would need its own measurement |
| J7 | Cap on very large roots | no cap in the first release (it matches the measured arm); require the audit to be configured when a root above 20,000 eligible members would be visited, or warn in the plan | a cap changes the method away from what Z5 measured; no cap leaves the tail to the audit and rollback |
| J8 | Review surface for held-back members | `set=held_back` on the node members route plus a Cropwright count and link | a wire change and a small UI change; without it held-back items are only counted, which makes them easy to forget |
| J9 | Detector mix | key by normalized name only (class identity by name), report the detector set per key in the survey and the plan | two detectors that name the same class differently stay separate roots, which costs effort but never mislabels |
| J10 | Judge for the visual check | the hosted judge only after it passes the planted-sheet agreement on Open Images animals; otherwise the local VLM ranks and a person decides | the hosted judge failed on Open Images vehicles; the local 4B VLM was killed as an authority (pair accuracy 0.51) |

Precondition on J5 and J7 (not a finding, a requirement on unwritten code): any new route
parameter (`set=held_back`) and the larger write volume per node need the same project-isolation
and lock-rule review every curation write path gets before merge.

---

## 11. File-by-file change list

### 11.1 Z5 (research; docs and research scripts only)

| File | Change |
|---|---|
| `scripts/datasets/openimages_urls.py` (new) | `BBOX_URL`, `IMAGES_URL`, `IMAGE_DOWNLOAD_TMPL`, `CLASS_DESC_URL` for validation, test and train |
| `scripts/datasets/fetch_openimages_plates.py` | import the URL tables from the new module; delete its own copies |
| `scripts/datasets/fetch_vehicle_eval.py` | train split; streaming CSV reader; header check; pre-download counts written before any image fetch |
| `scripts/datasets/fetch_cluster_eval.py` | `fetch_csv_text` callers move to the streaming reader (one path) |
| `scripts/datasets/vehicle_eval_selection.py` | quotas and target from arguments (already partly), record box `Source` distribution |
| `scripts/datasets/manifests/class_name_map_coco_oi.json` (new) | name map, data only |
| `scripts/datasets/manifests/openset_train_z5.json`, `splits/openset_train_z5.json` (new) | the pin and split |
| `scripts/research/cluster_diag/build_labels.py`, `z4_run.py` | read the name map; accept the Z5 inputs; stamp the registration sha |
| `scripts/research/cluster_diag/z5_bars.py` (new, only under J2 option B) | B-d; otherwise reuse `z4_bars.py` unchanged |
| `tests/test_vehicle_eval_selection.py`, `tests/research_scripts/test_openimages_stream.py` (new), `tests/research_scripts/test_z4_bars.py` (extend for B-d if chosen) | red/green as in sections 3.1 and 7.1 |
| `docs/research/cluster_config_z5.md` (new) | registration first, results later |
| `docs/benchmarks/cluster_config_z5/` (new) | backing JSON + `SHA256SUMS` |
| `docs/research/README.md`, `docs/design/README.md`, `docs/research/cluster_program_outcome.md` (append a dated line) | index rows; outcome |

### 11.2 Z5-I (product; only on PASS and the owner's yes)

| File | Change |
|---|---|
| `src/domain/loop_tree.py` | path grammar `(r\d+|k[0-9a-f]{12})`; `root_key_for_name(name)`; `parse_path` returns `(root_key: str, indices)`; `best_first_key` tie-break on `(kind, int or str)` |
| `src/services/curation/assist_loop/partition.py` (+ `partition_detector.py` if split) | `DetectorClassPartition`; snapshot `kind` and `root_name`; `partition_reader` factory; `RootRef.cluster_id` becomes optional or moves into a kind-specific field |
| `src/services/curation/assist_loop/guard.py` (new) | `held_back_ids(vectors, member_ids, sampled_ids) -> list[str]`, `GUARD_PERCENTILE = 95.0` |
| `src/services/curation/assist_loop/vectors.py` (new) | paged float32 vector reader; `splitter.fetch_embeddings` deleted and its callers moved |
| `src/services/curation/assist_loop/node_step.py` | guard step in `_propagate`, `hold_back` event; `root_key=` from the parsed string key (`node_step.py:303`) |
| `src/services/curation/assist_loop/propagate.py` | `held_back` in `SKIP_REASONS` |
| `src/services/curation/assist_loop/store.py` | replay `hold_back` events into `NodeRecord.held_back` |
| `src/services/curation/assist_loop/progress.py` | `items_held_back` |
| `src/services/curation/assist_loop/prepare.py`, `runtime.py` | the factory; kind-specific refusals |
| `src/services/curation/assist_loop/plan.py` | `roots_preview` with names |
| `src/services/curation/assist_loop/views.py` | `_order` on string keys; `set=held_back` in `member_page`; `root_kind`, `root_name` |
| `src/services/curation/assist_loop/driver.py` | `partition.kind` from the reader |
| `src/schemas/curation/assist_loop.py` | `LoopScope.partition`, `detector_class_names`, validation; `LoopGate.hold_back`; `LoopNode.root_kind`, `root_name`; path description |
| `src/routers/curation/assist_loop.py` | `set` query parameter on the members route |
| `contracts/` | regenerated in the same commit |
| `frontend/src/lib/types_assistLoop.ts`, `frontend/src/lib/components/assistLoop/*`, `frontend/src/lib/assistLoop/assistLoopController.svelte.ts` and their tests | partition selector (or none, per J5), root names, held-back set |
| `scripts/research/cluster_diag/loop_sim.py` | none; the product tests import it for parity (tests may import research scripts; production code may not) |
| tests | section 7.3 |
| `docs/CURATION.md`, `docs-site/docs/guides/auto-label-loop.mdx`, `docs/design/curation_api_contract.md`, `docs/design/auto_label_loop_plan.md` (status line), `CHANGELOG.md` | as section 6.6 |

---

## 12. Suggested phasing

| Phase | Content | Gate | Where |
|---|---|---|---|
| P0 (E-0) | fetcher train support, streaming reader, name map, tests | offline tests green | any time; docs and research scripts only, one small PR or a lane of the research branch |
| P1 | registration committed (J1, J2, J3 decided) | owner's yes on the registration text | any time |
| P2 (E-1) | fetch (counts first), ingest, export, simulate, score, write results | registered rule applied | RW-1 (GPU) plus a CPU slot, after the v0.7.0 tag |
| P3 | visual check | judge agreement, 5 % human re-judge | RW-1 or API budget |
| P4 | decision | owner reads the note | |
| P5 (Z5-I) | product change as one feature branch, one PR, waves per the house lane model (one worktree per lane, targeted tests per lane, one full sweep on the merged branch) | full offline sweep, contracts-check, frontend set, browser check | after B2 2E (and after B5c if T14 applies) |
| P6 | one empirical loop run on a calibration-half project | section 7.4 | the next live window, never a new one |

If P2 ends FAIL or INCONCLUSIVE, P3 to P6 are not run, and #398 closes with the note.

---

## 13. Libraries and documented patterns consulted

| Need | Library / document | What it says that this plan relies on |
|---|---|---|
| Percentile with interpolation | NumPy `numpy.percentile` (<https://numpy.org/doc/stable/reference/generated/numpy.percentile.html>) | default `method='linear'`; the simulator uses the default, so the product does too |
| AHC and k-means split | scikit-learn `AgglomerativeClustering`, `KMeans` (<https://scikit-learn.org/stable/modules/clustering.html>) | already used by `refine_members` and `splitter._kmeans_labels`; nothing new |
| Wilson interval | statsmodels `proportion_confint(method='wilson')` (<https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html>) | judge purity reporting |
| Bootstrap interval for B-d | SciPy `scipy.stats.bootstrap` (<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html>) | upper bound of the exposure rate (option B only) |
| Composite aggregation with missing values | OpenSearch composite aggregation `missing_bucket` (<https://docs.opensearch.org/latest/aggregations/bucket/composite/>) | the `none` key bucket |
| Open Images train files | Open Images V7 download page (<https://storage.googleapis.com/openimages/web/download_v7.html>) | train box and image CSV locations; about 1.7M train images with boxes (vendor-reported) |
| Constraint handling | research plan section 3.2 last paragraph | the guard is pre/post-processing around a library split, not a new clustering method; no maintained library offers it |

No custom clustering code is added: the guard is a 10-line NumPy function mirrored from the
simulator, and everything else reuses the loop's existing split and gate.
