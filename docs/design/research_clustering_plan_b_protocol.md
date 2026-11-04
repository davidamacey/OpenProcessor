# Clustering research, part b: prototype protocol, experiments and the visual judge

| Field | Value |
|---|---|
| Superseded parts | the landing lanes, waves and file ownership of this plan's section 8, its GPU lock and its GPU-hour estimates are superseded by the program (review RR-16, owner answers of 2026-11-04): lanes in [program/03b](program/03d_stage4_job_families.md) (W2.3a and W2.3b), [program/03d](program/03b_stage2_control_data.md), [program/03c](program/03f_stage6_scheduler_scaleout.md), [program/05](program/05_b4_stateless_api.md) and [program/03g](program/06e_w5e_plate_text.md); one lock `gpu<N>.lock` per card shared by every topic ([program/00b](program/00b_traceability.md) X-14); short tiered runs under a cap of about 10 GPU hours before the tag ([program/00f_research_track.md](program/00f_research_track.md) section 2.1). A fresh agent loads those files first |
| Status | plan, research. Nothing here has run. Written 2026-11-04 against `origin/main` at `497315954` |
| Part a | [research_clustering_plan.md](research_clustering_plan.md): summary, measured numbers, literature, candidate methods, the rebuild seams, production design, lanes, owner questions (J1-J17) |
| What this part holds | how every clustering and judge claim is measured: code location, venvs and pins, data and sampling, metrics and bars, the judge and the blind human re-judge in full (absorbing the former #406 plan), experiments CL-0 to CL-9, GPU etiquette, windows, stop rules, where results go, and the failure register (absorbing the former #346 plan) |
| When | R0 (CL-0, docs and CPU scripts) any time; every other experiment after the v0.7.0 tag (MQ-9) |
| Names | experiments `CL-n`, windows `CRW-n` (the GPU-pipeline research uses `GP-`), so ledgers and peer announcements cannot collide |

## 1. Rules

| Rule | Meaning here |
|---|---|
| Real data decides | quality claims come from real photos only. Public numbers use COCO, Open Images, CUB-200, SOP, LFW family. Synthetic vectors may carry runtime and memory only; `scripts/bench/scorecard.py` refuses quality fields on a dataset with `synthetic: true` |
| Visual check on every quality claim | every method that passes its machine bars is checked on contact sheets by a calibrated judge and a blind human re-judge (section 5) |
| Faces included | at identity level, local judge and person only |
| Pre-register | bars, splits, seeds, sample sizes and stop rules are committed in a docs commit before any score is computed; a bar moved after a read needs a new registration and a written reason |
| Cut unproven research | each line has a stop rule (section 7); a failed line is parked on #346 with its evidence and the product keeps its method |
| Libraries first | scikit-learn, FAISS, cuVS, cuML, cuGraph, leidenalg, SciPy, statsmodels, ppi_py; custom code only for glue (graph construction from library kNN, carry-over bookkeeping, sheet aliasing), with the reason in the script header |
| Class identity by name | ground truth, roots, open-set membership and cross-dataset maps use class names; a COCO-to-Open-Images name map is committed as data |
| Keep benchmark data | `docs/benchmarks/cluster_eval/`, `retrieval_baseline/`, `wb2_standard_benchmarks/`, `cluster_config_baseline_3000.json` and `artifacts_local/bench/cluster_search/w8/` are read, never changed or pruned; new results go to new folders |
| No private names | set P is "set P"; no paths, folder or file names, owner or organisation names, no private-repo content |

## 2. Code location, venvs and pins

### 2.1 Where the code lives

**Recommended: tracked `scripts/research/clustering/`**, next to the existing tracked research code
(`scripts/research/cluster_diag/`, `judge_*.py`, `leiden_refine_proto.py`, `gpu_methods.py`,
`scripts/bench/cluster_eval*.py`), with tests in `tests/research_scripts/` and `tests/research/`.

| Option | Verdict | Why |
|---|---|---|
| Tracked `scripts/research/clustering/` with exact pins | ✅ | [docs/research/CLAUDE.md](../research/CLAUDE.md) requires a reproduce command from a clean checkout for every result; the consolidated note's section-4 numbers rest on untracked scratch scripts (its section 9 item 10) and cannot be quoted in a paper until landed; tracked code is linted (ruff, file size, complexity) and its metric helpers get red/green tests; the existing harness is already tracked and is reused, not copied |
| Gitignored scratch directory | ❌ | unreproducible, unreviewed, lost with the machine |
| A separate repository | ❌ | splits the harness from the product code it imports (`src/services/labeling/contact_sheet.py`, `src/clients/pe_encoder.py`) |

Layout:

```text
scripts/research/clustering/
  README.md                 # one paragraph per script; reproduce commands live in the research notes
  requirements-cpu.txt      # exact pins for the named libraries
  requirements-cpu.lock.txt # pip freeze of the venv the R0 commit was tested with
  requirements-gpu.txt
  requirements-gpu.lock.txt
  bundle.py                 # vector bundle I/O: one embedding_field + preprocess_id per bundle, refuses mixes
  graph.py                  # mutual kNN from FAISS / cuVS output (wraps graph_lib.py, no second copy)
  methods.py                # arms as thin calls into scikit-learn, leidenalg, cuGraph, cuML
  carry_over.py             # Jaccard + linear_sum_assignment id carry-over, MONIC transitions
  incremental.py            # prototype assignment with isotonic bands (sklearn IsotonicRegression)
  run_cl.py                 # one CLI: run <experiment> <arm> --bundle ... --seed ... --out ...
  faces.py, viewpoint.py, images.py
scripts/research/judge_human.py   # blind human re-judge (section 5.3)
```

Research code may import `src` read-only helpers (the sheet builder, the PE encoder, the metric
modules); nothing in `src/` imports it (the layering ratchet).

### 2.2 Venvs

Two plain venvs outside the repository, created with `python3.12 -m venv`, never the product
`.venv` (its `faiss-cpu` must not meet `faiss-gpu`) and never a Docker build:

| Venv | Contents (exact pins for these; the rest from the lock file) | Used by |
|---|---|---|
| `$SCRATCH/venv-cluster-cpu` | `numpy`, `scipy`, `scikit-learn==1.9.1`, `faiss-cpu==1.15.1`, `igraph==1.0.0`, `leidenalg==0.12.0`, `statsmodels`, `ppi-python==0.2.3`, `pillow`, `httpx`, `pytest`; plus `requirements-test.txt` for the repo imports | every CPU arm, scoring, judge tooling |
| `$SCRATCH/venv-cluster-gpu` | the CPU set plus `cuvs-cu13==26.10.0`, `cuml-cu13==26.10.0`, `cugraph-cu13==26.10.0`, the matching `cupy-cuda13x`, and `faiss-gpu-cu12` (no cu13 wheel exists on PyPI; it runs on the CUDA 13 driver) | GPU kNN graphs, ECG, cuML HDBSCAN |

Versions were read on PyPI on 2026-11-04; R0 re-reads them, installs, runs the research tests and
commits the lock files. Every run header records `pip freeze` digest, git sha, driver and CUDA
versions. The existing unpinned `scripts/research/requirements.txt` gets the same pins in R0.

### 2.3 Vectors

Vectors come from the product's encoders so the numbers describe the product:

1. **Preferred:** export from a throwaway project on the running stack (existing images, own
   `-p op-rs-crw<n>` name), read-only point-in-time export (`scripts/research/fetch_project_vectors.py`,
   `scripts/bench/export_project_embeddings.py`).
2. **When no stack is up:** encode in the GPU venv through `PEEncoder.encode_images`
   (`src/clients/pe_encoder.py`) only if a 1,000-crop parity check against stack vectors gives
   cosine >= 0.999 on every row; else the bundle is refused.

Every bundle records `embedding_field`, `embedding_model`, `preprocess_id`, the pin manifest hash and
the B5 parity status. Bundles of different spaces or preprocessing variants are never mixed; the
scorer refuses a bundle with more than one.

## 3. Data and sampling

### 3.1 Sets

| Tier | Set | Size | Truth | Used by |
|---|---|---|---|---|
| Public | 3,000-item pin `cluster_eval_3000.json` | 944 images, 3,961 crops, 2,950 labelled, 86 classes | COCO and Open Images boxes at IoU >= 0.5 | regression row only (held-out half worn by three waves) |
| Public | Y1 pin `cluster_eval_fresh.json`; Z2 and Z4 Open Images sets | 3,355 crops; 33,580 crops; 7,209 labelled held out | Open Images boxes | regression rows |
| Public | **new fresh acceptance set F1**: COCO val2017 (5,000 images, about 36,800 instances) plus an Open Images V7 validation sample (split holds 41,620 images), disjoint from every earlier pin by file and content hash | about 10,000 images | human boxes | CL-2, CL-5 acceptance |
| Public | COCO train2017 | 118,287 images, about 860,000 instances | human boxes | CL-4 scale with quality; CL-9 agreement gate |
| Public | **new box set B1**: the public car-wheel example (COCO cars, wheel regions from the region stage) plus Open Images V7 part classes (for example Wheel, Tire) matched at IoU >= 0.5 | at least 5,000 boxes | Open Images part boxes | CL-3 |
| Public | `coco_bench_2000.json` | 2,000 images, 5 captions each | COCO categories | CL-8 |
| Public | CUB-200-2011, SOP | 11,788 / 120,053 images | author labels | CL-2 secondary rows (fine-grained and instance level) |
| Local only | CARS196 | 16,185 images | author labels | CL-2 secondary row (at the ceiling, read only as a sanity row) |
| Public, local images | LFW, CALFW, CPLFW | 13,233 / 12,174 / 11,652 images | identity labels | CL-6 quality (aggregates published only if J10) |
| Public | Open Images V7 "Human face" boxes (pin of 1,500 images, plus up to 100,000 train images) | as stated | no identities: chaining judged by a person on samples | CL-6 scale and chaining |
| Local only | CO3D v2 (car, motorcycle, bicycle), Objectron (bike, shoe) | sequences with camera poses | identity = sequence; view = azimuth | CL-7, only on J11 |
| Public vectors | big-ann-benchmarks real descriptor sets (Deep1B 96-d and Yandex Text-to-Image 200-d, 1M and 10M slices) | 1M, 10M | none | CL-4 and CL-9 time and memory only, never quality |
| Owner-run | set P | see 3.3 | none (judge + person) | CL-2, CL-5, CL-6 repeats of the winners; private |

Every new set gets a pin manifest in `scripts/datasets/manifests/` (counts, licence, `manifest_sha256`,
`disjointness.json` against every earlier pin) and a row in the #346 "real-world dataset testing"
checklist.

### 3.2 Splits

- By image, never by crop; seeded; 50 % dev, 50 % test; calibration (cuts, gamma, bands, floors) on
  dev only; the test half read once per method family.
- Near-duplicate groups (frame cosine >= 0.98) are held out together (owner answer 2026-11-04).
- Open-set splits are class-disjoint; face splits are identity-disjoint.
- A set whose test half has been read by two method families is retired for acceptance and kept as a
  regression row.
- Bootstrap intervals resample images (1,000 resamples), identities for faces, sheets for the judge.

### 3.3 Set P

Set P has no ground truth, so its rows are label-free metrics plus judged purity:

| Sample | Draw | Use |
|---|---|---|
| S-P1 | the existing set P project's 91,184 crop vectors, read-only point-in-time export | stability (permutation ARI, id stability after a 10 % addition), cluster-size profile, run cost, judged purity on a seeded sheet sample |
| S-P2 | 20,000 crops, stratified by detector class (at most 2,000 per class), all crops of a drawn image together, seed registered | judged purity per arm on the CL-2 finalists |
| S-P3 | faces detected on a seeded 50,000-photo sample, local only | CL-6 chaining and incremental assignment, judged by the local VLM (ranking) and the owner |

Set P vectors, sheets, verdicts and numbers stay in `artifacts_local/research/clustering/setp/`
(gitignored). Committed text says only "run on set P", or, with the owner's yes (J7), one PASS/FAIL
line. Set P sheets go to the local VLM and the owner only; a hosted judge sees set P only for a
sample the owner names (J8).

## 4. Metrics, harness and bars

### 4.1 Metrics

| Group | Metrics | Notes |
|---|---|---|
| Partition | BCubed P, R, F (headline); pairwise P, R, F; V-measure (homogeneity, completeness); purity and inverse purity; AMI; false-merge and false-split rates | strata: low resolution (< 32 px), elongated (> 2:1), rare, similar classes, person, open-set, view bin |
| Abstention | unassigned and review fractions, risk-coverage curve | BCubed counts an abstained item as a singleton |
| Stability | permutation ARI on identical vectors (must be 1.000), seed ARI, id stability (share of parent clusters keeping their uid), protected moves (must be 0) | |
| Effort | loop U1 and U2 from the loop simulator (oracle and 10 % noisy labeller rows side by side) | through the auto-label plan's simulator |
| Faces | persons per identity, identities per person, chaining incidents (persons holding more than one identity), incremental accept precision at coverage, names and merges kept across a re-run | identity labels only |
| GPU | kNN graph recall@k against exact, ARI and purity delta of GPU-fed partitions against CPU | |
| Judge | raw agreement, Cohen's kappa, PABAK, flag precision and recall, false-"same" and false-"different" rates, judge-vs-human agreement, PPI purity interval | section 5.4 |
| Cost | wall, peak RSS and VRAM, `uptime` at start and end | a run starting at 1-minute load > 16 is noisy and repeated |

### 4.2 Harness (reuse; R0 adds only what is missing)

Reuse: `scripts/bench/cluster_eval.py` and `cluster_eval_metrics.py` (pairwise, BCubed, AMI,
risk-coverage, protected moves), `scripts/bench/scorecard.py`, `scripts/research/cluster_diag/`,
`graph_lib.py`, `leiden_refine_proto.py`, `gpu_methods.py`, `bench_driver.py`, `move_audit.py`,
`knn_recall.py`, the judge scripts and `scripts/bench/retrieval_eval*.py`.

R0 adds, each with a hand-computed unit test watched failing on a wrong formula first:

| Helper | Library | Red case |
|---|---|---|
| `v_measure` in `cluster_eval_metrics.py` | `sklearn.metrics.homogeneity_completeness_v_measure` | swapping homogeneity and completeness fails the asymmetric fixture |
| `purity`, `inverse_purity` | numpy | a fixture where purity and inverse purity differ |
| `pabak`, `raw_agreement` in `judge_stats.py` | formula 2p - 1 | the all-"different" fixture: kappa 0.0, PABAK and agreement non-zero |
| `ppi_purity_ci` | `ppi_py.ppi_mean_ci` | a fixture with a biased judge: the PPI interval covers the true mean, the judge-only mean does not |
| `carry_over` | `scipy.optimize.linear_sum_assignment` | permuted identical input keeps every uid; a split records `split` with two predecessors |
| `bundle` refusal | - | a bundle with two `embedding_field` values raises |

### 4.3 Bars (registered in R0, read on the test half of a fresh set)

| Line | Bar (all must pass) |
|---|---|
| CL-2 items (class) | the corrected acceptance A1-A7 of [cluster_config_decision.md](../research/cluster_config_decision.md) section 6: BCubed P >= 0.90; non-person P >= 0.90; low-res P >= 0.85; BCubed R >= 2 x IVF's; pair R >= 3 x IVF's; no stratum with >= 100 crops below IVF by more than 0.02; permutation ARI 1.000. Plus: loop U1 not worse than IVF's; judged purity (PPI interval) not below the ground-truth BCubed P by more than 0.05 (the judge agrees with the labels); on S-P2 the judged purity interval's lower bound >= 0.85 |
| CL-3 boxes | the CL-2 bars scored per box against today's MiniBatchKMeans |
| CL-4 scale | the CL-2 bars on COCO train2017 test half; 860k items within 2 h wall on CPU, or on a validated GPU path; peak RSS within the worker's limit |
| CL-5 stability | protected moves 0; id stability >= 0.95 after a 10 % addition; 100 % of anchors, names and constraints kept; permutation ARI 1.000; incremental accept precision >= 0.95 at the registered coverage (class level) |
| CL-6 faces | BCubed P >= 0.97 on LFW + CPLFW identities with >= 2 images; chaining incidents fewer than union-find at matched pairwise recall; incremental accept precision >= 0.99; corrections kept 100 %; memory O(n k) (peak RSS at 100k real faces under 2 GB); approximate kNN pair recall >= 0.99 against exact at 200k |
| CL-7 viewpoint | same-identity pair recall in the 150-180 degree bin at a false-merge rate of 0.05 above chance by the registered margin; nearest-of-P prototypes beats the centroid at accept precision 0.95 |
| CL-8 images | hierarchical k-means label-set Jaccard purity not below flat IVF at equal or more clusters; judged coherence reported |
| CL-9 GPU | kNN graph recall@30 >= 0.99 against exact; on COCO train2017, the GPU-fed partition has purity within 0.01 of the CPU partition and ARI >= 0.95 against it; a GPU method used end to end (ECG, cuML HDBSCAN) must also pass its line's quality bars |
| CL-1 judge | per kind: pair kappa >= 0.7 and flag precision >= 0.8 against truth; judge-vs-human kappa >= 0.7 on the random half with a bootstrap lower bound >= 0.5; a layout chosen by a non-vacuous calibration or "not distinguishable" |

A gain counts only when it is larger than twice the run-to-run spread and its bootstrap interval
excludes 0.

## 5. The visual judge and the blind human re-judge

### 5.1 Fix the calibration before any new judge call (CPU, R0)

1. `make-calibration` gains `--clean-fraction` (default 0.5) and draws clean or contaminated by a
   seeded permutation of exact counts (`round(n * clean_fraction)` clean), contaminated cases from the
   E0 constant `BAD_COUNTS` (1, 2, 3, 5, 8); `bad_kind` and `clean_fraction` recorded.
2. `recommend_layout` in `judge_agreement.py` returns `recommended: null` with a reason when: the
   truth clean share is outside [0.4, 0.6] ("unbalanced sample"); the judge's "same" share is 0 or 1
   at every layout ("judge answered one way at every layout"); no layout reaches kappa 0.7. Otherwise
   it compares each eligible layout with the best by the paired per-case accuracy difference and its
   bootstrap interval (`scipy.stats.bootstrap(..., paired=True)`), and picks the smallest layout whose
   lower bound is above -0.05. No recommendation keeps the default 6 x 4 at 160 px
   (`src/services/labeling/contact_sheet.py`), which is a result, not a failure.
3. Report the shown contaminated-cell share per layout: synthetic contaminants are the farthest
   members and fill the outlier slots first (2 on a 3 x 3 sheet, 3 on a 6 x 4), so a "smaller grid is
   better" reading may only reflect a higher contaminated share.
4. Proof on existing raw data, no judge calls: recompute `cal_oi` from the E0 JSONL; it must return
   `null` with "judge answered one way" (today it returns 3 x 3).

Red tests: calibration clean share exactly 0.5 for n = 10, 30, 50; the all-"different" over 20 %
clean fixture returns `null`; the paired rule picks a different layout than the fixed 0.05 rule on a
constructed case (the test that pins today's rule is replaced).

### 5.2 Sheets and sampling

- Builder `src/services/labeling/contact_sheet.py` unchanged (version `cs1`): seeded member selection
  (prototypes, outliers, near members, random fill), letterboxed cells, shuffled order, bitmap
  digits, PNG without timestamps, manifest with `png_sha256`, refuse on a changed crop.
- Per experiment at most 20 cluster sheets, 10 pair sheets and 20 search sheets for the hosted judge
  (owner decision O3); half a seeded uniform draw, half targeted (largest clusters, borderline
  merges, arm disagreements), never pooled.
- About 10 % planted known-same and known-different controls; a batch whose controls fail (false
  accept or false reject above 0.2) is discarded. About 10 % re-rendered in a second order for
  position bias.
- Pair sheets of 6 rows for local judges (the 2 x 12 sheet may not suit a small model);
  `--pairs-per-sheet` on `make-e0`.
- Every new sheet directory passes `visual_judge.py verify` (byte-identical rebuild) before any call.
- Real-cluster sheets for CL-2 to CL-8 are built from the arm's partition through the same builder,
  with roles from the partition's prototypes.

### 5.3 Blind human re-judge (`scripts/research/judge_human.py`)

The person sees a PNG with numbered cells and the same question the model saw, and never sees,
before finishing: the sheet id, case id, layout, sample role, truth, any judge verdict, or the
disagreement list.

```text
judge_human.py draw     --sheets DIR [--sheets DIR ...] --per-kind N --targeted M --controls K --seed S --out HUMAN_DIR
judge_human.py export   --plan HUMAN_DIR/plan.json --exchange EXCH_DIR
judge_human.py collect  --plan HUMAN_DIR/plan.json --exchange EXCH_DIR --rater RATER_ID
```

- `draw` runs before any new judge verdict is read; the targeted half (`--targeted-only`) is drawn
  after the judges ran. `plan.json` lives outside the exchange directory.
- Alias = `sha256(f'{salt}:{sheet_id}')[:12]` with a random per-plan salt kept only in `plan.json`;
  export order shuffled; one sheet per case (never two layouts of a case or a twin).
- `export` writes only `<alias>.png`, `<alias>.prompt.txt` and `answers.csv`.
- `collect` validates each CSV row through the existing `parse_verdict`
  (`scripts/research/judge_protocol.py`), records invalid rows as invalid (never repaired: "same"
  with odd cells listed is invalid), and appends records in the judge run shape with
  `model = "human:<rater>"`, `backend = "human"`, so `judge_agreement.py report` reads human runs as
  it reads model runs. No cache argument: human runs never enter a judge cache.
- The sheet-file backend also writes aliased names from the plan's alias map, so hosted subagents
  cannot read roles from file names either; each subagent gets at most one sheet per case.
- CSV columns: cluster `alias, same (y/n), odd_cells, name, confidence, seconds`; pair
  `alias, row1 ... rowN (yes/no/unsure), seconds`.
- The person confirms in writing (one line in the note) that the disagreement lists were not opened
  before `collect`; the 20-item spot check of disagreements comes after the blind pass.

Sample per kind (J14 for a second rater):

| Kind | Random | Targeted (reported apart) | Controls | Who |
|---|---|---|---|---|
| COCO crops, cluster sheets | 40 | 10 | 4 | owner or delegate |
| COCO crops, pair sheets | 10 sheets | 0 | 0 | owner or delegate |
| Open Images vehicles, cluster sheets | 30 | 20 (judge "different", truth "same") | 4 | owner or delegate |
| LFW faces, cluster sheets | 30 | 0 | 4 | local only, on this machine |
| Each CL-2 to CL-8 finalist arm | 20 | 10 | 2 | owner or delegate |
| Human-human ceiling (J14) | 30 COCO sheets judged by two people | 0 | 2 | owner and delegate |

About 150 sheets for CL-1 (about 2 h at 30-60 s per cluster sheet) plus 30 per finalist arm.

### 5.4 What the numbers mean

- **Agreement:** raw agreement, kappa and PABAK side by side, plus the judge's own "same" share; a
  judge with one answer is reported as degenerate, never as kappa 0 "with information".
- **Purity estimates:** prediction-powered inference with `ppi_py.ppi_mean_ci`: the judge's per-sheet
  purity on all judged random sheets (unlabelled side) and the human's on the random human subset
  (labelled side, a uniform subset of the same draw). The interval is valid even if the judge is
  biased; the judge only narrows it. Targeted sheets never enter it. A run whose interval half-width
  exceeds 0.10 reports "insufficient sample", not a pass.
- **Judge vs truth** on labelled public data first, per kind, before a judge's verdict counts on that
  kind (bars in 4.3). Below them the judge only ranks sheets for a person.
- **Vehicles:** if the person also rejects most of the 20 targeted vehicle sheets (human-vs-truth
  kappa < 0.7), the vehicle-type labels are not a usable truth at that level: no more hosted spend on
  vehicle type. If the person sides with the truth, one registered prompt revision (`p2`, text
  committed in R0, defining the level in words such as "the same body type: car, van, truck, bus,
  motorcycle") is allowed, scored as a new kind.

### 5.5 Judges and budgets

| Judge | Runs | Budget |
|---|---|---|
| Haiku (hosted) | balanced COCO calibration: 50 cases (25 clean) at 3 x 3, 4 x 4, 6 x 4 at 160 px (150 sheets); COCO E0 top-up (45 cluster, 13 pair sheets); test-retest 25 sheets in a fresh context with a distinct model label (zero cache hits asserted); reserve 67 for vehicles only if 5.4 allows | at most about 300 sheets (J13: Message Batches with an API key, else sheet-file subagents); model id recorded per call |
| Local VLMs (CRW-3) | rerun of the second catalog entry at 2,048 output tokens on e0 and new calibration sets; Qwen3-VL-4B on the new calibration sets; one larger catalog entry only if both small models fail COCO | about 2 GPU hours |
| Human | 5.3 | about 2-4 h for CL-1, plus finalists |

Thumbnail size (128 / 160 / 192) is calibrated with the local judges only; Haiku stays at 160 px
unless the local runs show a size effect larger than the paired interval.

### 5.6 Privacy

- Face, identity-text and plate sheets never go to a hosted judge: the existing refusal by declared
  kind in `scripts/research/judge_protocol.py`, checked before any budget reservation, stays; owner
  decision OI-2 (2026-11-04) adds no further check, so pool provenance is a procedure: a face-bearing
  pool is always built with a private kind, and the run header records the pool's pin manifest.
- Set P per 3.3. Face material and set P material never leave `artifacts_local/`.
- Research model names come from the CLI (`--model`), never as literals (the naming-leak scan bans a
  word in the default catalog VLM's id).

### 5.7 FG-0 by a person

`scripts/research/fine_grained/fg0_run.py collect` gets a required `--reviewer`
(`machine:<label>` or `person:<rater>`; no default) replacing the hard-coded machine string, and a
`from-csv` converter into the schema `fg0.parse_verdict` validates. Scoring unchanged: PASS needs
model-level decided accuracy >= 0.95 and rejection of >= 0.80 of the same-make plants; the strict
reading (unsure counts as not confirmed) reported beside it. About 2 h. It decides whether the WB
model-level numbers stop being provisional; it does not reopen the encoder decision.

## 6. Experiments

| ID | Question | Data | Arms | Compute | Window | Output |
|---|---|---|---|---|---|---|
| CL-0 | R0: register bars (4.3), splits, seeds, sample sizes and stop rules; pins and lock files; harness helpers (4.2); judge fixes (5.1, 5.3, 5.7); fetchers for F1, B1 and the COCO train2017 crops; COCO-to-Open-Images name map | none | - | about 2 CPU h | any time | registration notes committed before any score; tests red then green |
| CL-1 | Judge per kind (#406) | COCO, Open Images vehicles, LFW, set C (FG-0) | 5.5 | Haiku about 300 sheets; GPU about 2 h; person about 4-5 h | Haiku and person after the tag; local half in CRW-3 | `docs/research/visual_judge_calibration.md` (registration first, results appended); `docs/benchmarks/visual_judge_calibration/` |
| CL-2 | Class-level items on F1 | F1; CUB, SOP rows; S-P1, S-P2 | IVF (today), IVF with K from data, average-linkage HAC on the mutual 30-NN graph (blocks of at most 150k), Leiden CPM (gamma log-grid 0.0003-0.03 on dev) with and without the CHAMELEON filter, each with and without the detector-class soft prior | GPU about 1 h (F1 ingest or encode); CPU about 6 h | CRW-1 + CPU slot | `docs/research/cluster_global_eval.md` |
| CL-3 | Region boxes | B1 | today's MiniBatchKMeans (n / 800) against the CL-2 arms per box space | GPU about 1 h (region stage on B1); CPU about 2 h | CRW-1 + CPU | `docs/research/cluster_region_box_eval.md` |
| CL-4 | Scale with quality | COCO train2017 (about 860k crops); Deep1B and Text-to-Image 1M / 10M slices for time | CL-2 finalists | GPU about 4 h (ingest or encode); CPU about 12-16 h | CRW-2 + CPU | section of the CL-2 note |
| CL-5 | Stability and incremental assignment (only on a CL-2 or CL-4 PASS, else on IVF alone for the S-core design) | F1; S-P1 | carry-over (Jaccard 0.5, Hungarian); anchored members pinned (`is_membership_fixed` or contraction); 10 % addition re-run; incremental assignment with isotonic accept / review / unassigned bands against nearest-centroid | CPU about 3 h | CPU slot | `docs/research/cluster_partition_stability.md` |
| CL-6 | Faces | LFW, CALFW, CPLFW; Open Images faces up to 100k; S-P3 | union-find 0.7; Immich-style core point (0.5, core 3, score 0.7); HDBSCAN (scikit-learn, cuML); average and complete graph HAC on mutual kNN; Leiden CPM with negative same-image edges; each with and without the same-image cannot-link; incremental nearest-of-8-prototypes with isotonic bands; approximate (FAISS IVF, HNSW, cuVS) against exact kNN at 200k | GPU about 3 h (detection, embedding); CPU about 4 h; person about 1.5 h | CRW-3 + CPU | `docs/research/face_grouping_eval.md` |
| CL-7 | Viewpoint and prototypes (J11) | CO3D v2, Objectron (local); F1 near-duplicate and similar-class strata | frozen PE-Core and MobileCLIP; centroid against nearest of P prototypes (P in 1, 2, 4, 8), with and without the margin rule; azimuth bins 0-30, 30-90, 90-150, 150-180 | GPU about 1 h; CPU about 2 h | CRW-2 | `docs/research/cluster_viewpoint_eval.md` |
| CL-8 | Whole images (low priority) | `coco_bench_2000`, Open Images val sample | flat IVF against hierarchical k-means (Vo 2024 code) on MobileCLIP | CPU about 2 h | CPU slot | `docs/research/cluster_image_eval.md` |
| CL-9 | GPU agreement and scale | COCO train2017 (agreement); Deep1B and Text-to-Image slices (time) | cuVS `all_neighbors` (`brute_force`, `ivf_pq`, `nn_descent`) and FAISS GPU kNN graphs feeding the CPU methods; cuGraph ECG; cuML HDBSCAN (faces) | GPU about 4 h | CRW-2 | section of the CL-2 note; a `cluster_backend_validations` row shape per passing version |

Owned by the parallel research plans, listed for order and interface only: Z5 detector-class roots
(#398, auto-label plan; this plan supplies the harness and the partition seam); near-duplicate
thresholds (#416); crop preprocessing and encoders (#347, #365) and the frame vector (#435)
(embeddings plan); plate text (#401). Attribute signals (#400) stay parked until CL-5's constraint
machinery exists.

### 6.1 Windows

| Window | Contents | GPU hours | Order |
|---|---|---|---|
| CRW-1 | CL-2 vectors (F1), CL-3 region stage on B1 | about 2-3 | first after the tag |
| CRW-2 | CL-4 vectors (COCO train2017), CL-9 GPU arms, CL-7 encodes | about 9-10 | after CL-2's verdict exists |
| CRW-3 | CL-1 local VLMs, CL-6 face detection and embedding | about 5 | any time after CRW-1 |
| CPU slots | CL-0, all scoring, CL-2 and CL-4 HAC and Leiden, CL-5, CL-8 | about 35-40 CPU h | between program sweeps |

A window never overlaps a program, v0.8.0 or GPU-pipeline research window or a live-harness run. A
GPU-pipeline adoption that changes preprocessing must pass the embedding parity gate before CRW-2,
else every stored vector is stale.

## 7. Stop rules

| Line | Stop when | Then |
|---|---|---|
| CL-1, per kind | pair kappa < 0.7 or flag precision < 0.8 on the topped-up sample, or judge-vs-human kappa < 0.7 | that judge ranks sheets for a person on that kind; at most one registered prompt revision; no tuning loops |
| CL-1 layout | the calibration returns `null` | 6 x 4 at 160 px stays; no second calibration |
| CL-1 local | the rerun and (if run) one larger entry both fail COCO | local judges rank only; faces keep the local judge as a ranking aid and a person decides |
| CL-2 | every arm fails on F1's test half | IVF stays as the served method; graph refinement parked for good on #346; CL-4, CL-9 quality runs and CL-5's method part are not run |
| CL-3 | no arm beats MiniBatchKMeans on the box bars | today's method stays, recorded on the partition seam |
| CL-4 | the CL-2 winner fails at 860k, or misses 2 h wall on CPU and on every validated GPU path | the winner ships only below its measured size limit, or not at all |
| CL-5 | protected moves > 0 or id stability < 0.95 | no new method ships (a partition that destroys review work is worse than IVF) |
| CL-6 | no arm beats union-find on chaining at matched recall | union-find stays; only the job move and the partition record ship |
| CL-7 | 150-180 degree bin near chance | record "no-go for frozen-embedding object identity"; class-level grouping stays the scope; no re-ID data without a licence decision |
| CL-8 | hierarchical k-means not better | flat IVF stays as navigation |
| CL-9 | agreement gate fails | GPU builds nothing in the product; CPU result stands |
| Any INCONCLUSIVE | one repeat with a re-registered, larger set | a second INCONCLUSIVE closes the line as not established |

## 8. Repetition, GPU etiquette and recording

- **Repetition:** three seeds and one row permutation per headline row; 20 seeds for loop
  simulations; report mean and spread; one factor changed at a time.
- **GPU:** GPUs 0 and 2 only (the A6000s), never GPU 1. `CUDA_DEVICE_ORDER=PCI_BUS_ID` always
  (without it `CUDA_VISIBLE_DEVICES=2` selects the 12 GB card). One GPU job at a time on the host,
  serialised with `flock "$SCRATCH/locks/gpu<N>.lock" <command>`; check `nvidia-smi` and ask the peer
  sessions before using a card, and pause at once when a peer asks. Never cap the address space of a
  CUDA process (the bench driver's `RLIMIT_AS` broke cuGraph). Stop the VLM after its run.
- **Host:** `uptime` at start and end of every run; CPU work at `nice -n 19` with at most 8 threads;
  a start above load 16 is marked noisy and repeated.
- **Stacks:** `docker compose -p op-rs-crw<n>` only, never the default project; snapshot containers,
  volumes and networks before and after.
- **No wait loops:** waits use the Monitor tool or a healthcheck, never `pgrep -f` or `until`/`sleep`
  loops.
- **Run header:** git sha, pin manifest hash, lock-file digest, model and engine versions,
  `embedding_field`, `preprocess_id`, seed, GPU id, `nvidia-smi` snapshot, load, wall, peak RSS and
  VRAM.
- **Where results go:** raw output in `artifacts_local/research/clustering/<exp>/` (gitignored);
  validated artefacts moved out of `/tmp`; small JSON backing a published number in a new
  `docs/benchmarks/<exp>/` folder with `SHA256SUMS`; the note in `docs/research/` (Setup, Method,
  Results, Findings, Open issues, Reproduce), rows in `docs/research/README.md` and
  `docs/design/README.md`; one results row per experiment on #346. Set P per 3.3.
- **Bars before scores:** for every experiment, `git log` shows the registration commit older than
  the first results commit and its benchmarks folder; existing `docs/benchmarks/` files show no `D` or
  `M` in any diff.

## 9. Failure register (ways the program can look done and not be)

1. A guarded arm silently unread because the set was sized for another arm (Z5 lesson: size per arm).
2. A multi-GB annotation CSV loaded whole; stream and filter, print rows dropped per filter (a
   licence column that differs passes everything).
3. Open-set or class membership from indices; use names and the name map.
4. Crop near-duplicate recall measured globally while the product runs per IVF bucket.
5. Mixed spaces or preprocessing variants in one bundle.
6. Acceptance claimed on a worn held-out half.
7. Near-duplicates straddling dev and test.
8. Row order: permutation ARI not checked before trusting a method.
9. A method that abstains a lot looking precise; always report unassigned and review fractions.
10. Oracle-only loop rows hiding labeller noise; report the 10 % noisy row beside it.
11. A vacuous calibration (one answer at every layout) read as kappa 0 with meaning.
12. Kappa 0 returned as a number, not "degenerate": the guard looks at the answer share.
13. Unsalted aliases reversible by hashing guessable ids; plan order leaking draw order.
14. Two layouts of one case, or a twin, in one human plan.
15. Human records written into a judge cache; a retest answered from the cache.
16. `--limit` selecting the first sorted ids (it mixed twins into E0's sample); select by a seeded
    id list.
17. Planted controls cited as judge accuracy (they are easier than real sheets).
18. cuGraph Leiden reported as CPM; FAISS GPU `k` above 2048; a GPU "range search".
19. Partition ids renumbered on each run while every quality metric stays green; only id stability
    and protected moves see it.
20. A research run executed as a product task inside a Celery time limit; research never runs as a
    product task, and a landing sizes its limit from the measured wall time.
21. A synthetic number in a hand-written quality table; every quality number links its scorecard JSON.
22. Benchmark data pruned by a cleanup; the diff check in section 8 guards it.
23. Loaded-host timings presented as absolute.
24. A face crop entering a generic object pool under a non-private kind.
25. Set P material in a committed file or a hosted call.
