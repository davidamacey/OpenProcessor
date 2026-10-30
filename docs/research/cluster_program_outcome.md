# Cluster research program: outcome

Status: closing summary, written 2026-10-30 for the 0.7.0 cycle. It collects what the cluster and
embedding experiments of waves W0 to W8, WB and WB2 found, what each arm's decision is, and what to read
or run to check any claim. Every number below is quoted from the linked note, which holds the method, the
registered bar and the limits. Nothing in this program changed product code.

## 1. The short version

- **Nothing new ships from the clustering experiments.** The production configuration stays: IVF
  clustering (k-means with at most 512 centres) over the PE-Core-L14-336 embedding, and the auto-label loop
  that sits on top of it. Every challenger either failed a bar that was written before it was scored, or
  could not be read.
- **The one real lead** is class-level: using the detector class as the loop's first grouping ("root")
  needs about a third of IVF's labelling decisions at the same label precision. A cheap, label-free
  hold-back guard removes most of its one risk. The check that would allow adoption (Z5) is still to run.
- **Fine-grained grouping (make and model) is an embedding problem, not a clustering problem.** The
  choice of grouping method moved results far less than the choice of encoder, and the encoder ranking
  from one weakly labelled set did not survive the standard benchmarks.
- **Graph refinement (Leiden, ECG, GPU arms) stays parked** in issues #342 to #346. It works on synthetic
  data and is unproven on real data.

## 2. What was tried

| Wave | Question | Note |
|---|---|---|
| W0, W0b | How good is the stored IVF partition, and how do average linkage, complete linkage, k-means, Leiden and ECG compare on stored vectors of public COCO and Open Images crops? | [cluster_eval_baseline.md](cluster_eval_baseline.md), [cluster_config_baseline.md](cluster_config_baseline.md) |
| W0c | Does grouping inside the detector class (average linkage with a per-class cut) meet product bars? Do whitening, k-reciprocal re-ranking or DBA help? Can novelty be flagged? | [cluster_config_experiments.md](cluster_config_experiments.md) |
| W0d | The corrected bars A1 to A7 on a fresh public set; a label-free per-class cut rule; chaining; splitting by descent | [cluster_config_fresh_set.md](cluster_config_fresh_set.md) |
| W0e | The auto-label loop's gate simulated on IVF, one cluster per detector class, and within-class linkage; a judge-gated merge | [cluster_config_loop_cost.md](cluster_config_loop_cost.md) |
| Decision | Pulls W0b to W0e together, sets the registered bars | [cluster_config_decision.md](cluster_config_decision.md) |
| Z4 (W0f) | Detector-class roots with and without a hold-back guard, against within-class roots and IVF, on an open-set-rich set | [cluster_config_z4.md](cluster_config_z4.md) |
| W8 | A visual judge on contact sheets (Haiku and a local VLM) | [visual_judge_e0.md](visual_judge_e0.md) |
| WB | Make and model grouping on Commons set C: five encoders, preprocessing variants, grouping methods, new-model queue | [fine_grained_benchmark_plan.md](fine_grained_benchmark_plan.md), [fine_grained_benchmark_results.md](fine_grained_benchmark_results.md) |
| Labelled benchmarks | Which author-labelled benchmarks can replace weak labels | [labelled_benchmarks.md](labelled_benchmarks.md) |
| WB2 | The WB encoder question re-run on CARS196, CUB-200-2011 and Stanford Online Products | [wb2_standard_benchmarks.md](wb2_standard_benchmarks.md) |
| Leiden and GPU arms (#342) | Global kNN graph plus Leiden against IVF plus average linkage, on CPU and GPU | [leiden_refinement_prototype.md](leiden_refinement_prototype.md) |

The contract for how each wave was run is the program plan in
[ship_track_plan.md](../design/ship_track_plan.md) (state of the product lanes, measurement protocol,
Z5 definition in section 7). Bars were committed before scores in W0e, Z4, WB and WB2; the notes say
where a bar was missed and by how much, and none was moved afterwards.

## 3. Decision per arm

"Shipped" means in the product already or merged by this program. "Parked" means it may return with new
evidence. "Killed" means a registered bar failed and no follow-up is planned.

| Arm | Decision | Evidence |
|---|---|---|
| IVF clustering (K = min(512, n // 6)) | **Shipped (stays)** | Cheapest to run; best of the arms on `rare` items; see section 4 |
| Average linkage inside the detector class, per-class cut | **Parked** | Fails the corrected acceptance on a fresh set (A4 0.288 against 0.329; `rare` stratum below IVF); the cut moves between sets (0.346, 0.345, 0.515) and does not transfer to one of them |
| Label-free per-class cut rule | **Killed** | Fails recall bars on the held-out half and on the fresh set |
| Whitening, k-reciprocal re-ranking, DBA on crops | **Killed** | Precision gain at matched recall is negative: whitening -0.033 and -0.065, k-reciprocal -0.014, DBA -0.011 |
| Novelty flag by distance (sheep key) | **Killed** | AUROC 0.838 and 0.851 against a bar of 0.90; no cut reaches flag precision 0.80 (best 0.733 and 0.761) |
| Judge-gated merge of small neighbours (Z3) | **Killed** | Adds 2.2 % effort instead of saving the 10 % bar |
| Split by descent | **Killed** | 10 of 14 impure clusters become pure (71 %) on one half, 1 of 11 (9 %) on the other; bar 80 % |
| Within-class roots for the loop | **Killed** | Fails the effort bar in Z4 (0.526 decisions per item against a limit of 0.417); 813 small roots save little |
| Detector-class roots (with or without the guard) | **Parked as the lead** | Passes precision and effort in Z4; the open-set bar was unreadable (section 6) |
| Graph refinement: Leiden CPM, CHAMELEON filter | **Parked** (#342 to #346) | Merges fragments on synthetic data; on real crops only a stricter filter beat the baseline on both axes (section 7) |
| Consensus Leiden, HDBSCAN leaf, FINCH, adaptive thresholds | **Killed** | Consensus costs 5 times the time for equal quality; HDBSCAN labels about 80 % of points noise; FINCH has no granularity control ([leiden_refinement_prototype.md](leiden_refinement_prototype.md), findings 6 to 10) |
| GPU Leiden/ECG (cuGraph) | **Parked** | One 20k synthetic result only; nothing above 20k, no real data, no 1M run |
| Another encoder replacing PE-Core-L | **Not adopted** | No challenger passes the registered gain on CARS196 (section 5) |
| Letterbox preprocessing for whole images | **Not adopted** | No gain reaches 0.03 on CARS196 for any encoder |
| Visual judge (Haiku) | **Shipped as research tooling only** | Usable on COCO crops, over-flags vehicle labels; "same" is trustworthy, "different" needs a person (section 8) |
| Local VLM judge (Qwen3-VL-4B) | **Killed as an authority** | Pair accuracy 0.51 on COCO crops, near chance; may rank clusters for review only |
| Region-box id-sort fix | **Out of this branch** | Already in PR #389; the within-class method itself is not shipping |

## 4. What stays in production

- **IVF is the default clustering.** On the 3,000-item set (944 images, 3,961 crops, 2,950 labelled in 86
  classes, public COCO and Open Images) it scores held-out pair precision 0.716 and BCubed precision 0.784
  but very low recall (pair recall 0.027, BCubed recall 0.186)
  ([cluster_eval_baseline.md](cluster_eval_baseline.md), "BCubed rows on the 3,000-item set"). It is
  conservative: clusters are mostly right and many.
- **Why nothing beat it.** Average linkage reaches higher recall only by giving up precision (at pair recall
  0.126 its held-out precision is 0.330 against IVF's 0.734 at 0.029;
  [cluster_config_baseline.md](cluster_config_baseline.md), "E1 on the heldout half"). On `rare` items the
  incumbent is hard to beat, which is why bar A6 failed twice.
- **PE-Core-L14-336 stays the embedding.** On the standard benchmarks no challenger passed the registered
  Recall@1 gain, and PE-Core-G, the best on all three sets (+0.008 on CARS196, +0.040 on CUB, +0.024 on
  SOP), costs about eight times the encode time and five times the memory
  ([wb2_standard_benchmarks.md](wb2_standard_benchmarks.md), findings 9.6). The text tower is also needed
  for search, and four of the five encoders have one.
- **The auto-label loop stays as designed.** Its gate certifies a cluster at 90 % homogeneity from a
  sample. Class-level effort figures for the loop are in section 6. The loop branch itself is outside this
  research PR ([ship_track_plan.md](../design/ship_track_plan.md), section 7).

## 5. Fine-grained results (make and model)

Wave WB, Commons set C (2,943 images, 26 makes, 50 models, 74 generations; 13,301 rows with Z2 car
distractors). Findings in [fine_grained_benchmark_results.md](fine_grained_benchmark_results.md):

- Replacing the production vector with MobileCLIP2-S2 raised model-level precision at 10 by 0.11 and cut
  the review cost from 0.54 to 0.32 at the same registered precision bars; letterbox added 0.04 to 0.11
  for MobileCLIP2 and SigLIP 2.
- Average linkage inside the car key met the registered bar for four representations. K-reciprocal
  re-ranking, DBA, Leiden and ECG all failed. New-model detection by distance failed (FG-6).
- Haiku can judge make-level naming but not model-level (FG-7: make pass, model fail).
- The labels come from Commons category names and were only machine-reviewed (FG-0 inconclusive), so
  every model-level figure is provisional.

Because of that, WB2 re-ran the encoder question on author-labelled sets
([wb2_standard_benchmarks.md](wb2_standard_benchmarks.md), section 9):

| Encoder (variant) | CARS196 R@1 | CUB R@1 | SOP R@1 |
|---|---|---|---|
| PE-Core-L (letterbox), production | 0.961 | 0.867 | 0.796 |
| PE-Core-G (letterbox) | 0.969 | 0.907 | 0.820 |
| MobileCLIP2-S2 (centre) | 0.964 | 0.752 | 0.674 |
| SigLIP 2 (letterbox) | 0.975 | 0.744 | 0.790 |
| DINOv2-L (centre) | 0.825 | 0.892 | 0.576 |

- The set C ranking does not hold: MobileCLIP2 is 0.115 below PE-Core-L on CUB and 0.122 below on SOP.
  The earlier "swap the encoder" signal is therefore not a decision.
- The registered sanity check was not met (CARS196 Recall@1 above 0.95 for four models; the NMI chance
  bar was set wrongly), so every WB2 comparison is labelled exploratory in the note.
- Average linkage keeps BCubed precision at 0.90 or more on 3 of 3 sets for PE-Core-G and 2 of 3 for
  PE-Core-L (it fails on CUB), at pair recall 0.05 to 0.38. At the true number of clusters it is clearly
  worse than k-means.

## 6. The lead for later: detector-class roots and the hold-back guard

Wave W0e ([cluster_config_loop_cost.md](cluster_config_loop_cost.md)) simulated the loop's gate on a
vehicle-heavy Open Images set (33,580 crops, 7,476 labelled in the held-out half, 20 seeds, oracle labeler):

| Arm | Decisions per item | Propagated precision |
|---|---|---|
| IVF (incumbent) | 0.776 | 0.988 |
| One cluster per detector class | 0.374 | 0.989 |
| Average linkage inside the class | 0.326 | 0.996 |

Z4 ([cluster_config_z4.md](cluster_config_z4.md)) then compared class roots (with the loop's own
splitter), the same with a label-free hold-back guard, within-class roots and IVF on a second set (7,209
labelled held-out crops, 395 of them open-set animals the detector cannot name):

| Arm | Decisions per item | Precision | Open-set items propagated wrongly per run |
|---|---|---|---|
| `key` (class roots) | 0.293 | 0.982 | 9.05 (0 to 72 by seed) |
| `key_guard` (plus hold-back) | 0.377 | 0.988 | 0.25 |
| `within_key` | 0.526 | 0.990 | 0.95 |
| IVF (cheaper of the two IVF rows) | 0.834 | 0.979 | not comparable |

The registered rule needed at least 30 open-set items propagated per run to read the open-set bar. Only
2 of 20 seeds reached that (the set held 395 held-out open-set crops against a registered 2,000), so the
verdict is **inconclusive and IVF stays**. The direction is clear: class roots cost 0.35 times IVF's
effort, the guard costs 29 % more effort than class roots alone and removes 97 % of the open-set exposure.
Z5 would repeat Z4 on the Open Images train split with at least 1,200 open-set crops
([ship_track_plan.md](../design/ship_track_plan.md), section 7.3), then integrate the roots and the guard
into the loop (Z5-I, section 7.4) only if a bar passes. No Z5 note exists yet.

The decision note also records why this does not carry over to make and model: inside the `car` key the
detector class gives no help ([cluster_config_decision.md](cluster_config_decision.md), 0A.3).

## 7. Parked issues (#342 to #346)

The Leiden prototype ([leiden_refinement_prototype.md](leiden_refinement_prototype.md)) is the work for
phase 1 (#342):

| Phase | Issue | State |
|---|---|---|
| Prototype: global kNN plus Leiden against the current pipeline | #342 | Done as research; result below |
| Constrained split/merge, stable cluster ids, versioned partitions | #343 | Parked; start with a held-out set and threshold calibration, not product code |
| Continuous local refinement in the refresh worker | #344 | Parked |
| GPU Leiden/ECG/HDBSCAN, FINCH, HIT-Leiden, embedding adapter | #345 | Parked; GPU runs only reach 20k |
| Epic: graph-based global refinement | #346 | Parked |

Results behind the parking:

- Synthetic planted partitions (20k, 100k, 500k): Leiden CPM at gamma 0.003 lifts pair recall from about
  0.08 to 0.79 to 0.86 and the CHAMELEON-style filter restores precision to 0.953 to 0.954.
- On 539 real labelled COCO crops the unfiltered pass trades precision for recall; only the stricter
  filter (RI 0.5, RC 0.95, tuned in the same sample) beat the baseline on both axes (precision 0.944 and
  recall 0.195 against 0.905 and 0.088). The authors' verdict is a conditional go with a weak case.
- Modularity Leiden over-merges (false-merge 0.236 at 20k to 0.609 at 500k). cuGraph's Leiden is
  modularity-only, so it is not a drop-in; cuGraph ECG scored 0.953 precision and 0.899 recall at 20k
  synthetic but nothing larger was run.
- Cost on CPU: 150 to 220 s and 1.2 to 1.5 GB at 500k points, 128 dimensions. Calibration of gamma and
  the filter floors per embedding distribution is the open engineering problem. 1M was not run.

Other open follow-ups: GPU clustering via cuML UMAP is described in [gpu_clustering.md](gpu_clustering.md).

## 8. Judge and contact sheets

[visual_judge_e0.md](visual_judge_e0.md) covers the harness (contact-sheet builder in
`src/services/labeling/contact_sheet.py`, runners under `scripts/research/`). Hosted judges are refused
for face, identity-text and plate sheets before any call.

| | COCO crops | Open Images vehicles |
|---|---|---|
| Sheets judged | 55 | 55 |
| Pair kappa, flag precision (bars 0.7, 0.8) | 0.821, 0.827 (pass) | 0.611, 0.454 (fail) |
| Clean sheets wrongly rejected | 0.217 | 0.765 |
| Contaminated sheets accepted | 0.094 | 0.000 |

Reading: Haiku's "same" is reliable and its "different" needs a person. The local VLM is not a judge on
these sheets. The layout calibration was vacuous (Haiku said "different" for all 50 sheets), so no
default sheet layout was chosen. The 5 % human re-judge and the FG-0 label check are for a person and have
not been done.

## 9. Data sets downloaded and licence notes

Nothing below is committed except small manifests and pins; image data lives on the local dataset store.
Details: [cluster_proxy_datasets.md](cluster_proxy_datasets.md) and
[labelled_benchmarks.md](labelled_benchmarks.md).

| Set | Use here | Licence note |
|---|---|---|
| COCO crops (3,000-item set, COCO-200) | Baselines, decision experiments | COCO terms; public |
| Open Images V7 validation and test vehicles, animals, faces | Fresh sets, Z2, Z4, judge pools | Images CC BY 2.0, annotations CC BY 4.0 |
| Commons set C (2,943 images) | WB make and model | Per file; author and licence in `scripts/datasets/manifests/commons_set_c.json`; share-alike applies |
| LFW | Face proxy, judge pool | No licence published; local use only |
| CARS196 (16,185 images), CUB-200-2011 (11,788), Stanford Online Products (120,053) | WB2 | Non-commercial research or marketplace photos; local evaluation only |
| VMMRdb (285,086 images, 24 GB), CALFW, CPLFW | Downloaded and indexed, not scored | No licence found for VMMRdb; its zip does not match the published counts; CALFW and CPLFW licence not stated |
| CompCars, VeRi, VehicleID, BoxCars, IJB-B/C, MS-Celeb-1M, VGGFace2 and others | Not fetched | Signed forms, dead hosts or withdrawn data; reasons in [labelled_benchmarks.md](labelled_benchmarks.md) |

Public data only was used for every figure above. The private photo set was not used, and no set P crop
was sent to a hosted model.

## 10. What was not established

- Whether class roots plus the guard hold on open-set animals (Z5).
- Any make or model verdict with person-checked labels (FG-0), on street photos (not curated Commons
  photography) or at the 13,301-row scale of FG-3 (set C-large not fetched).
- Faces: no face clustering experiment was run; the notes treat them as a separate wave.
- Timings: the host was loaded (load average 9 to 79) in every run, so only within-table comparisons hold.
- GPU graph methods above 20k points, and 1M.

## 11. How to reproduce

Each note ends with a Reproduce section; this list says where to start. Research dependencies (igraph,
leidenalg and the encoders) are listed in `scripts/research/fine_grained_requirements.txt` and are not part
of the product requirements; tests that need them skip when absent.

| To reproduce | Start from |
|---|---|
| Fetch the pinned public sets | `scripts/datasets/fetch_cluster_eval.py`, `fetch_vehicle_eval.py`, `fetch_vehicle_proxy.py`, `fetch_face_proxy.py`; pins in `scripts/datasets/manifests/` |
| Fetch and index the author-labelled benchmarks | `scripts/datasets/prepare_labelled_benchmark.py` ([labelled_benchmarks.md](labelled_benchmarks.md), Reproduce) |
| Export vectors, score baselines, run E1 and E5 | `scripts/research/cluster_diag/` ([cluster_config_baseline.md](cluster_config_baseline.md), Reproduce) |
| Loop simulation and the Z4 bars | `scripts/research/cluster_diag/loop_sim.py`, `z4_run.py` ([cluster_config_z4.md](cluster_config_z4.md), Reproduce) |
| WB encoders, grouping and bars | `scripts/research/fine_grained/` ([fine_grained_benchmark_results.md](fine_grained_benchmark_results.md), Reproduce) |
| WB2 embeddings, scoring, bars | `scripts/research/standard_benchmarks/`; raw JSON in `docs/benchmarks/wb2_standard_benchmarks/` ([wb2_standard_benchmarks.md](wb2_standard_benchmarks.md), Reproduce) |
| Judge runs and contact sheets | `scripts/research/visual_judge.py` ([visual_judge_e0.md](visual_judge_e0.md), Reproduce) |
| Leiden and GPU arms | `scripts/research/leiden_refine_proto.py`, `bench_driver.py`, `gpu_methods.py` ([leiden_refinement_prototype.md](leiden_refinement_prototype.md), section 9) |
| Summary of the 3,000-item baseline | `docs/benchmarks/cluster_config_baseline_3000.json` |
| Offline tests for all of the above | `pytest tests/research tests/research_scripts tests/bench tests/test_contact_sheet.py tests/test_proxy_dataset_manifest.py tests/test_vehicle_eval_selection.py tests/test_cluster_eval_selection.py -q --no-cov` |

Raw run output (`artifacts_local/`) is gitignored and not in the repository; the notes name the paths
and the commands that regenerate them.
