# Fine-grained grouping benchmark (wave WB): vehicle make and model

Status: evaluation plan, registered 2026-10-29, before any fine-grained data is scored. Nothing here
is measured. The bars in section 5 are fixed now. Changing one after a read needs a new registration
and fresh data. Origin: section 0A of [cluster_config_decision.md](cluster_config_decision.md). No
product code changes in this wave.

## 1. Why a separate benchmark

The class-level work (W0b to W0e) answered a different question. It showed that the detector key
plus the loop's gate labels classes cheaply. Inside the `car` key that help is gone: every crop has
the same key. Nothing from the class-level results transfers: not the cut, not the key partition's
precision, not the effort figures.

The question here is whether any frozen embedding, preprocessing and within-key method groups car
crops by make and by model well enough for two uses:

- a person names whole groups from contact sheets;
- the loop (#286) propagates labels at model level.

## 2. Data

| Set | Content | Use | Constraints |
|---|---|---|---|
| **Set C** (Wikimedia Commons) | 2,943 images; 26 makes, 50 models, 74 model generations; 31 to 40 images per generation and 40 to 120 per model. Labels are `taxonomy_path` (make, model) plus `generation`, taken from Commons categories. No view labels | Main labelled set for every bar except FG-3 | Per-file licence recorded, redistributable. Fetched by `scripts/datasets/fetch_vehicle_proxy.py` with `commons_set_c.json` (branch `feat/cluster-w1-datasets`). The labels are unverified: the 294-row review sample has not been filled in yet (FG-0) |
| **Set C-large** (to fetch) | The same fetcher on Commons model categories with more images: up to 200 per model, disjoint from set C by file name and hash | FG-3 only. The loop gate needs about 35 answers, so 40-image groups leave it almost nothing to propagate (Z1's lesson) | If Commons cannot give at least 150 images per model for at least 15 models, FG-3 is reported as unreadable, not passed |
| Distractors | The car-key crops of the Z2 vehicle set: 7,961 labelled at class level, no make labels | Added to the car key as unlabelled members, never scored. They give the key a realistic density and test whether set C's structure survives in it | Public (Open Images) |
| Set P car and truck keys | 14,009 and 2,769 crops of the 91,184-crop project | Sizes per cut, and make-level purity from the local judge on a sample the owner approves | Numbers only. Never sent to a hosted model: car crops can show plates (plan 7.2) |
| Not available locally | Stanford Cars, CompCars, VMMRdb | | Allowed locally if the operator supplies a copy (owner decision O2). Without CompCars, viewpoint effects are not measured |

**Splits.** Set C is split by image, stratified by model generation, seeded, 50/50 into calibration
and held-out halves. For FG-6 there is also a class-disjoint split: in makes with two or more
models, half the models (seeded) are held out as new models.

**Crops.** Set C goes through the product path in a throwaway project (own `-p` name, GPU 2, decode
once). The largest car or truck box of each image is its labelled crop; the image's other crops stay
unlabelled.

## 3. What is compared, one factor at a time

**Representations, screened first at the current preprocessing:**

- PE-Core-L14-336 (production; stored vectors).
- MobileCLIP2-S2 (already in Triton).
- DINOv2-L/14.
- SigLIP 2 (So400m/16 or L).
- PE-Core-G, only if it fits GPU 2.

Models not shipped run in the research venv or Docker, on GPU 2, under the GPU lock. Nothing is
installed into the product.

**Preprocessing, on the top two representations of the screen only** (no full factorial):

- centre-crop (current);
- letterbox to square;
- 10 % context margin;
- whole image. This one is a set C diagnostic only: set C photos mostly show one vehicle filling the
  frame, while set P images hold many vehicles, so whole-image is never a product option.

**Methods inside the car key, on the best representation, then the runner-up:**

- **Average linkage**, cut chosen on calibration: the baseline.
- **k-reciprocal re-ranking** (k1 20, k2 6, lambda 0.3) and **DBA**, computed within the key. They
  were closed for class level (X3), but re-ranking comes from re-identification, which is closer to
  make and model, so they are re-tested here.
- **Leiden CPM** and **ECG** on the key's mutual 30-NN graph, seeded, input sorted by id.

**Same-image cannot-link is not tested at model level.** Two cars of one model often share a photo
(parking, traffic), so the constraint is wrong at class and model level. It belongs to the
identity-level experiments: faces (E6) and plate-linked bursts (W12).

Not re-tested: IVF cells, global cuts, HDBSCAN, FINCH (all killed at class level, and nothing about
a single key changes why).

## 4. Metrics

- **Neighbours** (no clustering), among labelled crops, at make, model and generation level: P@1,
  P@10, mAP@R, and 10-NN vote accuracy.
- **Partition:**
  - BCubed precision, recall and F at model level and at make level.
  - Item-weighted purity per make and per model.
  - The share of model-level false merges that stay within one make. Those still give a correct
    make-level name.
- **Review cost R1** (a person names groups from 6 x 4 contact sheets):
  - For each cluster of n crops: ceil(n / 24) sheets at one decision each, plus one decision per odd
    cell (a crop not of the cluster's plurality model, as flagged by an oracle).
  - Each singleton costs one decision.
  - R1 = decisions / labelled crops. The oracle flagger is optimistic; real flag recall comes from
    FG-7.
- **Loop cost at model level.** U1, U2 and U3 with the #286 gate exactly as in Z2: theta 0.9, alpha
  0.05, max dissent 1, oracle answers at model level. A second row, not barred, uses answers at make
  level only.
- **Stability.** ARI under a row permutation and across four seeds.
- **Cost per variant:** encode throughput and peak memory on GPU 2, vector size, and the re-embed
  time at 100k and 1M crops.

## 5. Bars (registered now)

Bars are read once, on the held-out half, fold 1. Fold 2 is reported. The decision is reopened if
any bar is missed by more than 0.03 in fold 2.

| ID | Bar | Source of the number |
|---|---|---|
| FG-0 | Set C label accuracy at model level >= 0.95 on the 294-row review sample, checked by a person (Wilson interval reported). If it is lower, model-level bars are not read and only make-level bars are | Label noise caps any model-level score |
| FG-1 | A challenger representation goes forward only if held-out model-level P@10 >= PE-Core's + 0.03. A preprocessing variant goes forward only if model-level P@10 rises by >= 0.03, with no loss above 0.01 at make level, nor in class-level P@10 on the 3,000-item set (it changes the shared crop path). Ties keep PE-Core and centre-crop | The plan's WB rule (E2, E3) |
| FG-2 | Best representation, average linkage, operating point chosen on calibration by section 6's rule moved to model level: the largest target model-level pair recall in {0.05, 0.1, 0.2, 0.3} whose calibration model-level BCubed precision is >= 0.90. On held out: model-level BCubed precision >= 0.90, make-level >= 0.95, R1 <= 0.5, ARI 1.000 | 0.90 is the plan's K2 bar. Make errors cost more, hence the higher make-level bar. 0.5 is the owner's halve-effort factor, as for U-b |
| FG-3 | On set C-large: model-level U2 >= 0.95 and U1 <= 0.5 | The plan's propagated-label bar; halve effort |
| FG-4 | Leiden, ECG, k-reciprocal or DBA replaces average linkage only if model-level BCubed precision at matched BCubed recall (within 0.02) rises by >= 0.02 in both folds, with no make-level loss above 0.01. For Leiden and ECG, also ARI >= 0.95 across seeds and permutations | X3's rule; plan principle 9 |
| FG-5 | A second embedding vector is added only if it passes FG-1 and gives R1 <= 0.9 x PE-Core's at FG-2's precision bars | A second vector means a re-embed of every item (#365). 10 % is the same margin as U-c |
| FG-6 | A new-model queue is built only if, on the class-disjoint split, novelty AUROC is >= 0.90 in each make with two or more held-out models, and flag precision is >= 0.80 at a threshold chosen on calibration. Otherwise new models are found by a person naming groups | X4's bars, unchanged |
| FG-7 | The judge is used at a level (make or model) only if E0 on set C sheets at that level gives pair kappa >= 0.7 and flag precision >= 0.8. Haiku runs within the O3 budgets; the local VLM judges the full sample. On set P, only levels that pass, and only with the local VLM | The plan's E0 bars |

Set P has no bar. For car and truck it reports sizes per cut (Y3's metrics) with the winning
representation, and local-judge make-level purity at the chosen cut.

## 6. Order, cheapest first

1. **F0:** the person fills in set C's review sample (294 rows; about 30 min; no GPU). This decides
   FG-0.
2. **F1:** ingest set C into a throwaway project on GPU 2 and export the vectors.
3. **F2:** the FG-1 representation screen. Four or five models encode about 3,000 crops each on GPU
   2, minutes per model.
4. **F3:** preprocessing variants on the top two.
5. **F4:** FG-2 and FG-4 (CPU).
6. **F5:** FG-7, judge agreement at make and model level on set C sheets.
7. **F6:** fetch set C-large and read FG-3 (one more ingest).
8. **F7:** FG-6 on the class-disjoint split.
9. **F8:** set P sizes and the local judge (after F5).

If F2 finds no challenger and F3 no variant, F4 to F8 still run on PE-Core with centre-crop. A failed
bar ends that branch, not the wave.

## 7. Faces (separate, identity level)

Faces use ArcFace, an identity-trained space, and run as E6 (LFW and a pose set):

- union-find at 0.7;
- Immich-style core points (0.5, 3);
- average and complete graph HAC;
- Leiden.

The same-image cannot-link is valid here. Bars are the plan's (BCubed precision >= 0.97 and the
rest of its row 8.2). The judge is the local VLM only. Nothing from the class-level or make/model
work transfers except the protocol.

## 8. Limits known before any run

- **Size.** Set C has 31 to 40 images per generation, too few for the loop gate. FG-3 needs set
  C-large.
- **Labels.** Set C labels come from Commons categories, and set C has no view labels. Commons photos
  are curated shots, mostly front three-quarter, so they are easier than set P street photos.
- **Viewpoints.** Linking the front and rear of one model is not measured without CompCars or a view
  sample checked by a person (plan 6.2).
- **Generation level** is reported only, with no bar: there are too few images per generation.
