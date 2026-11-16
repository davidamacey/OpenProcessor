# 16: the cluster, label and train loop, on the record and the Rust backend

| Field | Value |
|---|---|
| Status | plan, revision 2 (2026-11-16): revision 1 with its independent review resolved (1 blocker, 18 major, 12 minor; section 9). Nothing here is built |
| Companions | [16a](16a_label_train_loop_slices.md): the slices in plan 18a's template, the exact changes 18b and 18 need; [16b](16b_label_train_loop_schema.md): the DDL, the concurrency rules, erasure |
| Evidence | [labelling_decisions_m16_0.md](../../research/labelling_decisions_m16_0.md) (written "M16-0" below, with its section): the labelling measurement on public sets under two cost models, re-tabulated from the raw rows for this revision |
| Reads with | [14](14_roadmap.md) stage 4; [18](18_backend_rust_port_plan.md) sections 0, 2.3, 2.5, 3.5, 8; [18a](18a_backend_rust_port_slices.md) sections 1.1, 2.2; [18b](18b_backend_rust_port_later_slices.md) R3a, R5, R5e, R8 to R11; [15a](15a_storage_schema.md); [02d](02d_research_seam_schema.md) section 2.2; [09c](09c_research_clustering_protocol.md); [VISION_AND_GOALS.md](../../VISION_AND_GOALS.md) |
| Not in scope | face grouping, scenes and events, video; the region stage (18b R7); the gallery's views |

## 0. The person's job, and what "better" means

A person points OpenProcessor at their photos. The detector finds the objects (car, bird,
airplane). The person labels **their own** classes (car makes and models, bird species) by looking
at groups of similar object crops, trains a model on those labels, and the model labels new photos.

What matters, in this order:

1. **The labels and the trained model are right.** A design that looks cheaper but mislabels is
   worse.
2. **Decisions per object** to reach that: the person's answers divided by the objects in the
   library. It is the cost to label a pool, not the cost to train a good model (that relation is
   measured by M16-6, section 4).
3. **The wait between decisions**: no page waits on a job; suggestions reflect a label within a
   bound measured in labels, not ticks.

Two cost models bound the second number (M16-0 section 3), and every number in this plan names its
model:

| Model | Plain meaning | Bias |
|---|---|---|
| **Gate** | the person answers a sample of a group; when the sample proves purity, the unseen rest take the class, errors included | labels reach objects nobody saw; the cheapest cluster settings were picked with the truth (oracle-tuned) |
| **Glance, seen only** | the person sees every object on pages of 30 and pays one decision per page plus one per wrong object | optimistic: every wrong object is spotted, a page of 6 costs what a page of 30 costs |

Where the two models disagree, the choice waits for M16-1 (how a real person reviews a page) and
this plan says so. Part A (section 2) moves what the person has today; Part B (section 3) holds every
improvement, each behind a named measurement.

## 1. What the data says, and what it does not

All rows from M16-0, five seeds, public sets (Stanford Cars 8,041 objects and 196 models; CUB-200
11,788 and 200 species; FGVC-Aircraft 10,000 and 100 variants; mixed pools with COCO objects whose
detector class is **simulated**).

| Question | Gate model | Glance, seen only | Source |
|---|---|---|---|
| Clusters or a classifier on the person's labels (cars, CUB, aircraft)? | the class-mean classifier: 0.26 ± 0.04 / 0.38 ± 0.11 / 0.57 ± 0.11 against clusters 0.49 / 0.54 / 0.68 (oracle-tuned); held-out 0.23 / 0.28 / 0.43 against 0.48 / 0.54 / 0.68. Truth-free knobs at 98 %: classifier 0.56 / 0.82 / 0.84, clusters 0.98 to 1.00 | clusters (today's rule): 0.20 / 0.20 / 0.37; classifier 0.23 / 0.23 / 0.31. Clusters cheaper on two sets, classifier on aircraft | M16-0 section 4 |
| Does a library size flip the answer? | the classifier is cheaper at every size from 1,000 up on the fine sets | no: clusters cheapest or tied in 12 of 13 size rows (1,000 to 40,000); the classifier wins only aircraft 10,000 and ties the 29,829 pool. Revision 1's "crossover at about 8,000 objects" is withdrawn | M16-0 section 5 |
| Which classifier? | the class mean on all three sets | the class mean on all three (aircraft 0.31 against logistic regression 0.32; the earlier 0.46 was a reporting error, M16-0 section 9) | M16-0 sections 4, 9 |
| One fit per detector class? | helps on the mixed pools (40,000: 0.66 to 0.49), costs on cars (0.49 to 0.60), nothing on CUB and aircraft | the same direction (40,000: 0.29 to 0.23; cars 0.20 to 0.25) | M16-0 section 6 |
| ... and is that evidence for the person's case? | no: the mixed-pool gain rests on a simulated detector class that equals the truth 92.1 % of the time; the fine sets, which are the person's case, show none | the same | M16-0 sections 2, 6 |
| One classifier per detector class? | helps only at 40,000 (0.69 to 0.43) | costs on every pool (0.27 to 0.34; 0.48 to 0.69; 0.33 to 0.41) | M16-0 section 6 |
| How many clusters? | flat from n // 40 to n // 160; clusters of 10 the worst | clusters of 6 to 20 the cheapest; n // 40 costs more at every size (cars 0.32 against 0.20) | M16-0 section 7 |
| Can the classifier's confidence be trusted? | the global estimate is conservative (0.91 against 0.96 to 0.97), under random-then-least-certain labelling; the worst proposed class still has 37 % to 70 % wrong above 0.95 | n/a | M16-0 section 8 |
| Finding the classes | random order until every class has 5 examples: 0.34 (cars), 0.21 (CUB), 0.12 (aircraft); one object per cluster as a start beats random on CUB and aircraft | n/a | M16-0 section 8 |

Not measured (M16-0 section 10): how a real person reviews a page; a noisy labeller; a real
detector on a mixed library; pools above 40,000 in decisions; how labels turn into a trained model's
accuracy; a holdout drawn from biased labels; calibration under group-naming labels; a stale
classifier; refits during a session.

What follows from the table: **no improvement in this plan rests on one model where the other
disagrees.** The classifier's gain exists under the gate (labels on unseen objects) and largely
vanishes when every label is seen; the group-count rule flips between the models; per-class scopes
show a gain only on simulated data. Part B gates each on a measurement that can tell them apart.

## 2. Part A: parity, today's loop on the record and in Rust

Part A is plan 18b's R5 (labelling), R8 (clustering) and R10 (machine helpers) refined. Its
promise: **the person meets the same loop**. "Same" means the same screens and fields, the same
rules and defaults, groups of the same quality (equal algorithm and parameters, results inside the
run-to-run spread of today's library), and the same refusals. What changes is underneath: claims,
the fold, membership rows, Rust, PostgreSQL. Two correctness additions are made (rows marked
**added**) because the record must not lose a person's decision; both are listed as intended
changes with their frontend edit.

### 2.1 What the person has today (read in code, 2026-11-16)

| Area | Today | File |
|---|---|---|
| Groups | spherical k-means (faiss, 50 iterations, seed 42), K = min(512, n // 6), trained on a seeded sample of at most 50,000; region boxes by scikit-learn `MiniBatchKMeans`, K = min(max(8, round(n / 200)), n) | `src/services/clustering/run.py:39-132` |
| Pool | embedded objects that are not validated, not excluded, and whose class source is not a confident one (classifier, VLM, human, import, propagated); boxes proposed or accepted | `src/services/curation/clustering/pool_snapshot.py:30-66`, `src/domain/ingest_class_sources.py:90-103` |
| Identities across refits | member overlap (Jaccard) and the Hungarian method; anchored pinning exists but nothing writes an anchor | `src/services/clustering/carry_over.py` |
| Serving | profile activation `human` by default: a fit serves only after `PUT /curation/projects/{project}/clusters/activation`; `auto_when_safe` serves a fit with stability at least 0.95 and no anchored or validated moves | `src/services/clustering/activation.py:27-61` |
| Refresh | the `cluster_refresh` tick every 120 s (`OP_CLUSTER_REFRESH_INTERVAL_S`): a fit when 200 objects were added and ingest is quiet, or after 1,800 s of deferral; items only; never while a run or an assist loop is live | `src/services/clustering/refresh_schedule.py:314-343`, `src/config/cluster_profile.py:14-44` |
| New photos | placed at ingest on the serving partition's nearest prototype, with score and margin, marked `incremental` | `src/services/clustering/assign.py:165-196`, `src/services/curation/ingest_index.py:176-230` |
| The group list | cards of size, validated and labelled counts, dominant class, label agreement, representatives, detector top class, review counts, flags; 24 cards per screen; eight sorts; facets; three empty states | `src/services/curation/cluster_cards.py`, `cluster_facets.py`, `cluster_empty_state.py`; `frontend/src/lib/clusters/clusterListController.svelte.ts:46` |
| Naming a group | "Label all N as ..." labels every member by filter (up to 20,000) after a dry-run count; selected members by id; Z undoes the operation | `frontend/src/lib/components/clusters/ClusterScopeHeader.svelte:143-152`, `src/routers/curation/crops_write.py:108-158` |
| Review | six tabs (to label, check machine labels, disagreements, new class proposals, audit sample, regions), pages of 30 | `src/services/curation/review_tabs.py:19-45`, `frontend/src/lib/review/reviewQueueLoader.svelte.ts:35` |
| Audit | a stratified sample of machine-labelled objects per detector class for the person to label | `src/services/curation/audit.py`, `src/domain/audit_math.py` |
| Test holdout | a deterministic 20 % per class (at least a floor) of **validated** objects, frozen | `src/services/curation/holdout.py:233-275` |
| Machine helpers | auto-label (manual; VLM over unvalidated objects, or every unvalidated member of one cluster); assist runs (suggestions, applied by the person); assist loop (manual; cluster-sampled roots of at least 40, a VLM sample per node, a statistical gate, propagation as unvalidated machine labels, rollback) | `src/services/curation/auto_label_run.py`, `assist/`, `assist_loop/` |
| Classes | rename, merge (refused with frozen holdout members), deprecate (refused while referenced) | `src/routers/curation/classes_write.py`, `classes_merge.py` |

### 2.2 The same loop on the record

| Area | Same for the person | What changes underneath |
|---|---|---|
| Groups | the same algorithm, K rule, iterations, seed and training-sample cap; "same" = objective within the two-run spread of faiss and agreement with faiss (adjusted Rand index) not below that of two faiss seeds, on 100,000 and 1,000,000 real crop vectors (M12, 18 section 8) | Rust CPU in `op_numeric` (16a L16-1); no Python fallback for any clustering kind (section 2.4) |
| Region boxes | groups of the same quality | the same spherical k-means at today's box K rule, accepted when BCubed precision and recall on recorded box vectors are within the two-run spread of today's `MiniBatchKMeans` (09c CL-3's metric); if not, `op_numeric` gains mini-batch k-means (scikit-learn's algorithm, [MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)) in the same slice. Nothing in Part B touches boxes, so neither is rewritten |
| Pool, identities, serving, refresh | identical rules and defaults (the anchored pinning has no writer and is not ported) | the pool read is one binary `COPY` per fit (15a section 6.4); the activation is a pointer flip with OCC (02d); `cluster_projection_sync` is never ported (18b 7.2), so an activation serves at once instead of after a projection job (intended change, frontend edit in `clusterRunController.svelte.ts`) |
| New photos | placed in the serving groups within seconds of ingest | membership rows `assigned_by = 'incremental'` written by the `cluster_place` job, queued by the ingest commit after it commits (the tick is the backstop); the ingest commit (18b R1c) carries no numeric code |
| The group list and pages | every field and sort of today | sizes and labelled counts are read-time aggregates over the serving partition's memberships joined to `item_state` (16b 1.1): a partition's own summaries stay immutable as 02d requires |
| Naming a group, labels, undo | the same acts and routes (18b R5) | claims and withdraws (15a section 3); **added**: what the page or the dry run showed is the act's precondition, so a stale tab or a second person never overwrites a newer human decision (16b 3.1; intended change: `class_version` on the item wire, `selection_token` from the dry run) |
| Review, audit, holdout | the same tabs, audit and freeze | holdout and audit are claims (`holdout_frozen`, `audit_sampled`) |
| Machine helpers | the same (18b R10) | opinions with `op_id` (15a 2.3); the lock rule by data shape (15a 4.4) |
| Classes | the same refusals | registry rows; names joined at read; every machine-made reference resolved by 15a's one function (16b 3.4) |
| Erasure | erasing a photo removes what was derived | memberships deleted, the centroid of every cluster that held the photo recomputed from the remaining members, the medoid key no longer stored (16b 1.3) |

### 2.3 Seams Part A carries, and those it does not

| Seam Part B needs | Carried by Part A? | Why |
|---|---|---|
| A label-scope key on partitions and activations | no; B4 adds it by migration (16b 2.2) | an unused column would be dead code and its default would claim a scope design the data does not support yet |
| A place for the classifier's proposals | no; B5 adds its tables (16b 2.3) | results tables are new rows, not a change to Part A's |
| The centroid as a column | yes (16b 1.1) | today's placement needs it; Part B reads it unchanged |
| The precondition on label acts | yes | correctness today; Part B's suggestion pages use it unchanged |
| A whole-fit interface in `op_numeric` that a GPU backend can implement | yes (section 2.4) | Part B's group-count rule may need a GPU; Part A's call sites never change |

### 2.4 Where Part A's computation runs, and the numeric library's seam

Today's rule bounds the fit: at most 512 clusters, trained on at most 50,000 objects. Arithmetic: 2
x 50,000 x 512 x 1,024 operations per iteration, 50 iterations, about 2.6e12 operations per fit at
any library size; assigning 1,000,000 objects to 512 centroids is about 1.1e12. That is seconds on
the worker's CPU cores (estimate), so **Part A runs on the CPU in Rust**; the decision that M12
settles for Part A is CPU or GPU inside Rust (bar in 16a section 4), never Rust or Python (owner's
rule: Python only for `probe`, `train`, `model_export`, `bakeoff`, `promote`, `embedding_viz`).

The seam: `op_numeric`'s backend interface is **one whole fit** and **one assignment batch** (host
vectors in, host centroids and labels out), not one matrix product. A CPU backend does the products
on the host; a GPU backend uploads the vectors once per call and keeps vectors and centroids on the
card for all iterations. A per-product interface would force a transfer per iteration and a rewrite
(review major 16); a whole-fit interface does not. The cost it leaves is one upload per call (2 GB
for 1,000,000 x 1,024 halves, a fraction of a second at PCIe rates), stated in L16-2.

## 3. Part B: improvements, each behind a measurement, in order

Each item says what changes for the person, the measurement and its bar, what of Part A it reuses
unchanged, and what it adds. Nothing starts before Part A's slice it builds on has merged.

| Order | Improvement | Measurement and bar | Reuses unchanged | Adds |
|---|---|---|---|---|
| B1 | **A random check sample and an unbiased test set.** The person labels a small uniform sample of a library's objects (labelled or not, seen through the audit tab); the app reports how accurate the group labels and the suggestions really are, per class with counts, and can freeze that sample as the test set | M16-6: does a test set drawn from labels made group-first or least-certain-first misjudge the trained model's accuracy? Bar: if the difference to a random test set exceeds 0.02 on any of the three fine sets (or its interval is wider than 0.02), B1 is built and owner question O5 sets the default | the audit tab, `audit_start`, `holdout_frozen`, the review page | a payload field (16b 2.1); the sample draw; the accuracy report |
| B2 | **A group-count rule for large libraries.** Today's rule makes 512 groups of 58 at 29,829 objects and 78 at 40,000 | candidate fixed in advance: K = max(min(512, n // 6), n // 20), equal to today up to 10,259 objects. Data so far: glance 0.27 against 0.39 (29,829) and 0.29 against 0.45 (40,000); gate 0.85 against 0.80 and 0.88 against 0.84 (fixed gate level 0.9): **the models disagree**. Bar: M16-1's measured cost model, replayed at 29,829, 40,000 and on M16-3's 100,000, puts the candidate at or below today's rule beyond the seed spread, in seconds per object | Part A's fit, carry-over, activation, pages | a profile default; L16-2 (GPU) only if M16-3 shows the CPU fit of n // 20 misses its bar at 1,000,000 |
| B3 | **Groups keep their labelled members.** Today a labelled object leaves the pool at the next refit, so a named group shrinks or vanishes | none can show it in decisions; it changes what a group shows by default, so it is owner question O2 | Part A's pages (unlabelled first is a sort) | a pool filter change |
| B4 | **Label scopes**: groups inside one detector class (or a set the person merges, such as car, truck, bus) | M16-2: the three fine sets in one library and COCO objects **through the product's detector** (no simulated agreement), set P on a sample. Bar: per-detector-class groups cost less than one fit over the library by more than the seed spread under both models on the real-detector mixed library, and not more than the spread on the fine sets and set P. Fails: not built | Part A's fit per scope, placement, activation | scopes and the scope key (16b 2.2), the scope list screen, the served merge hint |
| B5 | **Suggestions from a classifier on the person's labels**: pages of objects proposed as one class, a least-certain queue, novel objects (furthest from every class) | M16-1 first: the measured miss rate and seconds per page and per answer, put into the M16-0 replay with every label seen. Bar: the class-mean classifier's pages cost fewer **seconds per object** than today's groups on at least two of the three fine sets beyond the seed spread, and its labels' error stays under the target. If the owner chooses labels on unseen suggestions (O1 b), M16-4 must pass and the gate model applies. Fails both: not built. Before entry also: M16-6 (does least-certain labelling help or hurt the trained model per label?) and M16-3's head rows | Part A's labelling acts with their precondition, the review tab table, the audit (B1) as the honest check of the suggestions | `op_numeric::head` and calibration; the head tables and the training revision (16b 2.3); `head_fit`; suggestion pages; tabs `least_certain` and `novel` |
| B6 | **Growth without constant refits.** Today a refit runs after every 200 new objects once ingest is quiet, and moves objects the person was working on | M16-7: replay a library growing in ten steps; bar: refitting only when the placed share passes a setting gives decisions per object within the spread of refitting each step, with fewer objects moved between groups the person already saw | Part A's placement and tick | a refresh setting |

B5's details that the review asked for, fixed now so the slice has no open choice:

- **Which classifier**: the class mean (each class the normalised mean of its labelled vectors; a
  proposal is the nearest mean, its score and the margin to the second). It is the cheapest head in
  both models on all three sets (section 1).
- **Over what**: one classifier per project and item space, over the whole library: per-scope heads
  cost more under the glance on every pool measured (section 1). M16-2 may change this; the head
  tables then get a scope key by drop and create (16b 2.3).
- **When suggestions appear**: none until a measured minimum of labels and classes (the replay in
  M16-1 sets it; the simulation's cold start was 100 decisions; the proposed default is 100 labels
  over at least 5 classes with 5 labels each). Below it the screen says how many more are needed.
- **Its confidence**: per proposed class, the out-of-fold precision of that class on the labelled
  part with its count and a Wilson interval (as `src/domain/audit_math.py:78` computes for the
  audit), never a sum of one global calibration; when a random check sample (B1) exists, its
  per-class precision is shown beside it as the unbiased one. M16-1's replay checks the estimate on
  a classifier trained from group-named labels (typical objects first).
- **Freshness**: no tick in the path; a label queues a refit and a refit that ends behind the
  labels queues the next (16b 3.3). The bound in labels comes from M16-3.
- **A proposal is never a label**: proposals live in their own tables; the fold never reads them;
  a test proves no wire serializer maps a proposal into `class_name` (minor 6).
- **A page confirm** is a `class_set` with `via = 'batch'` and `label_source = 'human_confirmed'`
  (an existing value), so training and the audit can tell a glance confirm from a typed label.

## 4. Measurements before code

Each runs in the coordinator's hold under plan 17 section 5's evidence clause, raw files under
`artifacts_local/`, a tracked summary under `docs/research/`, set P as numbers only. Every bar
names its cost model.

| Id | Question | Data | Method | Bar | Changes |
|---|---|---|---|---|---|
| M12 (plan 18) | Part A's fit in Rust at scale | 100,000 and 1,000,000 real crop vectors (18 section 8), with their real detector class names (C7) | `op-numeric-bench` against faiss CPU with today's parameters (K = min(512, n // 6), 50,000-object sample, 50 iterations); assignment of every object | objective within faiss's two-run spread; adjusted Rand index to faiss not below two faiss seeds' minus their spread; fit at most 60 s and assignment at most 60 s at 1,000,000 on the worker's CPU (proposed) | Part A's go row; a miss is a lane in L16-1 (threads, f32 products), not a GPU |
| M16-1 | How does a real person review a page? | pages of 30 built from Cars, CUB, Aircraft with known truth and 0 to 6 planted wrong objects; pages of 6, 15 and 30; then pages of set P groups | **on today's product** (Part A's screens are today's: the cluster detail grid and the review page; B5's suggestion page uses the same page component, so the result carries over): the owner reviews 60 pages and answers 60 single objects (O4); time per page and answer, objects deselected. A VLM stand-in is recorded separately and never counted as the person. Then the M16-0 replay is rerun with the measured miss rate and seconds, every label seen | none to pass: it replaces both models with a measured one | sets B2's and B5's bars in seconds per object; the minimum before suggestions; the calibration check under group-named labels |
| M16-2 | Do label scopes pay with a real detector? | the three fine sets and COCO train2017 objects through the product's detector, as one library; set P with owner labels on a sample (a stand-in counted separately) | M16-0's harness, arms A and B, per detector class against one fit, both models, 5 seeds, held-out seeds | B4's bar | B4 built or not; whether the head gets a scope key |
| M16-3 | Fit, placement and the classifier at scale; freshness | M12's vectors; a 100,000-object labelled subset (COCO train2017 crops plus the three fine sets) | `op-numeric-bench` for B2's candidate K at 100,000 and 1,000,000, CPU and (if built) GPU; decisions per object at 100,000 for today's rule and B2's candidate (both models, then M16-1's); `head_fit` end to end against PostgreSQL holding the vectors; a replay where suggestions lag by the measured refit time | B2's candidate fit at 1,000,000 in at most 10 minutes on CPU, else L16-2 (GPU) with 09c CL-9's agreement (purity within 0.01, adjusted Rand index at least 0.95); `head_fit` at most 10 s at 40,000 and 120 s at 1,000,000; the lagged replay within the spread of the fresh one | B2's backend; B5's freshness bound in labels; "write changed proposals only" if the head misses |
| M16-4 | Only if O1 (b): can unseen suggestions be accepted safely by a person who errs? | the three fine sets and the mixed pools of M16-0 | 09c's 10 % noisy labeller and M16-1's miss rate; stop rules with per-class cross-validated accuracy | held-out accuracy at least the target on every set, held-out cost within 0.03 of the selection cost | the accept-unseen act for suggestions is built or not |
| M16-5 | How long from "train" to a served model? | a public preset on a COCO subset of 10,000 labelled objects | export, trainer start to first epoch, evaluation, promotion, first request | proposed: export within 60 s, first epoch within 60 s of the click on a free card | owed before 18b R9; a miss is a lane in R9 or the trainer |
| M16-6 | How do label count, labelling order and label errors turn into the trained model's accuracy, and is a test set from those labels honest? | the three fine sets; labels drawn group-first, least-certain-first and at random, at 1, 2, 5, 10, 20 % of objects; label noise 0, 2, 5 % | a linear probe on the crop vectors for every cell (CPU), plus the product's trainer on one fine set at three label counts (one GPU hold); each model scored on a random held-out test set and on a test set drawn from its own labels | none to pass: it gives the learning curve; B1's bar reads the test-set difference; B5's entry reads whether least-certain order costs model accuracy per label | B1's default; B5's queue order; the served "enough labels?" hint |
| M16-7 | Refit on growth or place first? | the three fine sets and the mixed 40,000 pool, ingested in ten steps | M16-0's harness with placement between refits; refit every step against refit past a placed share of 10, 20, 40 % | B6's bar | B6 built or not |

Not measured on purpose: logistic regression or kNN on the large pools (the class mean is cheapest
in both models where all three ran); farthest-point starts (never the best where run,
M16-0 section 8); pages grouped by cluster for the classifier (a count cannot show it).

## 5. Jobs

On the Rust runner (18 section 2.5); times are the estimates of section 2.4 until M12 and M16-3.

| Kind | Plain meaning | Part | Exclusive group, scope | Reads | Writes |
|---|---|---|---|---|---|
| `cluster_run` | fit the groups (items or boxes) | A | `cluster_run`, target, level, field (B4 adds the scope) | the pool by one binary `COPY`; the parent partition's memberships | partition, clusters with centroids, memberships, pointer (auto-activation as the profile says) |
| `cluster_place` (new) | put new objects into the serving groups | A | `cluster_place`, target and field | unplaced objects (anti-join on the membership key) and their vectors; serving centroids | membership rows `incremental` |
| `label_op`, `label_undo`, `class_merge` | label acts, undo, merge | A (18b R5) | as today | claims, the selection | claims, state; B5: after commit, queue `head_fit` |
| `autolabel`, `assist_loop`, `assist_run` | machine helpers | A (18b R10) | as today | as today | opinions with `op_id` |
| `head_fit` (new) | refresh the classifier and its suggestions | B5 | `head_fit`, project and space | labelled and unlabelled parts and the training revision, one snapshot | head rows, proposals, pointer; a follow-up when behind (16b 3.3) |
| `train`, `promote`, `model_export`, `bakeoff`, `probe`, `embedding_viz` | training and the 2-D map | Python, unchanged | as today | | |
| `cluster_projection_sync` | rewrote every item for a flip | never ported (18b 7.2) | | | |

The `cluster_refresh` tick (ported by 18b R3a as it is) keeps today's rule and, from R8, queues the
Rust `cluster_run` and backstops `cluster_place`; with B5 it backstops `head_fit` (16b 3.3).

## 6. Slices, schema

The slices, in order, with 18a's template, and the exact changes 18b and 18 need:
[16a](16a_label_train_loop_slices.md). The DDL, the concurrency rules, the class and erasure
matrices: [16b](16b_label_train_loop_schema.md).

## 7. Deliberately not built now

| Not now | Why | Where it attaches |
|---|---|---|
| Accepting suggestions for objects nobody saw | the gate's cheap settings were picked with the truth; a person who errs is unmeasured; owner question O1 | M16-4, then a `via` value and a dry run on the suggestion act |
| Logistic regression or kNN heads | the class mean is cheapest in both models (M16-0 section 4) | `op_numeric::head` takes another head if a measurement shows a gain |
| A GPU fit | Part A's fit is bounded (section 2.4) | L16-2, only for B2 at 1,000,000 |
| Per-scope classifiers | cost more under the glance on every pool | M16-2 |
| Several people with their own suggestions | one classifier per project serves all; the claim log has `actor_user_id` | after authentication (18 section 6.1) |
| Face grouping, video | other plans | `op_numeric`'s fit serves them with target `face` |

## 8. Risks and questions

### 8.1 Risks and early signals

| Risk | Early signal | Answer |
|---|---|---|
| The glance is optimistic: group labels carry errors nobody sees | B1's random sample disagrees with group labels above the target | M16-1; B1 reports it per class; O1 |
| A class's suggestions are mostly wrong while the global estimate looks fine | per-class out-of-fold precision low with a wide interval; B1 disagrees | per-class estimates with counts on every page; no threshold accept |
| The first hour: rare classes are found late | groups with no labelled member remain late; the novel queue keeps finding new names | B5's novel queue; today's "unlabeled" sort stays |
| Two people label one group | `changed_since_shown` counts in act answers | the precondition (16b 3.1) |
| The trigger's row lock serialises label commits of one project | label p99 under two sessions in R5's `db` tier | one statement per chunk; a per-space counter split if the two-session test misses |
| The fit at 1,000,000 with B2's rule is slow | M16-3's row | L16-2; or keep today's capped rule |
| cuBLAS absent from the worker image | L16-2's first GPU run | a CUDA runtime base for the GPU worker (C1) |

### 8.2 Questions for the owner (product choices; batched)

| # | Question | Options | Recommendation |
|---|---|---|---|
| O1 | Labels on objects the person did not see. Today "Label all N as ..." labels a whole group by filter after a count, without showing every member | (a) keep it as today, with the new rule that only the objects counted in the dry run are labelled; (b) group labels apply only to objects shown on pages; (c) as (a), and later also accept classifier suggestions unseen once M16-4 passes | (a) now: it is what the person has, and B1's random sample will show its real error per class; decide (b) or (c) from M16-1 and B1's numbers |
| O2 | After labelling, should a group keep its labelled members (shown after the unlabelled), or shrink at the next refit as today? | (a) today; (b) keep them | (b): the person sees each group's progress and a named group does not vanish; no measurement can rank this, and it changes what a group shows, so it is your call |
| O3 | Should a project's **first** groups appear without a click? Later fits keep today's rule | (a) yes; (b) today: the person activates every fit | (a): a first fit moves nothing the person saw |
| O4 | Will you review 60 pages and 60 single objects (about 45 minutes) for M16-1, public pages first, then set P? | (a) yes; (b) a stand-in only (its numbers are not a person's) | (a): the review model decides B2 and B5 |
| O5 | The test set: frozen from your labels as today, or from a random sample you label (B1)? | (a) today's; (b) the random sample by default, its size about 20 objects per class; (c) offered, not default | decided by M16-6: (b) if your labels' test set misjudges the model by more than 0.02, else (c) |
| O6 | If B5 is built: suggestions on new photos before any model is trained? | (a) yes, as suggestions only; (b) only after a model is trained | (a): they never become labels without the person |

### 8.3 Questions for the coordinator

| # | Question | Recommendation |
|---|---|---|
| C1 | cuBLAS `gemm` in cudarc 0.19.10 is an `unsafe fn` (verified on docs.rs, `cublas/safe/trait.Gemm.html`) and f16 needs cudarc's `f16` feature; server crates forbid unsafe code | only if L16-2 is built: a `blas` module in `gpu_frame` (where the workspace's `unsafe` lives), cudarc's `cublas` and `f16` features added to the workspace line |
| C2 | Placement leaves the ingest commit for `cluster_place` queued after commit | yes; 16a section 4 lists the 18b edit |
| C3 | The selection tables ride in R5's `0013_claims_cutover`; Part A's cluster DDL in R8's `0016`; each Part B revision gets the next number in chain order when its gate passes | yes (16b) |
| C4 | Plan 18 section 2.3 and D5 still say "else Python" for `cluster_run` | apply 16a section 5's text in the next revision of 18 |
| C5 | 13b lines 102-103 say clusters are computed within a detector class | not true in Part A; true only if B4 passes; the 13b owner edits the sentence then |
| C6 | 15b's read shapes: group list with read-time labelled counts at 1,000,000; members of one group; B5's suggestion page and least-certain queue | measured in 15b's round before R8 (R8's entry names 15b M10) |
| C7 | M12's vector build must carry each vector's real detector class name | add the column to the build (M16-2, M16-3 reuse it) |
| C8 | The M16-0 harness is gitignored | commit it under `scripts/research/` with its README so a clean checkout reruns M16-0 and M16-1's replay |

## 9. Review trace (review of revision 1, 2026-11-16)

| Finding | Now |
|---|---|
| B1 Python fallback for `cluster_run` | removed; Part A is Rust CPU by arithmetic (2.4); M12 decides CPU or GPU inside Rust; 18's text change in 16a section 5 |
| M1 headline numbers from the gate model | every number names its model; section 1 has both side by side; O1 no longer quotes a lower bound as "the measured cost" |
| M2 class mean "cheapest or tied" | re-tabulated: cheapest in both models on all three sets once the aircraft reporting error is fixed (0.31, not 0.46); the mixed-pool one-head cells were kNN and are now separate (M16-0 sections 4, 6, 9) |
| M3 crossover at about 8,000 | withdrawn; M16-0 section 5 has the full curve; no rule rests on it |
| M4 scopes default from a circular simulation | no scope in Part A; B4 gated by M16-2 with a real detector; the classifier has no scope key (B5) |
| M5 calibration under another labelling order | per-class out-of-fold precision with counts; B1's random sample as the unbiased check; M16-1's replay checks group-named labels |
| M6 no trained-model target | M16-6 (learning curve, order, noise, test-set bias); B1; section 0 says what decisions per object is |
| M7 the first hour | today's unlabeled sort stays (Part A); B5 adds the novel queue and refuses suggestions below a measured minimum; group-page cold start measured in M16-1's replay |
| M8 stale suggestions | follow-up refit from the finishing job, no tick in the path; bound in labels; autovacuum on proposals; lagged replay in M16-3 (16b 3.3) |
| M9 claim-id watermark | replaced by counter rows updated in the changing transaction; documented PostgreSQL behaviour cited (16b 3.2) |
| M10 rename, merge, delete, undo | one matrix for every machine-made reference; the training trigger catches every change (16b 3.4, 2.3) |
| M11 two tabs, two people | precondition by `class_version` and the dry-run selection (16b 3.1), in Part A |
| M12 immutable partition summaries | summaries describe the fit; live counts at read time; placed rows marked at insert (16b 1.1) |
| M13 object-to-scope mapping | resolved inside the reading statement by a view; scope identity by id; scope revision recorded and checked (16b 2.2) |
| M14 schema exactness | DDL written out with constraint names, composite keys, RLS, FK indexes, NULL ordering, autovacuum, the analyser CHECK (16b) |
| M15 erasure of derived data | centroids recomputed, medoid key dropped, heads deleted per space, catalogue rows (16b 1.3, 3.5) |
| M16 numeric trait | whole-fit seam; head code in its own B5 slice (2.4, 16a) |
| M17 slice size | split by concern: storage and jobs, routes and frontend, deletions (16a) |
| M18 evidence not readable | `docs/research/labelling_decisions_m16_0.md` |
| Minor 1 shipped K on mixed pools | in M16-0 section 6 (0.71 gate, 0.45 glance at 40,000) |
| Minor 2 the shipped rule described wrongly | section 2.1: min(512, n // 6), with the box rule separate |
| Minor 3 "flat" K region | M16-0 section 7, both models; it is flat only under the gate |
| Minor 4 rare classes | stated as too few objects to carry a rule (M16-0 section 8) |
| Minor 5 anchored identities | not ported: nothing writes an anchor today |
| Minor 6 proposal never the class | a wire test (B5) |
| Minor 7 page confirm's source | `via = 'batch'`, `label_source = 'human_confirmed'`, no DDL |
| Minor 8 holdout on pages | hidden as today, counted in the hidden counts |
| Minor 9 "decisions spent" unit | dropped from the served progress; the product shows labelled, unlabelled and B1's accuracy, not a per-object cost it cannot know |
| Minor 10 M16-1 page format | runs on today's screens; B5 reuses the page component, condition stated |
| Minor 11 arithmetic of 3.9 | stands (labelled estimates); Part A's bound is in 2.4 |
| Minor 12 crate claims | verified on docs.rs (16a section 2) |
