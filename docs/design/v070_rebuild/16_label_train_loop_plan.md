# 16: the cluster, label and train loop, on the record and the Rust backend

| Field | Value |
|---|---|
| Status | plan, revision 1 (2026-11-16). Replaces revision 0 (branch `lane/plan16`, written before the owner's Rust decision of 2026-11-15 and before the labelling measurement). Section 9 says what is kept from revision 0, what is dropped and why. Nothing here is built |
| Companion | [16a_label_train_loop_slices.md](16a_label_train_loop_slices.md): the slices in plan 18a's template, the exact changes 18b needs, the crate placement |
| Evidence | the labelling measurement "M16-0: what cuts the number of labelling decisions", finished 2026-11-16: gitignored `artifacts_local/speed_1113/cluster_rule/results.md` (written "results" below, with its section and table), tables in `out/`. It reports facts and conditions, no recommendation; this plan draws the conditions, and where a product choice remains it is a question in section 8 |
| Reads with | [14](14_roadmap.md) stage 4; [18](18_backend_rust_port_plan.md) sections 2.1, 2.3, 2.5, 8; [18a](18a_backend_rust_port_slices.md) sections 1.1, 2.2; [18b](18b_backend_rust_port_later_slices.md) stage C; [15](15_storage_plan.md), [15a](15a_storage_schema.md), [02d](02d_research_seam_schema.md) section 2.2 (the cluster tables); [09c](09c_research_clustering_protocol.md) (the cost model the measurement extends); [09e](09e_research_auto_label_assist.md); [13b](13b_crop_vectors_plan.md) (the crop vectors); [VISION_AND_GOALS.md](../../VISION_AND_GOALS.md) |
| Not in scope | face grouping, scenes and events, video (section 7); the region stage and the segmenter (plan 18b R7); the VLM helpers beyond the changes listed for 18b R10; the gallery's views |

## 0. The person's job, and what "better" means here

A person points OpenProcessor at their photos. The detector finds the objects (car, bird,
airplane). The person labels **their own** classes (car makes and models, bird species) by looking
at pages of similar object crops, and trains a model on those labels; the model then labels new
photos. This loop is the product's core job ([VISION_AND_GOALS.md](../../VISION_AND_GOALS.md),
"The user's job").

Two numbers measure it:

1. **Decisions per object** to reach a target label accuracy. A decision is one answer about one
   object (name a class, confirm, fix), or one glance at a page that confirms it (09c's cost model
   as results section 2 extends it).
2. **The wait per decision**: how long the person waits for the next page after an answer, and how
   long a label takes to change what the app suggests next.

What the person has today (the Python product, clusters only, k-means with K = min(512, n // 6),
`src/services/clustering/run.py:80-85`), against the best measured alternative, at 98 % final label
accuracy (results section 3, the three set tables; "glance" = the page-glance model V, section 1):

| Set (objects, classes) | Today, decisions per object | Best measured, decisions per object | Under the page glance, today / best |
|---|---|---|---|
| Stanford Cars (8,041, 196) | 0.51 | 0.26 (a classifier on the person's labels) | 0.20 / 0.20 |
| CUB-200 birds (11,788, 200) | 0.54 | 0.38 (the same) | 0.20 / 0.20 |
| FGVC-Aircraft (10,000, 100) | 0.76 | 0.54 (clusters, then the classifier) | 0.37 / 0.32 |
| Mixed library (40,000, 571) | 0.66 (clusters over everything, best K; the shipped K was not tabled at 98 % here) | 0.43 (one classifier per detector class) | 0.29 / 0.23 (clusters per detector class) |

"Better" in this plan therefore means: at the person's target accuracy, decisions per object at or
below the best row above for the way the person actually reviews (section 1.2 says why that is
still open), and no wait on any computation: a page read inside plan 15's page bars (p50 at most
50 ms, p99 at most 200 ms at 8 clients, [15](15_storage_plan.md) section 6), and the suggestions
reflecting a label within a bound measured by M16-3 (section 5).

## 1. What the measurement decides, and what it leaves open

### 1.1 Terms, plainly

| Plain name | What it is (results section 2) |
|---|---|
| Clusters only (arm A) | k-means groups; the person labels a group, splits or fixes it |
| The classifier, or "head" (arm B) | a classifier trained on the labels the person has given so far, proposing a class for every unlabelled object; the cheapest one measured is the **class mean** (each class is the mean of its labelled vectors; an object takes the nearest mean) |
| Clusters, then the head (arm C) | the first groups by clusters, then the head |
| Gate model | the person answers a sample of a group; when the sample proves purity, the unseen rest take the class (errors stay) |
| Page glance (model V) | the person sees every object on a page of 30 (Cropwright's review page size) and pays one decision per page plus one per wrong object; optimistic: it assumes every wrong object is spotted |
| Label scope | the objects one fit or one head works over: one detector class (car), several merged (car, truck, bus), or the whole library |

### 1.2 Conditions the numbers support

| Question | What the numbers say | Source |
|---|---|---|
| Clusters only or a head, if the person accepts unseen objects after a sample (gate)? | the head: 0.26 / 0.38 / 0.57 against 0.49 / 0.54 / 0.68 for the best cluster rule at 98 %, and the order holds on held-out seeds (0.23 / 0.28 / 0.45 against 0.48 / 0.54 / 0.68) | results 3, 4 |
| The same if every labelled object is seen on a page (glance)? | no gain: 0.21 / 0.23 / 0.32 for the head against 0.20 / 0.20 / 0.37 for clusters; clusters are cheapest below about 8,000 objects, the head ties or wins from about 8,000 | results 3, 5 |
| Which head? | the class mean is cheapest or tied on all three sets; logistic regression gives no gain (cars 0.44 against 0.26), kNN is in between; labelling the least certain object next beats any fixed order (0.57 against 0.81, cars at 3,000, 95 %) | results 4, "Classifier" |
| One fit and one head per detector class, or over everything? | per detector class helps when the library holds many detector classes (40,000 mixed: clusters 0.66 to 0.49, heads 0.65 to 0.43); it does nothing when one coarse class holds the fine classes (three fine sets: 0.57 / 0.59), and it costs a little when the detector splits one fine class across coarse classes (cars split car / truck / bus: 0.60 against 0.49) | results 3 (cars table), 6 |
| How many clusters? | flat from K = n // 20 to n // 160 (cars 0.79 to 0.83 at gate level 0.9, sd 0.01); small clusters are the expensive end: n // 10 is the worst (0.89 to 0.94), and today's K = 512 makes clusters of 14 at 8,041 objects and costs more than n // 40 at every size up to 3,072 objects | results 4 "Cluster rule", 5 |
| Small libraries (1,000 objects, 5 to 10 per class)? | nothing is cheap under the gate (clusters 0.90 to 0.92, heads 0.64 to 0.85); under the glance clusters are clearly cheapest (0.54 to 0.56 against 0.67 to 0.73); a head cannot propose a class it has not seen | results 4, 5 |
| The first hour: finding the classes | labelling in random order until every class has 5 examples costs 0.34 (cars), 0.21 (CUB), 0.12 (aircraft) decisions per object; "one per cluster, largest first" as the first labels beats random order on CUB and aircraft (random wins on cars, within the spread); the head finds classes only through objects the person is shown | results 7 "Cost of finding", 4 "Classifier" |
| Rare classes | classes under 10 objects end at 0.91 (head) and 0.92 (clusters then head) final accuracy against 0.97 for clusters only; classes of 10 or more are 0.97 to 0.99 for every arm | results 4 "Tail classes" |
| Can the head's own confidence be trusted? | the class mean's estimate is conservative (0.91 estimated against 0.96 to 0.97 true on the unlabelled part); but per proposed class the worst class still has 37 % to 70 % wrong above confidence 0.95 (a class with a handful of accepted objects) | results 7, calibration table |
| Is a fixed setting safe (the winner's curse)? | on the three fine sets the cost order holds on held-out seeds; on mixed pools one head is unstable (40,000: chosen at 0.64, 0.90 on held-out seeds; one seed stopped below 98 %); heads per detector class were stable (0.981 to 0.986 accuracy at 0.43); 41 of 296 held-out comparisons differ by more than 0.03; the cheapest cluster settings at 98 % used gate level 0.6 to 0.7, whose proof is weak (they reach 98 % because the clusters happen to be pure). With knobs a product can set without the truth (gate level = target), clusters cost 0.91 to 0.96 at 95 % (nothing saved) and the head 0.33 / 0.57 / 0.69 | results 3 (last two bullets), 4, 6 |
| Run time and memory at the sizes run | k-means K = 512: 2.3 s (8,041), 3.6 s (11,788), 18.8 s (29,829); per detector class 7.6 s (40,000); class-mean fit 0.14 to 0.71 s; out-of-fold calibration 0.8 to 4.2 s; kNN 1.9 to 45 s; peak memory of a job 0.5 GB (1,000) to 3.8 GB (40,000). One niced thread on a shared host: indicative, not a result | results 7 "Cost rows" |

### 1.3 What it leaves open (not measured, results section 9)

| Open | Why it matters to this design | Closed by |
|---|---|---|
| How a real person reviews a page: misses at a glance, seconds per page against seconds per single answer | the glance model and the gate model rank the arms differently (1.2 rows 1 and 2); the default page source and whether "accept unseen" is offered depend on it | M16-1 |
| A noisy labeller (09c's 10 % row), a person who stops early | the gate and any stop rule rest on the oracle | M16-4 (only if "accept unseen" is wanted, owner question O1) |
| A real detector on the mixed pool (the 92.1 % agreement was simulated) and set P | whether label scopes per detector class are the right default for real detector boxes (car / truck splits, misses) | M16-2 |
| 1,000,000 objects; fit and head refresh time on the product's code; whether K = n // 160 equals n // 40 at large n | where each computation runs at scale (CPU or GPU) and the scope's refresh bound | M16-3 (with plan 18's M12) |
| Logistic regression on the 29,829 and 40,000 pools | not needed: the class mean is cheapest or tied where both ran (1.2 row 3); not run again | not planned |
| Faces, video | other plans (section 7) | |

## 2. The loop as the person meets it

Each step names the mechanism of section 3 behind it.

1. **A new library.** The person ingests photos. When ingest goes quiet, the app fits groups inside
   each label scope (3.2) and serves them without a click (owner question O3). The labelling screen
   opens on the list of **label scopes** with their object counts ("car 6,064", "truck 1,918",
   "bird 11,788"), each with its served progress (3.7): objects, labelled, decisions so far, groups
   still unnamed.
2. **Finding the classes (the first hour).** In a scope the person starts on **unnamed groups**:
   one page per group, largest group first, its unlabelled members nearest the group's centre
   first. The person types a class name (a new class is created by name), deselects the outliers
   and confirms: one decision for the page, one per fix. This is the "one per cluster, largest
   first" start, which finds classes faster than random order on two of three sets (1.2), and the
   glance-cheapest path for small libraries.
3. **The head starts.** Once a scope holds labels of at least two classes, a head is fitted for it
   (3.3) and refreshed after every label act in the background. The person never waits for it.
4. **Labelling by suggestion.** Next to the unnamed groups, the scope offers **suggestion pages**:
   the unlabelled objects the head proposes as one class, most confident first, with the class's
   estimated accuracy and how many labels support it, and a **least certain** queue (the measured
   best order for the head, 1.2). A group's page also shows the head's proposed class, so naming a
   group becomes one key. Which page source the screen leads with is decided by M16-1 (section 5);
   until then both are served and the person chooses.
5. **Rare classes and new classes.** Unnamed groups stay available to the end: they are where
   classes the head has never seen are found (1.2 "Rare classes"). The scope shows how many groups
   have no labelled member.
6. **Review.** The six review queues of today (`src/services/curation/review_tabs.py:19-26`) stay,
   built by plan 18b R5; this plan adds the least-certain queue to them (3.6). Every label the
   person gives is a claim (15a section 3) and can be undone (3.8).
7. **Train.** When the person decides the labels are enough (the served estimates help, they do not
   decide), they freeze the test holdout (18b R6b), export and train; training, evaluation and
   promotion stay Python (18 section 2.3), the dataset is assembled by Rust (18b R9) by class name.
8. **The model labels new photos.** A promoted model serves as a detector or classifier profile; its
   classes reach new objects as machine opinions under the lock rule (15a section 4.2). New objects
   are placed in their scope's groups and get head suggestions within the refresh bound (3.4, 3.3),
   so the person's next session starts with suggestions on the new photos.

## 3. The mechanisms

### 3.1 Label scopes

| Point | Design | Why |
|---|---|---|
| What a scope is | a named set of detector class names; an object belongs to the scope that holds its **scope detector's** class name. Default: one scope per detector class name, created on first sight. The person may merge detector classes into one scope (for example car, truck, bus as "vehicles") or split them again | per detector class helps on mixed libraries and costs a little when the detector splits a fine class (1.2 row 4); the merge answers the second case without giving up the first (owner question O2) |
| The scope detector | the detector named in the project's cluster profile, default the ingest profile's primary detector. Detections of other analysers (a promoted fine-class model, open vocabulary) never set the scope | otherwise a promoted model that outputs "Ford F-150" as its detector class would create a scope per fine class after promotion |
| Objects without a detection of the scope detector | drawn or imported objects: scope `''` (the no-class scope), same rules | they are few (results section 1: 28 of 8,041 cars objects had no detector class) |
| Where scopes are stored | in the cluster profile's revisioned body (`src/services/config_store/`, ported to Rust by 18b R11): `label_scopes: [{name, detector_classes: [..]}]`; absent classes default to their own scope. A merge or split is a new profile revision, undone by reverting the revision | the config store already gives revisions, `ETag` and revert; no new table |
| Class identity | scope keys are detector class **names**, never a model's class index | rule of [VISION_AND_GOALS.md](../../VISION_AND_GOALS.md) |
| A served hint | when one class name has labels in two scopes ("Ford Ranger" in car and truck), the scope list says so and offers the merge | the car / truck split is the measured cost of per-class scopes (1.2 row 4); the backend serves the decision, the frontend renders it |

### 3.2 Groups (clusters) inside a scope

| Point | Design | Basis |
|---|---|---|
| Computation | spherical k-means (Lloyd's iterations on unit vectors, centroids re-normalised after each iteration), as the product fits today and as the measurement fitted it (faiss `Kmeans`, `spherical=True`, 50 iterations, `run.py:44-45`). Initial centroids by a seeded sample; seed `seed XOR crc32(scope name)`; input sorted by `item_key`, so arrival order never changes a fit | faiss `ClusteringParameters` (`spherical`: "normalize centroids after each iteration", `niter`, `seed`, `max_points_per_centroid` 256: [Clustering.h](https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.h)) |
| Cluster count | K = max(1, n // 40) per scope, n the scope's object count. Inside the measured flat region (n // 20 to n // 160), and n // 40 is exactly faiss's own floor of 39 points per centroid (`min_points_per_centroid`, same page). M16-3 checks whether n // 160 (4 times cheaper to fit) is equal at 100,000 objects; if so, scopes above a measured size use it | results 4, 5 |
| Pool | every object of the scope that is not excluded, **including labelled ones**, so a labelled group stays on screen with its labelled count after a refit (today labelled objects leave the pool and a fully labelled group vanishes, `src/services/curation/clustering/pool_snapshot.py:34-37`); holdout objects are in the pool and never in a label act | revision 0 section 2.6 |
| Data read | the scope's crop vectors of the project's primary item space (`pe_core_l14_crop224`, 1,024 dimensions, `halfvec`, 13b section 7, 15a section 6.1) with `item_key`, in **one** binary `COPY (SELECT ...) TO STDOUT` statement, so one snapshot (15a section 6.4; [COPY](https://www.postgresql.org/docs/18/sql-copy.html)). Measured read rate on the record so far: 69,630 rows per second (15a section 6.4) | |
| Where it runs | the Rust worker role (`op-server worker`), library crate `op_numeric` (16a section 2). CPU at the sizes measured. A GPU path only if M12 / M16-3 show the CPU fit misses its bar at 1,000,000 objects and the GPU fit agrees with the CPU fit (09c's CL-9 bar: purity within 0.01, adjusted Rand index at least 0.95) | 3.9 |
| Identities across refits | carried from the parent partition by member overlap and the Hungarian method, anchored (named) identities kept, as `src/services/clustering/carry_over.py` does; in Rust with `pathfinding`'s `kuhn_munkres_min` (18 section 2.2, graph and assignment row) | |
| Result storage | `cluster_partitions` (one per fit and scope), `partition_clusters` (with the centroid as a `halfvec` column, 3.10), `cluster_memberships`, written by `COPY` into an `ON COMMIT DROP` temporary table and one `INSERT ... SELECT` (15a section 1, row-level security refuses `COPY FROM`); then the partition's summaries are filled and the scope's serving pointer flips (`cluster_activations`, with OCC on `revision`, 02d section 2.2); no object row is rewritten (plan 15 section 13) | |
| When it runs | (a) the first fit of a scope after ingest goes quiet (today's rule: no new photo for two ticks, or 1,800 s, `src/services/clustering/refresh_schedule.py:70-99`); (b) on the person's request; (c) when the scope's objects placed since the last fit exceed a share of the pool (a profile setting, default 10 %: revision 0 recorded that a refit moves 24 to 46 % of earlier objects per added 10 %, 13 row T7, so a refit is never automatic for small growth) | |
| Region boxes | box partitions use the same fit with today's box rule K = max(8, round(n / 200)) (`run.py:80-85`), not measured for labelling; one k-means path, so `minibatch_kmeans` leaves (16a, R8a) | one implementation, no second path |

### 3.3 The head inside a scope

| Point | Design | Basis |
|---|---|---|
| Computation | the class mean: for each class name with live labels in the scope, the normalised mean of its labelled unit vectors; an unlabelled object's proposal is the class with the highest cosine, its score, the **margin** to the second class, and a calibrated confidence | the cheapest or tied head on all sets (1.2) |
| Calibration | isotonic regression of "proposal correct" on the margin, fitted on **out-of-fold** predictions of the labelled part only (5 folds), as the measurement did; served as the per-object confidence and summed into an estimated accuracy per proposed class and per scope | results 2 (arm B), 7; isotonic regression: `linfa-linear` 0.8.1 `IsotonicRegression` ([docs.rs](https://docs.rs/linfa-linear/0.8.1/linfa_linear/)), checked against scikit-learn's `IsotonicRegression` on recorded inputs ([scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html)) |
| Training set | the objects of the scope whose class is decided by a live human or validated import `class_set` (`item_state.class_validated`, resolved name, 15a section 4.3), **not excluded and not in the test holdout** | the holdout is the test split |
| Refusal (fail closed) | fewer than two classes: no head, served reason `need_two_classes`. Fewer than 5 labels in a class, or fewer than 100 labels in the scope: proposals are served without a confidence and marked `uncalibrated` | the measurement's cold start was 100 decisions (results 2); calibration without data would be a guess |
| Data read | labelled part: `(item_key, class name, vector)`; unlabelled part: `(item_key, vector)`; both from one binary `COPY` per part inside one `REPEATABLE READ READ ONLY` transaction, which also reads the claim watermark (the newest `claim_id` visible) ([SET TRANSACTION](https://www.postgresql.org/docs/18/sql-set-transaction.html)) | one snapshot for the whole fit |
| Where it runs | Rust CPU, worker role, `op_numeric`. The score matrix (objects x classes x 1,024) is 4.7e10 operations at 40,000 objects and 571 classes and 2e12 at 1,000,000 and 1,000 classes (arithmetic): seconds on the host's 24 cores; at 1,000,000 the vector read (about 14 s at the measured 69,630 rows per second) dominates, so a GPU does not pay for the head (estimate; M16-3 records it) | |
| Result storage | `head_models` (one row per fit: scope, space, watermark, digest of the training pairs, counts, calibration knots, metrics), `head_class_means`, `head_proposals` (one row per unlabelled object), and the serving pointer `head_serving` per scope (3.10). Machine results only: never a claim, never a `class_opinions` row, so **the fold is untouched** and plan 18b R1b's frozen oracle stays valid | a separate table keeps a proposal out of the effective class: a global threshold accept would let the worst class carry 37 % to 70 % wrong (1.2) |
| When it runs | after each label act that touched the scope (the label job queues it, 16a R8b), and by the periodic tick when the scope's newest class claim is newer than the serving head's watermark (backstop) | |
| New photos | a new object placed in a scope (3.4) is scored by the next head refresh; the tick queues one when placed objects lack a proposal | |

### 3.4 Placing new objects

New objects join the **serving** partition of their scope: the nearest centroid, with score and
margin, in a job (`cluster_place`) that takes every object of the scope not yet placed in its
serving partition, in batches (one matrix product per batch). It runs after ingest commits, queued
by the periodic tick (today's tick interval `OP_CLUSTER_REFRESH_INTERVAL_S`, default 120 s,
`src/config/cluster_profile.py:14-15`). It is **not** in the ingest commit, which plan 18b R8 had
planned: the ingest's API CPU per photo is plan 18b R1c's go bar, and a batched product per
thousand objects costs less than one per commit group. Membership rows are inserted into the ready
partition (02d's trigger refuses only updates and deletes of a ready partition's children). A
scope with no partition yet leaves its objects unplaced until its first fit.

### 3.5 Pages

| Page | Rows | Order | Read |
|---|---|---|---|
| Unnamed group | members of one cluster of the serving partition | unlabelled first, then by score to the centroid (descending), `item_key` | membership join to `item_state` (15b's cluster shapes) |
| Group list of a scope | clusters with size, labelled count per class, unlabelled count, the head's proposed class and its share among the unlabelled members | unnamed first, size descending | summary of the partition (filled before the flip, 15b section 6.7) plus one aggregate over `head_proposals` of the serving head |
| Suggestion page | unlabelled objects of the scope proposed as one class by the serving head | confidence descending, `item_key` | keyset over `head_proposals (head_id, class_name, confidence DESC, item_key)` joined to `item_state` for "still unlabelled, not excluded, not in the holdout" |
| Suggested classes of a scope | class name, proposed count, estimated accuracy, labelled support | estimated accuracy descending | from `head_models.metrics` of the serving head (written by the job: no aggregate at read time) |
| Least certain | unlabelled objects of the scope | margin ascending | keyset over `head_proposals (head_id, margin, item_key)` |

A page holds 30 objects (Cropwright's `REVIEW_PAGE_SIZE`, `frontend/src/lib/review/reviewQueueLoader.svelte.ts:35`).
Every label act from a page names **the objects shown**, by id, never a filter: what the person
labels is what the person saw (owner question O1 on the alternative).

### 3.6 Writes the person makes

| Act | Write | Lock rule |
|---|---|---|
| Name a group or confirm a suggestion page, minus the deselected | one `label_operations` row and one `class_set` claim per object shown, `via` `'page'` (new value, 16a R8b), chunks of 1,000 (15a section 3.5) | holdout objects are skipped and reported in the served dry run, as `PUT /curation/projects/{project}/crops/batch_label` does |
| Fix one object on a page | `class_set` `via` `'single'`, or `excluded` | |
| Merge or split scopes | a cluster profile revision; the affected scopes are refit | |
| Rename a class | a registry row edit (15a section 3.2: configuration, not a claim); proposals carry the old name until the next head refresh; a page resolves names through the registry and drops unresolved names (fail closed) and queues a refresh | |

Machine writers here (`cluster_run`, `cluster_place`, `head_fit`) write only result tables; claims
are written only through `op_store::human` (18a section 4, module privacy), so no machine write can
change a person's decision, by construction (15a section 4.4).

### 3.7 Progress the person sees, and the loop's own measurement

Per scope, written by `head_fit` into `head_models.metrics` and served: objects, labelled,
unlabelled with a proposal, estimated accuracy of the proposals (calibrated, conservative,
1.2), groups with no labelled member, and **decisions spent**: the count of label operations and
single labels whose claims touch the scope, divided by the labelled objects. This makes the
measurement's unit (decisions per object) visible in the product on the person's own data, set P
included, with no extra instrument.

### 3.8 Staying correct while the person keeps labelling, and undo

| Hazard | Answer |
|---|---|
| A head refresh races a label | the job's snapshot has a watermark; pages filter "still unlabelled" at read time from `item_state`, so a just-labelled object never shows as a suggestion. The serving pointer moves only to a head with a higher watermark (OCC on `head_serving.revision`), so a slow older job never replaces a newer head |
| Two refreshes of one scope | exclusive group `head_fit` per scope (18 section 2.5, `uq_jobs_one_active`); a refused queue attempt is covered by the tick |
| A crashed job | the job's `head_id` is derived from its job id; a retry deletes its own `building` rows first, then writes again; readers see only `ready` heads through the pointer |
| A refit during a person's session | the person's page keeps reading the partition it started on (the page cursor carries the partition id); the next page list shows the new partition; a label is a claim on objects, independent of groups |
| Many API processes and workers | all state is in PostgreSQL (pointers with OCC, jobs with leases and fencing tokens, results keyed by job); no in-process lock or cache is the truth. The only in-memory data is a job's own working copy of vectors and centroids |
| Undo a label | `POST /curation/projects/{project}/label_ops/{op_id}/undo` (18b R5): withdraws, the fold restores; the next head refresh drops the withdrawn labels |
| Undo a fit | `PUT /curation/projects/{project}/clusters/activation` back to the scope's earlier partition (kept by retention) |
| Undo a scope merge | revert the cluster profile revision; the old scopes' partitions re-serve if retained, else refit |
| A bad head | heads are rebuildable results: the next refresh replaces it; nothing to undo in the record |
| Erasure | `head_proposals` and memberships go with the item (cascade, or the catalogue's explicit delete for memberships, 15a section 8); a head whose training set held an erased object is discarded and refit (a mean is derived from the erased vector); the erasure catalogue lists the head tables (16a R8b) |

### 3.9 Where each computation runs, at 40,000 and at 1,000,000 objects

| Computation | 40,000 objects | 1,000,000 objects | Basis |
|---|---|---|---|
| Fit of one scope (k-means, K = n // 40, 50 iterations) | Rust CPU: about 4e12 operations, seconds to tens of seconds on the worker's cores (estimate); the measured faiss fits at these sizes took 2.3 to 18.8 s on one niced thread | about 2.6e15 operations for one scope of 1,000,000 (quadratic in scope size, since K grows with n): tens of minutes to over an hour on CPU, about 100 s on one A6000 through a matrix multiply at an effective 25 TFLOP/s (estimates); ten scopes of 100,000 cost a tenth. **GPU here if M12 / M16-3 confirm** | arithmetic 2 x n x K x d per iteration; results 7 |
| Placement of new objects | Rust CPU, a batch of 1,000 against 1,000 centroids: 2e9 operations, milliseconds | 10,000 new against 25,000 centroids: 5e11 operations, about a second (estimate) | arithmetic |
| Head fit, calibration, scoring | Rust CPU, about 3 to 7 s end to end with the write (estimate from results 7) | Rust CPU; the read of 1,000,000 vectors (about 14 s) and the write of up to 1,000,000 proposal rows dominate (estimate 30 to 90 s); M16-3 measures and sets the refresh bound | results 7; 15a 6.4 |
| Pages, group list, suggested classes | SQL on the record | SQL on the record; the class list is precomputed by the job | 15b shapes; plan 15 section 6 bars |
| "More like these" for a page | the vector search of 18b R6a (pgvector, exact under a small filter, [pgvector](https://github.com/pgvector/pgvector) "exact nearest neighbor search" by default) | the same, gate B's index | 15a section 6.4 |
| The 2-D map | Python `embedding_viz` (18 section 2.3), outside the loop's critical path | | |
| Training, evaluation, promotion | Python workers on the job tables (18 section 2.5) | | |

### 3.10 Schema additions (in revision `0016_cluster_cutover`, 18b R8)

Written against 15a and 02d; wipe and re-ingest, no data migration. Every new table has
`project_id` and the row-level policy of 02a section 1.4 ("RLS(02a)"); `PROJECT` as in 15a
section 1.

```sql
-- label scopes on the cluster tables (02d section 2.2)
ALTER TABLE cluster_partitions        ADD COLUMN label_scope text NOT NULL DEFAULT '';
ALTER TABLE cluster_activations       ADD COLUMN label_scope text NOT NULL DEFAULT '';
ALTER TABLE cluster_activation_events ADD COLUMN label_scope text NOT NULL DEFAULT '';
-- the activation's unique key becomes (project_id, target, level, embedding_field, label_scope)
ALTER TABLE cluster_activations DROP CONSTRAINT <the 02d unique key>,
  ADD CONSTRAINT uq_cluster_activations_scope UNIQUE NULLS NOT DISTINCT
      (project_id, target, level, embedding_field, label_scope);
-- the centroid as a vector column; prototypes jsonb keeps count and the medoid item key only
ALTER TABLE partition_clusters ADD COLUMN centroid halfvec NOT NULL;
-- method vocabulary: one k-means path ('ivf' = spherical k-means, today's wire value);
-- 'minibatch_kmeans' and the never-implemented 'detector_class' leave the CHECK and src/domain/clustering_vocab.py
-- analysers: kind 'head' added to ck_analysers_kind (15a section 2.3); one row per method version, not per fit

CREATE TABLE head_models (
  head_id         uuid PRIMARY KEY,                 -- derived from the job id (retry-safe)
  PROJECT,
  label_scope     text NOT NULL,
  space_id        uuid NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE,
  analyser_id     integer NOT NULL REFERENCES analysers(analyser_id),   -- kind 'head', name 'class_mean'
  claim_watermark bigint NOT NULL,                  -- newest claim_id the training read saw
  training_digest text NOT NULL,                    -- sha256 of the sorted (item_key, class name) pairs
  n_labelled integer NOT NULL, n_classes integer NOT NULL,
  calibration     jsonb NOT NULL,                   -- isotonic knots; '{}' when uncalibrated (3.3)
  metrics         jsonb NOT NULL,                   -- per class and per scope: estimates, counts, decisions spent (3.7)
  status          text NOT NULL CHECK (status IN ('building','ready','failed','discarded')),
  job_id          uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  created_at      timestamptz NOT NULL DEFAULT now(), finished_at timestamptz NULL
);
CREATE TABLE head_class_means (
  head_id uuid NOT NULL REFERENCES head_models(head_id) ON DELETE CASCADE, PROJECT,
  class_name text NOT NULL, class_id_hint integer NULL,   -- by name; the id is a hint only (15a section 3.3)
  n integer NOT NULL, mean halfvec NOT NULL,
  PRIMARY KEY (head_id, class_name)
);
CREATE TABLE head_proposals (
  head_id uuid NOT NULL REFERENCES head_models(head_id) ON DELETE CASCADE, PROJECT,
  item_key bigint NOT NULL REFERENCES items(item_key) ON DELETE CASCADE,
  class_name text NOT NULL, class_id_hint integer NULL, second_class text NULL,
  score real NOT NULL, margin real NOT NULL, confidence real NULL,   -- NULL when uncalibrated
  PRIMARY KEY (head_id, item_key)
);
CREATE INDEX ix_head_proposals_page   ON head_proposals (head_id, class_name, confidence DESC, item_key);
CREATE INDEX ix_head_proposals_margin ON head_proposals (head_id, margin, item_key);
CREATE TABLE head_serving (
  PROJECT, label_scope text NOT NULL,
  space_id uuid NOT NULL REFERENCES embedding_spaces(id) ON DELETE CASCADE,
  head_id  uuid NOT NULL REFERENCES head_models(head_id),
  revision bigint NOT NULL DEFAULT 1, updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, label_scope, space_id)
);
-- claim payload (15a section 3.3): class_set.via gains 'page'
```

Retention: a head is deleted (cascade to its rows) once a newer head of its scope serves; the
current and the previous ready head are kept. Sizes at 1,000,000 objects in one scope:
`head_proposals` about 60 bytes per row plus two indexes, about 0.2 GB per head (estimate, 15a's
tuple arithmetic).

## 4. Jobs

On the Rust runner's tables (18 section 2.5: claim by `FOR UPDATE SKIP LOCKED`, lease and
fencing token, exclusive groups by `uq_jobs_one_active`, pause and yield, `auto_resume`); Python
kinds through the claim loop. Times are the estimates of 3.9; M16-3 replaces them with numbers.

| Kind | Plain meaning | Runner | Exclusive group, scope | Pausable or sliced | Reads | Writes | 40,000 objects | 1,000,000 objects |
|---|---|---|---|---|---|---|---|---|
| `cluster_run` | fit the groups of one label scope | Rust (D5: Rust if M12 passes, 18 section 2.3; else Python reading and writing the same tables) | `cluster_run`, scope key = target, level, field and label scope | cancel between iterations; not sliced | scope pool by `COPY`; parent partition's memberships | partition, clusters, memberships, summaries, pointer flip | seconds to tens of seconds (CPU, estimate from results 7) | tens of minutes on CPU, about 2 minutes on a GPU, one scope of 1,000,000 (arithmetic) |
| `cluster_place` (new) | put new objects into their scope's serving groups | Rust | `cluster_place`, scope = label scope | no | unplaced objects' vectors; serving centroids | membership rows | under a second per 1,000 new objects | about a second per 10,000 new objects |
| `head_fit` (new) | refresh the classifier and its suggestions for one scope | Rust | `head_fit`, scope = label scope | no | labelled and unlabelled parts by `COPY`, one snapshot | head rows, proposals, pointer | 3 to 7 s | 30 to 90 s, read and write bound |
| `label_op`, `label_undo` | write or withdraw a label act's claims | Rust (18b R5) | as today | as today | | claims; after commit, queues `head_fit` for the touched scopes (16a R8b) | | |
| dataset export (18b R9) | assemble the training set by class name | Rust | R9's | | current state, one `REPEATABLE READ READ ONLY` read | files under `derived/` | M16-5 | M16-5 |
| `train`, `promote`, `model_export`, `bakeoff`, `probe` | train, gate and serve a model | Python, unchanged | as today (`src/config/job_kinds/gpu_train.py`, `promote.py`) | | | | not changed by this plan | |
| `embedding_viz` | the 2-D map | Python | as today | | | | | |
| `cluster_projection_sync` | rewrote every item for a flip | never ported (18b section 7.2) | | | | | | |

The periodic tick `cluster_refresh` (ported by 18b R3a) decides per scope: the first fit after a
quiet ingest; a refit when the placed share passes its setting; `cluster_place` when objects are
unplaced; `head_fit` when the newest class claim of a scope is newer than its head's watermark.

## 5. Measurements before code

Only those that decide an open choice. Each runs in the coordinator's hold under plan 17 section
5's evidence clause, raw files under `artifacts_local/`, set P as numbers only (its images, file
names and paths never enter a tracked file). Lists by the dataset registry's names (gitignored
`artifacts_local/datasets/REGISTRY.md`).

| Id | Question | Data | Method | Bar | What each outcome changes |
|---|---|---|---|---|---|
| M16-1 | How does a person review a page: how many wrong objects are missed at a glance, and how long does a page take against a single answer? | pages of 30 built from `cluster_eval` (Stanford Cars), `crop_cub`, `crop_aircraft` with known truth and 0 to 6 planted wrong objects per page; then pages from set P groups | on today's product (the Python reference, whose cluster and review pages exist): the owner reviews 60 pages and answers 60 single objects (about 45 minutes, owner question O4); the screen logs time per page and per answer and the objects deselected. A VLM stand-in on the same pages is recorded **separately** and never counted as the person. The simulator of M16-0 is then re-run with the measured miss rate and seconds | none to pass; it fixes the cost model | miss rate at most 2 % (a proposed threshold): the glance model holds, the screen leads with groups below the measured crossover size (about 8,000 objects per scope, results 5) and with suggestions above it. Miss rate above 2 %: rerun the arms with that rate; the cheaper arm in **seconds per object** at 98 % leads. Seconds per page against seconds per answer gives the wait budget per decision |
| M16-2 | Are label scopes per detector class right for a **real** detector? | COCO val2017 and `o365_200` objects through the product's detector (no simulated 92.1 %), the three fine sets with their real detector boxes; set P objects with make and model labels from the owner on a sample (or a stand-in labeller, counted separately) | M16-0's harness, arms A and B, per detector class against all objects against merged vehicle classes, 5 seeds, held-out seeds as in results 4 | per detector class kept as the default if its cost at 98 % (gate and glance) is not above the better of the other two by more than the seed spread on the mixed pool and on set P | kept: default as 3.1. Merged vehicles cheaper on set P beyond the spread: the default profile merges the detector's vehicle classes. Whole-library cheaper: scopes become opt-in |
| M16-3 | Where does the fit run at scale, and how fresh can suggestions be? (with plan 18's M12) | M12's shared build of 100,000 and 1,000,000 real crop vectors (18 section 8), with their real detector classes; decisions curve on a 100,000-object labelled subset (COCO train2017 crops plus the three fine sets) | `op_numeric`'s bench binary (16a L16-1) against faiss CPU and GPU: fit time per scope size, objective, agreement; head refresh end to end against PostgreSQL holding the vectors; placement of 10,000 new objects; K = n // 40 against n // 160 in decisions per object at 100,000 | M12's bar for the fit (objective within the two-run spread of faiss, time not above faiss GPU by more than the spread); proposed: one scope of 100,000 fits in at most 60 s, 1,000,000 in at most 10 minutes; head refresh at most 10 s at 40,000, at most 120 s at 1,000,000 | CPU meets the 1,000,000 bar: no GPU path (L16-2 not built). Misses and GPU agrees (CL-9): L16-2. n // 160 equal: large scopes use it. Head refresh misses: proposals written only where they changed, measured again |
| M16-4 | Only if the owner wants "accept unseen" (O1): does a stop rule hold with a noisy person? | the three fine sets and the mixed pools of M16-0 | M16-0's harness with 09c's 10 % noisy labeller and M16-1's miss rate; stop rules with per-class cross-validated accuracy against the global one | held-out accuracy at least the target on every set at the chosen setting, with held-out cost within 0.03 of the selection cost | passes: the accept-unseen act is built with that rule. Fails: not built; the served estimates stay information only |
| M16-5 | How long does the person wait from "train" to a served model? (13 row T6, never measured) | a public preset on a COCO subset of 10,000 labelled objects | time of export, trainer start to first epoch, evaluation, promotion, first request | proposed: first epoch within 60 s of the click on a free card; export within 60 s | owed before 18b R9; a miss is a lane in R9 or the trainer, with its number; changes nothing in this plan's design |

Not measured on purpose: logistic regression at the large pools (1.3), farthest-point starts
(never the best, results 9), pages grouped by cluster for the head (a count cannot show it,
results 9).

## 6. Slices

In [16a_label_train_loop_slices.md](16a_label_train_loop_slices.md): one new crate, two
library slices, two refinements of 18b R8, and the exact changes 18b needs. In order: L16-1
(numeric library, CPU, starts now), M16-3 with M12, L16-2 (GPU path, only on M16-3's verdict),
R8a (groups per label scope on the record), R8b (the head, suggestion pages, progress).

## 7. Deliberately not built now

| Not now | Why | Where it attaches later |
|---|---|---|
| Accepting labels for objects the person did not see (the gate) | the gate's cheap settings rest on purity, not proof (1.2); a noisy person is unmeasured | owner question O1; M16-4; a `via` value and a dry run on the label route |
| Logistic regression or kNN heads | no gain over the class mean where measured (1.2) | `op_numeric::head` takes another head behind the same trait if a measurement shows a gain |
| A GPU fit | not needed at the sizes measured; quadratic only at scale | L16-2 after M16-3 |
| Growth without any refit (place, then fit only the unplaced into new groups) | the refit-on-share rule of 3.2 covers new photos; faiss's `frozen_centroids` shows the pattern if needed | a measurement of moves and decisions, then a `cluster_run` mode |
| Face grouping | identity level, another bar (09c CL-6) | `persons_cluster` (18b R2b) and the photo-library people plan; `op_numeric`'s k-means and assignment serve it with target `face` |
| Video | planned once after the gallery (14 section 10) | objects from frames become items; scopes and heads apply unchanged |
| Several people labelling one project with their own suggestions | the claim log already records `actor_user_id` (15a section 2.2); one head per scope serves all people now | per-person heads would key `head_serving` by user; authentication first (18 section 6.1) |
| The VLM labelling the groups | plan 18b R10's family; its roots change (16a section 4) | R10 |

## 8. Risks, questions

### 8.1 Risks and their early signals

| Risk | Early signal | Answer |
|---|---|---|
| The glance model is optimistic | M16-1's miss rate; in production, fixes per page found later by review (claims withdrawn or changed after a page act) | M16-1's branch; the least-certain queue as the default |
| One head over a mixed scope is unstable (results 6) | the served estimate jumps between refreshes of one scope | scopes per detector class (stable in the measurement); the person merges only where M16-2 or the served hint says so |
| Rare classes stay unfound | unnamed groups remain late in a scope | unnamed groups stay served to the end (2, step 5) |
| A class's suggestions are mostly wrong while the scope's estimate looks fine | per class: the worst class had 37 % to 70 % wrong above 0.95 (results 7) | per-class estimates with their support on every suggestion page; no threshold accept |
| The fit grows quadratic in scope size | M16-3's 1,000,000 row | K = n // 160 if equal; GPU path; a person-made split of a huge scope |
| Writing 1,000,000 proposals per refresh is too slow | M16-3's head row | write changed rows only; refresh less often for that scope |
| cuBLAS is absent from the worker image | L16-2's first GPU run | a CUDA runtime base for the GPU worker (coordinator question C1) |
| Car and truck carry the same make and model | the served hint of 3.1 fires | the merge |
| Names drift (rename, merge) | unresolved names on pages | fail closed and refresh (3.6) |

### 8.2 Questions for the owner (product choices only)

| # | Question | Options | Recommendation |
|---|---|---|---|
| O1 | May a group label apply to objects the person did not see? | (a) never: every label is on a page the person saw; (b) offered after a sampled check, with the person's target accuracy, once M16-4 passes; (c) default on | (a) now, (b) only after M16-4. Under (a) the measured cost is the glance model's 0.20 to 0.37 decisions per object (1.2), and at a million objects the trained model, not the person, labels the rest (step 8) |
| O2 | Label scopes: per detector class, or something else by default? | (a) one scope per detector class, the person merges; (b) the whole library; (c) vehicles merged by default | (a), and revisit after M16-2 on set P (the car / truck split is the measured cost of (a)) |
| O3 | Should a library's first groups appear without a click? | (a) yes, the first fit of a scope serves at once (it moves nothing the person saw); (b) today's default, the person activates | (a) |
| O4 | Will you review 60 pages and 60 single objects (about 45 minutes) for M16-1, on public pages and then set P? | (a) yes; (b) a stand-in only (its numbers are not a person's) | (a): the review model is the one fact that decides the default page source |
| O5 | Suggestions on new photos before any detector is trained? | (a) yes, as suggestions only, never as labels; (b) only after a model is trained | (a): the head is cheap and the label rule is unchanged |

### 8.3 Questions for the coordinator

| # | Question | Recommendation |
|---|---|---|
| C1 | cuBLAS's `gemm` in cudarc 0.19.10 is an `unsafe fn` (`cudarc/src/cublas/safe/gemm.rs`), and server crates forbid unsafe code (18 section 7). Where does the GPU path live? | in `gpu_frame` (plan 17's crate, where every `unsafe` block already is) as a small `blas` module, with cudarc's `cublas` feature added to the workspace line; `op_numeric` calls it behind a `gpu` feature; a GPU worker on a CUDA runtime base. Only if L16-2 is built |
| C2 | Placement moves from the ingest commit (18b R8's note) to the `cluster_place` job | yes (3.4); 16a section 4 lists the 18b edit |
| C3 | The head tables and the scope columns ride in `0016_cluster_cutover` (R8's revision), not a new revision | yes; R8b adds the head tables in the same file if R8a has not merged, else `0016b` in chain order (coordinator's merge order) |
| C4 | 18 section 2.2's k-means row says "matrix-multiply kernels in our own GPU crate first"; a hand-written matrix multiply would redo cuBLAS | read it as "cuBLAS through cudarc, wrapped in our GPU crate, plus one small kernel for the top two per row"; the next revision of 18 says so |
| C5 | 13b line 102-103 says "clusters are computed within a detector class" (roadmap gap G1) | true from R8a in the form "within a label scope, by default one detector class"; the 13b owner edits the sentence when R8a merges |
| C6 | The 15b revision in review: this plan needs the read shapes of 3.5 (group list per scope with labelled counts and the head's proposed class; members of one group, unlabelled first; suggestion page by class and confidence; least certain by margin; scope list with progress) at plan 15's page and dashboard bars at 1,000,000 objects | measured in 15b's round that precedes R8 (18b R8's entry already names 15b M10) |
| C7 | M12's vector build must carry each vector's real detector class name, so M16-2 and M16-3 reuse it | add the column to the build |

## 9. What changed from revision 0

| Revision 0 | Now | Why |
|---|---|---|
| Lane L16-1: clusters within the detector class in the Python product, with a fit-scale runner | dropped; the rule is built once, in Rust (R8a) | no new Python product code (owner, 2026-11-15) |
| K_c = n_c // 80 per detector class, "0.707 against 0.971 decisions per object" | K = n // 40 per label scope | the earlier gain did not reproduce on the product's 224 vectors (results 8: no gain on cars, CUB or aircraft); the K sweep is flat, n // 40 sits inside it |
| The few-labels head as logistic regression (scikit-learn), a Celery job, results as a `class_opinions` source `head` | the class mean in Rust; its own result tables; the fold untouched | the class mean is cheapest or tied (results 4); a `class_opinions` source would change the fold and its frozen oracle and make a proposal the effective class |
| Head bar "0.5 x the per-class cluster loop on two of three sets" | the head is built; the default page source waits for M16-1 | the head halves the gate cost on cars (0.26 against 0.49) but ties under the glance (results 3) |
| Placement inside the ingest commit | the `cluster_place` job | R1c's CPU bar; batching |
| M16-1 fit time, M16-2 growth without refit, M16-3 head replay | M16-3 (with M12); growth without refit not planned; the head replay is done (M16-0) | |
| M16-4 SAM 3 cost per prompt, lane L16-6 | moved out: the region stage's plan (18b R7) | it does not change labelling decisions |
| M16-5 training timing | kept as M16-5, owed to R9 | |
| The S6 addendum to plan 15c | replaced by 16a section 4's changes to 18b R8 | plan 15c's Python slices are superseded (18 section 3.2) |
| Lock rule, class identity by name, fail closed, one wire serializer, claims and the fold, the trainer protocol | kept as written in revision 0 section 1 and in 15a | |
