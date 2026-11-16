# 16a: the cluster, label and train loop, the slices and the changes to plan 18b

| Field | Value |
|---|---|
| Status | plan, revision 1 (2026-11-16), companion of [16_label_train_loop_plan.md](16_label_train_loop_plan.md) (written "16" below: the design, the jobs, the measurements, the questions) |
| What a lane takes from here | its slice in section 3, in the template of [18a](18a_backend_rust_port_slices.md) section 1.1, with 18a's lane rules (section 1.2), workspace rules (section 2), test and parity rules (section 3) and fail-closed rules (section 4), and the gates of [18](18_backend_rust_port_plan.md) section 7 |
| What this file does not do | it does not edit 18b; section 4 lists the exact changes 18b needs, for the coordinator to apply |

## 1. Order, and why nothing is written twice

| Step | Plain name | Kind | Entry | Why here |
|---|---|---|---|---|
| L16-1 | the numeric library on the CPU | lane | now (R0w merged) | pure computation with no input or output; M12 and M16-3 need a Rust arm to measure; whatever R8a, R8b and 18b R10 compute is this code, so it is written once |
| M16-3 with M12 | fit, placement and head refresh at 100,000 and 1,000,000 objects | coordinator | L16-1 merged; M12's vector build | decides D5 for `cluster_run` (18 section 3.5) and whether L16-2 exists |
| M16-2 | label scopes on a real detector and set P | coordinator | none (runs on M16-0's harness) | fixes R8a's default scopes before R8a writes them |
| L16-2 | the GPU path | lane, conditional | M16-3 says the CPU misses at 1,000,000 and the GPU agrees | adds a backend behind L16-1's trait; L16-1 is not changed |
| R8a | groups per label scope on the record | 18b slice (refines R8) | 18b R8's entry (R6b and R7 merged, 15b M10, M12) plus M16-2 | the cluster family moves once, already per scope |
| M16-1 | how a person reviews a page | coordinator, with the owner | today's product (no new code) | decides R8b's default page source; can run any time before R8b |
| R8b | the head, suggestion pages, the loop's progress | 18b slice (new, after R8a) | R8a merged; M16-1 recorded | the head reads R8a's scopes and pages; built once on the record |
| R9, R10 | training inputs; machine helpers | 18b slices | as 18b, with section 4's changes | they read scopes and heads; nothing they need is rewritten |

## 2. Crates and layering

| Crate | Holds | Created by | May depend on (server crates) | Row to add |
|---|---|---|---|---|
| `op_numeric` (new) | spherical k-means, nearest-centroid assignment (top two), the class-mean head, out-of-fold isotonic calibration, per-class estimates; a backend trait for the matrix products with a CPU backend and, behind the cargo feature `gpu`, a GPU backend | L16-1 | none on CPU; `gpu_frame` under the feature `gpu` (L16-2) | 18a section 2.2's table and `CRATE_DEPENDENCIES` in `scripts/codegen/check_layering.py`: `'op_numeric': frozenset({'gpu_frame'})` |
| `op_store` (exists from R1b) | gains `clusters` (pool reads, partition writes, placement, activation per scope) in R8a and `heads` (training reads, head writes, serving pointer, pages) in R8b | 18b R1b | adds `op_numeric` | `'op_store': frozenset({'op_db', 'op_domain', 'op_numeric'})` |
| `gpu_frame` (plan 17's) | gains a `blas` module (L16-2): the cuBLAS matrix multiply through cudarc, which is `unsafe fn gemm` in cudarc 0.19.10, so it lives where every other `unsafe` block of the workspace lives | plan 17 | unchanged | none; plan 17's owner is asked (16 question C1) |

Why a new crate and not `op_domain`: `op_domain` holds the fold and other pure rules that the
oracle checks; numeric kernels with a GPU feature and a bench binary are a different concern
(split on concern, not size). Why not inside `op_store`: the bench binary and M12 must run the
computation without a database.

Job bodies are plain functions in `op_store::clusters` and `op_store::heads`; their `op_jobs`
registrations sit in `op_server`'s worker role, as for every family kind (the layering table lets
`op_server` depend on both).

Libraries, each against its alternative (versions from the crates.io records, 2026-11-16; licences
inside `native/deny.toml`'s allow list unless said):

| Need | Choice | Alternatives and why not | Documentation |
|---|---|---|---|
| Matrix products on the CPU | faer 0.24.4 (MIT, released 2026-06): "high performance for algebraic operations on medium/large matrices", a `rayon` parallel backend | matrixmultiply 0.3.11 is already in `native/Cargo.lock` but single-threaded unless its `threading` feature is on; ndarray 0.17.2 adds an array layer we do not need. The lane records one product's time with each at L16-1's first commit (no database, no GPU) and keeps the faster | https://docs.rs/faer/0.24.4/faer/ |
| k-means | our own spherical Lloyd's loop over the backend trait (about 200 lines, estimate) | linfa-clustering 0.8.1 (MIT or Apache-2.0, 2025-12-23): "parallelises the assignment step", "the update step ... single thread", Euclidean `L2Dist` by default, a modified algorithm "m_k-means", no centroid re-normalisation, so not the measured spherical k-means. faiss 0.13.0 bindings (2025-11-15): need a system `libfaiss_c.so` built from C++ (or the `static` feature with a C++ compiler and BLAS); the measured algorithm, but a C++ build in the server image and a binding a year without release. cuvs 26.10.0 (Apache-2.0): has `cluster::kmeans` (`fit`, `predict`, `cluster_cost`) on device memory through DLPack, linking a prebuilt `libcuvs_c` found through conda or a Python environment (`cuvs-sys`'s build reads `CONDA_PREFIX`, `LIBCUVS_USE_PYTHON`); not proven on this card or in our image; 18 decision 3 allows it only if proven | https://docs.rs/linfa-clustering/0.8.1/linfa_clustering/struct.KMeans.html ; https://docs.rs/faiss/0.13.0/faiss/ ; https://docs.rs/cuvs/26.10.0/cuvs/cluster/kmeans/index.html ; https://docs.rs/crate/cuvs-sys/26.10.0/source/build.rs |
| The algorithm's rules | faiss's: `spherical` re-normalises centroids each iteration; `max_points_per_centroid` 256 subsamples the training set; an empty cluster is re-seeded by splitting a large one (`Clustering.cpp`, `split_clusters`) | our loop follows these so its objective is comparable to the measured fits (M12's bar) | https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.h ; https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.cpp |
| Isotonic regression | linfa-linear 0.8.1 `IsotonicRegression` (MIT or Apache-2.0) | a pool-adjacent-violators loop of our own (about 40 lines) if linfa-linear's output differs from scikit-learn's on the recorded cases; the lane decides by the case replay, not by taste | https://docs.rs/linfa-linear/0.8.1/linfa_linear/ ; https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html |
| Logistic regression | not used | linfa-logistic 0.8.1 exists; the class mean is cheapest or tied (16 section 1.2) | https://docs.rs/linfa-logistic/0.8.1/linfa_logistic/ |
| Half floats | half 2.7.1 (already in `native/Cargo.lock`) for `halfvec` bytes | | https://docs.rs/half/2.7.1/half/ |
| Vectors in PostgreSQL | pgvector 0.4.2 crate (18 section 2.2, feature `halfvec`) for the reads and the centroid column | | https://docs.rs/pgvector/0.4.2/pgvector/ |
| GPU matrix multiply (L16-2 only) | cudarc 0.19.10's `cublas` feature (already a workspace dependency of `gpu_frame`, today without that feature): `CudaBlas`, the `Gemm` trait for f16, bf16, f32 and f64 | a matrix multiply written by hand as an NVRTC kernel would redo cuBLAS; a small NVRTC kernel through `gpu_frame`'s existing kernel loader takes the top two per row of the product | https://docs.rs/cudarc/0.19.10/cudarc/cublas/index.html ; https://docs.rs/crate/cudarc/0.19.10/features |
| Assignment for identity carry-over | pathfinding 4.16.0 `kuhn_munkres_min` (18 section 2.2) | | https://docs.rs/pathfinding/4.16.0/pathfinding/kuhn_munkres/fn.kuhn_munkres.html |

## 3. The slices

### L16-1. The numeric library on the CPU

| Row | Content |
|---|---|
| Goal | the groups and the suggestions a person sees are computed in Rust exactly as the measurement computed them |
| Measurement | needs none; this slice is the Rust arm that M12 and M16-3 measure |
| Entry | R0w merged (done) |
| Crate and files | new crate `native/op_numeric` (`#![forbid(unsafe_code)]`): `matrix` (row-major f32 blocks decoded from `halfvec` bytes, the `Products` backend trait, `CpuProducts`), `kmeans` (spherical Lloyd's loop; parameters k, iterations 50, seed, training subsample at 256 points per centroid, empty-cluster split; an injectable initial set for tests; `k_for(n, items_per_cluster)` = max(1, n // items_per_cluster), default 40), `assign` (top two centroids per row: index, score, margin; batched), `head` (class means by class **name**, top two classes, score, margin), `calibrate` (5 folds assigned by a hash of `item_key`, out-of-fold class-mean predictions, isotonic fit, per-class and per-scope estimates; refusals of 16 section 3.3), and the binary `op-numeric-bench` (reads a vector file of rows of 1,024 f16 values with a JSON sidecar naming n, d and the group per row; runs fits, placements and head refreshes for a size list; prints JSON rows with wall time, peak resident memory from `/proc/self/status`, thread count, CPU model). Shared files: `native/Cargo.toml` (one `members` line, its dependency lines), `scripts/codegen/check_layering.py` (one row), the new generator `scripts/codegen/numeric_cases.py` and its drift hook line in `.pre-commit-config.yaml` |
| Replaces | nothing yet: the Python `cluster_run` runs until R8a |
| Contract | `contracts/json/numeric_cases.json` (new, generated): small inputs (300 vectors of 16 dimensions, three scopes, given initial centroids) with faiss 1.15.1's spherical k-means labels and objective, numpy's class-mean proposals and scikit-learn 1.9.1's isotonic output (both pinned in `requirements-test.txt`), as 18a section 3.1's rule for logic moving to Rust |
| Tests | each seen failing first (the commit message says how): Lloyd's loop with the given initial centroids reproduces faiss's labels (ties aside) and objective within 1e-4 relative (red: an update without re-normalisation); `k_for` at n = 0, 1, 39, 40, 79, 80, 8,000 gives 1, 1, 1, 1, 1, 2, 200; rows permuted with their keys give the same labels after the key sort; a scope's fit does not change when another scope is added (seed per scope name); assignment's top two and margin equal a brute-force reference; class-mean proposals equal numpy's; isotonic output equals scikit-learn's within 1e-6; one class refuses with `need_two_classes`; under 100 labels or under 5 in a class the output is uncalibrated; class identity: renaming a class in the input changes only names, never indices in the output |
| Selective checks | cargo gates (`fmt`, `clippy -D warnings`, `doc`, `test`, `deny`), jscpd over `native/**/*.rs`, the generator's drift, the layering check, hooks on touched files |
| Owed to the full sweep | none in the sweep; the bench rows are M16-3's and M12's (coordinator, CPU hold) |
| Go bar | every case equal; M12's bar then decides D5 for `cluster_run` |
| Deletes | nothing |

### L16-2. The GPU path (only on M16-3's verdict)

| Row | Content |
|---|---|
| Goal | a library of a million objects gets its groups in minutes, not hours |
| Measurement | M16-3 and M12: the CPU fit misses the 1,000,000 bar, and a GPU fit agrees with the CPU fit (09c CL-9: purity within 0.01, adjusted Rand index at least 0.95, on real crop vectors) |
| Entry | L16-1 merged; the coordinator's and plan 17's yes on `gpu_frame` (16 question C1) |
| Crate and files | `gpu_frame::blas` (cuBLAS `gemm` for f16 inputs with f32 output, through cudarc's `cublas` feature; the one new `unsafe` call, documented); a top-two-per-row kernel compiled by `gpu_frame`'s NVRTC loader; `op_numeric` feature `gpu` with `GpuProducts`; the workspace's cudarc line gains `cublas` (coordinator, 18a section 2.3); a GPU worker service of the same `op-server` binary built with `gpu`, on a CUDA runtime base that ships cuBLAS, claiming only the GPU-backed kinds, with the card claim of the ported GPU arbiter (18b R3a). `cluster_partitions.backend = 'gpu'` and a passing `cluster_backend_validations` row as the start condition (02d section 2.2: "the `cluster_run` start refuses `backend = 'gpu'` without a passing row") |
| Replaces | nothing |
| Contract | none |
| Tests | CPU-only unit tests of the backend selection and the validation refusal (red: a GPU fit started without a passing row) |
| Selective checks | cargo gates with and without the feature (the feature builds without a card, since cudarc loads CUDA dynamically) |
| Owed to the full sweep | one GPU hold: the agreement rows (CL-9) on 1,000,000 real vectors and the fit times, writing the `cluster_backend_validations` row |
| Go bar | CL-9 met; the 1,000,000 bar of M16-3 met |
| Deletes | nothing |

### R8a. Groups per label scope on the record (refines 18b R8)

| Row | Content |
|---|---|
| Goal | the person's objects are grouped inside each label scope, groups stay on screen after labelling, and new photos join their groups without waiting for a refit |
| Measurement | 18b R8's (15b M10, M12) plus M16-2 recorded (it fixes the default scopes) |
| Entry | 18b R8's (R6b and R7 merged); L16-1 merged; D5 decided |
| Crate and files | `src/db/migrations/sql/0016_cluster_cutover.sql` with 16 section 3.10's scope columns, the activation key, the `centroid` column and the method vocabulary (the head tables come with R8b, in the same file if R8a has not merged, else in the next revision in chain order); `op_store::clusters` (pool read by binary `COPY` per scope, partition write by temporary-table `COPY` then `INSERT ... SELECT`, carry-over with `pathfinding`, summary fill before the flip, activation per scope with OCC, placement); job kinds `cluster_run` (per scope) and `cluster_place`; the `cluster_refresh` tick's decision per scope (16 section 4); `op_api` module `clusters` (the 8 cluster operations of 18b R8) with the `label_scope` filter; the cluster profile body (ported by R11) gains `label_scopes` and the scope detector; the served hint of 16 section 3.1 |
| Replaces | `src/services/clustering/` (`run.py`, `job.py`, `assign.py`, `activation.py`, `carry_over.py`, `partitions.py`, `refresh_schedule.py`, `projection.py`, `serving_cluster.py`), `src/services/curation/cluster_cards.py`, `cluster_facets.py`, `cluster_representatives.py`, `cluster_empty_state.py`, `src/services/curation/clustering/`, the cluster routers; job kinds `cluster_run` (Python) and `cluster_projection_sync`; the `cluster-worker` service if no kind is left on it |
| Contract | intended changes, each with its frontend edit: `GET /curation/projects/{project}/clusters` and `PUT /curation/projects/{project}/clusters/activation` take `label_scope`; `GET /curation/projects/{project}/label_scopes` (new): scopes with counts and the merge hint; the partition wire shows `label_scope`; method values lose `minibatch_kmeans` and `detector_class`; region box clusters (`GET /curation/projects/{project}/regions/clusters`) follow the one k-means path. Cropwright: the cluster screen opens on the scope list and filters by scope (`frontend/src/lib/` cluster API and page files, listed by the lane with `rg -ln "clusters" frontend/src`) |
| Tests | red first: a scope's fit never mixes scopes; a detection of a non-scope detector never sets a scope; labelled objects stay members after a refit; placement takes only the scope's serving centroids and leaves a scope without a partition unplaced; the first partition of a scope serves without a click (if owner question O3 says yes); the tick queues `cluster_place` for unplaced objects and a refit only past the placed share; activation with a stale revision is refused; no `claims` write exists in `op_store::clusters` (module privacy, compile-time) |
| Selective checks | 18a section 1.1's |
| Owed to the full sweep | differential cases of the cluster family against the reference stack (intended changes listed); `tests/live/test_live_clusters.py` and the cluster browser pages at desktop and 430 px; the `db` feature tests (partition write under row security, placement insert into a ready partition, two `cluster_place` jobs of one scope refused); 15b's cluster shapes at 1,000,000; a box-cluster quality row (BCubed precision and recall of the one k-means path against today's `MiniBatchKMeans` on recorded box vectors, within the two-run spread, 09c CL-3's metric) |
| Go bar | owed green; group list and group page inside plan 15's bars; the first groups of a 40,000-object library readable within 60 s of the quiet point (proposed; M16-3 sets the 1,000,000 value) |
| Deletes | the replaced modules by last importer (`rg -l` at entry), their tests with ledger rows (18a section 3.2) |

### R8b. The head, suggestion pages and the loop's progress (new 18b slice after R8a)

| Row | Content |
|---|---|
| Goal | after a few labels in a scope, the person confirms whole pages of suggested objects, works the least certain ones, and sees how far the scope is and how many decisions it took |
| Measurement | M16-1 recorded (it sets the default page source); M16-3's head rows (they set the refresh bound) |
| Entry | R8a merged; R5's label service and review tab table in Rust |
| Crate and files | the head tables of 16 section 3.10 (in `0016` or its successor, see R8a); `op_store::heads` (training read and scoring read in one `REPEATABLE READ READ ONLY` transaction with the claim watermark, head writes, the serving pointer with OCC and the watermark rule, retention of two heads, pages of 16 section 3.5, the scope progress of 16 section 3.7); job kind `head_fit`; the label job (`label_op`, R5) queues `head_fit` for the scopes its items fall in after it commits; the tick's backstop; the review tab table (R5) gains `least_certain`; the claim payload gains `via = 'page'`; the erasure catalogue (R5e) gains the head tables (a head that trained on an erased object is discarded) |
| Replaces | nothing in Python: the head is new. It supersedes the "check machine labels" sort plan of revision 0 (no Python counterpart was built) |
| Contract | new: `GET /curation/projects/{project}/label_scopes/{label_scope}/suggestions` (classes with proposed counts, estimated accuracy, support, head status or the refusal reason), `GET /curation/projects/{project}/label_scopes/{label_scope}/suggestions/page` (`class_name` as a query parameter, keyset cursor, 30 objects), the scope list gains progress fields; `GET /curation/projects/{project}/review/{tab}` serves `least_certain` with a `label_scope` filter; the cluster list gains the proposed class and its share per group; `PUT /curation/projects/{project}/crops/batch_label` accepts `page_source` (`group` or `suggestion`) with explicit ids only, recorded as `via = 'page'`. Cropwright: the suggestion pages (one class per page, deselect wrong ones, confirm), the least-certain queue, the scope progress, the proposed class on group pages; each edit names its file at entry |
| Tests | red first: the training read never contains a holdout or excluded object or an unresolved name; a proposal is never a claim and never changes `item_state` (the fold's oracle cases replayed unchanged); a head with an older watermark never replaces a newer one; a page shows no object labelled after the head's snapshot; a page act writes claims only for the ids sent; a retried `head_fit` with the same job leaves one set of rows; with one class the scope serves `need_two_classes`; the per-class estimate and support are on every suggestion page; class names, never indices, in proposals, pages and the payload |
| Selective checks | 18a section 1.1's; `npm run check` for the frontend edits |
| Owed to the full sweep | differential cases where a Python route is touched (the label and review routes); live tests of a scripted loop on `crop_cub` (label 2 classes, refresh, confirm a page, undo it, suggestions return); the browser pages at desktop and 430 px, viewed; the `db` feature tests (two concurrent `head_fit` of one scope, a label racing a refresh, erasure discarding a head); head refresh time at 40,000 objects within M16-3's bar |
| Go bar | owed green; suggestion pages inside plan 15's page bars; refresh within M16-3's bar; on `crop_cub` replayed through the routes by a scripted oracle, decisions per object at 98 % within the seed spread of results section 3's arm B (0.38) under the gate count, or of the glance count, whichever M16-1 made the default |
| Deletes | nothing in Python (the probe uncertainty queue stays: it reads a trained detector, a different thing) |

## 4. The changes 18b needs (the coordinator applies them; this file does not edit 18b)

| 18b place | Today's text | Change |
|---|---|---|
| Section 3, row R8 | "`cluster_run` per D5 (Rust GPU k-means or Python reading the record's vectors)"; "placement of new photos in the ingest commit as membership rows" | R8 becomes **R8a** and **R8b** (section 3 here). Entry adds L16-1 merged and M16-2 (R8a), M16-1 and M16-3's head rows (R8b). Kinds: `cluster_run` per label scope; new `cluster_place`, `head_fit`. Placement by the `cluster_place` job after commit, not in the ingest commit. `0016` carries 16 section 3.10 |
| Section 3, row R9 | entry "R8 merged" | entry "R8a merged" (training inputs need no head); export reads only claims and current state, never `head_proposals` |
| Section 3, row R10 | "`autolabel`, `assist_loop` ... in Rust" | `autolabel`'s own embed and IVF retrain step is deleted: it queues `cluster_run` for the scopes it works on. The assist loop's roots: label scopes or bands of the head's proposals instead of groups of about 40 objects (09e records that the VLM's 1:20 yield needs nodes of 700 to 1,100 members); chosen by 09e's replay AL3 before R10 codes. The VLM's selection skips objects whose head proposal is confident only if AL3 shows a saving |
| Section 1, R1c | (no placement text; R8 had it) | nothing to build in the ingest commit for clusters; R1c's commit group needs no prototypes |
| Section 1, R3a | ports the `cluster_refresh` tick | port today's schedule step (`refresh_schedule.step`) as it is; R8a adds the scope key and the queuing of `cluster_place` and `head_fit` to the same tick (an extension, not a rewrite) |
| Section 2, R5 | label acts and review tabs | the claim payload model leaves room for `via = 'page'` (added by R8b with its route); the review tab table is the place R8b adds `least_certain`; the label job exposes an after-commit hook R8b uses to queue `head_fit` |
| Section 2, R5e | the erasure catalogue | R8b adds the head tables and the "discard a head trained on an erased object" step; the catalogue completeness test covers them |
| Section 2, R11 | `cluster_profiles` 7 operations | the Rust profile body is the place R8a adds `label_scopes` and the scope detector |
| Section 7.2, job kinds | 44 kinds; `cluster_run` in R8 per D5 | 46: add `cluster_place` (R8a) and `head_fit` (R8b) |
| Section 6, row S6 | "`0016`, cluster summaries, placement" | add "per label scope; placement as a job; the head tables (plan 16)" |
| 18a section 2.2 | the crate table | add `op_numeric` (section 2 here); `op_store` may depend on it |

## 5. Mapping to plan 18b

| This plan | 18b slice | Relation |
|---|---|---|
| L16-1 | none (new library slice before stage C) | new; feeds M12 (18 section 8) and R8a, R8b, R10 |
| L16-2 | none | new, conditional |
| R8a | R8 | refines: scopes, placement as a job, one k-means path |
| R8b | after R8 | new slice |
| M16-5 | R9 | owed before R9 |
| The VLM helpers | R10 | changes listed in section 4 |
| Faces | R2b (`persons_cluster`) | unchanged; may use `op_numeric` |
