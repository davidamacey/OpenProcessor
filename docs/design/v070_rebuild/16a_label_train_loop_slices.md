# 16a: the cluster, label and train loop, the slices and the changes to plans 18b and 18

| Field | Value |
|---|---|
| Status | plan, revision 2 (2026-11-16), companion of [16](16_label_train_loop_plan.md) (the person's loop, the data, Part A and Part B) and [16b](16b_label_train_loop_schema.md) (DDL and concurrency) |
| What a lane takes from here | its slice in section 3, in the template of [18a](18a_backend_rust_port_slices.md) section 1.1, with 18a's lane rules (1.2), workspace rules (2), test and parity rules (3) and fail-closed rules (4), and the gates of [18](18_backend_rust_port_plan.md) section 7 |
| What this file does not do | edit 18, 18a or 18b: sections 4 and 5 list the exact changes, for the coordinator |

## 1. Order, and why nothing is written twice

| Step | Plain name | Kind | Entry | Why here |
|---|---|---|---|---|
| L16-1 | the numeric library on the CPU | lane | now (R0w merged) | pure code; M12 measures it; Part A's fit, placement and carry-over call it; its interface is the whole-fit seam (16 section 2.4), so a GPU backend later adds code and changes none |
| M12 | Part A's fit in Rust at 100,000 and 1,000,000 | coordinator | L16-1 merged; the shared vector build | R8-1's entry: CPU or GPU inside Rust (never Python) |
| R5 (18b, with section 4's additions) | labelling acts with the precondition | 18b slice | as 18b | the precondition is part of the label acts; Part B's pages use it unchanged |
| R8-1 | groups on the record: storage and jobs | 18b slice (R8 split) | R6b, R7 merged; 15b M10; M12; L16-1 | Part A's clustering, written once |
| R8-2 | the cluster routes and screens on Rust | 18b slice (R8 split) | R8-1 merged | one concern: HTTP, contract, frontend edits, Python deletions |
| R10 (18b) | machine helpers | 18b slice | as 18b | unchanged by this plan |
| M16-1, M16-6, M16-2, M16-3, M16-7 | the measurements of 16 section 4 | coordinator (M16-1 with the owner) | M16-1 and M16-6 any time (today's product, public sets); M16-2 and M16-3 on M12's vector build | each gates one Part B item |
| B1 | random check sample | lane | M16-6 bar met; R5 merged | first: it gives later gates a real-data check |
| B2 | group-count rule | lane | M16-1 and M16-3 bars met; R8-1 merged | a default and a rule function |
| L16-2 | GPU fit backend | lane, conditional | B2 passed and M16-3 shows the CPU misses at 1,000,000 | adds a backend behind L16-1's interface |
| B3 | groups keep labelled members | lane | owner O2 = (b); R8-2 merged | a pool filter and a sort |
| B4-1, B4-2 | label scopes: storage and jobs; routes and screens | lanes | M16-2 bar met; R8-2 merged | after B2 and B3, so the per-scope fit reuses their rules |
| B5-1, B5-2, B5-3 | the classifier: numeric; storage and job; routes and screens | lanes | M16-1 bar met (or O1 (c) and M16-4); M16-6 and M16-3 recorded; R8-2 and B1 merged | B1's sample is its honest check |
| B6 | growth without constant refits | lane | M16-7 bar met; R8-1 merged | a tick setting |

Part B items whose gate fails are not built and their rows here are deleted in the same change that
records the failure.

## 2. Crates and libraries

| Crate | Holds | Created by | May depend on (server crates) | Row to add |
|---|---|---|---|---|
| `op_numeric` (new) | `matrix` (f32 row blocks decoded from `halfvec`), `kmeans` (spherical Lloyd's fit, the whole-fit `ClusterBackend` interface and `CpuBackend`), `assign` (top two per row), `carry` (member overlap and the Hungarian method), `k_rules`; B5-1 adds `head` and `calibrate` | L16-1 | none; `gpu_frame` only under the feature `gpu` (L16-2) | 18a section 2.2's table; `CRATE_DEPENDENCIES` in `scripts/codegen/check_layering.py`: `'op_numeric': frozenset()`, and `'op_numeric': frozenset({'gpu_frame'})` only in L16-2, with a check that `gpu_frame` is used only under `cfg(feature = "gpu")` |
| `op_store` (from R1b) | gains `clusters` (R8-1), `heads` (B5-2) | 18b R1b | adds `op_numeric` | `'op_store': frozenset({'op_db', 'op_domain', 'op_numeric'})` |
| `gpu_frame` (plan 17's) | gains `blas` only in L16-2 | plan 17 | unchanged | none (question C1 of 16) |

Why a crate of its own: `op_domain` holds the fold and pure rules the oracle checks; numeric kernels
with a bench binary are a different concern, and M12 runs them without a database. Job bodies are
functions in `op_store::clusters` and `op_store::heads`, registered in `op_server`'s worker role.

Libraries, each read on its documentation page on 2026-11-16 (versions from crates.io):

| Need | Choice | What the documentation says, and alternatives | Page |
|---|---|---|---|
| CPU matrix products | faer 0.24.4 (MIT, 2026-06-24) | `rayon` is a default feature and "enables global parallelism by default"; f32 and f64 only, so `halfvec` rows are converted to f32 on read (16b 1.1 keeps centroids in f32). matrixmultiply 0.3.11 (already in `native/Cargo.lock`) is single-threaded unless its `threading` feature is on; the lane times one product with each at its first commit and keeps the faster | https://docs.rs/faer/0.24.4/faer/ ; https://docs.rs/crate/faer/0.24.4/features |
| k-means | our own spherical Lloyd's loop (about 200 lines, estimate), following faiss's rules: re-normalise centroids each iteration, split a large cluster into an empty one, today's training sample cap of 50,000 (`run.py:107-112`) and `max_points_per_centroid` = max(256, sample // k + 1) | linfa-clustering 0.8.1 `KMeans`: `KMeans::params` uses `L2Dist`; a custom distance is possible but the centroid update stays a plain mean, no re-normalisation, so not today's algorithm. faiss crate 0.13.0 (2025-11-15): exposes `ClusteringParameters::set_spherical`, but binds faiss's C API 1.7.2 only and needs `libfaiss_c` (or a C++ build with the `static` feature). cuvs 26.10.0 (2026-10-08): `cluster::kmeans` `fit`, `predict`, `cluster_cost` on device memory; its params take a `DistanceType` (`CosineExpanded`, `InnerProduct` exist) but whether k-means honours a non-L2 metric is not documented; `cuvs-sys` links a prebuilt `libcuvs` found by CMake (`LIBCUVS_USE_PYTHON` or `CMAKE_PREFIX_PATH`), so the image would need libcuvs and CMake | https://docs.rs/linfa-clustering/0.8.1/linfa_clustering/struct.KMeans.html ; https://docs.rs/faiss/0.13.0/faiss/cluster/struct.ClusteringParameters.html ; https://docs.rs/crate/faiss/0.13.0 ; https://docs.rs/crate/cuvs/26.10.0/source/src/cluster/kmeans/mod.rs ; https://docs.rs/crate/cuvs-sys/26.10.0/source/cmake.rs ; https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.h |
| Hungarian method | pathfinding 4.16.0 `kuhn_munkres_min` | weights `C: Bounded + Sum + Zero + Signed + Ord + Copy`, so overlaps are integer counts (exact, no scaling); O(n³); checked against SciPy's `linear_sum_assignment` on recorded inputs | https://docs.rs/pathfinding/4.16.0/pathfinding/kuhn_munkres/fn.kuhn_munkres_min.html |
| Isotonic regression (B5-1) | linfa-linear 0.8.1 `IsotonicRegression` (MIT or Apache-2.0) | pool adjacent violators, features of shape (n, 1); checked against scikit-learn's on recorded cases; a 40-line loop of our own only if they differ | https://docs.rs/linfa-linear/0.8.1/linfa_linear/struct.IsotonicRegression.html |
| Vectors in PostgreSQL | pgvector 0.4.2 with features `postgres` and `halfvec` | no default features; `HalfVector` (`half::f16`) behind `halfvec`, `Vector` for f32 centroids | https://docs.rs/crate/pgvector/0.4.2/features ; https://docs.rs/pgvector/0.4.2/pgvector/struct.HalfVector.html |
| Half floats | half 2.7.1 (in `native/Cargo.lock`) | | https://docs.rs/half/2.7.1/half/ |
| GPU products (L16-2 only) | cudarc 0.19.10, features `cublas` (and the f16 support the `Gemm` page puts behind "the `f16` crate feature") | `CudaBlas` handle; `Gemm` implemented for f32, f64, and f16 and bf16 behind that feature; `gemm` is `unsafe fn` ("improper arguments may lead to invalid memory accesses"). The workspace pins cudarc with `dynamic-loading`, which needs no CUDA library at build time | https://docs.rs/cudarc/0.19.10/cudarc/cublas/safe/trait.Gemm.html ; https://docs.rs/crate/cudarc/0.19.10/features |

Owed checks (who): the exact name of cudarc's f16 feature on its features page and a `cargo check`
with `cublas` on a host without a GPU (L16-2's lane, at entry); whether cuvs k-means honours an
inner-product metric (only if L16-2 considers cuvs; the lane reads the cuVS C API page or tests it).

## 3. The slices

### L16-1. The numeric library on the CPU

| Row | Content |
|---|---|
| Goal | the person's groups are computed in Rust exactly as today's product computes them |
| Measurement | needs none; it is the Rust arm M12 measures |
| Entry | R0w merged (done) |
| Crate and files | new crate `native/op_numeric` (`#![forbid(unsafe_code)]`): `matrix`; `kmeans` with `trait ClusterBackend { fn fit(&self, x: VectorsRef<'_>, p: &FitParams) -> Result<Fit, NumericError>; fn assign(&self, x: VectorsRef<'_>, c: &Centroids) -> Result<Assignments, NumericError>; }` and `CpuBackend` (whole calls only: no per-product method is public); `FitParams` = k, iterations 50, seed (default 42), sample cap 50,000, `max_points_per_centroid` rule, an injectable initial set for tests; `assign` (top two: index, cosine score, margin; batched); `carry` (member overlap counts, `kuhn_munkres_min`, today's stability figure, `carry_over.py` without the anchored branch nobody writes); `k_rules` (`items_today(n) = max(1, min(512, n / 6))`, `boxes_today(n) = min(max(8, round(n / 200)), n)`); binary `op-numeric-bench` (reads rows of f16 values with a JSON sidecar; runs fits and assignments for a size list; prints JSON rows: wall time, peak resident memory from `/proc/self/status`, threads, CPU model). Shared files: `native/Cargo.toml` (members line, dependency lines), `scripts/codegen/check_layering.py` (one row), the generator `scripts/codegen/numeric_cases.py` and its drift hook line in `.pre-commit-config.yaml` |
| Replaces | nothing yet (the Python `cluster_run` runs until R8-1) |
| Contract | `contracts/json/numeric_cases.json` (new, generated): 300 vectors of 16 dimensions with given initial centroids and faiss 1.15.1's spherical Lloyd's labels and objective; scikit-learn's `MiniBatchKMeans` box groups on a recorded set (for R8-1's box bar); SciPy's `linear_sum_assignment` on overlap matrices; brute-force top two (generator pins in `requirements-test.txt`) |
| Tests | each seen failing first: Lloyd's loop from given centroids reproduces faiss's labels (ties aside) and objective within 1e-4 relative (red: an update without re-normalisation); `items_today` at n = 0, 1, 5, 6, 3,071, 3,072, 3,078, 100,000 gives 1, 1, 1, 1, 511, 512, 512, 512; `boxes_today` at 1, 7, 1,700, 1,900, 10,000 gives 1, 7, 8, 10, 50 (Python's `round` takes ties to even: 8.5 gives 8, so Rust uses `round_ties_even`); rows permuted with their keys give the same labels after the key sort; the sample of 50,000 is the same for the same seed and key order; assignment equals brute force; the Hungarian assignment equals SciPy's; the interface has no method taking one matrix product (a compile-fail doc test) |
| Selective checks | cargo gates (`fmt`, `clippy -D warnings`, `doc`, `test`, `deny`), jscpd over `native/**/*.rs`, the generator's drift, the layering check, hooks on touched files |
| Owed to the full sweep | none; M12's rows are the coordinator's |
| Go bar | every case equal; then M12's bar (16 section 4) |
| Deletes | nothing |

### R8-1. Groups on the record: storage and jobs (18b R8, first half)

| Row | Content |
|---|---|
| Goal | the person's groups, placement of new photos and the refresh rule work on the record exactly as today |
| Measurement | 15b M10 and M12 recorded |
| Entry | 18b R8's (R6b and R7 merged); L16-1 merged; M12 bar met |
| Crate and files | `src/db/migrations/sql/0016_cluster_cutover.sql` with 16b 1.1's statements and the trigger change of 16b 1.3; `op_store::clusters`: pool read (one binary `COPY` per fit with today's pool filter), partition write (temporary-table `COPY`, then `INSERT ... SELECT`, 15a section 1), carry-over through `op_numeric::carry`, activation with OCC and today's `auto_when_safe` rule, the first-fit rule if owner question O3 says (a), placement (anti-join on the membership key, centroids cached per process by partition id); job kinds `cluster_run` (items and boxes) and `cluster_place` registered in `op_server`'s worker role; the `cluster_refresh` tick (ported by R3a) queues the Rust kinds and backstops `cluster_place`; `op_ingest`'s commit queues `cluster_place` after commit (one line, owned here); the erasure catalogue rows of 16b 1.3 (R5e's catalogue file) |
| Replaces | job kinds `cluster_run` (Python) and `cluster_projection_sync`; `src/services/clustering/` (`run.py`, `job.py`, `assign.py`, `activation.py`, `carry_over.py`, `partitions.py`, `refresh_schedule.py`, `projection.py`), the placement calls in `src/services/curation/ingest_index.py` and `reprocess_images.py`, `src/services/curation/clustering/`; the `cluster-worker` service and its compose, CLI and installer entries |
| Contract | none (routes are R8-2's) |
| Tests | red first: today's pool filter, row by row (validated, excluded, confident sources, holdout); a fit's parameters equal today's for both targets; carry-over equals `carry_over.py` on recorded parents; a stale activation revision is refused; the first fit serves without a click only when O3 says so; placement writes `incremental` rows into the serving partition only and nothing without one; no `claims` write exists in `op_store::clusters` (module privacy) |
| Selective checks | 18a section 1.1's |
| Owed to the full sweep | the `db` feature tests (partition write under row security; insert into a ready partition; two `cluster_place` of one target refused; erasure recomputes a centroid and deletes an emptied cluster, 16b 1.3); the box bar (BCubed precision and recall on recorded box vectors within the two-run spread of today's `MiniBatchKMeans`; a miss adds mini-batch k-means to `op_numeric` in a fix lane); a 40,000-object library's first groups readable within 60 s of the quiet point; M12's 1,000,000 row |
| Go bar | owed green |
| Deletes | the replaced modules by last importer (`rg -l` at entry), their tests with ledger rows (18a section 3.2) |

### R8-2. The cluster routes and screens on Rust (18b R8, second half)

| Row | Content |
|---|---|
| Goal | every cluster screen the person uses today answers from Rust, with the same fields |
| Measurement | 15b M10 (the read shapes) |
| Entry | R8-1 merged |
| Crate and files | `op_api` module `clusters`: the cluster operations of 18b R8 (`GET /curation/projects/{project}/clusters`, `GET /curation/projects/{project}/clusters/{cluster_uid}`, its facets, `GET /curation/projects/{project}/clusters/partitions`, `GET /curation/projects/{project}/clusters/partitions/{partition_id}`, `GET /curation/projects/{project}/clusters/representatives`, `POST /curation/projects/{project}/clusters/runs`, `PUT /curation/projects/{project}/clusters/activation`, the empty states) and `GET /curation/projects/{project}/regions/clusters`; card fields as today, sizes and labelled counts at read time (16b 1.1) |
| Replaces | the cluster routers, `src/services/curation/cluster_cards.py`, `cluster_facets.py`, `cluster_representatives.py`, `cluster_empty_state.py` |
| Contract | kept: every field and sort. Intended changes, each with its frontend edit: activation answers when the pointer moved (no projection job to follow): `frontend/src/lib/clusters/clusterRunController.svelte.ts`; method values lose `detector_class` (and `minibatch_kmeans` if R8-1's box bar passed) |
| Tests | red first: card fields on a seeded partition equal the reference's; labelled counts follow a label written after the fit; holdout and excluded objects are hidden and counted as today |
| Selective checks | 18a section 1.1's; `npm run check` |
| Owed to the full sweep | differential cases of the family against the reference stack; `tests/live/test_live_clusters.py`; the cluster browser pages at desktop and 430 px, viewed; 15b's cluster shapes at 1,000,000 within plan 15's page bars |
| Go bar | owed green; p50 and p99 not above the reference |
| Deletes | the replaced routers and services by last importer |

### B1. The random check sample (Part B, first)

| Row | Content |
|---|---|
| Goal | the person learns how right their group labels and the suggestions really are, per class, and can test the model on objects that were not chosen by how they were labelled |
| Measurement | M16-6 recorded and its bar met (16 section 3, B1) |
| Entry | R5 merged; M16-6 |
| Crate and files | `op_store::human`: `audit_start` with `sample_kind = 'random'` (a seeded uniform draw over the library's objects, labelled or not, excluding holdout; size per owner question O5); the payload model change of 16b 2.1 in `src/domain/claim_payloads.py` and `op_domain`; the audit report gains per-class agreement of earlier labels and of suggestions (once B5 exists) with Wilson intervals (`audit_math` rules); "freeze this sample as the test set" through the existing holdout freeze |
| Replaces | nothing |
| Contract | `POST /curation/projects/{project}/audit/start` gains `sample_kind`; the audit report gains the per-class rows. Cropwright: the audit tab offers the random sample and its report |
| Tests | red first: the draw is uniform and reproducible for a seed; a labelled object in the sample is shown without its label; the report's interval equals `wilson_interval` |
| Selective checks | 18a section 1.1's; `npm run check` |
| Owed to the full sweep | differential cases of the audit routes; live test of a sample, its labels and the report; browser page viewed |
| Go bar | owed green |
| Deletes | nothing |

### B2. The group-count rule for large libraries

| Row | Content |
|---|---|
| Goal | in a library of tens of thousands of objects, groups are small enough to name at a glance |
| Measurement | M16-1 and M16-3 bars met for the candidate fixed in 16 section 3 |
| Entry | R8-1 merged |
| Crate and files | `op_numeric::k_rules::items_large(n) = max(items_today(n), n / 20)`; the cluster profile default switches to it (R11's profile body); nothing else |
| Replaces | today's default for new projects |
| Contract | the profile's `n_clusters` documentation |
| Tests | red first: the rule equals today's up to 10,259 objects and n / 20 from 10,260 |
| Owed to the full sweep | the fit at 1,000,000 within M16-3's bar on the backend it names |
| Go bar | M16-3's fit bar |
| Deletes | nothing |

### L16-2. The GPU fit backend (only if B2 passed and the CPU misses)

| Row | Content |
|---|---|
| Goal | a library of a million objects gets its groups in minutes |
| Measurement | M16-3: the CPU fit of B2's rule misses the 1,000,000 bar; a GPU fit agrees with the CPU fit (09c CL-9: purity within 0.01, adjusted Rand index at least 0.95) |
| Entry | B2 merged; the coordinator's and plan 17's yes on `gpu_frame::blas` (16 C1); the owed cudarc checks of section 2 |
| Crate and files | `gpu_frame::blas` (cuBLAS f16 `gemm` with f32 output, the one new `unsafe` call, documented); a top-two-per-row kernel through `gpu_frame`'s NVRTC loader; `op_numeric` feature `gpu` with `GpuBackend` implementing `ClusterBackend` (one upload per call, vectors and centroids resident for all iterations); a GPU worker of the same `op-server` binary built with `gpu`, claiming only GPU-backed kinds, with the card claim of the ported arbiter (18b R3a); `cluster_partitions.backend = 'gpu'` only with a passing `cluster_backend_validations` row (02d) |
| Replaces | nothing |
| Contract | none |
| Tests | CPU-only tests of backend selection and of the validation refusal (red: a GPU fit started without a passing row) |
| Selective checks | cargo gates with and without the feature |
| Owed to the full sweep | one GPU hold: CL-9 rows and fit times at 1,000,000, writing the validation row |
| Go bar | CL-9 met; M16-3's 1,000,000 bar met |
| Deletes | nothing |

### B3. Groups keep their labelled members (only if owner question O2 says (b))

| Row | Content |
|---|---|
| Goal | a group the person named stays on screen with its progress |
| Measurement | needs none (a product choice) |
| Entry | R8-2 merged; O2 = (b) |
| Crate and files | `op_store::clusters` pool filter keeps validated and confidently machine-labelled objects (excluded and holdout stay out); group pages and the detail grid order unlabelled first |
| Replaces | today's pool filter |
| Contract | none new; the card counts already carry labelled counts |
| Tests | red first: a labelled object stays a member after a refit; the detail grid lists unlabelled members first |
| Owed to the full sweep | live test of a refit after labelling; the browser page viewed |
| Go bar | owed green |
| Deletes | the old filter |

### B4-1. Label scopes: storage and jobs

| Row | Content |
|---|---|
| Goal | groups never mix detector classes the person keeps apart |
| Measurement | M16-2 bar met |
| Entry | R8-2 merged (and B2, B3 where built) |
| Crate and files | the revision of 16b 2.2 (tables, view, partition and activation keys); `op_store::clusters` reads the pool per scope through `item_scopes` in the same statement; `cluster_run`, `cluster_place` and activation per scope; the tick refits the scopes a scope edit changed; the default (one scope per detector class, plus the rest scope) created on the first fit |
| Replaces | Part A's single partition per target |
| Contract | none (routes in B4-2) |
| Tests | red first: a fit never mixes scopes; a detection of a non-detector analyser never sets a scope; a partition of an older scope revision is never activated; placement uses the current mapping |
| Owed to the full sweep | the `db` feature tests (scope edit OCC; activation refusal) |
| Go bar | owed green |
| Deletes | nothing |

### B4-2. Label scopes: routes and screens

| Row | Content |
|---|---|
| Goal | the person sees the scope list with counts and can merge or split scopes |
| Entry | B4-1 merged |
| Crate and files | `op_api` routes `GET /curation/projects/{project}/label_scopes` (scopes, counts, the merge hint when one class name has labels in two scopes), `PUT /curation/projects/{project}/label_scopes` (the whole definition with `expected_revision`); the cluster routes take `scope_id` |
| Contract | the routes above (new); `scope_id` on the cluster routes. Cropwright: the cluster screen opens on the scope list (files named at entry with `rg -ln "clusters" frontend/src`) |
| Tests, checks, owed, go bar | red first: a stale revision is refused; the hint appears for a shared class name. 18a's checks, `npm run check`; differential and browser pages viewed; owed green |
| Deletes | nothing |

### B5-1. The classifier: numeric

| Row | Content |
|---|---|
| Goal | suggestions are computed in Rust as the measurement computed them |
| Measurement | M16-1's bar met (or O1 (c) with M16-4) |
| Entry | L16-1 merged |
| Crate and files | `op_numeric::head` (class means by class name, top two, score, margin, the novelty score = cosine to the nearest mean); `op_numeric::calibrate` (5 folds by a hash of `item_key`, out-of-fold predictions, isotonic fit, per-class out-of-fold precision with count and Wilson interval; the minimum before suggestions from M16-1) |
| Contract | `contracts/json/numeric_cases.json` gains numpy class-mean proposals, scikit-learn 1.9.1 isotonic output and per-class precision cases |
| Tests | red first: proposals equal numpy's; isotonic within 1e-6 of scikit-learn's; below the minimum the output is a refusal with the count missing; renaming a class in the input changes only names |
| Selective checks, owed, go bar, deletes | 18a's; none owed; every case equal; nothing |

### B5-2. The classifier: storage and job

| Row | Content |
|---|---|
| Goal | suggestions follow the person's labels within a bound counted in labels |
| Measurement | M16-3's head rows; M16-6 recorded |
| Entry | B5-1, R8-2 and B1 merged |
| Crate and files | the revision of 16b 2.3 (tables, trigger, analyser CHECK); `op_store::heads` (training and scoring reads in one snapshot with the revision, writes, the pointer rule, retention of two heads, the follow-up rule of 16b 3.3); job kind `head_fit`; the label, undo and merge jobs queue `head_fit` after commit; the erasure rows of 16b 3.5 |
| Replaces | nothing (new) |
| Contract | none (routes in B5-3) |
| Tests | red first: the training read never holds a holdout, excluded or unresolved object; a proposal is never a claim and never changes `item_state` (the fold's oracle cases replayed); a head of an equal or lower revision never replaces the serving one; a fit that ends behind queues one follow-up; a retried job leaves one set of rows |
| Owed to the full sweep | `db` feature tests: two sessions labelling while a fit runs (no lost label, one follow-up), the trigger's lock under the two-session label test, erasure deleting the heads; `head_fit` at 40,000 within M16-3's bar |
| Go bar | owed green |
| Deletes | nothing |

### B5-3. The classifier: routes and screens

| Row | Content |
|---|---|
| Goal | the person confirms whole pages of suggested objects, works the least certain and the novel ones |
| Entry | B5-2 merged |
| Crate and files | `op_api` routes `GET /curation/projects/{project}/suggestions` (classes with proposed counts, per-class precision and interval, B1's check where present, status or the refusal with the labels still needed) and `GET /curation/projects/{project}/suggestions/page` (`class_name`, keyset cursor, 30 objects with `class_version`); the review tab table gains `least_certain` and `novel`; the cluster cards gain the proposed class and its share |
| Contract | the routes above (new); the tab ids. Cropwright: the suggestion page (one class, deselect, confirm with the shown versions), the two tabs; files named at entry |
| Tests | red first: a page confirm writes claims only for the ids shown and skips changed ones (16b 3.1); no wire serializer maps a proposal into `class_name`; uncalibrated pages order by margin |
| Owed to the full sweep | live: a scripted loop on CUB (label to the minimum, refit, confirm a page, undo it, suggestions return); browser pages at desktop and 430 px, viewed; a scripted oracle through the routes reproduces M16-1's replay cost within the seed spread |
| Go bar | owed green; pages inside plan 15's bars |
| Deletes | nothing |

### B6. Growth without constant refits

| Row | Content |
|---|---|
| Goal | a growing library keeps its groups steady while the person works |
| Measurement | M16-7 bar met |
| Entry | R8-1 merged |
| Crate and files | the `cluster_refresh` rule gains the placed-share setting (profile `refresh.placed_share`), default from M16-7 |
| Tests, owed, go bar | red first: no refit below the share; a fake-clock tick test; owed green |
| Deletes | `refresh.growth_threshold` if M16-7 replaces it |

## 4. The changes 18b needs (complete; the coordinator applies them)

| 18b place | Today's text | Change |
|---|---|---|
| Section 1, R1c | (no placement text; R8 had it) | after its commit group commits, the ingest queues `cluster_place` (the line is R8-1's); no numeric code in the commit |
| Section 1, R3a | ports the `cluster_refresh` tick | port today's rule as it is (`refresh_schedule.py:314-343`), queuing the Python `cluster_run` until R8-1 switches it to the Rust kind |
| Section 2, R5 | `0013_claims_cutover`; label acts, review tabs | `0013` also creates `label_selections` and `label_selection_items` (16b 1.2); every label act carries the precondition of 16b 3.1 (`class_version` per shown object; `selection_token` from the dry run for filter acts; `expected_class_version` on the single label); the item wire gains `class_version`; the frontend sends them (`api_cropWrites.ts`, `api_select.ts`, `selectionActionController.svelte.ts`); the `db` tier's two-session label test covers a stale tab |
| Section 2, R5e | the erasure catalogue | R8-1 adds 16b 1.3's rows and the trigger change; B5-2 adds 16b 3.5's rows; the completeness test covers them |
| Section 3, row R8 | "`cluster_run` per D5 (Rust GPU k-means or Python reading the record's vectors)"; "placement of new photos in the ingest commit as membership rows" | R8 becomes **R8-1** and **R8-2** (16a section 3); entry adds L16-1 merged and M12's bar; kinds: `cluster_run` in Rust (items and boxes), new `cluster_place`; placement by `cluster_place` queued after the ingest commit; `0016` carries 16b 1.1 |
| Section 3, row R9 | entry "R8 merged" | "R8-1 merged"; export reads claims and state, never `head_proposals` |
| Section 3, row R10 | the machine helpers in Rust | unchanged: the assist loop keeps today's cluster-sampled roots on the serving partition (one partition per target, or per scope if B4 is built) |
| Section 6, row S6 | "`0016`, cluster summaries, placement" | "`0016` (16b 1.1), placement as the `cluster_place` job, read-time group counts" |
| Section 7.2, job kinds (44) | `cluster_run` "R8 per D5 (M12)" | 45: `cluster_run` R8-1 (Rust); add `cluster_place` (R8-1); `head_fit` (B5-2) makes 46 only if B5 is built |
| Section 7.3, ticks | `cluster_refresh` (R3a) | unchanged list; R8-1 switches what it queues |
| 18a section 2.2 | the crate table | add `op_numeric` (section 2); `op_store` may depend on it |
| 18a R1j | the job runner | for B5-2 only: a handler may return "requeue"; the runner inserts the follow-up row in the same transaction as the terminal state (16b 3.3) |

## 5. The changes plan 18 needs (next revision)

| 18 place | Today's text | Change |
|---|---|---|
| Section 2.2, row "k-means and kNN at scale" | "matrix-multiply kernels in our own GPU crate first; cuvs ... ; linfa-clustering 0.8.1 on CPU as the small-set path" | "`op_numeric` (plan 16a L16-1): our own spherical Lloyd's loop on faer products on the CPU, behind a whole-fit interface; a GPU backend (cuBLAS through cudarc in `gpu_frame`) only if plan 16's B2 needs it; linfa-clustering is not today's algorithm (L2, no re-normalisation)" |
| Section 2.3, row `cluster_run` | "Rust if M12 or M13 passes ...; else Python reading and writing its own result tables" | `cluster_run` is Rust, CPU or GPU decided by M12; no Python fallback (owner's list: `probe`, `train`, `model_export`, `bakeoff`, `promote`, `embedding_viz`). The same applies to `item_scores`, `near_dup_score`, `near_dup_core_group` and `persons_cluster` (outside this plan; flagged) |
| Section 3.5, D5 | "Numeric kinds in Rust or Python" | "Numeric kinds on CPU or GPU, in Rust" |
| Section 8, M12 | "D5 for `cluster_run`: Rust when the objective is within the two-run spread of faiss and the time is not above faiss GPU by more than the spread" | M12 measures today's parameters (K = min(512, n // 6), a 50,000-object sample) at 100,000 and 1,000,000; bar of 16 section 4; it decides CPU or GPU for Part A |

## 6. Mapping to plan 18b

| This plan | 18b slice | Relation |
|---|---|---|
| L16-1 | none (new library slice) | feeds M12 and R8-1 |
| R8-1, R8-2 | R8 | R8 split by concern |
| the precondition, selection tables | R5 | additions (section 4) |
| B1 to B6, L16-2 | none | new, each after its gate |
| M16-5 | R9 | owed before R9 |
| machine helpers | R10 | unchanged |
| faces | R2b (`persons_cluster`) | may use `op_numeric` |
