# 18a: the backend in Rust, how slices run and the slices that start now

| Field | Value |
|---|---|
| Status | plan, revision 1 (2026-11-15), brought in step on 2026-11-16 with decisions 14 to 20 and what stage 0 built (the "Built" lines of section 5), companion of [18](18_backend_rust_port_plan.md) (decisions, order, windows, storage design, measurements, cloud benchmark) and [18b](18b_backend_rust_port_later_slices.md) (the slices after the record exists, the plan 15c mapping, traceability) |
| What a lane takes from here | section 1 (the template and lane rules), section 2 (the workspace, its crates and who owns each shared file), section 3 (tests and parity), section 4 (fail-closed rules), and its own slice in section 5, plus the gates of 18 section 7 |
| Reads with | [17](17_native_engine_plan.md) sections 1.1 and 3.1 (the engine protocol and the Rust gates); plan 15c on `int/storage` (the storage design each record slice builds) |

## 1. How every slice runs

### 1.1 The slice template

| Row | Content |
|---|---|
| Goal | one sentence for the person who uses OpenProcessor |
| Measurement | "needs none", or the measurement ids that must be recorded before entry (18 section 8) |
| Entry | merged slices, engine phases and measurements, by id |
| Crate and files | the crates and modules it creates or edits; the shared files it owns (section 2.3) |
| Replaces | Python routes (by contract operation), modules, job kinds, processes |
| Contract | the routes kept in behaviour; intended changes, each with its frontend edit; generated files touched |
| Tests | the tests it writes, each seen failing first (the commit message says how) |
| Selective checks (the lane runs them) | its Rust unit tests, case replays, contract checks, `cargo` gates and hooks on touched files, the offline Python tests of the directories it touches, `npm run check` when a frontend file changes. No database, compose, live, browser, soak or GPU run in a lane |
| Owed to the full sweep | its database-tier tests, differential cases, live and browser tests, one targeted stack or GPU run where named, its measurement rows |
| Go bar | the measured bar that keeps the slice; a miss goes back to one fix lane with its number |
| Deletes | the Python modules whose last importer it removes (found with `rg -l` at entry and listed in the brief), and the tests of deleted modules with their ledger rows (section 3.2) |

### 1.2 Lane rules (global rules, restated where a lane needs them)

- One lane = one agent in its own git worktree on its own branch; one commit per concern; hooks on,
  never `--no-verify`; never push, merge or check out another branch.
- Every lane brief starts from the documented pattern or library for what it touches, cites the pages
  read (18 section 2.2 lists the first set) and justifies any custom code against them.
- A lane edits only its own crate, its own modules and the shared files section 2.3 assigns it.
- From a worktree, any compose command uses `-p <name>`; lanes run none anyway.
- The full sweep (offline suite, database tier, live tier, browser tier, `--all-files` pre-commit and
  pre-push, the Rust database tier, differential runs, the benchmark regression run) runs once per
  large merge of several finished slices, on the integration branch, by the coordinator.

## 2. The Cargo workspace (decision 11)

### 2.1 One workspace: `native/`

Decision: the server crates join the engine's workspace `native/`. Reasons, each checked:

| Reason | Evidence |
|---|---|
| One `Cargo.lock`, so tokio, tonic, prost and hyper are one version on both sides of the engine protocol | `native/Cargo.toml` pins tokio 1.54.1, tonic 0.14.9, prost 0.14.6 in `[workspace.dependencies]`; `native/Cargo.lock` already holds axum 0.8.9, hyper 1.12.0, hyper-util 0.1.21, serde 1.0.229, serde_json 1.0.154, sha2 0.11.1, tracing 0.1.44, prometheus-client 0.26.0 |
| One set of gates already wired | `.pre-commit-config.yaml` scopes `cargo fmt`, `clippy`, `doc`, `test`, `deny` to `^native/`; `[workspace.lints]` and `native/clippy.toml` hold the complexity, documentation and suppression-reason rules; jscpd covers `native/**/*.rs` |
| No duplicated shared code | the KServe protos have one copy (`native/gpu_frame/proto`), read by each crate's `build.rs` with protox (`frame_engine/build.rs`, `frame_engine_loadgen/build.rs`); the server's engine client reads the same copy. The BYTES-tensor codec is shared by the two clients (Q5 in 18) |
| Cost, accepted | every `native/` commit's hooks build and test the whole workspace; server lanes and engine lanes both append to `[workspace.dependencies]` and `Cargo.lock` (rule 2.4) |
| Risk, settled | Cargo unifies features across the packages of one build, and serde_json's `arbitrary_precision` breaks internally tagged enums and `#[serde(flatten)]` (serde-rs/json issue 505); R0w's test failed with `frame_engine` turning it on (18 Q3), so the engine's plan reader (`wire/json.rs`) is its own and the workspace's serde_json carries no behaviour-changing features (`arbitrary_precision`, `preserve_order`); `op_config/tests/workspace_features.rs` guards it, and a server crate must not enable them; serde_json's features are set once, in the workspace table: `float_roundtrip` on, `arbitrary_precision` and `preserve_order` never |

Two workspaces would give two lock files with possibly different tokio and tonic versions across the
protocol, two gate setups, and a second copy of the protos or a cross-workspace path dependency.

### 2.2 The crates

The engine crates stay plan 17's: `gpu_frame`, `frame_engine`, `frame_engine_loadgen`. One dependency
runs from an engine crate to a server crate: `frame_engine_loadgen` uses `op_engine_client`'s tensor
codec (18 Q5, built in R1e); `frame_engine` is a dev-dependency of `op_engine_client` and nothing
else. The server crates, with the lane that creates each (R0w creates the stubs of the early ones so no two lanes ever
create the same crate) and the lane that fills it:

| Crate | Holds | Stub created by | Filled by | May depend on (server crates) |
|---|---|---|---|---|
| `op_config` | reading `OP_*` settings from the environment into typed structs (one helper), the retired-variable refusal, and the one function that installs the process-wide TLS provider (`install_tls_provider`, rustls with ring, `native/op_config/src/tls.rs`; 18 section 2.2), which every crate that makes a TLS connection calls | R0w | R0w (helper), R1a (retired variables) | none |
| `op_storage` | key layout, landing zone, local read paths, streaming SHA-256, durable object store (`object_store`), mover, derived files and cache, delete and erase primitives, reconcile listing | R0w | R1s | `op_config` |
| `op_media` | imohash port, ExifTool `-stay_open` pool by path, capture field mapping; later the fast path (M16) | R0w | R1x | `op_config` |
| `op_engine_client` | the engine protocol v1 client: BYTES codec, plan builder, answer parser, channel pool, readiness; the model server's readiness and load calls | R0w | R1e | `op_config`; `frame_engine` as a dev-dependency only (the in-process stub pipeline of its round-trip tests) |
| `op_db` | the pool, `ProjectTx`, the statement registry, the `LISTEN` connection, the test harness that loads the schema dump | R0w | R1j | `op_config` |
| `op_jobs` | the job runner, claim, lease, heartbeat, fencing, terminal states, exclusive groups, the periodic framework and later every Rust kind's registration | R0w | R1j; kinds by their slices | `op_db`, `op_config` |
| `op_ingest` | the upload receiver (R1u); the ingest pipeline and commit (R1c) | R0w | R1u, R1c | `op_config`; `op_storage`, `op_media`, `op_engine_client`, `op_jobs`; `op_store` from R1c; `op_upload_loadgen` as a dev-dependency only (R1u's tests build their browser-shaped request bodies with its multipart writer) |
| `op_api` | router assembly, middleware, error shapes, health, metrics, the forwarder; one module per route family added by its slice | R0w | R1a; families by their slices | `op_config`, `op_db`, `op_storage`, `op_ingest`, `op_jobs`, `op_store` |
| `op_server` | the binary `op-server` and its roles | R0w | R1a (`api`), R3a (`worker`, `periodic`), R12 (`migrate`) | every server crate |
| `op_upload_loadgen` | the binary `op-upload-loadgen` (18 section 5.2) | R0w | R0g | `frame_engine_loadgen` (its `cgroup` and `arith` modules, no server crate) |
| `op_domain` | pure ports: fold, claim matching, tallies, review tabs, geometry, filter grammar, cursor codec | R1b | R1b, then families | none (no input or output) |
| `op_store` | record repositories (photos, items, results, state, vectors, analysers, events, claims) | R1b | R1b, then families | `op_db`, `op_domain`, `op_numeric` |
| `op_numeric` | the numeric kernels of the labelling loop: row blocks, spherical k-means behind a whole-fit interface (CPU first), top-two assignment, member-overlap carry-over, the group-count rules; no database, no input or output (plan 16a section 2; a GPU backend only if 16a's L16-2 is built) | L16-1 (plan 16a) | L16-1 | none; `gpu_frame` only under the feature `gpu` (L16-2) |

`op_domain` and `op_store` are not stubbed early: only R1b creates them, months after stage 0 starts,
so no empty crate waits; `op_numeric` is created by its own library slice (L16-1 of plan 16a). There
is no `op_contracts` crate: the contract is kept contract-first (decision 14, 18 section 2.4).
`scripts/codegen/check_layering.py` (`CRATE_DEPENDENCIES`) enforces the right-hand column (18
section 7) and holds the same table as this one for every crate that exists today, dev-dependencies
included: the rows for `frame_engine_loadgen`, `frame_engine` and `op_engine_client` are the
engine-side facts above. Its remaining entries for `op_contracts` (a row, and a member of
`op_server`'s set) are stale and are removed by the first slice that edits the script; the rows of
`op_domain` and `op_store` are already there, and the slice that creates `op_numeric` adds its row
and `op_store`'s dependency on it.

### 2.3 Shared files and their owners

| File | Owner | Rule for others |
|---|---|---|
| `native/Cargo.toml` `members` | R0w writes every early member at once | later crates (`op_domain`, `op_store`, `op_numeric`) add one line each in their creating slice |
| `native/Cargo.toml` `[workspace.dependencies]` | each lane for its own dependencies | one entry per line, alphabetical (the file's existing rule), never edit another lane's line; a version change goes through the coordinator |
| `native/Cargo.lock` | the coordinator at merge | rule 2.4 |
| `native/deny.toml` | the coordinator | a lane that needs a licence outside the allow list stops and asks |
| `scripts/codegen/check_layering.py` (crate rule) | R0w | a slice that adds a crate adds its row |
| `scripts/codegen/export_api_contracts.py`, `contracts/README.md`, the contract hook lines of `.pre-commit-config.yaml` | R0c, then R1a's contract part (after R0c merges) | families regenerate, never edit by hand |
| `docker-compose.yml`, `docker-compose.dev.yml`, `openprocessor`, `scripts/lib/*.sh`, `tests/compose/`, `tests/installer/` | R1a in stage 0; then R3a, R3b, R2b and R12, which never run in the same wave | a slice that adds, renames or removes a service updates all of them in the same change |
| `docker/op-server/Dockerfile` | R1a (with ExifTool and Perl from the start) | others ask the coordinator |
| `tests/conftest.py`, `frontend/e2e/conftest.py` (pending-list hooks), `tests/pending_rust.txt` | R0d builds the mechanism; each family slice removes its own ids | never add an id outside your slice |
| `requirements-test.txt` | R0d (adds `schemathesis==4.33.4`) | others ask the coordinator |
| `native/frame_engine_loadgen/src/client.rs` | R1e for its codec commit (18 Q5), with plan 17's lanes holding the file in that wave | plan 17 owns it otherwise |
| `src/db/migrations/sql/`, the Alembic revisions that run them | the slice that writes each revision, in the fixed order 0010 to 0018 (18b section 6) | the coordinator merges revisions in chain order |
| `CLAUDE.md` | R0w (the `native/` row), then the slice that changes a service, a command or a layout row | one row per change |

### 2.4 Lock-file merges

The lanes pin exact versions (`=x.y.z`). When two lanes both changed `native/Cargo.lock`, the
coordinator takes the integration side and runs `cargo metadata --manifest-path native/Cargo.toml
--format-version 1 > /dev/null` without `--locked`, which adds the missing entries and changes no
existing one; the lock diff must show additions only, then the hooks run with `--locked` as usual.

## 3. Tests and parity

### 3.1 What happens to the Python tests

| Kind of test | Rule |
|---|---|
| Pure domain functions whose logic moves to Rust | the slice that ports a module adds a generator under `scripts/codegen/` writing `contracts/json/<module>_cases.json` from the Python functions, with a drift hook; Rust tests replay it. When the Python module is deleted, its last file is frozen under `native/<crate>/tests/goldens/` and its Python tests go with it, each with a ledger row |
| The fold, claims and `apply()` | no Python generator on the integration branch: R0x exports the oracle of `int/storage` once, frozen (decision 1: not extended) |
| Route tests with fakes (`_curation_app.py` fixtures) | replaced by the family's differential cases and Rust handler tests; deleted with the Python route, each with a ledger row |
| Service tests with fakes | become Rust unit tests of the ported logic |
| Database-tier tests of a Python repository | ported to Rust tests behind the cargo feature `db` (testcontainers-modules, schema from section 3.3); schema and migration tests stay Python until R12 |
| Live tier (`tests/live/`) | black-box over HTTP; must pass against the integration branch at every sweep, or be in the pending list (3.4). A slice edits a live test only for an intended change it lists, and moves a switched family's seeding to the HTTP seeder (3.6) |
| Browser tier and vitest (`frontend/`) | unchanged except a listed frontend edit |
| Codegen and ratchet tests | extended to `native/` (layering by crate, file size, naming leaks) |
| Tests of Python workers that stay (18 section 2.3) | unchanged |

Totals to account for, from the inventory: 10,157 offline tests, 5,936 test functions in
`tests/curation/`, 1,714 database-tier tests, 186 live tests in 41 files. Each family slice's ledger
(3.2) states how many of its deleted tests map to which replacement; each sweep reports the running
totals.

### 3.2 Parity before deletion (decision 5)

Both servers exist during the port: the reference stack (`int/b4-stateless` at a pinned commit, its
current store) and the integration branch's stack. Both ingest the same photos and run the same seed
of acts over HTTP (3.6). For each family, before its Python handlers are deleted:

1. **Differential cases from the contract.** `tests/differential/` (R0d) uses Schemathesis 4.33.4's
   Python API to generate requests for every operation of the family from `contracts/openapi/
   curation.json` and `contracts/openapi/core.json`, sends each to both stacks, and compares parsed
   JSON after the family's normalisation list (generated ids that differ by store, timestamps,
   durations, ordering of sets, the intended changes of 15c section 2 and of the slice). Recorded
   pairs from the live tier are kept as extra cases. The lane reads Schemathesis's documentation for
   4.33.4 (https://schemathesis.readthedocs.io/en/stable/) and records how cases are generated and
   sent without its default checks.
2. **Every operation has cases.** The sweep fails if any operation of a route being deleted has zero
   differential cases.
3. **Live tier.** The family's live and browser tests pass against the Rust stack.
4. **Ledger.** `tests/parity/ledger/<slice>.tsv` has one row per deleted Python test id: `<test id>
   <tab> <replacement>`, where the replacement is a Rust test path and name, a differential case id
   pattern, a contract check, or `gone: <reason>` (a behaviour removed on purpose, named in the slice).
   `scripts/codegen/check_parity_ledger.py` (R0d), run by a hook on commits that delete files or test
   functions under `tests/`, fails when a deleted id has no row.
5. **Ground truth** where an output changes on purpose (R2a: the engine's preprocessing replaces the
   API's): the quality rows named in the slice, not bits.
6. **Cost.** p50 and p99 at 8 clients and API host CPU per request not above the reference (measured
   in the sweep), and lower where the slice claims a gain.

Allowed differences everywhere: the `msg` text of validation errors (not `loc` or `type`, never "String
should match pattern '...'"); the OpenAPI cosmetics of 18 section 2.4; float spelling (parsed values
compared); key order; header case; the `server` header.

### 3.3 The schema for Rust database tests

Alembic owns the schema until R12 (18 section 3.2). `scripts/codegen/dump_schema.py` (R1j) runs
`pg_dump --schema-only` on the harness database after `db-migrate` and writes
`native/op_db/schema/head.sql` with the Alembic revision it holds in its first line. The coordinator
runs it in the sweep and commits the result; a sweep step fails when the regenerated dump differs from
the committed one. `op_db::testing` (R1j) starts PostgreSQL with testcontainers-modules 0.15.0, loads
`head.sql`, then applies every `src/db/migrations/sql/00NN_*.sql` newer than that revision, so a lane's
new migration and its Rust database tests meet without a Python install in the Rust run and without a
database in the lane.

### 3.4 The pending list

`tests/pending_rust.txt` (R0d) holds one line per test that is red until a named slice: `<node id>
<slice>`. The conftest of `tests/` and of `frontend/e2e/` marks each listed id
`pytest.mark.xfail(strict=True, reason="pending <slice>")`
(https://docs.pytest.org/en/stable/how-to/skipping.html#strict-parameter), so a listed test that
passes fails the run. R1c writes the ids its switch turns red (the families that read the search
store); each family slice removes its own; the list is empty at R12.

### 3.5 The job conformance suite

One scenario file, `tests/job_conformance/scenarios.json` (R1j): claim race between two runners,
lease expiry and recovery, cancel while running, pause and resume, yield, exclusive group refusal,
a stale fencing token's write refused, `auto_resume` after `interrupted`. The Rust runner replays it
behind the `db` feature (R1j); the Python claim loop replays it in the database tier (R3a). Both must
give the same row states.

### 3.6 Seeding over HTTP

`scripts/curation/seed_live_harness.py` writes the search store directly, which neither the Rust
record nor the differential runs can use. `scripts/curation/seed_http/` (R0e) seeds a stack only
through its HTTP routes: one module per family (`projects.py`, `ingest.py` in R0e; each family slice
adds its own acts module), driven by `scripts/curation/seed_over_http.py`. States only a machine writer
produces (region results, VLM opinions) come from running that writer against the harness's fake model
servers, once that writer is on the stack. `seed_live_harness.py` is deleted by the slice that removes
its last caller (R12 at the latest).

## 4. Fail-closed rules that must survive

| Rule today | In Rust | Proof |
|---|---|---|
| Row-level security through the transaction (`src/db/session.py`, role `op_app` without `BYPASSRLS`) | `ProjectTx`; the pool is private to `op_db`, so no handler opens a bare transaction | the cross-project leak cases of `tests/curation/cross_project_leak/` ported to the `db` feature; a repository call without a scope does not compile |
| The project guard on OpenSearch (`src/services/projects/guard*.py`) | never ported for curation (the record replaces the indexes); R2b deletes it with the core indexes, or, if D4 keeps OpenSearch, a guarded client type built only from a bound project | deleted with OpenSearch, or its 24 test files' cases as Rust tests |
| The lock rule (a machine writer never overwrites a human or import decision) | claims are inserted only through `op_store::human` (module privacy); machine writers live in other modules and crates with no claim function; the fold decides state | the oracle cases (R0x), the fold property test, the lock-rule cases of 15a section 6 |
| Unbound or unknown project | 404 as today, from the project extractor | differential cases |
| Retired environment variables | start refusal from a generated `contracts/json/retired_env.json` (source `src/config/retired_env.py` until R12) in every role | one test per variable: the role exits non-zero naming it |
| The vector DDL role `op_vector` only in the embedding-space job | a separate pool used only by that job | the database test of plan 15c's S1a-ii design, in Rust |
| Upload and batch limits (`src/core/upload_limits.py`, `CurationConfig`) | the same served limits; tower-http body limit at the byte cap; the idle-body limit | request-edge tests |
| Low disk (`DISK_GUARD`) | the same thresholds on the landing volume (R1u) and on the core ingest upload routes (R2b) | request-edge tests |
| Paths outside the configured roots | 18 section 4.4 | property test |

## 5. The slices that start now (decision 11)

Order of merges in stage 0: R0w first (hours); then any of the others as they finish, except R1u
after R1s's first merge and R1a's contract part after R0c and the front (M15 is done: decision 14). None needs the engine switch (N3)
or the record (R1b).

| Slice | Plain name | Measurement needed | Can start |
|---|---|---|---|
| R0 | measurements M1 to M18 (coordinator) | not a lane | now |
| R0w | workspace preparation | none | now, first |
| R0c | core contract baseline | none | after R0w |
| R0s | contract spike (M15) | it is M15 | after R0w |
| R0d | differential runner, ledger, pending list | none | after R0w |
| R0e | seeding over HTTP | none | after R0w |
| R0g | upload load generator | none | after R0w |
| R0x | fold oracle export (on `int/storage`) | none | now |
| R1s | storage layer | none (M18 is its sweep row) | after R0w; in two merges |
| R1a | server skeleton and front | none (M15 is done: contract-first) | after R0w |
| R1j | job runner core | none | after R0w |
| R1e | engine client | none | after R0w |
| R1x | ExifTool pool and hashing | none | after R0w |
| R1u | upload receiver up to the landing zone | none | after R1s's first merge |

### R0. Measurements

**Goal.** The numbers that fix the bars and the open decisions exist before the slices that need them.
**What.** 18 section 8, each before the slice its row names. Coordinator holds; no lane, except R0s
(M15), which is a lane because it writes code. **Deletes.** Nothing.

### R0w. Workspace preparation

**Goal.** Every early lane finds its crate ready, so no two lanes create the same crate.
**Measurement.** Needs none. **Entry.** None.
**Crate and files.** Adds to `native/Cargo.toml` `members`: `op_config`, `op_storage`, `op_media`,
`op_engine_client`, `op_db`, `op_jobs`, `op_ingest`, `op_api`, `op_server`, `op_upload_loadgen`.
Each gets `Cargo.toml` (`[package]` fields `.workspace = true`, `[lints] workspace = true`) and
`src/lib.rs` (or `src/main.rs` for the two binaries) with its crate documentation (what it will hold,
from section 2.2) and `#![forbid(unsafe_code)]`; `op_server`'s `main` refuses to start with "no role
built yet". `op_config` gets its one helper: `Env` (a snapshot of the process environment) with
`get<T: FromStr>(key, default)` and `require<T>(key)`, returning a typed error that names the key, so
every crate parses settings the same way. `scripts/codegen/check_layering.py` gains the crate rule
over `cargo metadata` with section 2.2's table. `CLAUDE.md` gains the `native/` server crates in its
layout table.
**Tests.** `op_config` unit tests (a missing key names itself; a bad value names key and value);
`native/op_config/tests/workspace_features.rs`: round-trips an internally tagged enum and a struct
with `#[serde(flatten)]`, both holding integers and floats, through serde_json under the workspace's
unified features (18 Q3); the layering rule's test with a fixture metadata file.
**Selective checks.** cargo gates on `native/`; `tests/` for `check_layering.py`.
**Owed.** Nothing beyond the sweep's cargo gates. **Go bar.** Gates green. If the feature test fails,
the coordinator raises Q3 with plan 17 before R1a's error and wire modules land.
**Deletes.** Nothing.

### R0c. Core contract baseline

**Goal.** The 66 core routes (`/detect`, `/faces`, `/embed`, `/ocr`, `/analyze`, `/search`, ...) get
a committed contract, so their port has a baseline to compare with.
**Measurement.** Needs none. **Entry.** R0w merged.
**Crate and files.** `scripts/codegen/export_api_contracts.py` also writes
`contracts/openapi/core.json`: every operation of `app.openapi()` outside `/curation`, `/v1` twins
included, same sorting rules; `scripts/codegen/generate_contracts.py`, `make contracts` and
`contracts-check` pick it up; the drift hook's `files` pattern includes it; `contracts/README.md`
describes it; `scripts/docs/check_docs_vs_code.py` accepts routes found in either file.
**Replaces.** Nothing. **Contract.** A new generated file; no route changes.
**Tests.** `tests/test_codegen_api_contracts.py` extended: the core file holds every non-curation
route of the app and no curation route; it is stable across two runs.
**Selective checks.** That test file, the contract hooks, the docs check.
**Owed.** Nothing. **Go bar.** `make contracts-check` green. **Deletes.** Nothing.
**Built (merged, `contracts/openapi/core.json`).** 112 paths, 118 operations, 141 components. Facts a
port needs: six operations declare no response shape (`GET /`, `/connection_pool_info`, `/health`,
`/health/pe_text`, `/metrics`, `/ready`); `POST /ingest`'s 200 answer (duplicate) has no body shape; 13
schemas hold free-form objects (17 places with `additionalProperties: true`, among them `JobView`,
`SearchResult`, `FaceVerifyResponse`, `TestInferenceResponse`). A Rust type cannot be held to a
schema that says nothing, so a decided shape for each is owed before its route is ported (the
entries of R2a and R2b in 18b say so).

### R0s. Contract spike (M15)

**Goal.** The contract mechanism (code-first or contract-first, 18 section 2.4) is chosen on evidence
before any route is ported.
**Measurement.** It is M15. **Entry.** R0w merged.
**Crate and files.** A scratch crate `native/spike_contract/` in the lane's worktree only: it is never
merged. It writes, both ways, the two discriminator schemas, two `prefixItems` schemas and three
`const` fields of `contracts/openapi/curation.json` (the lane picks them with `jq` and lists them),
compares each result with the committed schema by the normalised diff of 18 section 2.4, and times a
`cargo run --locked -p spike_contract` hook on a warm and a cold target directory.
**Output.** `artifacts_local/speed_1113/plan18_contract_spike/` (gitignored): per schema, equal or
the diff; for code-first, the custom `ToSchema` impls needed; the hook times. The coordinator records
D6 in 18 section 3.5. **Deletes.** The worktree's branch after the decision.
**Built (done, never merged).** M15 ran; its output is `artifacts_local/speed_1113/plan18_contract_spike/results.md`
(source kept beside it). D6 is B, contract-first (decision 14, 18 section 2.4): code-first fails the
pass rule of 18 section 2.4. No lane builds anything from this slice any more.

### R0d. Differential runner, ledger and pending list

**Goal.** No Python route is deleted without its behaviour checked against the reference on every
operation (section 3.2).
**Measurement.** Needs none. **Entry.** R0w merged.
**Crate and files.** `tests/differential/` (pytest, marker `differential`, excluded from the offline
suite as `live` is): `runner.py` (cases from the two contract files through Schemathesis, sent to
`OP_DIFF_REFERENCE_URL` and `OP_DIFF_CANDIDATE_URL`, results as JSON lines), `compare.py` (parsed
JSON comparison with a normalisation list), `families.toml` (operation id to family, from the
contract's tags and path prefixes, every operation in exactly one family), `normalise/<family>.toml`
(one per family, written by that family's slice; R0d writes the shared defaults). `tests/parity/
ledger/` with its README; `scripts/codegen/check_parity_ledger.py` and its hook. `tests/pending_rust.txt`
(empty) and the strict expected-failure hooks in `tests/conftest.py` and `frontend/e2e/conftest.py`.
`requirements-test.txt` gains `schemathesis==4.33.4`.
**Tests.** Offline, red first: `compare.py` on fixture pairs (a float spelling difference passes; a
missing key fails; a set in another order passes only when listed); `families.toml` covers every
operation of both contracts; the ledger check on a fixture diff (a deleted test without a row fails);
the pending hook (a listed test that passes fails the run).
**Selective checks.** Those tests; hooks on touched files.
**Owed.** One differential run with both URLs pointing at the same reference stack: empty diff on
every operation (proves determinism and the normaliser), in the first sweep after merge.
**Go bar.** That run empty. **Deletes.** Nothing.
**Built (merged).** `tests/differential/` (runner, parsed-JSON compare, reviewed normalisation list
`normalise/_defaults.toml`, `families.toml`, a fixtures provider with projects as the example; the
runs are `test_stacks.py`, selected by `OP_DIFF_*` variables), `tests/parity/ledger/` with
`scripts/codegen/check_parity_ledger.py` and its hook, `tests/pending_rust.txt` with the strict
expected-failure hooks, `schemathesis==4.33.4` in `requirements-test.txt`. The runner's
`--check-schema` also checks the candidate's answers against the committed OpenAPI file: that is
the response-conformance check of decision 14 (b), so a family's slice uses it and builds no second
one. Hypothesis draws from literals of loaded local modules, so the runner empties that pool while
it draws (`cases.without_local_constants`); without it one seed gave different cases. Owed to the
sweep: the dry run over every operation (5,040 cases for the 388 curation operations) and the
same-stack read run (`--methods get,head,options --seed 0`, no schema check when both URLs are the
reference). Writes are compared only across two separately seeded stacks.

### R0e. Seeding over HTTP

**Goal.** Both stacks are filled with the same data through their own routes, so a comparison
compares behaviour, not seeding.
**Measurement.** Needs none. **Entry.** R0w merged.
**Crate and files.** `scripts/curation/seed_over_http.py` (driver: stack URL, list name, families to
seed) and `scripts/curation/seed_http/` with `projects.py` (create the harness projects and their
settings) and `ingest.py` (upload a fixed public list from the dataset registry through `POST
/curation/projects/{project}/ingest/upload`, the frontend's field order, httpx 0.28.1 as the existing
scripts use). The harness compose (`docker/test/`) gains a documented call of the driver; the existing
seeder stays for the live tier until each family moves.
**Tests.** Offline, red first, with httpx's mock transport: the request sequence per module; a 503
with `Retry-After` is retried after the stated delay; the same list gives the same requests.
**Selective checks.** Those tests.
**Owed.** One seeding of the reference stack in the first sweep after merge; the ingest response
counts recorded. **Go bar.** Seeding twice gives `duplicate` for every photo the second time.
**Deletes.** Nothing.

### R0g. Upload load generator

**Goal.** Uploads at the rate the cloud benchmark needs (about 520 MB/s of 20 MP photos) can be sent
the way the browser sends them, and timed.
**Measurement.** Needs none (it is the tool M10 and B1 to B4 use). **Entry.** R0w merged.
**Crate and files.** `native/op_upload_loadgen/`: binary `op-upload-loadgen`. A closed loop of N
in-flight requests over a list held in memory; each request is browser-shaped multipart built as a
streamed body (hyper 1.12.0 client, `http-body-util` 0.1.5 `StreamBody`; both in `native/Cargo.lock`):
`source`, `image_paths`, `run_id`, then the `images` parts, or the legacy order with `--field-order
legacy`; packing by `--max-images` (default 128) and `--max-bytes` (default 512 MiB). Stamps: when the
last frame of each part is taken by the connection (a client-side bound, stated as such in the row),
and each request's response time. One JSON line per row (`op-upload-loadgen/1`): photos/s, request
p50 and p99, failures by key, the per-part stamps file path, host CPU of named containers from cgroup
`cpu.stat` through `frame_engine_loadgen::cgroup`, and the evidence fields of plan 17 section 5.
**Tests.** Unit, red first: the multipart body parses with `axum::extract::Multipart` in an in-process
test server into the same fields and bytes; packing respects both caps; the row's arithmetic on fixed
stamps.
**Selective checks.** cargo gates. **Owed.** M10's rows (coordinator). **Go bar.** It sends at least
600 MB/s of 20 MP photos to a sink on loopback (estimate of the need with margin), measured in the
sweep. **Deletes.** Nothing.
**Built (merged).** `native/op_upload_loadgen`, with unit tests and in-process tests against an axum
server. It speaks plain `http` only; a run against a TLS front needs the generator to gain TLS first. Owed to the sweep: the 600 MB/s loopback row
against a sink the coordinator provides, and M10.

### R0x. Fold oracle export (on `int/storage`)

**Goal.** The Rust fold is held to the Python fold's verified behaviour without keeping the Python fold
alive (decision 1).
**Measurement.** Needs none. **Entry.** None; the lane's worktree is on `int/storage`.
**Crate and files.** `scripts/codegen/export_fold_oracle.py` on `int/storage`: writes, as JSON, (1) for
each of the 2,000 seeds of `tests/domain/test_fold_differential.py` the acts and the fold's state after
every step (the 45,000 sequences); (2) every row of the fold truth table
(`tests/domain/test_fold_truth_table.py`); (3) `claim_match` cases; (4) the classification of every
result table (`results.RESULT_SPECS`); (5) the outcome fields of `apply()` on the step-1 tests' inputs.
Each file stays under the repository's 10 MB large-file cap (`check-added-large-files --maxkb=10240`):
if the full sequence set does not fit, the committed files hold a covering subset (every truth-table
row and every claim kind at least 50 times, the lane reports the count) and the full set is kept under
`artifacts_local/speed_1113/fold_oracle/` for one full comparison in R1b's sweep. It also renders
migration 0010 as SQL with Alembic's offline mode (`alembic upgrade 0009:0010 --sql`,
https://alembic.sqlalchemy.org/en/latest/offline.html); if the revision needs a live connection, the
coordinator renders it from a database instead.
**Tests.** The exporter's own test: re-reading the files reproduces the step-1 tests' expectations on
a sample of 20 seeds.
**Selective checks.** That test; hooks. **Owed.** Nothing. **Output.** The files and
`0010_storage_record.sql`, which R1b commits under `native/op_domain/tests/goldens/` and
`src/db/migrations/sql/`. **Deletes.** Nothing (the step 1 is frozen, not extended).

### R1s. Storage layer

**Goal.** Every uploaded photo is stored once, organised by project and content, never half written,
on local disk or in S3 by the same code, and can be erased with everything derived from it.
**Measurement.** Needs none to start; M18 is measured in its sweep. **Entry.** R0w merged.
**Crate and files.** `native/op_storage/`. Two merges.

R1s-1, the local half (R1u starts from it):

| Module | Public interface (fixed here so R1u codes against it) |
|---|---|
| `keys` | `ProjectId(i64)`, `ContentHash([u8; 32])` (hex display), `StorageKey` (validated with `object_store::path::Path::parse`); constructors `original(ProjectId, &ContentHash)`, `derived(ProjectId, DerivedKind, photo_key: i64, name)`, `source(rel)`, the roots of 18 section 4.3; `parse(&str) -> Result<StorageKey, KeyError>` |
| `hash` | `Sha256Stream` (update with each chunk, `finish() -> ContentHash`) over `sha2` 0.11.1 |
| `local` | `LocalRoots { landing, sources }` from settings; `resolve(&StorageKey) -> Result<PathBuf, KeyError>` for landing and source keys with the canonical-prefix check of 18 section 4.4 |
| `landing` | `Landing::begin_part(upload_id: Uuid, index: u32) -> PartWriter`; `PartWriter::write(&mut self, chunk: &Bytes)` (async file write, hash update), `finish(self) -> LandedPart { upload_id, index, path, size, sha256 }`, `discard(self)`; `Landing::name_by_content(&self, ProjectId, LandedPart) -> LandedPhoto { project, key, path, size, sha256 }` (a move within the landing root that never replaces: the name is `landing/<project_id>/<sha256>.<upload_id>.<index>`, so a landing file has one owner); `Landing::remove(&LandedPhoto)`; `Landing::free_ratio()` for the disk guard |

R1s-2, the durable half:

| Module | Public interface |
|---|---|
| `durable` | `Durable::from_settings` builds `Arc<dyn ObjectStore>`: `LocalFileSystem::new_with_prefix` for `file://` or `AmazonS3Builder` for `s3://` (endpoint, allow-HTTP, checksum algorithm SHA-256, conditional put from settings; TLS by rustls with the ring provider and the platform's roots, no system OpenSSL and no aws-lc: decision 18, the provider installed by `op_config::install_tls_provider`); `make_durable(&self, &LandedPhoto) -> Result<DurableObject, StorageError>`: `PutMode::Create`; same-filesystem local store: `rename_opts` within one store when the landing root is the store's `landing/` prefix, else `put_multipart_opts` streaming from the file; `AlreadyExists` checks the stored size, never overwrites; `exists(&StorageKey)` (head) |
| `derived` | `put`, `get`, `delete_prefix` for kinds `crops`, `thumbs`, `exports`, `training`, `snapshots`; a local cache tier in front of a remote store only (a local store is already local disk): its own directory, a byte cap (least recently read evicted) and a maximum age per entry; `delete_prefix` removes the subtree on disk too |
| `erase` | `delete(&StorageKey)` and `delete_prefix(prefix)` through `delete_stream`, both idempotent; the reference check and locking belong to the erasure job (R5e) |
| `reconcile` | `list_originals(ProjectId)` streaming keys and ages, for the unreferenced-object part of the `upload_sweep` tick (R3a ports the tick, R1c adds this part) |

Settings (new `OP_*` variables, read through `op_config::Env`): `OP_STORAGE_URL`,
`OP_STORAGE_S3_ENDPOINT`, `OP_STORAGE_S3_ALLOW_HTTP`, `OP_STORAGE_CACHE_MAX_BYTES` (default 1 GiB, 0 =
no cache), `OP_STORAGE_CACHE_ROOT` (the cache directory: required with an S3 store and a cap above
0, ignored otherwise, never inside the landing root), `OP_STORAGE_CACHE_MAX_AGE_S` (default 86,400,
above 0), `OP_LANDING_ROOT`; source roots come from the existing `OP_SOURCE_ROOT`. The cache is
shared by the server processes of one host and its layout mirrors the key (decision 19). The erase
guarantee that follows: no process serves an erased file; this host's cached copy is removed at
once; another host's copy is gone within `OP_STORAGE_CACHE_MAX_AGE_S`. Credentials come from the standard AWS
environment variables `AmazonS3Builder::from_env` reads; none is logged.
**Replaces.** Nothing yet: `src/clients/storage/` and `persist_uploaded_bytes` are deleted by R1c with
their last importers.
**Contract.** None. Docs: `env.template` gains the variables with one line each.
**Tests.** Unit, red first: key constructors and `parse` (refuses `..`, empty segments, control
characters, unknown roots); the property test (generated keys never resolve outside their root, a
symlink leaving a root is refused); `PartWriter` hash equals a one-shot SHA-256; an unfinished part is
never visible under its landing name; two parts with equal bytes get two files and removing one leaves the other; `make_durable` on `object_store::memory::InMemory` and on
`LocalFileSystem` in a temporary directory: create, `AlreadyExists` with equal size accepted, unequal
size refused; a local same-filesystem move keeps the inode (no copy); erase twice is a no-op;
cache eviction at the cap.
**Selective checks.** cargo gates.
**Owed.** An S3-compatible server on loopback in the sweep: `make_durable` through `put_multipart_opts`
with the checksum; M18's rows. **Go bar.** Owed tests green; M18 recorded (its number is a budget
line, not a pass or fail). **Deletes.** Nothing.
**Built (both merges done).** `native/op_storage` as above, with a local same-filesystem move that
names a part by `link(2)` so it never replaces another file, and `make_durable` that `fsync`s the
file and the directories it touched. Owed to the sweep: the two tests in
`native/op_storage/tests/store/s3_tests.rs`, marked `#[ignore]`, which need an S3-compatible server
and an existing bucket that supports conditional put (the command is in the file's header: the
variables `OP_TEST_S3_ENDPOINT` and `OP_TEST_S3_BUCKET`, then `cargo test -p op_storage --test store
s3_ -- --ignored`), and M18's rows. `tests/test_env_surface.py` reads `native/**/*.rs`: a variable
read through `op_config::Env` counts as read by code (decision 16), so a lane documents each new
variable in `env.template` and adds no allowlist line.

### R1a. Server skeleton and the Rust front

**Goal.** The Rust server answers on port 4603 and passes every path it does not yet own to the Python
API, unchanged for the person.
**Measurement.** Needs none (M15 is done: D6 is contract-first, decision 14). **Entry.** R0w merged;
the contract part also R0c merged.
**Crate and files.** `op_config` (adds `retired`: the refusal from `contracts/json/retired_env.json`,
written by the new generator `scripts/codegen/retired_env_contract.py` from `src/config/retired_env.py`
with a drift hook); `op_api`: `routes.rs` (one line per family, alphabetical), `middleware`
(tower-http request id on `x-request-id`, CORS from the settings `src/core/cors.py` reads, the HTTP
metric names and labels of `src/app/factory.py`, tracing JSON logs with `request_id`, a concurrency
limit with load shedding at today's 512 answering 503, keep-alive 75 s, graceful shutdown 30 s),
`errors` (the error shapes of 18 section 2.4), `health` (`/live`; `/metrics` with the Rust registry),
`forward` (an axum `fallback` that forwards to `OP_PY_API_URL` over hyper-util's client with
streaming bodies both ways, server-sent events included, forwarding `x-request-id`,
`x-forwarded-for`, `x-forwarded-proto` and `host` so `src/core/request_origin.py` sees what it sees
today); `op_server` (`api` role). Compose: service `api` runs `op-server api`; the Python API becomes
`api-py` on internal port 4613, not published; Prometheus scrapes `api-py` directly for the Python
metrics. `docker/op-server/Dockerfile`: multi-stage, `-slim` base, non-root user, ExifTool and Perl
installed. The CLI, `scripts/lib/*.sh` health probes, `tests/compose/` and `tests/installer/` follow
the service change. Contract part (decision 14: D6 is contract-first, so there is no `op_contracts` crate and no printer;
the committed `contracts/openapi/*.json` stay the source of truth): the check of Rust wire types
against the committed file, written once here for the first family and copied by every later one.
The types are hand-written natural serde types; each family's Rust tests validate serialised
samples and negative cases against the committed schema with the `jsonschema` crate (a
dev-dependency; its licence, MIT-0, is added to `native/deny.toml` by the coordinator when the first
lane needs it); the sweep's Schemathesis response-conformance run (R0d's `--check-schema`) covers
every operation of the family against the Rust server. Each family's slice also stops
`export_api_contracts.py` from writing that family once Rust owns it. Of the older tools the
`oasdiff` hook and the intended-change file `scripts/codegen/contract_intended_changes.txt` remain
(they judge a hand edit of a frozen file against the integration head); the normalised diff tool
`contract_normalised_diff.py` is not built, because no second generated document exists to diff
against. Open, settled before R12: how a new route enters the contract once FastAPI is gone (a hand
edit of the OpenAPI file with `oasdiff` and the frontend type generation, or code-first revisited).
**Replaces.** uvicorn as the process on port 4603 (FastAPI keeps serving behind the front).
**Contract.** Unchanged.
**Tests.** Router tests with `tower::ServiceExt::oneshot`, red first: method, body, headers and an
event stream forwarded to a stub upstream; error shapes; load shedding at the limit; the
retired-variable refusal (one test per variable from the generated file); the compose and installer
tests for the renamed service; the contract part's check on fixtures (a serialised sample that
breaks the schema, a missing required field and an extra field each fail).
**Selective checks.** cargo gates; `tests/compose/`, `tests/installer/`, the codegen tests touched.
**Owed.** The whole live and browser tiers through the front (they prove the forwarder); forwarded p99
at 8 clients against direct to `api-py`; the image's Trivy scan. **Go bar.** Tiers green; forwarded
p99 within direct's spread plus 1 ms. **Deletes.** Nothing (the forwarder goes in R12).

### R1j. Job runner core

**Goal.** Rust can run jobs on the same tables, with the same guarantees, that the job route and the
Python workers read.
**Measurement.** Needs none. **Entry.** R0w merged.
**Crate and files.** `op_db`: `Pool` (deadpool-postgres over the database settings `src/db/engine.py`
reads, listed by the lane at entry), `project_tx(&Pool, ProjectId) -> ProjectTx` setting the scope as
`src/db/session.py` does, a statement registry (each statement a named constant, prepared in the
database-tier test), `Listener` (a dedicated reconnecting connection outside the pool that issues
`LISTEN` and yields notifications from `AsyncMessage::Notification`), and `testing` (section 3.3);
`scripts/codegen/dump_schema.py`. `op_jobs`: `kinds` from the generated
`contracts/json/job_kinds.json` (new generator `scripts/codegen/job_kinds_contract.py` from
`src/config/job_kinds/`, drift hook); `claim` (the `FOR UPDATE SKIP LOCKED` claim of 18 section 2.5
for the kinds the process registers, writing `owner`, `fencing_token`, `lease_expires_at`, a
`job_attempts` row); `lease` (heartbeat that extends the lease only with the current fencing token);
`terminal` (fenced writes of every terminal state the 02b trigger allows); `control` (cancel and pause
requests observed at heartbeats, yield and resume, `auto_resume`); `exclusive` (the
`uq_jobs_one_active` index refuses a second active job of a group and scope at insert, as 02b section
1 says; the reaper's resume runs each row in its own savepoint so a race skips the row, 02b rule b); `runner` (`run_worker`:
`LISTEN op_jobs` plus a 2 s poll, a bounded set of running jobs, graceful drain on shutdown);
`periodic` (`run_periodic`: a tick trait, the single-ticker advisory lock key of `src/tasks/beat_lock.py`,
a fake clock for tests); handlers `noop` and `sleep_noop`. `tests/job_conformance/scenarios.json`
(section 3.5). The roles are wired into `op-server` by R3a, not here.
**Replaces.** Nothing yet (Celery keeps running every kind until R3a).
**Contract.** `contracts/json/job_kinds.json` (new generated file).
**Tests.** Unit, red first, with a fake clock and a fake store trait: the state machine accepts only
the 02b transitions; a stale fencing token's write is refused; a cancel observed at the next heartbeat;
exclusive refusal; the listener reconnects and the poll covers a lost wake-up.
**Selective checks.** cargo gates; the job-kinds generator's drift.
**Owed.** The `db` feature tests on the schema dump: the conformance scenarios; two runners racing for
100 rows claim each once; lease expiry with a killed runner; `NOTIFY` to claim latency; the
statement-prepare test. **Go bar.** Owed green. **Deletes.** Nothing.
**Built (merged, and proven on PostgreSQL 18 on the merged tree).**
`native/op_db/schema/head.sql` (from `scripts/codegen/dump_schema.py`, with a fixed `pg_dump` restrict
key so the dump is reproducible) holds the schema at revision `0009_ingest_retry_at`; the harness
loads it into a PostgreSQL 18 container (`testcontainers`, started by image tag, not by digest). The
database-tier tests of `op_jobs` (`db_tier.rs`: nine; `db_proofs.rs`: seven) and the harness's own
passed with `cargo test -p op_db -p op_jobs --features db -- --test-threads=1`; the proofs include
2,000 jobs claimed exactly once, many runners, a cut connection, and cancel and pause. The claim
orders by `queued_at` alone, the order of the partial index `ix_jobs_queued`: ordered by
`(queued_at, id)` the planner chose a sort over a sequential scan (149 ms and about 100,000 pages per
poll at 100,000 queued rows, and a bulk start gives every row the same `queued_at`); by `queued_at`
alone it takes 3 ms and about 260 pages. Rows with equal `queued_at` are taken in no fixed order. TLS to PostgreSQL (decision 18): `OP_DB_SSLMODE` (`disable`, `allow` and
`prefer` (`allow` behaves as `prefer`), `require`, `verify-ca`, `verify-full`; default `prefer`) with
libpq's meanings, except that `prefer` does not retry in plain text after a failed handshake, and
`OP_DB_SSLROOTCERT` (a PEM file that is then the only trust root; refused with `disable`, `allow` or
`prefer`), through `tokio-postgres-rustls` 0.14.0 with ring; the pool, the `LISTEN` connection and
the session lock were checked under `verify-full` against a container with `ssl=on`, and a wrong
authority was refused. **Open finding, decided with R3a (18b):** a runner of one kind behind a large
backlog of queued rows of other kinds reads about 80,500 pages per poll, and no index tried is used
by the planner. B5-2 of plan 16 adds one more behaviour here: a handler may return "requeue", and
the runner then inserts the follow-up row in the same transaction as the terminal state (16b
section 3.3).

### R1e. Engine client

**Goal.** The Rust server can ask the engine for a photo's objects, vectors, crops, faces and text
with the same plan and the same reading of the answer as the Python client.
**Measurement.** Needs none. **Entry.** R0w merged.
**Crate and files.** `op_engine_client`: `build.rs` generating the KServe client from the one proto
copy `native/gpu_frame/proto` with protox (as `frame_engine_loadgen/build.rs` does); `codec` (the
BYTES-tensor pack and unpack, moved from `frame_engine_loadgen/src/client.rs`, which then imports it:
18 Q5); `plan` (the plan JSON from the ingest profile, equal to `src/domain/frame_engine_wire.py`'s
builder, checked against `contracts/json/frame_engine_plan_build_cases.json` from the new generator
`scripts/codegen/frame_engine_plan_build_cases.py`, with a drift hook); `answer` (typed results: the
photo status, boxes, classes, scores, vectors, crop JPEGs as `Bytes`, blur sums, faces, text lines,
as `src/clients/frame_engine.py` `_answer` and `frame_engine_wire.py` read them, statuses from
`contracts/json/frame_engine_v1.json`); `client` (a channel pool and an in-flight cap from settings, a
deadline per call, the readiness verdict cached 5 s as `src/services/curation/ingest_gpu_decode.py`
does, engine-unavailable as a typed error); `model_server` (the model server's `is_model_ready` and
`load_model` calls the readiness check makes). The engine's own decoder in `frame_engine/src/wire` is
not shared: the client stays written against the protocol, as the load generator is (plan 17 N0).
**Replaces.** Nothing yet (the Python client goes with the Python ingest path in R1c).
**Contract.** A new generated case file; the engine contract unchanged.
**Tests.** Unit, red first: the codec round-trips and matches bytes from the Python
`serialize_byte_tensor` (goldens); the plan-build cases; answers parsed from fixtures; a round trip
against the engine's stub pipeline started in-process (`frame_engine` as a dev-dependency only, no GPU,
as the capacity rows ran it in a CPU container).
**Selective checks.** cargo gates; the generator's drift; `frame_engine_loadgen`'s tests after the
codec move.
**Owed.** One run against the native engine on a card in the first GPU hold after N3: the four-way
ingest rows' answers equal the Python client's on `coco2k` (parsed fields). **Go bar.** Equal answers.
**Deletes.** The codec functions in `frame_engine_loadgen/src/client.rs` (moved, not copied).
**Built (merged).** `native/op_engine_client` with the modules named above plus `retry` (the policy of
`src/clients/frame_engine.py`: an overload is retried until the photo's 120 s deadline, an
unavailable engine 3 times, nothing else; the engine's own failure text is never retried),
`observer` and `settings` (`OP_FRAME_ENGINE_URL` and the pool sizes); the plan text is byte-equal
to the Python builder on `contracts/json/frame_engine_plan_build_cases.json` (generator
`scripts/codegen/frame_engine_plan_build_cases.py`), the answer vocabularies are closed, and the
codec move is done (`frame_engine_loadgen/src/client.rs` imports `pack_bytes` and `unpack_bytes`
from this crate, so the engine crate `frame_engine_loadgen` depends on a server crate: section 2.2).
`frame_engine` is a dev-dependency only, for the in-process round trips. Owed to the sweep and the
first GPU hold: the run against the native engine on a card (the go bar above).

### R1x. ExifTool pool and hashing

**Goal.** Capture time, place, orientation and camera are read from each new photo by path, with the
values today's code produces, and duplicates are found by the same sampled hash.
**Measurement.** Needs none (M5 and M16 decide later whether ExifTool stays on the hot path and whether
a fast path joins; the pool is needed either way). **Entry.** R0w merged.
**Crate and files.** `op_media`: `imohash` (the port of 18 section 2.2 over `murmur3` 0.5.2, goldens
`native/op_media/tests/goldens/imohash.json` from the new generator `scripts/codegen/imohash_goldens.py`
over sizes around the sampling threshold and four times the sample size, and over the committed test
images); `exiftool` (a pool of `exiftool -stay_open True -@ -` children, the arguments
`src/clients/exiftool_client.py` passes today (`-fast -G -n` and the rest, listed at entry), one read
per request framed with `-execute<n>`, a timeout per read, a child restarted after a failure or a
timeout, the pool size from settings); `capture` (ExifTool's JSON to the capture fields
`src/services/curation/capture_read.py` returns: time and offset, GPS under the location policy's
inputs, orientation, camera make and model, dimensions; checked against
`contracts/json/capture_cases.json` from the new generator `scripts/codegen/capture_cases.py`, which
runs the Python mapping over recorded ExifTool outputs committed as fixtures).
**Replaces.** Nothing yet (`capture_read.py` and `exiftool_client.py` go with their last importers in
R1c, after `source_backfill` moves).
**Contract.** Two new generated case files.
**Tests.** Unit, red first: imohash goldens; capture cases; the pool against a fake ExifTool (a fixture
script speaking the `-stay_open` protocol): framing, a timeout restarts the child, concurrent reads
spread over the pool.
**Selective checks.** cargo gates; the generators' drift.
**Owed.** The pool against real ExifTool in the `op-server` image over `coco2k` and `setp200`: values
equal to the Python reader on every file. **Go bar.** Equal values. **Deletes.** Nothing.
**Built (merged).** `native/op_media`: `imohash` (bit-identical to Python on the goldens
`tests/goldens/imohash.json` over 89 images), `capture` (equal to the Python mapping
on the recorded tag sets and the location-policy cases of `contracts/json/capture_cases.json`),
`exiftool` (the pool, 20 tests on a fake ExifTool; one run against the real binary passed on the
host). Settings: `OP_EXIFTOOL_BIN`, `OP_EXIFTOOL_POOL_SIZE`, `OP_EXIFTOOL_MAX_WAITING`,
`OP_EXIFTOOL_QUEUE_WAIT_S`, `OP_EXIFTOOL_READ_TIMEOUT_S`. The pool starts each child through
`setpriv --pdeathsig TERM --` (util-linux, `native/op_media/src/exiftool/child.rs`), so the
`op-server` image needs `util-linux` as well as ExifTool and Perl (R1a's Dockerfile). Owed to the
sweep: the pool against the real ExifTool in that image over `coco2k` and `setp200`, values equal to
the Python reader on every file (the go bar above).

### R1u. Upload receiver up to the landing zone

**Goal.** A person's upload of one or many files streams to local disk at the client's pace, with back
pressure, never holding a whole photo in memory, and each finished photo is handed on as soon as its
last byte arrives.
**Measurement.** Needs none (M10 is R1c's comparison). **Entry.** R1s-1 merged (its `landing`
interface).
**Crate and files.** `op_ingest`: `upload` (an axum handler for `POST /curation/projects/{project}/
ingest/upload`, built and tested here, mounted in the served router by R1c): fields `images` (parts),
`image_paths`, `source`, `run_id` in any order; validation and messages equal to
`src/routers/curation/ingest_upload.py` (`image_paths is not JSON: ...`, `image_paths must be a JSON
list of strings`, `image_paths has N entries but M images were sent`), the served limits (128 photos,
512 MiB) and the accepted extensions; each part streamed with `Field::chunk()` into
`Landing::begin_part`; `admission` (two semaphores, photos in work and bytes in flight; a bytes permit
taken at a part's first chunk); `RequestBodyLimitLayer` and `RequestBodyTimeoutLayer` on the route;
`disk_space` (the thresholds and messages of `src/core/disk_space.py` on the landing volume); the
hand-off: each `LandedPart` named by content and sent through a bounded `tokio::sync::mpsc` channel to
a `PartSink` trait (R1c's pipeline implements it), the receiver stopping its reads while the channel is
full; 503 with `Retry-After` when a request cannot start inside its admission bound.
**Replaces.** Nothing yet (Python serves the route until R1c).
**Contract.** The route's request and response schemas unchanged (checked when R1c mounts it).
**Tests.** In-process, red first, with an axum test router, a fake `PartSink` and a temporary landing
root: field order both ways; 129 photos and the byte cap refused with Python's messages; a refused
extension; a slow body is not timed out while frames arrive, an idle body is; with the sink full the
receiver stops reading (bytes read stay bounded); a full admission answers 503 with `Retry-After`; the
disk guard refuses below its threshold; a client that disconnects leaves no landing file.
**Selective checks.** cargo gates.
**Owed.** Nothing alone; R1c's sweep runs it through the stack. **Go bar.** Its tests green.
**Deletes.** Nothing.
**Built (merged, not mounted).** `native/op_ingest` (`upload/`, `admission.rs`, `disk_space.rs`,
`settings.rs`) with in-process tests; its request-edge cases are generated from the Python route
(`contracts/json/upload_validation_cases.json`, generator
`scripts/codegen/upload_validation_cases.py`). The seam to the pipeline is the trait `PartSink`
(`upload/sink.rs`): `accept` once per photo, in photo order, as soon as it has landed; `finish` once
at the end of a request that passed every check, with the late form fields (`source`, `run_id`),
returning one result per accepted photo; `abort` once instead of `finish` when the request is
refused or its client is gone. The permit rule: the `WorkPermit` in each part is the photo's share
of the admission bounds, and the sink drops it when its work on the photo is done, not when the
request ends. Landing-file ownership: the receiver owns a photo's file until `accept` takes it;
from then the sink owns and removes it, and the receiver still removes the files of an aborted
request. R1c implements `PartSink` and mounts the route. Settings: `OP_UPLOAD_MAX_IMAGES_PER_REQUEST`,
`OP_UPLOAD_MAX_BYTES_PER_REQUEST`, `OP_UPLOAD_ACCEPTED_EXTENSIONS`, `OP_UPLOAD_PHOTOS_IN_WORK`,
`OP_UPLOAD_BYTES_IN_FLIGHT`, `OP_UPLOAD_ADMISSION_WAIT_S`, `OP_UPLOAD_BODY_IDLE_TIMEOUT_S`,
`OP_UPLOAD_HANDOFF_QUEUE`, `OP_DISK_LOW_PCT`, `OP_DISK_CRITICAL_PCT`.
