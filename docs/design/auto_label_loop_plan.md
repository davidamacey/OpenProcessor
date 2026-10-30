# Cluster-sampled VLM auto-labeling loop over assist runs

Status: S1 to S8 (engine, worker, routes, contracts, docs) and S10 (fake VLM modes, live test) are
implemented on `feat/v070-assist-loop`; S9 (the Cropwright panel, on the clusters page, with the gaps below) is
implemented; S0 and S11 (acceptance runs) are open. Issue #286, milestone v0.7.0, wave W3 of
[`v070_wave_plan.md`](v070_wave_plan.md) (that file calls this doc `assist_loop_plan.md`; this is
the same plan under the name the issue thread settled on). Audience: a fresh implementation agent
with no conversation memory. Paths were checked with `rg` on `release/0.7.0` at `5f596793`;
symbols are named instead of line numbers, re-find them with `rg`.

Known S9 gaps: the panel has no pause button and no event log because the API serves neither (a
loop pauses only for the audit; the routes are cancel, resume and rollback), and no route was added
for them.

Notation: `{P}` = `/curation/projects/{project}`. Planned routes are written `{P}/assist/loops...`
because `scripts/docs/check_docs_vs_code.py` rejects a doc that names a route the app does not
serve. **Set P** is the owner-local collection of about 21 MP photos (hundreds of thousands); its
paths, names, counts per folder and per-image results stay in `artifacts_local/` (gitignored), and
its source volume is read-only to every tool in this plan. The tested local catalog models are the
two `tested` rows of `examples/vlm/catalog.tsv`; this doc calls the higher-ranked one "the tested
local catalog model" (the naming scan bans its family name outside the catalog).

Read first: `CLAUDE.md`, `docs/VISION_AND_GOALS.md`, `v070_wave_plan.md` (section 0 ground rules,
W3), [`selectable_assist_plan.md`](selectable_assist_plan.md) (sections 3.2 to 3.7),
`docs-site/docs/guides/selective-assist.mdx`, `docs/research/cluster_types.md` (3.1),
`docs/research/cluster_eval_baseline.md`.

## 0. Goal and the short answer

Owner goal: ingest very large photo sets (public COCO and Open Images, and set P), cluster them,
label them with a local VLM efficiently and safely, review, export, train a better vehicle
detector. One person running assist cluster by cluster does not scale to a million crops.

The loop: for each pinned cluster (a **root node**), draw a seeded sample of its unlocked
members, ask the VLM through the assist-run core, measure the vote distribution, and

- **propagate** the top class to the node's eligible members when a statistical gate passes,
- otherwise **split** the node into children and recurse on each child,
- stop on depth, minimum node size, budget (VLM crops, requests, wall time), cancel or audit
  failure; everything not labeled is **left for review** with its vote distribution.

It runs as a resumable job with a run record per level, a second-opinion audit of propagated
labels, and a rollback of everything one loop wrote. Cropwright gets a panel to plan, start, watch,
cancel, resume and roll back.

Decisions this plan makes (each is open to the owner in section 14):

| # | Decision | Why |
|---|---|---|
| D1 | Propagated labels are a new **machine** label kind (`class_source: assist_propagated`, not validated), written by the same apply core; explicit apply gains `label_source: "propagated"` so an explicit apply writes the identical document | Today apply writes `human` (validated, locked, exported). A loop writing `human` impersonates a person, locks thousands of machine decisions against every later fix, and sends them into the training export (`export_readiness.py` reads `class_validated`) |
| D2 | The loop never writes cluster fields. Its partition tree (root snapshot, children) lives in the job directory; splitting reuses the refine core (`refine_members`) with an in-memory writer | `cluster_subid` holds one level only, a refine rewrites it for the whole cluster, and the refresh worker and humans write it too. An internal tree is reproducible, recursive and invisible to everyone else |
| D3 | The loop reuses the assist-run **core** (scope rules, sampling, `process_chunk`, `compute_proposal`) without creating one assist-run record per node | Assist runs are one per project (`assist_busy`) and only the newest 100 are kept (`KEEP_RUNS`); thousands of node runs would block every human assist run and prune their own audit trail |
| D4 | The loop runs in a dedicated worker service (`curation-assist-loop-worker`, profile `curation`), trigger-file protocol like `scripts/curation/auto_label_worker.py` | A multi-hour job must survive API restarts and hot reloads; splitting a large node needs up to 512 MB and CPU seconds that do not belong in an API process |
| D5 | The gate is a one-sided exact (hypergeometric) bound on node homogeneity, not the assist gate (`answered >= 3`, `agreement >= min_agreement`) | 3 of 3 agreeing answers only show homogeneity above 0.37 at 95 % confidence (section 6) |

## 1. What exists (reuse, do not duplicate)

| Piece | Where | Use in the loop |
|---|---|---|
| Assist request/prepare | `src/services/curation/assist/prepare.py` `prepare_run`, `PreparedRun`; `models.py` `AssistRunRequest` | Pin the VLM endpoint (`resolve_run_vlm`, `require_run_vlm`), the pack, `images_per_call`; the same refusals (`vlm_not_configured`, `no_class_names`) |
| Scope and sampling | `assist/scope.py` (`filter_members`, `membership_overlap`, `membership_digest`), `src/domain/assist_sampling.py` (`stratified_sample`) | Root membership, drift measure, seeded draws |
| Chunk execution | `assist/chunk.py` `process_chunk` (per-item VLM class write through `vlm_class_batch.write_updates` + `ClassWriteGuard`, the `assist` slot via `slot_write.write_slots`), `assist/runner.py` (`CONCURRENCY = 4`, `heartbeat_ticker`) | Ask the VLM about a node's sample; the per-item answer is written exactly as an assist run writes it |
| Proposal | `assist/proposal.py` `compute_proposal`, `ClassAnswer`, `MIN_ANSWERED` | Vote distribution; the loop's gate (D5) is applied on top |
| Apply | `assist/apply.py` `apply_proposal` (`SAME_SCOPE_JACCARD = 0.9`, `WRITER = 'human:assist_apply'`), `crop_bulk.occ_bulk_human_relabel`, `class_label.class_label_update`, `ItemLabel` | The single class write both apply and the loop call (stage S3) |
| Lock rule | `src/clients/occ_locks.py` (`is_locked_class`: human marker, validated import, `test_holdout`), `class_write_guard.py` (`class_write_locked` adds any `class_validated`; `class_state_token`; `ClassWriteGuard`) | Every write re-checks at write time under OCC |
| Refine core | `src/services/curation/clustering/refine.py` `refine_members` (AHC complete/cosine, `MIN_REFINE_MEMBERS = 4`, `MAX_REFINE_MEMBERS` 8000 from `OP_MAX_REFINE_MEMBERS`), `refine_cluster` | Split a node (D2); `POST {P}/clusters/refine/{cluster_id}` stays the human tool |
| Cluster ids | `src/services/curation/cluster_ids.py` `RESIDUAL_CLUSTER_ID_OFFSET` (candidate clusters are `>= 10000`; class clusters have `cluster_id == class_id`) | Roots are candidate clusters |
| Job files | `src/services/curation/file_job.py` `FileJob` (atomic `state.json`, `heartbeat`, `cancel.flag`, `HEARTBEAT_STALE_S = 30`), `job_lock.exclusive_start_lock` | Run record and singleton |
| Worker pattern | `scripts/curation/auto_label_worker.py` (trigger claim by `unlink`, per-project binding, heartbeat) | Template for D4 |
| Refresh daemon | `scripts/curation/cluster_refresh_daemon.py`, `src/services/curation/clustering/refresh_schedule.py` (`Observation.busy`, `step`) | Holds retrain and auto-promote while a loop is live (section 8) |
| History/undo | `src/domain/history.py` `record_class_snapshot`, `restore_class_state` | Rollback restores the snapshot the loop's write recorded |
| External consent | `src/services/labeling/vlm_factory.py` `assert_may_connect`; `src/services/config_store/vlm_validation.py` `check_external` (`OP_VLM_EXTERNAL_POLICY`, endpoint `allow_external`) plus per-run `acknowledge_external` | Primary and audit endpoints each pass it |

Facts that shape the design:

1. Apply writes `ItemLabel.human`: `class_source: human`, `class_validated: true`,
   `cluster_id = class_id`, `cluster_subid = null` (`class_label_fields`). The item leaves its
   candidate cluster and is locked forever against automated writers.
2. An assist `classify` writes the VLM's class on each sampled unlocked item (`class_source: vlm`,
   `set_cluster=False`); the normalizer later moves those items to their class cluster. Sampling
   itself therefore shrinks a cluster, and an apply after a normalizer pass can see
   `scope_changed` caused by its own run. The loop's drift check discounts its own sampled ids.
3. At 1M crops the residual partition is 512 clusters of about 1,950 crops on average
   (`docs/research/v070_measurement_log.md`, item 7); some exceed the 8,000-member refine cap.
4. A propagated cluster label is the failure this codebase already paid for once:
   `clustering/auto_promote.py` documents a cohort poisoned by `cluster_majority_agreement`, and
   `scripts/curation/revert_class_cluster_promotions.py` exists to undo it. The #341 baseline shows
   8 % of crops outside their cluster's majority class (9.8 % at low resolution).

## 2. Module layout

Layering (`scripts/codegen/check_layering.py`): routers > services > clients; `domain` and
`schemas` import neither services nor clients. Every new file stays well under the 700-line cap
(`check_file_size.py`, warn band 640) and under the complexity baseline (no baseline line added).

| Path | Content |
|---|---|
| `src/domain/loop_stats.py` | Pure, stdlib only: `required_sample(n_members, theta, alpha, dissent) -> int` (exact hypergeometric, `math.comb`), `homogeneity_lower_bound(agree, n, N, alpha)`, `wilson_lower_bound(x, n, z)` |
| `src/domain/loop_tree.py` | Pure: node path grammar (`r10042`, `r10042.2`, `r10042.2.0`), `child_sample(parent_sampled, child_ids, k, seed, path)`, best-first priority key |
| `src/schemas/curation/assist_loop.py` | Wire models (section 10) |
| `src/services/curation/assist_loop/__init__.py` | Package docstring only |
| `.../assist_loop/partition.py` | `PartitionReader` protocol and `ClusterFieldPartition` (reads `cluster_id`); the #343 seam |
| `.../assist_loop/splitter.py` | `split_node(ids, embeddings, branching, seed)`: `refine_members` with a fixed `n_clusters` and a capture writer; seeded k-means above `MAX_REFINE_MEMBERS` |
| `.../assist_loop/budget.py` | Write-ahead reservations for crops, requests and wall time |
| `.../assist_loop/store.py` | Job directory: `FileJob`, append-only `events.jsonl`, replay into a tree |
| `.../assist_loop/evaluate.py` | One node: draw, ask (assist core), gate |
| `.../assist_loop/propagate.py` | Eligibility, drift check, the shared apply write |
| `.../assist_loop/driver.py` | State machine and scheduler (section 5) |
| `.../assist_loop/audit.py` | Second-opinion sample and pause rule |
| `.../assist_loop/rollback.py` | Restore everything (or chosen nodes) a loop wrote |
| `.../assist_loop/plan.py` | Dry run: roots, eligible items, cost estimate |
| `src/services/curation/assist/apply_write.py` | `write_class_to_ids(...)`, the write core shared by `apply_proposal` and the loop (stage S3) |
| `src/services/curation/assist/sample_eval.py` | `ask_sample(...)`, the chunked VLM execution extracted from `runner.run_assist` so both call it (stage S4) |
| `src/routers/curation/assist_loop.py` | Thin router, registered in `src/routers/curation/__init__.py` |
| `scripts/curation/assist_loop_worker.py` | Worker entry point (D4) |
| `frontend/src/lib/api_assistLoop.ts`, `types_assistLoop.ts`, `assistLoop/assistLoopController.svelte.ts`, `components/assistLoop/AssistLoopPanel.svelte` (+ `LoopLevelTable.svelte`, `LoopReviewList.svelte`) | Cropwright (section 11) |

The selectable-assist rule "nothing in the assist layer imports clustering" stays true:
`assist_loop/splitter.py` imports `clustering.refine`; the `assist` package does not.

## 3. Data model

### 3.1 Job directory (the source of truth for a loop)

Root: a new setting proposed here as `assist_loop_jobs_dir` (default `/jobs/assist_loop`, through
`project_jobs_dir` and `mark_project_dir` like `assist/job.py`; stage S7 picks the `OP_*` name,
following `OP_ASSIST_JOBS_DIR`, and adds it to `env.template` and `tests/test_env_surface.py`). It must not live inside the assist runs root: `assist/job.py`
`_run_dirs` treats every subdirectory there as a run and `_prune` deletes the oldest.

```text
<root>/projects/<slug>/
  start.lock                       exclusive_start_lock: one live loop per project
  trigger.json                     written by the API, claimed by the worker (unlink)
  <loop_id>/                       loop_id = alp_<UTC>_<6 hex>
    request.json                   normalized AssistLoopRequest
    state.json                     AssistLoopState (FileJob, atomic)
    heartbeat, cancel.flag, pause.flag
    roots.jsonl                    {root, cluster_id, n, digest, members_file}
    members/<root>.json            sorted member ids + class_state_token per id at snapshot
    events.jsonl                   append-only, fsync per line (3.2)
    budget.json                    {crops_reserved, crops_spent, requests_reserved, requests_spent}
    audit.jsonl                    {crop_id, node, propagated_class, audit_answer, agree}
```

### 3.2 Events (replayed on resume)

| Event | Fields | Written |
|---|---|---|
| `node_open` | `path`, `parent`, `n`, `digest`, `level` | before the node draws |
| `reserve` | `path`, `crops`, `requests` | before any VLM call (write-ahead) |
| `answers` | `path`, `{crop_id: class_name or null, proposed}` per chunk | after the chunk returns |
| `decision` | `path`, `kind` (`pass`, `split`, `review_leaf`, `review_small`, `stale`, `unvisited`), `class_name`, `agree`, `n`, `bound`, `distribution` | after the gate |
| `children` | `path`, `[child paths with member files]` | after a split |
| `propagate_begin` | `path`, `class_name`, `n_eligible` | before writing |
| `propagated` | `path`, `written`, `skipped` counts | after the bulk write |
| `audit` | batch result | after each audit batch |

### 3.3 Item fields

No mapping change. A propagated item (D1) gets, through `class_label_update(current, ItemLabel.propagated(...))`:

```text
class_id, class_name                  from the registry by name
class_source      assist_propagated   new constant PROPAGATED_CLASS_SOURCE in src/domain/ingest_class_sources.py
class_validated   false
label_source      assist_loop
cluster_id        class_id            class-cluster placement, as for any confident source
cluster_subid     null
class_detector    assist_loop         provenance (class_provenance)
class_detector_version  <loop_id>     keyword, indexed: rollback and review query by it
class_labeler     assist_loop
class_id_history  + restorable snapshot, writer "assist_loop:<loop_id>"
```

`PROPAGATED_CLASS_SOURCE` joins `confident_class_sources()`, so the residual pool and the
normalizer treat a propagated item like a VLM-labeled one (never re-clustered). It does not
contain `human`, so `is_locked_class` stays false and the item remains reviewable and fixable.

## 4. What is written, and by whom

| Write | Writer id | Path | Locked/validated/holdout/excluded items |
|---|---|---|---|
| Per-item VLM class on a sampled unlocked item | `assist_run` (unchanged) | `process_chunk` | never sampled (the loop samples eligible members only); defense in depth in `process_chunk` |
| `assist` suggestion slot on a sampled item, `run_id = <loop_id>/<path>` | `assist_run` | `write_slots` | n/a (only sampled items) |
| Propagated class | `assist_loop:<loop_id>` | `write_class_to_ids` under OCC + `ClassWriteGuard` | re-checked per document at write time; skipped and counted |
| Rollback restore | `assist_loop_rollback:<loop_id>` | `rollback.py` OCC merge | touches only docs whose current state is exactly what the loop wrote |

Nothing else: no cluster fields, no boxes, no registry entries (a VLM `__new__` proposal is
reported, never created), no holdout freeze.

## 5. State machine

### 5.1 Loop status

`dry_run` (response only) -> `queued` -> `running` <-> `paused` (audit or user) -> `completed` |
`cancelled` | `failed`; a dead worker (stale heartbeat) shows `interrupted` until resumed.
`stop_reason` on `completed`: `done`, `budget_crops`, `budget_requests`, `budget_time`.

### 5.2 Per node

```text
open -> draw -> ask -> gate --pass--> propagate -> done
                          \--fail--> depth < max_depth and n >= min_node_size ?
                                       yes: split -> children open (each re-enters at draw)
                                       no:  review_leaf / review_small
budget cannot cover the next draw   -> unvisited (left for review)
drift above 10 % at propagate       -> stale (left for review)
validated-member veto (5.4)         -> treated as fail
```

### 5.3 Draw, ask, gate

1. **Eligible members** of a node: snapshot members that are not `class_write_locked`, not
   `test_holdout`, not `class_excluded` at snapshot time. Locked members are never sent (they cost
   budget and carry a better answer: section 5.4).
2. **Sample size**: `k1 = required_sample(N, theta, alpha_stage, dissent=0)` with
   `alpha_stage = alpha / (max_dissent + 1)`; `N` is the eligible count. A node with
   `N <= k1` is asked whole (exact answer).
3. **Draw** (`loop_tree.child_sample`): the parent's sampled ids that fall in this node are reused
   (no new call); the rest is a seeded draw from the unsampled eligible members,
   `random.Random(f'{seed}:{path}')` over the sorted ids. Same members, seed and path give the same
   ids; input order does not matter.
4. **Ask** with `ask_sample` (the extracted assist core): same pack, endpoint, `images_per_call`,
   and per-item writes as an assist run. Failed or unreadable answers are not votes; they are
   re-drawn once from the unsampled pool, counted against the budget.
5. **Gate**: `answered` votes, top class `c` with `agree` votes, `d = answered - agree`.
   Pass iff `c` is a registry class by name, `d <= max_dissent`,
   `answered >= required_sample(N, theta, alpha_stage, d)` and the validated veto does not fire.
   With `d` in `1..max_dissent` and too few answers, top up once to `required_sample(..., d)`
   (second stage), then decide. `new_class_proposals` never pass.

### 5.4 Validated-member veto

Locked members are ground truth for free. If a node holds at least 3 members with a human or
validated-import class and fewer than `theta` of them carry `c`, the node fails (split or review)
whatever the VLM said. This catches the systematic confusion the VLM's own agreement cannot see.

### 5.5 Propagate

Eligible at write time: the node's snapshot members minus (a) sampled members whose own answer
differs from `c` (their direct VLM answer stays), (b) sampled members that agree (they already
carry the VLM's `c` from step 4), (c) members whose `cluster_id` is no longer the root's cluster
(**drifted**), (d) members whose `class_state_token` differs from the snapshot token. If
`drifted / snapshot eligible > 1 - SAME_SCOPE_JACCARD` (0.1, the apply rule), the node becomes
`stale` and writes nothing. Otherwise `write_class_to_ids(ids, class_name, label_kind, writer)`:
`occ_bulk_human_relabel` with a merge that returns `{}` when `class_write_locked(current)` or the
token changed, else `class_label_update(current, ItemLabel.propagated(...))`.

Equality with apply: for any document both would write, the merge output is byte-identical
except `updated_at`, `class_labeled_at` and the history entry's `at`/`writer`, because both call
the same `write_class_to_ids` with the same label kind. Apply's member selection (current members
of a stored (sub-)cluster) and the loop's (snapshot leaf members minus 5.5 a to d) differ by design.

### 5.6 Split

`split_node`: fetch `pe_embedding` for the node's eligible members (sorted by `crop_id`, so the
fit is order-independent), `AgglomerativeClustering(n_clusters=branching, linkage='complete',
metric='cosine')` through `refine_members` (new keyword `n_clusters`, default `None` keeps today's
`distance_threshold` behaviour) with a writer that captures `(id, subid)` pairs in memory. Above
`MAX_REFINE_MEMBERS`: seeded k-means (`sklearn.cluster.KMeans(n_clusters=branching,
random_state=seed_int(seed, path), n_init=4)`) on L2-normalized vectors. Children are ordered by
their smallest `crop_id`, which fixes child paths. A child with no unsampled eligible members and
too few reused answers goes straight to `review_small`. Children are scheduled, not recursed on
the stack.

### 5.7 Scheduling and stop rules

Best-first: a priority queue of open nodes keyed by `(-N, path)`, so the budget goes to the nodes
that label the most items per call first; deterministic for a fixed tree. Up to
`node_concurrency` nodes evaluate at once; all VLM requests share one semaphore of
`vlm_concurrency` in-flight requests (the per-run `CONCURRENCY = 4` of assist runs would cap a
loop near 2 crops/s, section 9). Stop: queue empty (`done`), budget (5.8), cancel, audit pause,
`max_wall_s`. Defaults: `max_depth 3`, `branching 4`, `min_node_size 40`, `theta 0.9`,
`alpha 0.05`, `max_dissent 1`.

### 5.8 Budgets (never exceeded)

`budget.reserve(crops, requests)` takes the in-process lock, refuses when
`spent + reserved + ask > limit`, appends a `reserve` event and rewrites `budget.json` (fsync)
**before** any request is sent; a returned chunk moves its share from reserved to spent. Failed
and retried requests are spent. On resume every reservation without its `answers` event counts as
spent, so a crash can waste at most `vlm_concurrency x images_per_call` crops but never exceeds a
limit. `max_vlm_crops` is required (no default: fail closed); `max_vlm_requests` defaults to
`ceil(max_vlm_crops / images_per_call) + vlm_concurrency`; `max_wall_s` defaults to 6 h. Audit
calls have their own cap (section 7) and are reported apart. Token budgets need `usage` capture,
which the labeler does not do today (`vlm_client.py`); the plan reports an estimate only (Q9).

## 6. Statistics: what a sampled agreement proves

Let a node have `N` eligible members, of which a share `h` would get class `c` from the VLM
(homogeneity of VLM answers). We sample `n` without replacement and see `x` votes for `c`. The
gate passes only if, were `h` just below `theta` (`K = ceil(theta N) - 1` agreeing members), the
chance of seeing at least `x` agreements is at most `alpha_stage`:

```text
P(X >= x | N, K, n) = sum_{k=x..n} C(K,k) C(N-K,n-k) / C(N,n)  <=  alpha_stage
```

For large `N` this is the one-sided Clopper-Pearson bound; with no dissent it is
`theta^n <= alpha`, so `n >= ln(alpha) / ln(theta)`. Required `n` (computed with `math.comb`,
these exact values are the unit-test table):

| theta | alpha | dissent 0 | dissent 1 | dissent 2 |
|---|---|---|---|---|
| 0.80 | 0.05 | 14 | 22 | 30 |
| 0.85 | 0.05 | 19 | 30 | 40 |
| 0.90 | 0.05 | 29 | 46 | 61 |
| 0.95 | 0.05 | 59 | 93 | 124 |

Two-stage default (`alpha_stage = 0.025`, finite `N`), the sizes the loop actually draws:

| theta | N=30 | N=60 | N=150 | N=500 | N=2000 | N=8000 |
|---|---|---|---|---|---|---|
| 0.80 (d0 / d1) | 11 / 17 | 14 / 20 | 16 / 23 | 17 / 25 | 17 / 26 | 17 / 26 |
| 0.90 | 18 / 24 | 24 / 34 | 30 / 44 | 34 / 51 | 35 / 53 | 35 / 54 |
| 0.95 | 25 / 30 | 36 / 48 | 55 / 78 | 65 / 97 | 70 / 106 | 72 / 109 |

The assist gate (`answered >= 3`, `agreement >= 0.6`) gives, at 3 of 3, a 95 % lower bound of
`0.05^(1/3) = 0.37`; it is a display gate, not a propagation gate.

What the bound does not prove:

- `h` is agreement with the VLM, not correctness. Propagated precision is about `h` times the
  VLM's accuracy on class `c`; a systematic confusion (two similar vehicle classes) passes the
  gate. The validated veto (5.4) and the audit (7) exist for that.
- It is per node. Among nodes whose true `h` is below `theta`, at most `alpha` pass; the run-level
  precision is measured by the audit, not assumed.
- Bonferroni over the two stages keeps the per-node error at or below `alpha`.

Cost consequence: at `theta 0.9` a node costs 35 to 54 crops, so one VLM crop per 20 labeled items
needs passing nodes of at least 700 to 1,100 members; at `theta 0.8`, 340 to 520. The root
clusters at 1M (about 1,950 members) can meet that; their children (about 490 with branching 4)
only at `theta 0.8` or with heavy answer reuse. This is the precision/cost dial for the owner (Q3).

## 7. Quality safeguards

1. **Gate and veto** (5.3, 5.4).
2. **Dissent stays.** A sampled member that disagreed keeps its own VLM answer (5.5 a).
3. **Second-opinion audit.** After each batch of propagated nodes (every 2,000 propagated items
   and at the end), draw `ceil(audit.rate x written)` propagated items, at least
   `audit.min_per_batch` (30), stratified by node, seeded. Ask `audit.vlm` with the same pack's
   classify call (no new prompt) and compare its class with the propagated one. Record in
   `audit.jsonl`; report `sampled`, `agreed`, the agreement and its Wilson 95 % lower bound
   (`z = 1.645` one-sided). Pause rule: when at least 30 are audited and the lower bound is below
   `audit.floor` (default 0.85), set `paused` with `pause_reason: audit`, stop scheduling, and keep
   the propagated labels until the user resumes or rolls back (`audit.on_fail: "pause"`, the
   default) or roll back the batch automatically (`"rollback"`). Examples: 95 of 100 agree, lower
   bound 0.90; 90 of 100, 0.84 (pauses at the default floor); 48 of 50, 0.886.
   - `audit.vlm` is any registered endpoint (deployment-wide registry, `vlm_endpoints`): the second
     tested local catalog model on the in-compose vLLM, or a remote OpenAI-compatible endpoint
     (for example a small Claude Haiku model behind an OpenAI-compatible gateway; verify with
     `./openprocessor vlm probe` before relying on it). A remote endpoint passes the existing
     consent unchanged: `OP_VLM_EXTERNAL_POLICY` must allow it, the endpoint must carry
     `allow_external`, and the loop request must set `audit.acknowledge_external: true`; otherwise
     the start is refused with the existing `vlm_external_*` refusal. No private crop leaves the
     deployment without all three.
   - Audit answers never write a class; they are stored in `audit.jsonl` only, so an external judge
     cannot change data.
   - Audit cost: `audit.max_crops` (default 500) is a separate hard cap, reserved like 5.8.
   - The human path: the review tab lists the audit sample (`class_detector_version = <loop_id>`),
     so a person can spot-check the same items in Cropwright.
4. **Rollback** (`rollback.py`): for every item with `class_source = assist_propagated` and
   `class_detector_version = <loop_id>` (optionally limited to node paths), an OCC merge restores
   `restore_class_state(snapshot)` from the history entry whose writer is `assist_loop:<loop_id>`,
   but only if the current `class_state_token` still equals the state the loop wrote (nobody
   touched it). Changed items are kept and counted `kept_changed`. Dry run first. Idempotent.
5. **Drift and stale** (5.5), **holdout untouched** (never eligible, `class_write_locked` re-check).
6. **Evaluation**: the #341 harness (`scripts/bench/cluster_eval.py score`) scores the loop's leaf
   partition (exported as `partition.jsonl`, `item_id`, `cluster_id` = leaf path) and the
   propagated labels against COCO ground truth.

## 8. Concurrency, idempotency, and clusters changing underneath

- **One loop per project**: `exclusive_start_lock` under the project's loop root; a second start is
  409 `assist_loop_busy`. Assist runs stay independent (D3): a human can run assist while the loop
  runs; both write through OCC and `ClassWriteGuard`, and a concurrent human write always wins.
- **Pinned partition**: at start the loop snapshots every root's member ids and class tokens
  (`members/<root>.json`) and the partition digest; `ClusterFieldPartition` is the only reader of
  `cluster_id`. New items ingested later are outside the snapshot: never propagated, counted as
  `new_since_snapshot` per root in the review list.
- **Refresh daemon**: `Observation` gains `loop_live` (from `GET {P}/assist/loops?active=true`);
  `step` treats it like `busy`, so neither the retrain nor the daemon's `auto_promote` starts for
  that project while a loop is live. `POST {P}/pipeline/auto_label/start` with `train_clusters`
  refuses with 409 `assist_loop_running` while a loop is live (cancel the loop first). Ingest-time
  nearest-centroid assignment continues. If ids move anyway (a manual action, an old job), the
  drift rule turns the affected nodes `stale`; nothing is written to a member that left its root.
- **Idempotency**: the item is the truth. On resume, a sampled item whose `assist.run_id` is
  `<loop_id>/<path>` already has its answer (read it, no call); a propagated item with
  `class_detector_version = <loop_id>` counts as written; replaying `events.jsonl` rebuilds the
  tree; open nodes restart at their last completed step with the same seeded draws.
- **Cancel**: `cancel.flag`; in-flight chunks finish (their answers are kept), no new reservation,
  no new propagation; status `cancelled` with counts. **Resume**: `POST .../resume` on an
  `interrupted`, `paused` or `cancelled` loop writes a trigger; the worker claims it.
- **#343 seam**: `PartitionReader` has `roots(scope) -> list[RootRef(key, member_ids, version)]`
  and `root_of(ids) -> dict[id, key]`. #343 adds a versioned reader keyed by `cluster_uuid` and
  `cluster_run_id`; the pinned `version` then replaces the snapshot digest and drift becomes "not in
  the pinned partition version". The driver, gate and writer do not change. #343's lock-rule tests
  reuse the per-class fixtures of S6.

## 9. Cost model for 1M+ crops

Measured so far (one A6000 shared with Triton, VLM at GPU memory utilization 0.4):

| Number | Value | Source |
|---|---|---|
| Label throughput, `POST {P}/vlm/label_batch`, 8 client threads x 32 crops | 10.0 crops/s (about 3 crops per request) | `docs/PERFORMANCE.md`, "Measured VLM labeling speed" |
| Continuous VLM worker | 9.8 crops/s | same |
| Tokens per 4-image label call, 81-class registry | 1,306 prompt + 529 completion = 1,835 (about 460 per crop), 9.4 s latency | `docs/research/vlm_serving_sizing.md` |
| Set P registry size | 86 classes (same order as the measured 81) | owner annotations, count only |

Unknown, measured in stage S0 before any budget default is fixed: the rate of the assist core at
`vlm_concurrency` 4 to 32 (at 4 in-flight 4-image calls of 9.4 s it would be about 1.7 crops/s);
both tested catalog models; images per call 1, 4, 8; crop preparation from 12 to 24 MP sources
(set B as the public stand-in, set P aggregates only), where full-frame decode per crop may become
the limit (#219 cuts it).

Planning arithmetic at 10 crops/s (replace with S0 numbers):

| Scenario (1.2M crops, 512 roots) | VLM crops | Wall | Items labeled |
|---|---|---|---|
| Wave-plan budget | 30,000 | 50 min | depends on pass rate; best-first spends it on the largest homogeneous nodes |
| 40 % of roots pass at theta 0.9 (about 45 crops each) | 9,200 | 15 min | about 470k (0.4 x 1.2M) |
| Plus level 1: 307 failed roots x 4 children, half pass, about 30 new crops each | +37,000 | +62 min | +about 290k |
| Plus level 2 on the rest, nodes about 120, mostly review | +70,000 | +2 h | +about 100k |

Target ratio (O5): 1 VLM crop per 20 labeled items at 60 % coverage, i.e. 36,000 crops for 720k
items at 1.2M; the arithmetic above lands near it only if most labels come from roots and level 1.
The run record reports the real curve (`crops_spent` against `items_propagated` per level). Remote
endpoints: cost = requests x tokens per request x price; the plan's estimate uses 460 tokens per
crop until `usage` is captured (Q9).

## 10. API and wire models

All in `src/schemas/curation/assist_loop.py` (`extra='forbid'` on requests), routes in
`src/routers/curation/assist_loop.py`, project-scoped, added to
`tests/curation/cross_project_leak/_route_tables.py` and `_route_bodies.py`.

| Route | Purpose |
|---|---|
| `POST {P}/assist/loops` | Body `AssistLoopRequest`; `dry_run: true` (default) returns the plan and writes nothing; `false` claims the project slot and writes the trigger; 409 `assist_loop_busy`, `vlm_not_configured`, `clustering_running` (an auto-label job is live), `vlm_external_*`; 422 `invalid_request`, `no_class_names`, `no_roots` |
| `GET {P}/assist/loops?active=&limit=20` | Newest first, `AssistLoopList` |
| `GET {P}/assist/loops/{loop_id}` | `AssistLoopState` |
| `GET {P}/assist/loops/{loop_id}/nodes?level=&decision=&after=&limit=100` | Paged `LoopNode` rows |
| `GET {P}/assist/loops/{loop_id}/nodes/{path}/members?after=&limit=500` | Member crop ids of a node (review: open as a selection) |
| `POST {P}/assist/loops/{loop_id}/cancel` / `/resume` | Signals |
| `POST {P}/assist/loops/{loop_id}/rollback` | Body `{node_paths: list[str] or null, dry_run: true}`; `LoopRollbackResponse` |

`AssistLoopRequest`:

```jsonc
{
  "scope": {"cluster_ids": null, "min_root_size": 40},     // null = every candidate cluster (>= 10000)
  "vlm": null, "prompt_pack": null, "acknowledge_external": false,
  "seed": 0,
  "gate": {"theta": 0.9, "alpha": 0.05, "max_dissent": 1},   // theta 0.7..0.99, alpha 0.001..0.1, dissent 0..3
  "recursion": {"max_depth": 3, "branching": 4, "min_node_size": 40},
  "budget": {"max_vlm_crops": 30000, "max_vlm_requests": null, "max_wall_s": 21600},
  "label_kind": "propagated",                               // D1; "human" only if the owner keeps that option (Q1)
  "audit": {"vlm": null, "acknowledge_external": false, "rate": 0.01, "min_per_batch": 30,
            "max_crops": 500, "floor": 0.85, "on_fail": "pause"},   // vlm null = audit off, stated in the record
  "vlm_concurrency": 16, "node_concurrency": 8,
  "dry_run": true
}
```

`AssistLoopState`: `loop_id`, `status`, `stop_reason`, `pause_reason`, `created_at`,
`started_at`, `finished_at`, `error`, `request`, `pinned` (`vlm {name, revision, model}`,
`prompt_pack {name, revision}`, `audit_vlm`, `partition {kind, digest, snapshot_at, roots,
eligible_items}`), `plan` (`roots`, `eligible_items`, `required_sample_table`,
`estimated_vlm_crops_root_level`, `estimated_wall_s` from the measured rate),
`budget` (`max_*`, `crops_spent`, `crops_reserved`, `requests_spent`, `wall_s`),
`levels: [{level, nodes, passed, split, review_leaf, review_small, stale, unvisited,
items_propagated, items_direct, crops_spent}]`, `totals` (`items_propagated`, `items_direct`,
`clusters_left_for_review`, `items_left_for_review`, `new_since_snapshot`, `skipped {locked,
holdout, excluded, drifted, dissent, token_changed}`), `audit` (`sampled`, `agreed`, `agreement`,
`lower_bound`, `crops_spent`, `status`).

`LoopNode`: `path`, `level`, `parent`, `n`, `eligible`, `sampled`, `answered`, `decision`,
`class_name`, `agree`, `bound`, `distribution` (`ClassVotes`), `new_class_proposals`,
`items_propagated`. `AssistApplyRequest.label_source` becomes `Literal['human', 'propagated']`.

`make contracts` regenerates `contracts/openapi/curation.json` and the TypeScript contracts;
commit `contracts/` in the same commit as the route. Docs to update when routes exist:
`docs/design/curation_api_contract.md` (new section), `docs/CURATION.md`,
`docs-site/docs/guides/selective-assist.mdx` (a "Loop" section), `docs/ARCHITECTURE.md` (job and
service lists), `CHANGELOG.md` and `frontend/CHANGELOG.md`.

## 11. Cropwright (minimal; renders backend decisions)

On the clusters page (`frontend/src/routes/p/[project]/clusters/+page.svelte`), an
`AssistLoopPanel`:

1. **Plan**: VLM and pack pickers (reuse `components/vlm/VlmRunPicker.svelte`), `max_vlm_crops`
   (required), a theta preset (0.8 / 0.9 / 0.95), depth, audit endpoint (picker; an external
   endpoint shows the existing consent checkbox). "Plan" posts `dry_run: true` and shows the
   server's roots, eligible items, the required-sample table and the estimated time.
2. **Run**: Start, then poll `GET .../loops/{id}` (pattern of `assistRunController.svelte.ts`):
   budget bar, `LoopLevelTable` (the `levels` rows), audit agreement with its lower bound, status
   and reasons, Cancel, Resume.
3. **Review**: `LoopReviewList` pages `nodes?decision=review_leaf|review_small|stale|unvisited`,
   shows the distribution, and opens a node's members as a selection with the existing Assist
   dialog or batch label.
4. **Rollback**: dry-run count, confirm, result.

Types by hand in `types_assistLoop.ts`, pinned by `frontend/src/lib/contract/assistLoopContract.test.ts`.
Stubbed Playwright test in `frontend/e2e/stubbed/test_assist_loop_panel.py`. Verify with
`cd frontend && npm run check && npm run build`, then open the page with `DISPLAY=:11` and view
full-page screenshots at desktop and 430 px width.

## 12. Test plan

Offline (`.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'`), fakes as in
`tests/curation/test_assist_run_classify.py` and `test_assist_proposal.py` (query-capable fake
OpenSearch, an in-process fake labeler mapping `crop_id -> answer`). Register new labeler call
sites in `tests/fixtures/labeler_call_sites.txt`. Each test is written first and seen failing.

| File | Asserts |
|---|---|
| `tests/curation/test_loop_stats.py` | The two tables of section 6 exactly; `N <= k` returns `N`; monotone in theta and dissent; `alpha` bounds |
| `tests/curation/test_loop_tree.py` | Child sample reuses the parent's ids, same seed and path give the same ids, input order irrelevant, path grammar round-trips |
| `tests/curation/test_assist_apply_write_equality.py` | For the same ids and class, `apply_proposal` (`label_source: propagated`) and the loop's propagate produce identical documents minus `updated_at`, `class_labeled_at`, history `at`/`writer` |
| `tests/curation/test_assist_loop_lock_rule.py` | One case per item class, each never written: `class_source` human, `human_move`, `vlm_human_confirmed`; validated import; `class_validated` by any writer (auto-promote); `test_holdout`; `class_excluded`; a human write landing between read and write (OCC). Written: an unvalidated import (`label_trust: suggestion`) and a VLM-labeled item. Boxes untouched on all |
| `tests/curation/test_assist_loop_budget.py` | Randomized fake answers over 200 seeds, failures and retries injected: `crops_spent + crops_reserved <= max_vlm_crops` at every event; same for requests; a crash after `reserve` counts the reservation as spent |
| `tests/curation/test_assist_loop_reproducible.py` | Two runs, same seed and fake: identical `events.jsonl` minus timestamps; different seed: different samples |
| `tests/curation/test_assist_loop_resume.py` | Kill after each event type; resume: no duplicate call for an answered item (recovered from `assist.run_id`), no duplicate or lost write, same final tree |
| `tests/curation/test_assist_loop_drift.py` | Retrain simulated mid-loop (ids moved): nodes over 10 % drift turn `stale`, moved members never written; own sampled ids do not count as drift |
| `tests/curation/test_assist_loop_veto_dissent.py` | Validated members disagreeing block the pass; dissenting sampled members keep their VLM class |
| `tests/curation/test_assist_loop_audit.py` | Audit pause at 90 of 100 with floor 0.85; `on_fail: rollback` restores the batch; external audit endpoint without consent refused |
| `tests/curation/test_assist_loop_rollback.py` | Restores exactly the pre-loop state; a human edit after the loop is kept (`kept_changed`); second rollback is a no-op |
| `tests/curation/test_assist_loop_routes.py` | Dry run writes nothing; 409/422 codes of section 10; contract shapes |
| `tests/curation/test_cluster_refresh_loop_hold.py` | `step` does not start a retrain while `loop_live`; `auto_label/start?train_clusters=true` returns 409 `assist_loop_running` |
| `tests/curation/test_propagated_label_kind.py` | Propagated items are not locked, are in `confident_class_sources`, are excluded from the export readiness count, stay out of the residual pool |

**T1 live** (`tests/live/test_live_assist_loop.py`, harness `docker/test/compose.yml`, own `-p`):
seed about 40 COCO val2017 vehicle images with `scripts/curation/seed_live_harness.py`, human-label
5, freeze 3 into the holdout, cluster, run a loop with the fake VLM (`docker/test/fake_vlm.py`
gains a `class_cycle` setting on its test-only control endpoint that cycles answers, so one root fails and
splits), then assert: budget respected, locked and holdout items unchanged, propagated items carry
`class_detector_version = <loop_id>`, rollback restores them, the record reports calls used, items
propagated and clusters left for review.

**T2 real VLM** (isolated stack, GPU 0, the tested local catalog model):

1. The #341 evaluation project (COCO, ground truth): propagated label precision against ground
   truth, false propagation on the similar-class stratum, the leaf partition scored by
   `cluster_eval.py`. Its clusters are tiny (about 4 labeled crops each), so it measures
   precision, not the cost ratio.
2. The 10k COCO rehearsal-lite project (`v070_wave_plan.md` 3.2) for coverage and VLM crops per
   labeled item, once with the stub VLM and once with the real VLM.
3. Set P: a copy of a sample (never the source volume), aggregates only into
   `artifacts_local/assist_loop/`.

Acceptance (issue criteria plus O5, owner to confirm): lock-rule violations 0 (offline per class
and a before/after audit query on the T2 projects); same seed gives the same samples; propagated
documents equal an explicit apply of the same proposal; budget never exceeded; precision at least
0.95 against ground truth on the evaluation project; false propagation on the similar-class
stratum at most 3 %; on the 10k project at least 60 % of eligible items labeled at no more than one
VLM crop per 20 items labeled; audit agreement reported with its bound.

## 13. Build order

Each stage is one commit or a few on branch `feat/v070-assist-loop` (cut from `release/0.7.0`
after W1 merges), with its targeted tests and `.venv/bin/pre-commit run --all-files`; one full
offline suite at the end. Route or wire changes run `make contracts` in the same commit. Nothing is
pushed without the owner's yes.

| Stage | Tier | Work | Verify |
|---|---|---|---|
| S0 | Sonnet writes, Haiku runs | Measurement: `scripts/bench/vlm_rate.py` drives `POST {P}/vlm/label_batch` at concurrency 4/8/16/32 x images per call 1/4/8 on 500 crops, records crops/s, requests/s, p50/p95 latency, `nvidia-smi dmon -s pucv` alongside; both tested catalog models; public COCO crops, set B crops, set P sample (aggregates). Raw JSON to `artifacts_local/bench/v070/w3/`; summary rows to `docs/research/vlm_serving_sizing.md` and `docs/PERFORMANCE.md` | Stack: `docker compose -p op286-m1 ... --profile vlm up -d` with `VLM_GPU_ID=0` and own ports per ground rule 1; `.venv/bin/python scripts/bench/vlm_rate.py --api http://127.0.0.1:<port> --project <p> --concurrency 4,8,16,32 --images-per-call 1,4,8 --n 500 --out artifacts_local/bench/v070/w3/<label>.json`; one row repeated 3x |
| S1 | Sonnet; Opus reviews math | `src/domain/loop_stats.py`, `src/domain/loop_tree.py` | `.venv/bin/python -m pytest tests/curation/test_loop_stats.py tests/curation/test_loop_tree.py -q --no-cov` |
| S2 | Sonnet; Opus reviews | Label kind `propagated`: `PROPAGATED_CLASS_SOURCE`, `ItemLabel.propagated`, `class_label_fields` branch, `confident_class_sources`, VLM selection exclusion (`autolabel/selection.py`, `scripts/curation/vlm_pending.py`) per Q6, `AssistApplyRequest.label_source`, contracts | `pytest tests/curation/test_propagated_label_kind.py tests/test_class_label_single_writer.py tests/curation/test_assist_proposal.py -q --no-cov`; `make contracts-check` |
| S3 | Sonnet | Extract `assist/apply_write.py` `write_class_to_ids`; `apply_proposal` calls it | `pytest tests/curation/test_assist_apply_write_equality.py tests/curation/test_assist_proposal.py -q --no-cov` |
| S4 | Sonnet | Extract `assist/sample_eval.py` `ask_sample` from `runner.run_assist` (shared semaphore parameter); runner calls it | `pytest tests/curation/test_assist_run_classify.py tests/curation/test_assist_run_lifecycle.py -q --no-cov` |
| S5 | Sonnet | `partition.py`, `splitter.py` (`refine_members(n_clusters=...)`, k-means fallback) | `pytest tests/curation/test_orchestrator_residuals_refine.py tests/curation/test_assist_loop_reproducible.py -q --no-cov` (split part) |
| S6a | Sonnet; Opus reviews | `store.py`, `budget.py` | `pytest tests/curation/test_assist_loop_budget.py -q --no-cov` |
| S6b | Sonnet; Opus reviews | `evaluate.py`, `propagate.py`, `driver.py` | `pytest tests/curation/test_assist_loop_lock_rule.py tests/curation/test_assist_loop_drift.py tests/curation/test_assist_loop_veto_dissent.py tests/curation/test_assist_loop_resume.py tests/curation/test_assist_loop_reproducible.py -q --no-cov` |
| S6c | Sonnet; Opus reviews | `audit.py`, `rollback.py` | `pytest tests/curation/test_assist_loop_audit.py tests/curation/test_assist_loop_rollback.py -q --no-cov` |
| S7 | Sonnet | Worker `scripts/curation/assist_loop_worker.py`; compose service in `docker-compose.yml` (profile `curation`) and `docker-compose.dev.yml`; Makefile `CURATION_SERVICES`; the jobs-dir setting (3.1) in `env.template` and `tests/test_env_surface.py`; refresh hold; `auto_label/start` refusal | `pytest tests/curation/test_cluster_refresh_loop_hold.py tests/test_env_surface.py -q --no-cov`; `.venv/bin/python -c "import src.main"` |
| S8 | Sonnet; Haiku checks contracts | Schemas, router, plan, cross-project leak rows, docs | `pytest tests/curation/test_assist_loop_routes.py tests/curation/cross_project_leak -q --no-cov`; `make contracts && make contracts-check`; `.venv/bin/python scripts/docs/check_docs_vs_code.py` |
| S9 | Sonnet | Cropwright panel (section 11) | `cd frontend && npm run check && npm run build`; stubbed e2e; screenshots viewed |
| S10 | Sonnet; Haiku runs | Fake VLM knob, T1 live test; `loop` stage in `scripts/bench/rehearsal_stages.py` once W0 has created it | `docker compose -p op-live-verify -f docker/test/compose.yml up -d --wait`; `.venv/bin/python -m pytest tests/live/test_live_assist_loop.py -q --no-cov -m live`; `down -v --remove-orphans` |
| S11 | Sonnet runs; Opus go/no-go | T2 runs, acceptance numbers into `docs/PERFORMANCE.md`, then the full offline suite once | section 12; `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'` |

## 14. Risks

1. **Wrong labels at scale.** A false merge passes the gate when the VLM shares the confusion.
   Mitigation: theta and alpha per node, validated veto, dissent kept, audit pause, unvalidated
   label kind (excluded from export by default), one-call rollback.
2. **Cost overruns the estimate.** The assist core's rate is unmeasured; children may be too small
   to pay off. Mitigation: S0 first; best-first order; budget hard caps; the record shows the curve.
3. **Clusters move under the loop.** Mitigation: snapshot, drift rule, refresh hold, retrain refusal.
4. **Resume duplicates or loses work.** Mitigation: write-ahead budget, item-as-truth recovery,
   per-event tests at every kill point.
5. **Concurrent writers** (ingest, VLM worker, auto-label, humans). Mitigation: OCC +
   `ClassWriteGuard` token at snapshot; a human always wins.
6. **Worker memory.** AHC at 8,000 members is about 512 MB transient; `node_concurrency` times that
   bounds the worker. Mitigation: at most 2 concurrent splits (a separate semaphore).
7. **External data leak** through the audit endpoint. Mitigation: the existing three-part consent,
   refusal on any missing part, no class writes from audit answers.
8. **Set P exposure.** Mitigation: aggregates only, `artifacts_local/`, no screenshots of set P
   projects, the source volume is read-only to the loop (it only reads crops through the API).

## 15. Open questions for the owner

| ID | Question | Recommendation |
|---|---|---|
| Q1 | Propagated labels as a machine kind (`assist_propagated`, unvalidated, reviewable) or as human-validated like today's apply? | machine kind (D1); apply keeps `human` as its default |
| Q2 | Worker placement: new service, the auto-label worker, or an in-API task with manual resume? | new service (D4) |
| Q3 | Gate defaults: theta 0.9, alpha 0.05, one dissent (35 to 54 crops per node), or theta 0.8 (17 to 26)? | 0.9 for the first runs; revisit with T2 precision |
| Q4 | Publish the loop's first-level split as `cluster_subid` for review, or keep the tree internal? | internal (D2); the review list opens nodes as selections |
| Q5 | Audit endpoint and rule: second local catalog model or a remote small model (with consent), rate 1 %, floor 0.85, pause rather than auto-rollback? | local second model; pause |
| Q6 | May the continuous VLM worker and auto-label relabel propagated items, and may propagated labels enter the training export (wave-plan O3)? | neither by default; export only after review |
| Q7 | Acceptance numbers (precision 0.95, 60 % coverage, 1 crop per 20 items) and which project proves the cost ratio (the #341 set is too fragmented)? | confirm; the 10k rehearsal-lite project for cost |
| Q8 | Hold the refresh retrain and auto-promote while a loop runs, and refuse `train_clusters` starts? | yes |
| Q9 | Capture `usage` tokens in the VLM client for a real token budget on remote endpoints (a separate small change)? | yes, before any remote primary VLM run |
