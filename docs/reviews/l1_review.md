# L1: fake VLM loop modes and the live auto-label loop test

Branch `feat/v070-assist-loop` (worktree `OpenProcessor-wt-loop`), merge base with `main`
`0e2d3fe7`. Issue #286, stage S10 of [`auto_label_loop_plan.md`](../design/auto_label_loop_plan.md).

## What changed

- `merge: bring origin/main into feat/v070-assist-loop` (`0259a440`): one conflict, `CHANGELOG.md`
  (kept both sides); `make contracts` regenerated, no contract diff left.
- `test(curation): follow the residual pool query rename in the propagated-kind test` (`00f7c600`):
  the 2-line edit that was uncommitted. `main` renamed `_residual_pool_query` to
  `residual_pool_query`, so the old test failed at import after the merge. It is the rename, not a
  change of the expected source.
- `feat(frontend): add the auto-label loop panel to the clusters page` (`ccb774fa`): written before
  the re-scope, committed, **not part of L1**. The owner decided the UI lane owns it (UI5). It is
  one self-contained commit and can be dropped with `git rebase --onto` or left for UI5 to start
  from. See the risks section.
- `fix(test): probe the harness API at its configured URL after the restart test` (`4d4a31e3`):
  `test_state_survives_an_api_container_restart` probed a hard-coded port 14701, so it failed on any
  second harness (the documented `VERIFY_*` override).
- `test(harness): add class_cycle, answers and seen modes to the fake VLM` (`b88ce765`).
- Last commit (this one): the loop worker in `docker/test/compose.yml`, `tests/live/test_live_assist_loop.py`,
  harness README, plan status line and the design README row.

## Acceptance

Bars are those of loop plan section 12 "T1 live" plus the L1 text of the ship-track plan.

| Bar | Measured | Met | Raw data |
|---|---|---|---|
| A dry run sizes the loop and writes nothing | `loop_id` null, 3 roots, 58 eligible items, no loop listed, items identical before and after, fake VLM saw 0 calls | yes | `test_a_dry_run_sizes_the_loop_and_writes_nothing` |
| Budget respected | loop 1: 43 of 400 crops, 16 of 150 requests; loop 2: 49 of 400, 18 of 150; loop 3 (cap 40): 39 of 40 crops, 14 of 30 requests, `stop_reason: budget_crops`; reserved plus spent never above a cap; crops spent equals images the fake received | yes | `LOOP-RECORD` lines of the run output |
| Locked and holdout items unchanged | 3 locked and 4 held-out items skipped by the loop, every human, validated, holdout and excluded item has identical class fields after the loop and after the rollback | yes | same tests |
| Propagated items carry `class_detector_version = <loop_id>` | 15 items of loop 1, all `class_source: assist_propagated`, `class_validated: false`, class `box`; the count equals `totals.items_propagated` | yes | same tests |
| Rollback restores them | dry run restores 15 and writes nothing; after a person edits one item, the real rollback restores 14 and keeps 1 (`kept_changed: 1`); restored items equal their pre-loop class fields; a second dry run restores 0 | yes | same test |
| The record reports calls used, items propagated, clusters left for review | `budget.vlm`, `totals.items_propagated`, `clusters_left_for_review`, `items_left_for_review` asserted; loop 2: 0 propagated, 5 clusters and 53 items left for review | yes | same tests |
| A cluster that fails the gate is split, then left for review | `class_cycle` makes a quarter of the answers dissent: every root is `split` (none `pass`), the parts end `review_leaf` or `review_small`, nothing locked is written, a review leaf lists its members | yes | `test_a_cluster_that_fails_the_gate_is_split_then_left_for_review` |
| Fake VLM modes are covered offline | 4 tests, 3 seen failing against the old `fake_vlm.py` | yes | `tests/test_fake_vlm_loop_modes.py` |

Not a quality claim: the fake answers by rule. Precision, coverage and cost ratio are L2 (S11).

## How it was tested

- Red first: `tests/test_fake_vlm_loop_modes.py`, 3 of 4 failed with the old `fake_vlm.py` (stashed),
  4 pass with the change. The live test was run against the harness while it was written: it failed
  four times for four real reasons (freeze percent capped at 50, a label moves the item out of its
  cluster so the locked members left the root, the validated-member veto splitting the root when the
  humans' class differed from the fake's, a retired slug), each fixed in the test, not relaxed.
- Offline: `-k 'loop or autolabel or propagated or contract'` 984 passed, 2 skipped (after the merge);
  `make contracts-check` and `scripts/docs/check_docs_vs_code.py` clean; pre-commit passes on every commit.
- Live harness `loopl1` (own compose project, ports 14711/14712/14714, no GPU, no Triton): the new
  module passes 4 of 4. The whole `tests/live` suite: 95 passed, 9 failed on the first full run.
  Reproduced on a pristine `origin/main` worktree and harness (own project): 8 of the 9 fail there
  too, see risks. The 9th was the hard-coded port, fixed above and re-run green.
- Frontend (the committed panel): `npm run check`, `npm run lint`, 4925 vitest tests and the build
  pass; 18 new tests, 3 seen failing under mutation. It was opened in a browser at 1600 and 430 px
  (axe: no findings on the panel) against a loop-branch API, but only in its idle state; the live
  run with data was not viewed, because the lane was re-scoped. Screenshots are in
  the host scratch directory (not tracked).

## Risks and what was not done

- **Eight live tests fail on `origin/main` itself** (not caused by this lane): the six
  `tests/live/test_live_vlm.py` VLM tests (`vlm_not_configured`), `test_live_clusters.py::test_auto_promote_apply_validates_only_the_high_purity_cluster`
  (`audit_required`) and `test_live_served_facts.py::test_settings_serve_monitoring_links`
  (`KeyError: monitoring_links`). For the VLM ones the cause is narrowed down: right after the
  session's `seed_live_harness.py --wipe`, the default project's `GET .../vlm/endpoints/active`
  answers `source: off`; restarting the API container makes it `env` and `label_batch` works.
  No issue was filed (owner gate); one should be.
- `answers` is keyed by image hash, not crop id, as the plan words it: the VLM wire carries no crop
  id. The fake VLM's `__seen` endpoint lets a test learn the hash the product sends for a crop (ask about one crop,
  read the last entry). Z5-I has to use that or ask for a product change.
- The plan's "one root fails and splits" in a single loop is not possible with a fake that sees only
  image bytes: concurrent calls interleave roots. The live test runs two loops instead (all pass, all
  fail and split).
- The loop worker is a new service in `docker/test/compose.yml` (same image, mounts and environment
  as `api`, shared through YAML anchors).
- The frontend panel is committed on this branch although UI5 owns it. A `paused` loop is shown with
  its reason and a resume button; there is no pause button and no event log, because the API serves
  neither (cancel and resume only; the node table stands in for the log).
- A stand-in run (before the re-scope) used a throwaway project and two VLM endpoints on the
  `optest1` deployment, with a loop-branch API and workers beside it on that stack's network. All of
  it was removed afterwards (project, endpoints, containers). Their effects that remain: the
  deployment-wide revision documents of the deleted endpoints in `optest1`'s global config index, and
  one finished auto-label job record in its jobs volume. Index mapping migrations of the newer
  branch code ran against `optest1` indexes. Nothing else of `optest1` was touched. No GPU was used.
- Not done: S0 (`vlm_rate.py`), S11 (T2 runs, GPU 0 under the lock), the real-VLM and audit
  precision numbers, the push, the PR.

## Decisions needed from the owner (gate ids)

- G1: PR of this branch to `main` (the plan: after L1, owner's yes).
- Whether the frontend panel commit stays on this branch or is dropped for UI5 to write.
- File the issue for the eight pre-existing live failures (or assign them to a lane).

## Five-minute check for the owner

```bash
# from the loop worktree root; MAIN is the main checkout that holds the venv
export VERIFY_COMPOSE_PROJECT=chk VERIFY_CONTAINER_PREFIX=chk VERIFY_API_PORT=14731 \
  VERIFY_OPENSEARCH_PORT=14732 VERIFY_FAKE_VLM_PORT=14734 \
  VERIFY_API_URL=http://localhost:14731 VERIFY_OPENSEARCH_URL=http://localhost:14732 \
  VERIFY_FAKE_VLM_URL=http://localhost:14734
docker compose -p chk -f docker/test/compose.yml up -d --build --wait
$MAIN/.venv/bin/python -m pytest tests/live/test_live_assist_loop.py -rP --no-cov -m live
docker compose -p chk -f docker/test/compose.yml down -v --remove-orphans
```

Expected: 4 passed in about 3 minutes, and three `LOOP-RECORD` lines (completed within budget with
15 propagated; completed with 0 propagated and 5 clusters left for review; stopped at the crop cap).
