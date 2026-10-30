# D1: installer and docs wave

Branch `fix/v070-installer-docs`, merge base `0e2d3fe7` (main after PR #386). Issues fixed here: #371, #363, #355, #383, #154.
Verified only (no code): #335, #373, #377, #378. Nothing is pushed, no issue is closed.

## What changed

| Commit | Subject |
|---|---|
| `279143c4` | feat(search): turn semantic text search on by default (#371) |
| `aac1d466` | fix(trainer): register models with a staging alias, not registry stages (#154) |
| `59ec5af5` | docs(clustering): correct comments and docs that contradicted the code (#355) |
| `3e8c3fbe` | test(ingest): assert ordering and thread identity, not elapsed time (#383) |
| `4a7f3a20` | fix(installer): interrupt message, tier-aware status, target-fs disk check, clearer summary (#363) |

The review file itself is added by the wave's last commit.

## Acceptance

| Bar (plan section 9) | Measured | Met | Evidence |
|---|---|---|---|
| #371 text search on by default, explicit off still works | Unset flag gives 200 on `GET .../search/text`; `0` and `false` give 400; `/methods` serves `semantic_search_enabled: true` when unset | Yes (router level) | `tests/curation/test_search_router.py::test_search_text_is_on_when_the_flag_is_unset`, `::test_search_text_explicit_off_still_returns_400`, `tests/curation/test_methods_router.py` |
| #371 "a fresh unattended install answers text search with 200" | Not run live (no stack in this wave). The installer-written `.env` is a copy of `env.template`, where the flag is a comment, and no compose file sets it, so a fresh install takes the code default | Not live | Run the five-minute check below on an isolated stack |
| #371 startup cost and memory of the text encoder warm-up | No change: `_warm_pe_encoder` in `src/app/lifespan.py` already ran at every API start whatever the flag said, so turning the flag on adds no startup or memory cost. Not measured on a 24 GB card | Not measured (nothing new to measure) | code reading |
| #363 each item fixed or documented, with a test where testable | 7 of 7 items done, see the table below | Yes | `tests/installer/` |
| #355 every listed line corrected; `backbone_embedding` removed or kept with reason | All six items corrected; field kept with the reason written in the mapping comment (owner decision D1-1) | Yes | `scripts/docs/check_docs_vs_code.py`: 0 problems in all five checks |
| #383 passes 20 times under load, seen failing first | Old tests failed 1 of 6 runs under 120 busy-loop processes (load average about 24, rising to over 200). New tests: 20 of 20 pass under 120 busy-loop processes (load average 37 at start, over 200 at the end). Both new tests fail when the code under test is broken (reads or decodes run on the loop thread) | Yes | `tests/test_ingest_file_read.py`, `tests/test_ingest_offload.py` |
| #154 trainer test asserts the alias call and no stage call | `test_register_model_sets_the_staging_alias_and_never_a_stage`; the fake client raises on `transition_model_version_stage`. Seen failing before the fix (the push was swallowed as "registry push failed") | Yes | `tests/curation/test_mlflow_callbacks.py` |
| #154 "no MLflow deprecation warning in a trainer run" | Not run (needs the training profile). The deprecated call is gone from the source | Not live | |

### #363 items

| Item | Result | Test |
|---|---|---|
| Ctrl-C trap with a resume message | Done: message on the saved terminal fd (the logging pipe's `tee` gets the signal too), exit 130 or 143, nothing rolled back | `tests/installer/test_installer_interrupt.py` (SIGINT and SIGTERM, red first) |
| `status` probes the stack's own ports and tiers; `HTTP 000000` | Ports already came from `.env`; now a service not in `docker compose config --services` prints "not part of this install" and is not probed, and a refused probe prints `HTTP 000` once | `tests/installer/test_cli_status.py` (3 tests, 2 red first) |
| Dry-run disk check on the target filesystem | Checks the nearest existing parent of the real install dir | `test_a_dry_run_checks_free_disk_on_the_target_filesystem` |
| Control-plane summary wording | Prints `API alive http://...:<port>/` and says `/health` is 503 by design; default install unchanged | `test_control_plane_only_summary_points_at_the_liveness_url_not_the_503` |
| Short SHA | 12 characters in messages and summary; `state.json` keeps the full SHA | `test_branch_mode_prints_the_short_sha_but_records_the_full_one` |
| Branch-mode `sha-<12 hex>` tag | CI does not publish these tags (no workflow or release script builds them), so documented in `INSTALLATION.md` and the drive log: tag a local build or pass `--image-tag` | docs check |
| Upgrade note for control-plane-only installs | In CHANGELOG upgrade notes and `docs-site/docs/operations/upgrade-and-backup.mdx` | docs check |
| `action install` logged during upgrade (cosmetic) | The log line now says the directory already holds an install | no test (cosmetic) |

`docs/research/installer_drive_log.md` section "Messages a first-time user found confusing" now lists each as fixed.

### Verify-only issues: evidence on `main`

All four fixes are on `origin/main` (merge `0e2d3fe7` contains them). Each fix's tests were re-run on this worktree (main plus D1) and then re-run with only the fix's source files reverted to the parent commit, to prove the tests catch the bug.

| Issue | Fix commits | Check | Result |
|---|---|---|---|
| #373 | `9de79579` | `tests/installer/test_gpu_plan.py`, `test_ingest_gpu_decode_profile.py`, `tests/test_gpu_image_prep_instances.py` (107 passed with #377's and #335's tests). Issue repro: 2x24 GB plans `standard`, decode 0; 1x32, 1x48, 40+24 and 32+48 GB hosts plan `standard`; only 2x48 GB plans `full` with decode 1 | Fixed. With `setup-openprocessor.sh` and `profiles/full.json` reverted to the parent, 13 of those tests fail |
| #377 | `e76c3574` | `tests/installer/test_model_seed_sync.py`; upgrade page re-embeds `parts: ["crop","region"]` and names `./openprocessor models install --only preflight`, not a checkout-only command; `export/sync_model_seed.py` replaces older `MODEL_VERSION` dirs | Fixed. With `scripts/lib/model_setup.sh` reverted, `test_the_preflight_group_syncs_instead_of_cp_rn` fails. A real 0.6.0-to-0.7.0 upgrade is still part of the release acceptance (plan 11.1 item 6), not run here |
| #378 | `b443a66e` | `sha256sum -c docs/benchmarks/cluster_eval/SHA256SUMS` all OK; re-ran the documented `cluster_eval.py score` command offline: 128 clusters, pair precision 0.9051, recall 0.0875, 28 mixed clusters, identical to the committed `score.json` on every key it shares (the scorer has since gained BCubed, AMI and risk-coverage keys); `v070_measurement_log.md` has a raw-data index. `MANIFEST.sha256` lists only large arrays kept outside git, by design | Fixed as documented. The `ensure_indexes` and worker-stage (86 against 220 ms) figures still have no raw data; the log says so and asks for a re-time (tracked with #362 in SP2) |
| #335 | `2b0be5d0`, `d8d02a74`, `869e9882`, `d3bac95d` | `tests/test_docs_screenshot_seed.py` (18 passed); all 17 listed states have a committed `-1600.png` under `docs-site/static/img/screenshots/cropwright/` | Code and images present. The seeded capture was not re-run here (needs the isolated GPU-less stack); the "re-run produces the same set of files" bar is therefore not re-verified |

## How it was tested

Red first, with the failing run observed:
- `test_search_text_is_on_when_the_flag_is_unset`, the `/methods` flag test and the feature-flag consistency test, before the default change.
- `test_register_model_sets_the_staging_alias_and_never_a_stage`, before the alias change.
- The four installer tests above; all seven new installer tests fail against the unmodified `setup-openprocessor.sh` and `openprocessor` and pass with the change.
- #383: both tests fail when the code is broken (read inline on the loop, decode inline on the loop), checked by temporary edits that were restored.

Suites run on this branch: `tests/installer` (527 passed, run twice), the targeted unit tests (233 passed), `scripts/docs/check_docs_vs_code.py` (0 problems), pre-commit on every commit (no `--no-verify`). The full offline suite was not run (plan: once at push). `make docs-build` (npm ci and the Docusaurus build, which fails on broken links and anchors) succeeded.

Host facts: no `stress-ng` here, so load came from 120 busy-loop Python processes (recorded PIDs, killed afterwards).

## Risks and what was not done

- The installer now sources a new library, `scripts/lib/install_summary.sh` (listed in `release-manifest.txt` and in `tests/installer/test_static.py`). `setup-openprocessor.sh` was at its size baseline (3,183 lines, one function by design), so the summary moved out and the baseline dropped to 3,141. An installer from this tree run with `--version` of an older release that lacks the file would fail after download; the bootstrap path re-fetches the installer from the same release, so it is consistent there.
- `exec {_OP_TTY_FD}>&2` leaves one extra open fd in child processes of the installer. Harmless; containers started with `up -d` do not inherit it.
- The default flip changes behaviour for existing installs that never set the flag: text search now answers instead of 400. Cropwright's search box appears for them.
- The `#346` epic text still says "FAISS IVF, AHC refinement inside one cluster, residual UMAP" and `POST /clusters/refine/{cluster_id}`. GitHub issue bodies are not edited from here; suggested text: "IVF k-means over the item vectors, AHC refinement inside one cluster (`POST /curation/projects/{project}/clusters/refine/{cluster_id}`); UMAP only feeds the 2-d scatter and never clustering".
- Other docs and comments still name some routes without the `/curation/projects/{project}` prefix (for example the `/curation/review/*` paths); only the lines listed in #355 were in scope.

## Decisions needed from the owner (gate ids)

- D1-1: keep `backbone_embedding` (written at ingest by the dual-head export path, read by nothing) or remove the field, the ingest writer, the config dimension and the export docs as dead code? I kept it and wrote why in `bodies_core.py`; removal needs a mapping migration rule for existing indexes and is a feature removal.
- G5: close #335, #373, #377, #378 (evidence above), and #371, #363, #355, #383, #154 once D1 is merged and the live checks below pass.
- G1: push, PR and merge of `fix/v070-installer-docs` (first in the merge order).

## Five-minute check for the owner

1. `git log --oneline origin/main..fix/v070-installer-docs` shows the five commits plus this review.
2. `.venv/bin/python -m pytest tests/installer -q --no-cov -n 8` gives 527 passed.
3. Text search default, on an isolated stack with an empty `OP_SEMANTIC_SEARCH_ENABLED`: `curl -s -o /dev/null -w '%{http_code}\n' "http://127.0.0.1:<api-port>/curation/projects/default/search/text?q=truck"` prints 200 (503 while the encoder warms, 400 only with the flag set to `0`).
4. `./openprocessor status` on a control-plane-only install prints "Triton ... not part of this install" and "Grafana ... not part of this install".
5. Start an install with `setup-openprocessor.sh --unattended ...`, press Ctrl-C during the image pull: the last line says to re-run the same command and the exit status is 130.
