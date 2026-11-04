# OpenProcessor: smaller-workspace TensorRT builds and 12 GB GPU guidance (issue #42)

Status: **PLAN ONLY, not implemented.** Written 2026-11-01 against branch
`build/41-slim-images` at `d370cd16` (issue #41, not yet merged), with `main` read for context.
All `file:line` citations are on that branch unless marked `main`. Executor: an implementer with
no memory of this session, working tests-first.

## 0. TL;DR

1. **One knob, one place.** The TensorRT workspace limit is set today in about ten places with
   three different values (4 GB, 8 GB, 8192 MiB), some as literals. Collapse them into one
   setting, `OP_TRT_WORKSPACE_GB` (default unchanged: 4, PE 8 stays as its per-model default
   unless J2 says otherwise), read by one Python helper and one shell helper, and exposed as
   `./openprocessor models install --workspace N` (and `repair`, `export`).
2. **Stamp it.** Put the workspace (and engine max batch) in the engine build stamp
   (`.install/precision.tsv`), so a changed value rebuilds instead of being skipped as current.
3. **Be honest about what the workspace does.** The workspace limit caps *build-time* tactic
   scratch memory. It is not what makes a 12 GB card run out at *serve* time. Runtime memory is
   engine weights plus per-execution-context activation memory sized by the engine's *max
   optimisation-profile shape*, times Triton instance count. The real 12 GB lever is (a) the
   instance count the `standard` profile writes and (b) the engine's max batch. The plan adds a
   `small` build target that lowers both workspace and engine max batch, and documents that.
4. **Guidance and measurement.** Add a 12 GB section to `gpu-sizing.mdx` with per-model instance
   counts derived from measured numbers, not estimates. The measurement step is gated (J1): the
   executor measures only on **GPU 0, under the host's GPU 0 lock (one GPU job at a time)**, and the
   real 12 GB card run (GPU 1) is an owner-coordinated step.
5. **Depends on #41**: every path in this plan (the `model-export` service, `_ms_export`,
   `export/README.md` "Where these scripts run") exists only on `build/41-slim-images`. Implement
   on a branch cut from it after it merges (or stacked on it; J5).

Not decided here: J1 to J6 (section 8).

## 1. Why the obvious approach is wrong

The issue reads "engines built for 48 GB cards can run out of memory on a 12 GB card" and asks
for "smaller-workspace engines". Two problems:

- Engines are **not** built per card size. Every exporter uses a fixed workspace (4 GB or 8 GB)
  whatever the card (section 2). An install on a 12 GB card already builds with the same limits
  as one on a 48 GB card.
- `set_memory_pool_limit(WORKSPACE, ...)` / `--memPoolSize=workspace:` is an upper bound on scratch
  memory the builder may let a tactic use. A lower cap can exclude the fastest tactics and change
  the engine's runtime scratch requirement, but it is not the dominant term. A max-batch-128
  ArcFace profile or a max-batch-64 SCRFD profile reserves activation memory for that shape in
  every execution context, and Triton creates one per instance.

So a workspace flag alone will likely not move the "peak below 11 GB" criterion much. The plan
still ships it (the build itself can OOM on a small card that Triton already shares, which is the
`permanent:oom` case), but pairs it with max batch and instance counts, and measures all three.

## 2. What is there today (verified)

### 2.1 Workspace limits, every site

| Site | Value | How set |
|---|---|---|
| `export/yolo_export_settings.py:10` | 4 GB | `WORKSPACE_GB = 4`, used by `yolo_trt_build.py:21,59-60`, `yolo_formats.py:142,145,322,325`, `export_models.py:90,429` |
| `export/export_yolo26.py:89,194` | 4 GB | its own `WORKSPACE_GB = 4` (duplicate constant) |
| `export/scrfd_constants.py:8` -> `export/scrfd_triton.py:15,59` | 4 GB | `WORKSPACE_GB = 4` |
| `export/export_face_recognition.py:48,339` | 4 GB | another duplicate `WORKSPACE_GB = 4` |
| `export/export_mobileclip_image_encoder.py:317` | 4 GB | literal `4 << 30` |
| `export/export_mobileclip_text_encoder.py:294` | 4 GB | literal `4 << 30` |
| `export/export_paddleocr_det.py:149` | 4 GB | literal `4 << 30` |
| `export/export_paddleocr_rec.py:235` | 8 GB | literal `8192 * (1 << 20)` |
| `scripts/lib/model_setup.sh:262` (`_ms_pe_trtexec`) | 8 GB | hard-coded `--memPoolSize=workspace:8G`; ignores `WORKSPACE` of `export/build_pe_trt.sh:47` |
| `export/build_pe_trt.sh:27,47,181,194` | 8 GB | `WORKSPACE="${WORKSPACE:-8G}"` env (manual path) |
| `scripts/lib/export.sh:33` | 4 GB | `TRT_WORKSPACE="${TRT_WORKSPACE:-4G}"`, used by `scripts/lib/paddleocr_export.sh:150,163,206,216` |
| `scripts/export_paddleocr.sh:67` | 8 GB | `TRT_WORKSPACE="8G"`, **unconditional**, contradicts `export.sh:33` |

No `OP_*` variable controls any of these (`rg 'OP_.*WORKSPACE|OP_TRT'` over `src scripts export`
is empty). `export/README.md:390-395` says "Workspace: 4GB" for all exports, which is false for PE
and PaddleOCR rec.

### 2.2 Engine max batch (the runtime term)

- `export/export_face_recognition.py:42` `MAX_BATCH_SIZE = 128`, `--max-batch` CLI at `:455`.
- `export/scrfd_constants.py:7` `MAX_BATCH_SIZE = 64`.
- `export/export_yolo26.py:62,69,74` per-variant `max_batch` 128 / 32 / 32.
- `export/pe_image_constants.py:27` `DEFAULT_MAX_BATCH = 32`; PE trtexec `--maxShapes=images:32x...`
  at `scripts/lib/model_setup.sh:261`.
- The `standard` profile sets Triton `max_batch_size` 32 and 2 instances per model
  (`docs-site/docs/deployment/gpu-sizing.mdx:24-28`), but the **engine** profile max shape is
  fixed at export; a profile cannot shrink activation memory already baked into the engine.

### 2.3 Build placement (on #41)

- One-shot builds run in the `model-export` service: `scripts/lib/model_setup.sh:229`
  `_ms_export() { dc run --rm --no-deps -T -w /app model-export "$@"; }`;
  `docker-compose.yml:330-363`, profile `setup`, GPU reservation `device_ids: ['${API_GPU_ID:-0}']`.
- The upload/export API path runs in `model-export-worker` (`docker-compose.yml:369+`, profile
  `export`, `export.export_worker --watch /jobs/model_export`).
- The PE engine is **not** built in `model-export`: `_ms_pe_trtexec` calls `_ms_triton trtexec`
  (`model_setup.sh:230,254-264`), i.e. a one-shot `triton-server` container on the Triton GPU.
- The build stamp's compute capability is read from `TRITON_GPU_ID`
  (`model_setup.sh:286-288`), while Python builds run on `API_GPU_ID`. Same-architecture hosts
  are fine; see edge case E7.

### 2.4 OOM handling

- `scripts/lib/model_setup.sh:66-75` classifies `out of memory`, `CUDA initialization failure with
  error: 2`, `cudaErrorMemoryAllocation` as `permanent:oom` (no retry).
- Hint at `:97-98`: "move Triton with --gpu-plan triton=N, or free the card, then re-run". It does
  not mention the workspace or the build target, and it names Triton even though, on #41, Python
  builds sit on `API_GPU_ID`.
- `export/yolo_formats.py:32-44` `check_gpu_memory` reads `total_memory` and logs it as "free";
  it only warns, never fails. A bug to fix in this change (it is the one pre-build check).

### 2.5 Stamps do not record workspace or max batch

- Python builds stamp `"trt-python-api ${precision} input=..."` (`model_setup.sh:392-400`).
- PE stamps a static `_MS_PE_FLAGS="min1/opt8/max32 336x336 workspace8G strongly-typed input=uint8"`
  (`:282`), a string literal that would go stale silently if the trtexec call changed.
- `model_setup_stamp_stale` (`:347-374`) checks precision, input dtype, compute capability and ONNX
  hash only. A rebuild with a different workspace or max batch would be skipped as current.

### 2.6 Docs today

`gpu-sizing.mdx:24-28` lists `standard` for 12-39 GB with 2 instances per model; `:108-113` gives
"1-2 instances on a 12-24 GB card"; nothing measured on a 12 GB card. Per-model memory figures
(#393) exist; `docs/PERFORMANCE.md:866+` holds the 3080 Ti INT8/`bmax` study (triage comment).

## 3. Requirement 1: a documented smaller-workspace build option

Recommended:

- New `export/trt_build_settings.py` (or extend `yolo_export_settings.py` if J3 says so) with one
  function `workspace_bytes(model: str) -> int` and `max_batch(model: str, default: int) -> int`.
  Defaults keep today's behaviour exactly: 4 GB for all but PE (8 GB) and PaddleOCR rec (8 GB).
  `OP_TRT_WORKSPACE_GB` (int, >= 1) overrides all models; `OP_TRT_BUILD_TARGET=small` selects the
  small table (workspace 2 GB; max batch halved, floor = the profile's Triton `max_batch_size`,
  see E3). Invalid values raise (fail closed), never fall back.
- Every Python site in 2.1 calls it. Delete the duplicate `WORKSPACE_GB` constants and the
  literals in the same change (no shim).
- Shell: one helper in `scripts/lib/model_setup.sh` (`_ms_trt_workspace MODEL` printing e.g. `8G`)
  used by `_ms_pe_trtexec` and `paddleocr_export.sh`. `scripts/lib/export.sh:33` and
  `scripts/export_paddleocr.sh:67` read it instead of their own values. `build_pe_trt.sh` keeps
  its `WORKSPACE` arg for manual use but defaults from the same table.
- CLI: `./openprocessor models install|repair --workspace N` and `--build-target small` set the
  env for the `model-export` run (pass with `-e` on the `dc run`, and to the `triton-server`
  trtexec run). The worker (`model-export-worker`) reads the same env from `.env`.
- Add both variables to `env.template` (advanced tier) so `check_docs_vs_code.py` accepts docs
  naming them.
- `permanent:oom` hint: append "or rebuild with `--build-target small`", and name the card the
  build actually ran on.

Rejected: a per-card auto-picked workspace (by free VRAM). Free VRAM at build time depends on what
Triton holds then, so the same host would build different engines run to run; non-reproducible.

Rejected: lowering the default for everyone. It can cost throughput on large cards; unmeasured.

## 4. Requirement 2: 12 GB instance guidance

Add a "12 GB cards" section to `docs-site/docs/deployment/gpu-sizing.mdx`: a table per model
(engine weights MiB, per-instance context MiB at the engine max batch, recommended instances for
12 GB) built **only** from the measurement in section 6. If the measurement shows `standard` (2
instances) does not fit 11 GB, the outcome is either a profile change (J4) or a documented
"build-target small + 1 instance" recipe. Fix `export/README.md:390-395` to state the real
per-model defaults and the new knobs, and add a Troubleshooting entry for `permanent:oom`.

## 5. Requirement 3: READY + live suite + peak memory on 12 GB

The acceptance criterion asks for a real 12 GB card (owner decision 2026-10-29: the 3080 Ti, GPU
1, after coordinating with other host users). That is J1; the executor does **not** use GPU 1.
What the executor runs instead is in section 6.

## 6. Verification plan

### 6.1 Offline (tests first; watch each fail before the fix)

1. `tests/test_trt_build_settings.py`: defaults equal today's values per model (4 GB / PE 8 GB /
   rec 8 GB); `OP_TRT_WORKSPACE_GB=2` applies to all; `OP_TRT_BUILD_TARGET=small` halves max batch
   with the floor; `0`, `-1`, `abc`, `small2` raise. Red today: module missing.
2. A ratchet test: `rg`-style scan over `export/` and `scripts/` that fails on any
   `set_memory_pool_limit(` or `--memPoolSize` not going through the helper, and on any remaining
   `WORKSPACE_GB =` constant. Red today: ten hits. Make it enumerate all hits, not stop at the
   first, and assert the count of allowed call sites is > 0 (an empty scan must not pass).
3. Stamp test (bats or pytest driving the shell helper, following the existing model_setup tests):
   a stamp written with workspace 4G is stale when the requested workspace is 2G; a stamp missing
   the new field is stale ("stamp predates workspace field"). Red today: `model_setup_stamp_stale`
   ignores it.
4. PE: `_MS_PE_FLAGS` derived from the same values passed to trtexec (test both strings come from
   one source). Red today: two literals.
5. `check_gpu_memory` reports free memory (`torch.cuda.mem_get_info`) and the test asserts on
   free, not total. Red today.
6. `permanent_hint permanent:oom` mentions `--build-target small`.
7. `.venv/bin/python scripts/docs/check_docs_vs_code.py` passes with the new `OP_*` names.

Run: `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'` (targeted files while
iterating, full offline suite once at the end), then `.venv/bin/pre-commit run --all-files`.

### 6.2 GPU measurement (executor)

**GPU 0 only, every command under the host's GPU 0 lock (one GPU job at a time).** Never GPU 1 or 2.
Use its own compose project name (`docker compose -p op42-measure ...`), never the default live
project. No sleep/poll loops; run long builds with background execution and wait for completion.

1. Build the `standard` model set twice into a scratch model repo: default target and
   `--build-target small`. Record per-engine build peak (`nvidia-smi --query-gpu=memory.used`
   sampled by `nvidia-smi dmon`/`--loop-ms` into a file for the duration, killed afterwards by PID)
   and build time.
2. Load each set in Triton with the `standard` profile, GPU decode off; record memory after READY
   and peak during a short live-suite run. This gives per-model numbers on an A6000; memory
   per context is architecture-dependent only weakly (both cards are sm_86), so it is a valid
   estimate, labelled as measured on GPU 0, not a 12 GB pass.
3. Parity: run the existing embedding parity gates on the small-target engines; cosine p01 >= 0.995
   required. A failure is a finding, not a threshold to relax.
4. Throughput on set A for both targets (public data only); record in `docs/PERFORMANCE.md`.

### 6.3 Real 12 GB acceptance (owner step, J1)

Same as 6.2 steps 2-4 on the 3080 Ti, after the owner coordinates the card. Pass: all `standard`
models READY, peak < 11 GB, live suite green. Optionally a CUDA MPS memory cap on GPU 0 as the
documented fallback (owner decision text), under the same lock.

## 7. Corrections to the issue's premises

1. "Engines built for 48 GB cards": there is no per-card build. Workspace is a fixed 4 or 8 GB
   on every host (2.1).
2. "Smaller-workspace engines" will mostly not fix serve-time OOM: runtime memory is driven by
   engine max-batch profiles and instance counts (2.2). The workspace knob mainly helps the
   *build* fit next to a loaded Triton.
3. `export/README.md:394` "Workspace: 4GB" for all exports is wrong: PE and PaddleOCR rec use 8 GB,
   and `scripts/export_paddleocr.sh:67` forces 8 GB over the 4 GB default of `scripts/lib/export.sh:33`.
4. On #41 the engine builds run in `model-export` on `API_GPU_ID`, not in Triton; only the PE
   trtexec build still runs in a `triton-server` container. The OOM hint names only Triton.
5. Changing any build setting today would not trigger a rebuild: stamps omit workspace and max batch.

## 8. Judgment calls

- **J1: 12 GB acceptance run.** The owner decided GPU 1 (3080 Ti) with host coordination; this
  plan's executor is restricted to GPU 0 under the lock. Recommend: executor ships code + A6000
  measurements; the owner schedules the GPU 1 run as a separate step. Trade-off: the issue
  cannot close on the executor's PR alone.
- **J2: small target values.** Recommend workspace 2 GB and max batch halved (floor = profile
  `max_batch_size`). Alternative: only workspace, no max-batch change (smaller diff, likely no
  serve-time gain). Final numbers should follow 6.2.
- **J3: new module vs extend `yolo_export_settings.py`.** Recommend a new
  `export/trt_build_settings.py`, since the setting is not YOLO-specific.
- **J4: change the `standard` profile if 2 instances do not fit 11 GB.** Options: split
  `standard` at 24 GB (new `compact` profile for 12-23 GB), or keep it and document the override.
  Recommend deciding after the measurement.
- **J5: branch base.** Recommend waiting for #41 to merge, then branch from main. Stacking on
  `build/41-slim-images` is possible but every #41 rebase drags this along.
- **J6: build placement mismatch.** Python builds on `API_GPU_ID`, PE build and the stamp's
  compute capability on `TRITON_GPU_ID`. Recommend recording the build card's compute capability
  per engine (stamp from the card that built it) in this change; the alternative is a separate
  issue.

## 9. File-by-file change list

Export (Python): new `export/trt_build_settings.py`; edit `yolo_export_settings.py`,
`yolo_trt_build.py`, `yolo_formats.py` (also the `check_gpu_memory` fix), `export_models.py`,
`export_yolo26.py`, `scrfd_constants.py`, `scrfd_triton.py`, `export_face_recognition.py`,
`export_mobileclip_image_encoder.py`, `export_mobileclip_text_encoder.py`,
`export_paddleocr_det.py`, `export_paddleocr_rec.py`, `pe_image_constants.py` (max batch source),
`build_pe_trt.sh` (default from table). Check `export/export_worker.py` passes env through.

Installer/CLI: `scripts/lib/model_setup.sh` (helper, `_ms_pe_trtexec`, `_MS_PE_FLAGS`, stamp
field, `model_setup_stamp_stale`, OOM hint, `-e` passing on `_ms_export`/`_ms_triton`);
`scripts/lib/export.sh:33`; `scripts/export_paddleocr.sh:67`; `scripts/lib/paddleocr_export.sh`;
the `models` subcommand argument parsing in `scripts/lib/cli_*.sh` / `openprocessor`.

Config/docs: `env.template`; `export/README.md`; `docs-site/docs/deployment/gpu-sizing.mdx`;
`docs/PERFORMANCE.md`; `INSTALLATION.md` CLI section for the new flags.

Tests: section 6.1. No API route or wire change, so no `make contracts`.

## 10. Edge cases and traps

- E1: A ratchet that greps only `export/*.py` misses `scripts/lib/*.sh` trtexec calls and
  `build_pe_trt.sh`. Scan both trees.
- E2: Stamps written before this change have no workspace field. They must be treated as stale
  once, not as matching "default". Otherwise an install that picked `--build-target small` keeps
  the old big engines and reports done.
- E3: An engine max batch below Triton's `max_batch_size` in `config.pbtxt` makes Triton fail to
  load or reject batches. The small target must floor at the active profile's value, and a test
  must cover the profile pairing.
- E4: `--workspace` passed on the CLI but not forwarded into `dc run` is silently ignored; the env
  var only exists in the host shell. Test that the run command line carries `-e OP_TRT_...`.
- E5: The `model-export-worker` (API upload path) reads `.env`, not CLI flags; a user who built
  with `--build-target small` and then uploads a model gets a default build. Document, or persist
  the choice in `.env` (recommended: the CLI writes it).
- E6: TensorRT may still pick a tactic needing more than a small workspace and fail the build
  rather than skip it on some layers; log and classify as a build failure, not `oom`.
- E7: compute capability is read from `TRITON_GPU_ID` while the build runs on `API_GPU_ID` (J6).
- E8: `check_gpu_memory` currently passes even with too little memory; after the fix, keep it a
  warning (the build is the real test), but log the true free figure.
- E9: A measurement that reads `memory.used` once after READY misses the load-time peak. Sample
  for the whole run.
- E10: The FP32 fallback path (`*_fallback` precision) roughly doubles engine size; a 12 GB result
  is only valid if every engine is the FP16 designed precision (`models status`).

## 11. Phasing

1. After #41 merges: settings module + helper + ratchet + stamp field + CLI flags + OOM hint +
   `check_gpu_memory` fix + README correction (offline tests only). One PR.
2. GPU 0 measurement (6.2) under the lock; docs tables and `PERFORMANCE.md` from it; settle J2/J4.
   Same PR if quick, else a follow-up.
3. Owner-scheduled GPU 1 acceptance run (6.3); close #42 on that result.

## 12. Security

No security finding bears on this design. Forward-looking constraint: build-setting env values
reach shell command lines (`--memPoolSize=workspace:$X`); validate them as integers in the helper
before use, never interpolate raw strings.
