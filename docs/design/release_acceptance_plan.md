# Release acceptance plan: one scripted end-to-end run per release

Status: **open** (partially done). Issue: #54. Related: #44 (installer live test on a clean
host), #43 (docs screenshots). `docs/releases/RELEASE_CHECKLIST.md` covers building, pinning and
publishing; `tests/live/` covers API-level checks against a running stack. Nothing yet runs the
whole product the way a new user would. This plan is that run: the gate between "release candidate
built" and "tag and publish".

A fresh agent with no memory should be able to execute this from the file. Each phase has pass
criteria. Record every step as PASS, FAIL, PASS-WITH-FINDING or BLOCKED (with reason), plus wall
time and an evidence path. A failing step becomes a finding; never relax a criterion to make it
pass. Confirm every route against the running stack's OpenAPI document
(the API's served OpenAPI document, or `contracts/openapi/curation.json`) before calling it.

## 1. Rules for the run

1. **Public data only.** Use license-filtered COCO 2017 through the repo's own fetchers and the
   `examples/` profiles. No private exports, weights, class names or project names anywhere,
   including slugs, screenshots and the findings text.
2. **Isolated stack.** Use a dedicated compose project name (`docker compose -p <name>`), a
   dedicated install directory, a non-default `--port-base`, and locally tagged images
   (for example `acc/<image>:<version>-rc1`). Never run bare `docker compose` or `make up` from a
   checkout while another stack uses the default project name, and never use a registry tag that
   an auto-updater could re-pull.
3. **Dedicated GPUs.** Pin every GPU key of the install to GPUs that nothing else uses. Record
   `nvidia-smi` before and after.
4. **Safety snapshot first.** Before touching anything, record the running containers (name,
   start time, image), volumes, networks and GPU memory. After the run, compare: nothing outside
   the run's project may have changed.
5. **Never read secret files.** Inspect configuration with `./openprocessor config show` (it
   redacts values). Pass tokens by file path, never on the command line, and grep every log for
   token leaks at the end.
6. Evidence (logs, JSON, screenshots) goes in a gitignored directory. The only file written to
   the repo is the findings document (section 4).

## 2. Entry gate

All must hold before phase A:

- every feature branch for the release is merged to `main`; record the commit SHA;
- the gate passes on that SHA: the full pytest suite (without `tests/live`), the pre-commit run
  over all files, and `scripts/codegen/generate_contracts.py --check`;
- the frontend release candidate is built from its own tagged commit and its contract check
  passes against the backend SHA;
- the release scripts are consistent with the installer: feed the lock file written by the
  release script into the installer's lock validator (a test, not by eye), and confirm the
  release manifest still lists every file the installer downloads.

## 3. Phases

Each phase maps to a user requirement. Budget about 12 to 15 hours on one 48 GB GPU; the natural
break is after phase G with the stack left up.

| Phase | What it proves | Pass criteria (all required) |
|---|---|---|
| A. Preflight | Safe to start | snapshot files written; install directory absent or empty; the target GPU has the free memory the tiers need |
| B. Build the release candidate | The release assets are real | all backend images and the frontend image build under local tags; the Trivy CRITICAL gate runs and passes (allowlist: `scripts/release/trivy-allowlist.txt`); `scripts/release/verify_release_assets.sh` passes; the class-identity end-to-end test exists and is green |
| C. Clean install | A stranger can install | the piped one-liner (`curl ... \| bash`) from an empty directory exits 0 using the release candidate assets; every health check is green; ports bind to loopback unless LAN access was chosen; no token appears in any log; the installer summary prints the URLs and the no-login warning |
| D. Projects and isolation | Datasets never mix | two projects created through `/curation/projects`; the default project is archive-only; API, event stream, logs and metrics of one project show nothing of the other; capacity is served and consistent with the shard budget |
| E. Data per project | Every data path works | a YOLO-format import and a COCO-format import (via `/datasets/preview` then `/datasets/imports`), an own-export round trip, and plain ingest, one per project; class names map by name, never by index; a second import of the same files is a no-op |
| F. Pipeline | Detect, regions, embeddings, VLM | region detection with the multi-box example profile (parent class to child boxes, text-free mode, VLM verify); embeddings and clusters populate; VLM labeling stamps provenance; the review UI (per-box keys, multi-box edit, undo, stale-edit guard) agrees with the API; prompt-pack and region-profile editors, test-on-crop and rollback work; keymap change, conflict, lock and reset work; VLM selection works through API, CLI and UI |
| G. Train, bake-off, promote, inference | Class identity survives every hop | freeze the test holdout; export (`export/yolo`) and check the registry files; train; run the bake-off; promote; run inference; at **every** hop (export registry, training labels, bake-off report, promoted model labels, inference output) class id and name match the project registry. Sharing a promoted model with another project is opt-in and surfaces unmapped names |
| H. Combine | A superset project is correct | `/curation/projects/combine/preview` then `/curation/projects/combine`: counts, class mapping, preserved test splits, flagged label conflicts and provenance are right; source projects are byte-identical before and after (compare index doc counts and a checksum of their registries) |
| I. Operations | Day-2 behavior | reprocess respects the human-lock rule and reports skipped items; import undo keeps human edits and a second undo dry run is zero; archive and delete guards fire (protected default, busy project); dashboards, log search and experiment tracking show live data; the capacity check agrees with the cluster's shard settings |
| J. Installer lifecycle | The CLI is safe to re-run | re-run is a no-op (no container recreated, model groups report up to date); repair rebuilds only a deliberately broken model group; upgrade then rollback restore the previous state; a stack from another compose project is untouched; uninstall removes only this install |
| K. Visual acceptance | The UI and docs look right | every frontend screen and every docs page captured at 1600 px and 800 px wide, each screenshot opened and inspected (not only DOM or API assertions); read-only live tests green |
| L. Teardown | Nothing leaked | the stack and volumes removed; the before/after snapshot comparison shows no change outside the run |

Notes that apply across phases:

- Monitoring ships with default credentials for Grafana. Record the use of the shipped default
  as a security finding in the report unless the installer prompts for a new one.
- The segmenter and the VLM share a GPU with training: confirm that train mode stops them and
  frees enough memory (at least 16 GB for the default training profile) before phase G, and that
  they restart afterwards.
- Capture screenshots at the moment each UI step runs, then review them together in phase K.

## 4. Findings document

Write one findings document per run in the repo, containing: run facts (SHAs, versions, GPU map,
ports), a step results table, a class-identity ledger (one row per hop in phase G), findings
(severity Blocker, Major, Minor, Doc), the screenshot review log and a timing summary against the
budget in section 3.

## 5. Go / no-go

The owner signs one line per phase. **Go** requires every phase ticked and zero open Blockers.
A Major needs a merged fix plus a re-run of the affected steps, or a recorded owner waiver with a
reason. The release steps in `docs/releases/RELEASE_CHECKLIST.md` follow only after a Go.

## 6. Status and what remains

Done: the release scripts and checklist (`scripts/release/`, `docs/releases/RELEASE_CHECKLIST.md`),
the installer with its static, dry-run and lifecycle tests (`tests/installer/`), and the
API-level live tests (`tests/live/`).

Open: the scripted run itself (phases A to L), a public-data fixture manifest that drives phases
E and F reproducibly (the COCO subset from #45 is the intended source), and a wrapper that
writes the findings document skeleton. The installer live test (#44) is phase C and J of this
plan and should be run as part of it.
