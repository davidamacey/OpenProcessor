# CLAUDE.md: research notes

Rules for adding or editing anything in `docs/research/`. The project rules in the root
`CLAUDE.md` also apply.

## What belongs here

Evidence: measurements, literature reviews, evaluation plans, trade studies. A decision derived from
the evidence goes in `docs/design/` and links back here. A note is a dated snapshot: put the date,
the commit it was measured at and the hardware at the top.

## Rules

- **Never invent a number.** A run that failed is reported with its cause. Vendor-reported figures
  are labelled as vendor-reported, and every external claim carries its URL.
- **Public data only in committed text.** COCO, Open Images and other public sets. The owner's
  private photo collection is called "set P" and is described by its numbers; no folder names,
  file names, paths or owner/organisation names. `scripts/codegen/check_naming_leaks.py` enforces
  this in pre-commit. A public model name that the scan bans is allowed only through a narrow
  entry in `scripts/codegen/naming_leak_allowlist.txt` with a reason.
- **Every result has a reproduce command** that works from a clean checkout, plus the setup (stack,
  GPU, driver, versions, load average). Use `-p <name>` for compose; never act on the default
  project.
- **Raw outputs are kept**: per-run JSON, logs and GPU/host samples go to `artifacts_local/`
  (gitignored) or a local scratch directory outside the repo, and the note names where. Small result files that back a
  published number may go to `docs/benchmarks/`. Do not bulk-delete bake-off or ladder run data;
  some of it backs the write-up.
- **Measure one factor at a time**, repeat (3 runs is the default), report spread, and say what
  else was running. A shared host is noisy: record the load average next to each row.
- **GPU etiquette.** GPU 1 belongs to other workloads. Serialize GPU-0 jobs with
  `flock "$SCRATCH/locks/gpu0.lock" <command>` (any shared lock file all jobs agree on); the live test stack may hold another card.
- **One decode per image.** New measurement or model paths must reuse the GPU-resident decoded
  frame, not decode the same image twice.
- **Docs are checked against code.** Routes, `OP_*` variables and file links must exist
  (`.venv/bin/python scripts/docs/check_docs_vs_code.py`). Write proposed routes without a leading
  slash and say they do not exist.

## Adding a note

1. Copy the structure of an existing log: Setup, Method, Results, Findings, Open issues, Reproduce.
2. Link the GitHub issues (`#NNN`); update the issue with the numbers.
3. Add the row to [README.md](README.md) and to the research table in `docs/design/README.md`.
4. Run `.venv/bin/pre-commit run --files <changed files>` (call the venv binary directly) and
   commit with a conventional message (`docs(research): ...`).

## Writing up

`v070_measurement_log.md` is the running source for any paper or report: tables there are
written to be quoted. When results are final, compile the write-up from the logs and the raw JSON
rather than re-running from memory, and keep the figures reproducible from the scripts in
`scripts/bench/`.
