# Combine preview: embedding_model_mismatch and region_profiles_differ warnings

Status: **implemented** in `src/services/projects/combine/warnings.py` (issue #55). Items carry no encoder id, so the check is the vector dimension only.

A fresh agent should be able to implement this from the file. Re-find code by symbol.

## Context

`POST /curation/projects/combine/preview` is a dry run that returns errors, warnings, a mapping
suggestion, per-source counts, dedup numbers and a `preview_sha`. `POST /curation/projects/combine`
starts the job and refuses a stale `preview_sha`. The planner lives in
`src/services/projects/combine/plan.py`; the router is `src/routers/curation/projects_combine.py`.
Warnings implemented today include `label_conflicts`, `holdout_recompute_contamination` and the
shard-budget warning. Two designed warnings are missing, and the embedding dimension check has no
test.

## 1. embedding_model_mismatch

Problem: an item copied into the target keeps its source embedding. If the source stamped a
different encoder id or vector dimension than the encoder the target (the process-wide encoder)
uses, the vectors are not comparable and clustering or search over the combined project is wrong.

Behavior:
- Per source, read the encoder id and dimension stamped on its items (the same fields ingest
  writes; find them with `rg "embedding_model|encoder" src/services/curation`).
- If both match the running encoder, copy the embedding as today.
- If either differs, do not copy the vector. Queue the item for re-embedding in the target (use
  the same `embedding_state` mechanism the selective-embedding work introduced, see
  `docs/design/generic_detector_and_selective_embedding_plan.md`) and add the preview warning
  `embedding_model_mismatch` with `{project, items, source_encoder, target_encoder}`.
- Preview counts the affected items so the user sees the re-embedding cost before starting.

Tests (must fail before the change):
- a source whose stamped dimension differs: preview carries the warning, the executed combine
  copies no vector for those items and marks them for re-embedding;
- a source with matching stamps: no warning, vectors copied (mutation: remove the dimension
  comparison and the first test must fail).

## 2. region_profiles_differ

Problem: sources that used different region profiles (different child classes, parent classes or
text rules) are merged, and the target keeps one profile (`settings_from` or the first source).
Boxes made under another profile may not satisfy the target's rules.

Behavior: when two or more sources have different active region profile names or content hashes,
add the warning `region_profiles_differ` with `{profiles: [{project, name, hash}], target_profile}`.
This is informational: do not block the combine. Document in the warning text that the target
uses the profile named by `settings_from`, or the first source if unset.

Tests: two sources with different profiles produce the warning; identical profiles do not.

## 3. Contract and docs

- Add both codes to the warning-code documentation in `docs/design/curation_api_contract.md`
  and to the docs-site page `docs-site/docs/guides/combine-projects.mdx`.
- If the warning model enumerates codes, regenerate contracts with
  `scripts/codegen/generate_contracts.py` and commit the generated files.

## Verification

`pytest tests/ -q --ignore=tests/live` plus pre-commit; then run a preview on two small test
projects whose items carry different encoder stamps and confirm the warning appears.
