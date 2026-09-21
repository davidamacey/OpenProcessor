## Summary

<!-- What does this PR change, and why? -->

## Type of change

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `docs` — documentation only
- [ ] `refactor` — no functional change
- [ ] `test` — adding/fixing tests
- [ ] `build` / `ci` — build system or CI
- [ ] `chore` / `style` / `perf` / `security`

## Checklist

- [ ] Commit messages follow this repo's conventional-commit format (`<type>(<scope>): <summary>`, imperative mood) — see `CONTRIBUTING.md`.
- [ ] `.venv/bin/python -c "import src.main"` passes.
- [ ] `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'` passes.
- [ ] `.venv/bin/pre-commit run --all-files` passes.
- [ ] New/changed behavior has test coverage.
- [ ] Docs updated if this changes user-facing behavior, config, or the API surface (`README.md`, `docs/CURATION.md`, `docs/design/curation_api_contract.md` as applicable).
- [ ] If this touches `src/`, `scripts/`, or `tests/` files under the region-field ratchet (`scripts/codegen/check_no_literal_region_fields.py`), newly-ported paths are added to `PORTED_PATHS` in this same PR.
- [ ] No internal/private paths, hostnames, or proprietary product names introduced — if you're porting code from an internal/private codebase, genericize it first (see `CONTRIBUTING.md`'s leak-discipline note).

## Related issues

<!-- Closes #123 -->
