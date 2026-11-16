# Parity ledger

When a slice of the Rust port deletes a Python route, it deletes that route's Python tests too. The
ledger says what now proves each deleted test's behaviour, so a deletion never silently lowers what the
suite checks (plan 18a, sections 3.1 and 3.2).

## One file per slice

`tests/parity/ledger/<slice>.tsv`, for example `r5.tsv`. One row per deleted test, two tab-separated
columns; blank lines and lines starting with `#` are skipped.

```
<test id><TAB><replacement>
```

The test id is what pytest prints from the repository root, without parameters:
`tests/curation/test_classes.py::test_merge_keeps_names` or
`tests/curation/test_classes.py::TestMerge::test_keeps_names`. A parametrized test is one row.

## The replacement

| Form | Meaning | What the check verifies |
|---|---|---|
| `rust: <path>::<name>` | a Rust test | the file exists and contains `fn <name>` |
| `diff: <family>/<glob>` | differential cases of `tests/differential/` | the family is in `families.toml` and the glob matches at least one operation id of `contracts/openapi/*.json` |
| `contract: <file> <what is checked>` | a check against a committed contract file | the file exists |
| `gone: <reason>` | a behaviour removed on purpose, named in the slice | the reason is not empty |

## The check

`scripts/codegen/check_parity_ledger.py` runs as a pre-commit hook on every commit. A test counts as
deleted when its `(class, function)` is gone from its file, in the staged change; moving a file and
keeping the names is not a deletion. Each deleted test without a row fails the commit, and the failure
prints the rows still to fill in. Run it by hand over a merged range with
`python3 scripts/codegen/check_parity_ledger.py --between <old> <new>`.

The ledger is also checked by itself: every row well formed and resolvable, no test with two rows, and
no row for a test that still exists.
