# API contracts

Generated, committed descriptions of the curation API's wire format, for
frontends to vendor and test against. Every file here is **generated**. Do
not edit by hand: change the Python source and regenerate.

| File | What it is | Source of truth |
|---|---|---|
| `ts/regionStatus.ts` | `RegionStatus` enum, `RegionStatusValue` union, `REGION_STATUS_VALUES`, `TERMINAL_REGION_STATUSES`, `PENDING_REGION_STATUSES`, `HUMAN_REGION_STATUSES` (the statuses an operator may write), `REGION_STATUS_ROLE` (each status's role), `CONFIRM_STATUS_VALUE` / `REJECT_STATUS_VALUE` / `FALSE_POSITIVE_STATUS_VALUE` — all derived from `REGION_STATUS_INFO` | `src/config/region_state.py` |
| `ts/itemWire.ts` | `ItemWire` interface (the item every item-returning endpoint emits), `ITEM_WIRE_KEYS`, `REGION_WIRE_KEYS` (the `region_*` attribute subset), and per-endpoint extra keys (`REVIEW_EXTRA_KEYS`, `SEARCH_EXTRA_KEYS`, `TRAINING_CANDIDATE_EXTRA_KEYS`) | `ItemDoc` in `src/routers/curation/_item_models.py`, checked against `serialize_item` in `src/services/curation/wire.py` |
| `json/item_wire.json` | The same item contract as JSON: ordered `item_keys`, `region_keys`, `extra_keys`, and the full `ItemDoc` JSON schema (types and nullability) | same as above |
| `json/keymap_actions.json` | The keymap action table: `actions`, `contexts`, `grammar` | `src/config/keymap_actions.json` |
| `ts/classSources.ts` | `CLASS_SOURCE_ROLES`, the `ClassSourceRole` union, and the `ClassSourceEntry` shape of `GET {prefix}/class_sources` | `src/services/curation/class_sources.py` |
| `json/frame_engine_v1.json` | The frame engine's protocol version 1 for the native engine: tensors, datatypes and shapes, the order of the front's checks and their statuses, request rules, statuses and decode routes, branch outputs, plan faults and JSON rules, the ping policy | `src/domain/frame_engine_wire.py`, `src/domain/decode_route.py`, `frame_engine/front.py`, `src/clients/frame_engine.py` (rendered by `scripts/codegen/frame_engine_contract.py`) |
| `json/frame_engine_plan_cases.json` | Every plan text of `scripts/codegen/frame_engine_plan_cases.py` with what `parse_plan` answers (the parsed plan, or the fault and its message, which is the status details; a `bad_json` case carries no message); the native parser must give the same on each case | `parse_plan` in `src/domain/frame_engine_wire.py` |
| `json/capture_cases.json` | Tag sets of one ExifTool read (hand-written and recorded from the committed test images, `native/op_media/tests/fixtures/exiftool/recorded.json`) with what `parse_capture` answers, and what `apply_location_policy` answers under each policy; the Rust capture mapping (`native/op_media`) must give the same on each case | `parse_capture` in `src/domain/capture_metadata.py`, `apply_location_policy` in `src/domain/location_policy.py` |
| `openapi/curation.json` | `app.openapi()` filtered to paths under the curation API prefix (default `/curation`) plus only the components those paths reference. Keys are sorted, and the app version is left out, so diffs show real contract changes only | the FastAPI app (`src/main.py`) |
| `openapi/core.json` | The rest of `app.openapi()`: every operation outside the curation API prefix (`/detect`, `/faces`, `/embed`, `/search`, `/models`, `/health`, ... and their `/v1` twins) plus only the components those paths reference. Same sorting and version rule as `curation.json`; the native core routes are held to it | the FastAPI app (`src/main.py`) |

The generator always runs with default configuration. `OP_*` variables in
your shell and any `.env` file are ignored, so everyone produces the same
output. Item keys are fixed wire names: an `OP_REGION_FIELD_*` storage
override never changes them.

## Regenerate

```bash
make contracts          # or: python3 scripts/codegen/generate_contracts.py
make contracts-check    # or: python3 scripts/codegen/generate_contracts.py --check
```

Individual generators (each supports `--check`):

```bash
python3 scripts/codegen/export_region_status_to_ts.py [--check]
python3 scripts/codegen/export_api_contracts.py [--check] [--only item-wire|class-sources|keymap-actions|openapi|core-openapi|frame-engine]
.venv/bin/python scripts/codegen/capture_cases.py [--check] [--record]   # --record needs exiftool
```

`export_api_contracts.py` imports the app. It re-runs itself under the
project venv (`.venv/bin/python`, including the main checkout's venv when
run from a git worktree), or under `$CONTRACTS_PYTHON` if that is set.

Three pre-commit hooks reject a commit that leaves a contract stale:
`region-status-ts-drift`, `api-contracts-drift` and `capture-cases-drift`. When a change to the
API or wire format moves a contract, regenerate and commit the output in
the same commit.

## Consume

Vendor the files into the frontend at a known revision instead of copying
values by hand:

```bash
git -C /path/to/openprocessor show main:contracts/ts/itemWire.ts > src/lib/contracts/itemWire.ts
git -C /path/to/openprocessor show main:contracts/openapi/curation.json > src/lib/contracts/curation.openapi.json
```

Record the source commit (`git rev-parse main`) next to the vendored copy.
Frontend tests can then assert against `ITEM_WIRE_KEYS`,
`HUMAN_REGION_STATUSES`, and similar constants, or validate recorded
responses against `openapi/curation.json`. A backend change that breaks
the contract then shows up as a diff when you re-vendor.
