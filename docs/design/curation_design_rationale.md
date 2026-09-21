# Curation subsystem — design rationale

Status: **living reference doc**, owned by this backend. This is the
canonical answer to "why is it built this way" for the `curation`
subsystem — the three configuration dataclasses, the storage/wire
split, the pre-commit ratchet exemptions carried by the files that
introduced them, and the gaps that are known and tracked rather than
accidental. It complements, and deliberately does not duplicate,
[`curation_api_contract.md`](curation_api_contract.md) (the HTTP wire
contract itself) and [`../ARCHITECTURE.md`](../ARCHITECTURE.md#curation-subsystem)
(the component map).

## 1. Where this subsystem came from

The `curation` subsystem — active-learning review queues, clustering,
VLM-assisted labeling, dataset export, and the training-job API under
`CurationConfig.api_prefix` (default `/curation`) — was genericized out
of a private, domain-specific reference implementation (a
vehicle/license-plate curation stack) built for one deployment. That
reference implementation hardcoded its domain everywhere: OpenSearch
field names, detector model names and geometry heuristics, index names,
filesystem paths. None of that is inherent to "curate a stream of
detector crops with human review and active learning" — it's one
deployment's parameter values. The genericization's job was to find the
seam between "generic curation mechanics" and "one deployment's
parameters" and turn every hardcoded value on the wrong side of that
seam into configuration.

That is the through-line for everything below: **the mechanics are
generic Python; the domain lives entirely in data** — three frozen
dataclasses a deployment constructs (or overrides via environment
variables) rather than a codebase it forks.

## 2. The four configuration dataclasses

### 2.1 `CurationConfig` (`src/config/curation.py`)

Holds index names, filesystem roots, and API-surface constants: which
OpenSearch indexes back images/items/labels/classes/clusters, where the
class registry and exported datasets live on disk, the crop-cache
directory, the API mount prefix, and embedding-dimension/HNSW tuning.

Before genericization, these were a hardcoded `KbIndex(str, Enum)` and a
scatter of module-level path constants. Two deployments never share
index names, cache paths, or a mount prefix by coincidence — they're
deployment data, not code — so they moved onto a dataclass with an
`IndexRole` + `index_name()` lookup (typo-proof; a role can't resolve to
the wrong deployment's index by a copy-paste mistake) and a
`from_env(prefix='OP_')` classmethod so an operator overrides one field
via environment variable without touching source. `get_curation_config()`
is the process-wide default, built through `from_env()` exactly once and
memoized; anything needing a *different* configuration (an overlay for
an existing deployment with different index names) constructs its own
instance and injects it rather than mutating the default.

### 2.2 `RegionFields` (`src/config/region_fields.py`)

See §4 below — it's substantial enough to warrant its own section.

### 2.3 `DetectionProfile` (`src/config/detection_profile.py`)

Describes one detectable "region of interest" type as data: aspect-ratio
and area heuristics for auto-confirmation, a text-hint pattern and
length range, which Triton models (detector/segmenter/OCR) back it, and
their input sizes and confidence floors. The reference cascade hardcoded
all of this for exactly one region type (a license plate on a vehicle).
A deployment describing a different region — a barcode on a package, a
tag on livestock — constructs its own `DetectionProfile` instance
instead of branching or forking the cascade code that consumes it.

`DetectionProfile.from_env(prefix='OP_DETECTION_')` now exists, so a
deployment can override individual fields via environment variable the
same way it can for `CurationConfig` and `RegionFields` (see
`env.template`'s `OP_DETECTION_*` block). **Known gap, still tracked:**
`DetectionProfile`'s *default* field values remain the reference
deployment's tuned numbers (its aspect-ratio range, its text-length
range, its OCR/segmenter model names) rather than domain-neutral
placeholders — a new deployment gets a working example, not a neutral
default, out of the box, and should expect to override most fields for
its own region type. Also, exactly one `DetectionProfile` (and one
`PromptPack`) is active per process today; there is no per-request
selection among multiple registered profiles yet, even though the
underlying `profile_registry` mechanism supports registering more than
one.

### 2.4 `RegionStatus` (`src/config/region_state.py`)

The canonical state-machine enum for the region-of-interest pipeline
(`pending_detection` → detector cascade → `detected` /
`verify_rejected` / `no_region_box`; `pending_verification` → VLM
verify → `detected` / `no_region_visible`; any path can short-circuit to
the terminal `detection_failed`, and a human reviewer can additionally
mark a detected box `false_positive` without deleting it, preserving
provenance for hard-negative training). On-disk string values are kept
byte-identical to what earlier code wrote directly as literals — this
is a Python-symbol rename, not an OpenSearch data migration, matching
the same no-reindex reasoning as `RegionFields` (§4).

## 3. The frozen wire-contract split

The subsystem draws a hard line between two independent naming systems
that happen to look similar:

- **HTTP JSON field names** on the Pydantic wire models under
  `src/routers/curation/_common.py` — the contract every API consumer
  (the labeling frontend, any future service) speaks. These are frozen:
  once shipped, a field is not renamed, repurposed, or dropped on the
  wire.
- **OpenSearch document field names** — how the backend actually stores
  a value internally. These are configurable per §4.

A router handler can read `doc[region_fields.status]` internally while
returning a response whose Pydantic attribute is a fixed name — a
storage-field rename never has to ripple into a wire-format change, and
a wire-format decision never constrains which OpenSearch field name a
deployment's existing data happens to use. See
[`curation_api_contract.md`](curation_api_contract.md) for the full
route-by-route contract and the specific model attributes this applies
to; this doc only states the principle because §4 depends on it.

## 4. `RegionFields` and why there is no reindex

`RegionFields` is the indirection that makes the wire/storage split in
§3 actually hold at the OpenSearch layer. Every place the backend reads
or writes a per-item "region of interest" sub-annotation — its bounding
box, status, detector provenance, verification state, OCR text, cluster
assignment — goes through a `RegionFields` instance rather than a
literal string. The generic OSS defaults are `region_*` names
(`region_status`, `region_bbox_norm`, …); a deployment whose existing
OpenSearch index already has data under different field names (the
reference deployment's were prefixed for its domain) constructs its own
`RegionFields` instance with those names instead.

**Why this matters enough to be its own module:** an existing
deployment adopting this generic code does not want to reindex however
many million documents just to satisfy a generic module's naming
opinion. Reindexing is a real-money, real-downtime operation on a
production dataset; renaming a Python-level indirection is not. By
routing every field access through `RegionFields`, a rename becomes a
config flip — point the dataclass at the existing field names — with
zero data migration and zero reindex risk. The alternative (hardcode
`region_*` everywhere and require every adopter to migrate their data to
match) would make the generic code effectively unusable by the one
deployment it was extracted from.

`RegionFields.from_env(env_prefix='OP_REGION_FIELD_')` gives the same
per-field environment-variable override that `CurationConfig` has, and
`get_region_fields()` is the equivalent memoized process-wide default.
Two things worth stating plainly because they have been a source of
confusion before:

- This indirection governs OpenSearch query bodies, `_source` lists,
  bulk-update documents, painless scripts, and index mapping bodies —
  **not** the HTTP wire contract (§3), which is frozen independently,
  and **not** enum *values*, which are untouched.
- `get_region_fields()` **does** call `from_env()` — every
  `OP_REGION_FIELD_*` override is live. (An earlier internal audit
  claimed otherwise, confusing this module with a since-fixed bug in
  `CurationConfig.get_curation_config()` that briefly had the same
  symptom. Do not "fix" this again; both are the correct, already-fixed
  form.)

`scripts/codegen/check_no_literal_region_fields.py` is the automated
half of this guarantee: a pre-commit hook that bans a raw
`'region_...'`-shaped domain literal (the exemplar the check was built
against was the reference deployment's own field prefix) from appearing
in any file it has been told to police, via a growing `PORTED_PATHS`
allowlist. Each wave that ports or writes new curation code appends its
paths to that allowlist in the same commit — once a module is covered,
it can never silently regress to hardcoding a storage field name again.
Two categories are deliberately exempt from the ban and documented
in-file: `RegionFields`' own docstrings (which must name the
illustrative override example) and the frozen Pydantic wire-model
attribute declarations from §3 (which are a different naming system
entirely and were never in this guard's scope).

## 5. The pre-commit ratchet exemptions

`.pre-commit-config.yaml`'s `max-file-size` hook caps source files at
700 LOC to keep modules reviewable and discourage grab-bag files. Eight
files that arrived with the curation port are grandfathered past that
cap:

- `src/clients/curation_opensearch.py`
- `src/services/curation/clustering/orchestrator.py`
- `src/services/training/triton_promote.py`
- `src/routers/curation_train.py`
- `src/services/training/jobs.py`
- `src/services/labeling/vlm_labeler.py`
- `src/services/detection/cascade_detect.py`
- `scripts/curation/worker/runner.py`

The reasoning is the same for all eight and worth stating once instead
of once per exclude-list comment: each corresponds to a genuinely
cohesive reference-implementation module that was already over the cap
*before* any porting work touched it. Splitting a file correctly
requires understanding its internal seams well enough to draw a
sensible boundary; doing that split *while simultaneously* genericizing
the file's contents would have produced a diff that mixed structural
reorganization with semantic changes in the same commit — exactly the
kind of diff a reviewer cannot meaningfully check line-by-line. The
porting decision was: land the file whole (readable, semantically
diffable against its reference counterpart), grandfather it explicitly
with an in-file comment naming which port chunk added it, and treat the
split as separate, tracked follow-up work rather than silently deferring
it.

Practical consequence for anyone extending one of these eight files: add
functionality to the existing file rather than treating the grandfathering
as license to keep growing it indefinitely, and do not add a ninth
undocumented exemption — a genuinely new oversize file should be split
before it is committed, following the same reasoning that will
eventually retire these eight.

## 6. Known gaps

These are documented so they read as "known and tracked," not
discovered-in-production surprises. None of them is fixed in this pass;
they're recorded here so the rationale for *why the code looks
unfinished in these specific ways* lives somewhere durable.

- **Ingest is thinner than the reference deployment's, by design.**
  `POST /curation/ingest/image` and `/ingest/batch` exist and create
  items (duplicate detection, quality-gate scoring, crop-cache
  population, bulk indexing), and `POST /curation/import_labels(/batch)`
  imports pre-existing YOLO-format labels. What did **not** port: the
  reference's dual-head domain detector runner, its fixed
  domain-specific class allowlist, and its region-status assignment
  policy tuned to one domain — those remain a future, deployment-specific
  overlay, not something this generic ingest service should hardcode.
- **The asynchronous half of the product now has a first-party home,
  but it is still opt-in and still needs a trainer/segmenter you
  supply.** `docker compose --profile curation up -d` starts the
  detection worker, VLM worker, auto-label worker, and cluster-refresh
  daemon against this codebase. There is still no shipped trainer
  container or segmentation-service container — `/curation/train/*`
  and the cascade's segmenter leg talk a documented HTTP/file protocol
  (see `docs/CURATION.md`) that a deployment implements or points at
  its own service; nothing here starts one for you.
- **Environment-variable and metric-name prefixes are fully
  reconciled on the `OP_`/`op_` convention** — the `KB_*` env vars and
  `kb_*` metric names from the original port have been renamed. See
  `env.template` for the current, complete surface.
- **`DetectionProfile`'s shipped defaults are domain-tuned, not
  domain-neutral** (§2.3) — a new deployment should construct its own
  instance (or override via `OP_DETECTION_*`) rather than relying on
  the defaults describing a sensible generic region. Only one
  `DetectionProfile`/`PromptPack` is active per process; there is no
  per-request selection among several registered profiles yet.
- **No authentication of any kind on the API** — see `SECURITY.md`.
  Several curation write routes are destructive
  (`DELETE /curation/models/{model_name}`) or read arbitrary
  server-side paths (`POST /ingest/directory`). Do not expose this
  service directly to the internet.
- **Coverage is uneven across the ported surface.** Some routers and
  services carry thorough test suites; others were ported with
  comparatively thin coverage because the reference implementation
  itself had thin coverage there. Restoring/extending coverage on the
  weakest surfaces is ongoing, tracked work rather than a silent gap.

None of the above blocks using the subsystem for its core loop — ingest
(direct API calls or the label-import path), browse, cluster, review,
label, and export. It constrains how far along the "turnkey for an
arbitrary new deployment, fully autonomous end to end" spectrum the
subsystem currently sits, which is why it ships labelled experimental
for this release (v0.3.0) — see `docs/CURATION.md`.

## 7. Labeling-assist item selection: `PromptPack`, `DetectionProfile`, and the frontend's annotation-slot model

The frontend's labeling-assist UX lets an operator pick which items or
classes they want assistance with — pallets, food items, license
plates, anything — and scope a run to just that selection. Getting this
right on the backend meant adding a fourth member to the
`CurationConfig`/`RegionFields`/`DetectionProfile` family (§2) —
`PromptPack`, in `src/services/labeling/vlm_prompts.py` — and making
both it and `DetectionProfile` discoverable over the wire, without
conflating either of them with the frontend's own concept of an
"annotation slot."

**Why `PromptPack` is a fourth, separate dataclass rather than a field
on `CurationConfig`.** `CurationConfig` holds names and paths — small,
uniformly-typed deployment data. A `PromptPack` is the opposite: a dozen
multi-paragraph prompt templates plus two vocabulary tables
(`class_descriptions`, `synonyms`), all specific to one labeling
domain. Folding that much text onto `CurationConfig` would turn a
lookup-table dataclass into a prompt-engineering dataclass; keeping it
separate means a deployment can swap its *vocabulary* (`PromptPack`)
independently of its *index names* (`CurationConfig`) or its *detection
heuristics* (`DetectionProfile`). `CurationConfig` only holds the
*pointer* to a pack — `prompt_pack_path` — resolved lazily by
`resolve_prompt_pack()` so a missing/malformed file degrades to the
built-in generic pack (logged warning) rather than crashing the VLM
labeler at import time.

**Why `PromptPack` and `DetectionProfile` are not the same concept,
even though they correlate per-deployment.** `DetectionProfile`
describes a Triton detection *cascade* — which detector model, what
aspect/confidence thresholds, how OCR is wired. `PromptPack` describes a
*VLM conversation* — what to ask a vision-language model and what
vocabulary to expect back. A deployment adding a "pallet" domain
configures both, and in practice they describe related things (the
pallet `DetectionProfile`'s region type and the pallet `PromptPack`'s
`class_descriptions` are about the same physical objects) — but nothing
in the code ties them together structurally. One deployment could run a
`DetectionProfile` with no VLM stage at all (pure CNN cascade, VLM
disabled), or a `PromptPack` with no custom `DetectionProfile` (VLM
classifies whole-item crops; no sub-region detection). Merging them
into one dataclass would force every deployment to configure both
whenever it only needed one.

**Why neither is the frontend's "annotation slot."** The
labeling-assist frontend reasons about
*annotation slots* — a UI-level grouping of what a human curator sees
and edits for one class. `PromptPack` and `DetectionProfile` are
backend pipeline concepts: one drives an automated VLM pass, the other
drives an automated detection cascade. They influence what shows up
*for* a human to review, but a slot is not required to have a matching
`DetectionProfile` or a bespoke `PromptPack` entry — the generic pack's
open-vocabulary prompt and the default profile work across every class
in the registry unless a deployment opts into something more
specific. The correlation between all three is `class_id`: a
`PromptPack`'s `class_descriptions`/`synonyms` are keyed by class name,
a `DetectionProfile` is bound to whichever classes route to it (see
`secondary_shape_groups` on the shipped default), and the frontend's
labeling-assist run now scopes on `class_id` directly (`POST
{prefix}/pipeline/auto_label/start?class_id=<id>`, below) — but the
join is data (a shared registry class id), not a code-level dependency
between the three dataclasses.

**Discovery**: both are advertised on `GET {prefix}/methods`
alongside the existing `cluster`/`score`/`sort`/`overlay`/`export` axes
(`src/services/curation/strategy_registry.py`), in the same
`{id, axis, label, status, default}` shape as the `export` axis. Each
axis lists exactly one entry today (the one configured
`DetectionProfile`, the one resolved `PromptPack`), always
`status='stable'`, `default=True` — but the underlying mechanism is not
capped at one:
`src/services/detection/profile_registry.py` is a real, process-lifetime
registry (`register_profile()` / `get_profiles()` /
`get_default_profile_name()`) that a deployment wanting more than one
detectable region type registers against, and the `prompt_pack` axis
reads whatever `resolve_prompt_pack()` actually resolves, so pointing
`OP_PROMPT_PACK_PATH` at a different file changes both the VLM's actual
behavior and what `/methods` reports in the same step — the two can
never drift out of sync because they share one resolution function.

**Worked example — configuring a "pallet" labeling-assist setup
end to end:**

1. Add pallet classes to the registry (`data/class_registry.json`, or
   wherever `OP_REGISTRY_PATH` points) — see
   `data/class_registry.example.json` for a worked warehouse/pallet
   registry.
2. Define a `DetectionProfile` for the region type you want the
   cascade to find (e.g. a pallet ID tag) and register it via
   `src.services.detection.profile_registry.register_profile()` at
   process startup, or construct it directly wherever the detection
   cascade is wired for your deployment — mirror
   `cascade_detect.REFERENCE_LICENSE_PLATE_PROFILE`'s shape.
3. Write a `PromptPack` JSON file describing the pallet vocabulary —
   copy `data/prompt_pack.example.json` (a worked warehouse/pallet
   pack) and edit its prompts/`class_descriptions`/`synonyms`.
4. Point `OP_PROMPT_PACK_PATH` at that file. `GET {prefix}/methods`'s
   `prompt_pack` axis now reports your pack's `name` instead of
   `generic_item_v1`.
5. Trigger `POST {prefix}/pipeline/auto_label/start?class_id=<pallet
   class id>&run_gemma=true` — the run scopes its unvalidated-item
   query to that one class (a `term` filter on `class_id`, added
   alongside the existing `class_validated`/`gemma_verify_completed_at`
   exclusions in `src/routers/curation/pipeline.py`) instead of
   labeling the entire pool.
