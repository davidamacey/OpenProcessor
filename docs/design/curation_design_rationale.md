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

## 2. The three configuration dataclasses

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

**Known gap, tracked rather than fixed here:** at the time of writing,
`DetectionProfile`'s *default* field values are still the reference
deployment's tuned numbers (its aspect-ratio range, its text-length
range, its OCR/segmenter model names) rather than domain-neutral
placeholders, and the dataclass has no `from_env()` of its own (unlike
`CurationConfig` and `RegionFields`). A new deployment today configures
correctly by constructing an explicit instance — it does not get a
neutral default for free, and cannot override individual fields via
environment variable the way it can for the other two dataclasses. Both
of those are recorded as follow-up work; fixing them is not part of
this pass.

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

- **Thinner ingest path than the reference by design.** The public
  ingest surface currently exposes read-only status/lookup endpoints;
  no route creates a new item. The generic mechanics for a full
  create-path (duplicate detection, quality-gate scoring, crop-cache
  population, bulk indexing) are more portable than the amount of
  domain-specific logic the reference ingest path had wrapped around
  them might suggest, and closing this gap with a genuinely generic
  ingest service is planned follow-up work, not abandoned scope.
- **The asynchronous half of the product — long-lived detection/label
  workers, a training container, a segmentation service — exists as
  ported code with no corresponding container or compose service yet.**
  The synchronous HTTP API (browse, label, cluster, export) is usable
  standalone; the asynchronous pipeline that would keep it fed
  automatically is a separate, larger integration effort.
- **Environment-variable and metric-name prefixes are not yet fully
  reconciled.** Some capability flags and Prometheus metric names still
  carry a legacy prefix from the reference deployment rather than the
  generic `OP_`/`op_` convention used elsewhere; this is a naming
  cleanup with no functional impact, tracked as follow-up rather than
  addressed opportunistically file-by-file (a partial, ad hoc rename
  would be worse than a consistent, deliberate one).
- **`DetectionProfile`'s shipped defaults are domain-tuned, not
  domain-neutral** (§2.3) — a new deployment must construct its own
  instance rather than relying on the defaults describing a sensible
  generic region.
- **Coverage is uneven across the ported surface.** Some routers and
  services carry thorough test suites; others were ported with
  comparatively thin coverage because the reference implementation
  itself had thin coverage there. Restoring/extending coverage on the
  weakest surfaces is ongoing, tracked work rather than a silent gap.

None of the above blocks using the subsystem for its core loop —
ingest via direct OpenSearch writes or the label-import path, browse,
cluster, review, label, and export — it constrains how far along the
"turnkey for an arbitrary new deployment" spectrum the subsystem
currently sits.
