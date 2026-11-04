# OpenProcessor: registration-plate identity text as a per-box attribute, with privacy modes (issue #401)

**Status: PLAN ONLY, NOT IMPLEMENTED.** The issue stays parked. This plan is the privacy design the
owner asked for before any implementation (triage comment of 2026-10-31). It also covers the key
and tier design, the research gate, and the product landing if the gate passes.

- Written 2026-11-04 against branch `int/b1-correctness` at `81c4e4ef` (the newest code; B1 merged).
- Builds on:
  - `docs/design/research_clustering_plan.md`: RQ-7 is this issue, experiment E-9, stop rule 7.1,
    open question OQ-6;
  - `docs/design/cluster_search_makemodel_plan.md` section 6.8 and wave W12 (the original design,
    parked);
  - `SECURITY.md` "Privacy: face data", the precedent this plan follows;
  - the #400 plan (attribute signals), which owns the generic must-link and veto edge interface
    this key plugs into.
- Naming: the code and this plan say **identity text** (a region whose text identifies one physical
  object). Registration plates are the worked example; serial numbers and bib numbers are others.
  Nothing in code names a vehicle or a plate.
- Audience: a fresh agent with no memory of this session. Symbols are cited as `path:LINE` at the
  commit above; re-find them with `rg` if lines move.

---

## 0. TL;DR

**What I recommend**

1. **Split #401 into a privacy baseline and the identity feature.**
   - **Phase A, the privacy baseline,** has value on its own and no research dependency. It is a
     sink inventory of every place a region reading is written, a canary test that proves no
     reading reaches logs, events, job files or hosted endpoints, a local-only rule for reading
     identity regions, and a deployment-level privacy ceiling.
   - **Phase B, the identity key with privacy modes and a purge,** lands only after the E-9 read
     gate passes.
   - **Phase C, link and veto edges,** lands only when #400's edge interface has a consumer (RQ-5).
2. **Privacy modes: `off` / `hash_only` (default) / `raw`, under a deployment ceiling.**
   - The ceiling is `OP_IDENTITY_TEXT_MAX_MODE`, with default `hash_only`. It is read live, and a
     malformed value refuses startup, like `OP_FACES_ENABLED` (`src/config/faces.py:1`).
   - A project can choose a mode at or below the ceiling. It cannot raise itself above it through
     the API, which has no authentication (`SECURITY.md` section 1).
3. **Keyed hashing with two secrets held apart.**
   - A **per-deployment master secret** lives in a read-only secrets mount, never in OpenSearch.
     It follows the existing pattern `./secrets/vlm:/run/secrets/op_vlm:ro`
     (`docker-compose.yml:184`, `src/services/labeling/vlm_endpoints.py:187`).
   - A **per-project random salt** (`secrets.token_bytes(32)`) lives in the project's own policy
     document.
   - The project key is `HMAC-SHA256(master, label || salt)`, with stdlib `hmac` and `hashlib`.
     The stored key is `HMAC-SHA256(project_key, version || folded_text)`.
   - Purge replaces the salt. Old keys become unrecomputable even by someone who holds the master
     and a backup of the index.
4. **Purge is a CLI action, never an API route.** It uses
   `./openprocessor identity-text purge --project <slug> [--dry-run] [--yes]`. This follows the
   face precedent exactly: "It is a command line action only: the API has no authentication, so it
   has no purge route" (`SECURITY.md`, face section; `scripts/purge_face_data.py:1`). This
   **corrects** the parked plan's proposed purge route.
5. **The research gate (E-9) cannot be read on the 300-plate pin.** A confident-key precision of
   at least 0.99 needs at least about 300 confident keys with zero errors (rule of three), or more
   with errors (Clopper-Pearson). The pin has 300 images in total. The fetcher must grow to the
   Open Images V7 train split (CC BY 2.0 only), with a registered minimum n before any read.

**What I am NOT deciding** (Judgment calls, section 10):

- whether Phase A moves into the v0.8.0 hardening milestone now (J1);
- equality-only keys versus keyed distance-1 neighbourhoods (J4);
- raw-mode retention (J6);
- what happens to existing projects that already hold readings when Phase B lands (J7);
- whether an external VLM may label items at all in a project with identity regions (J8).

---

## 1. Context: why the obvious approach is wrong here

The obvious approach follows the parked plan 6.8 literally. It adds `text_key` next to the text
fields, hashes with a secret in the project state directory, adds a purge route and builds edges.
That approach has six problems, and each changes the design.

1. **The reading is written in more places than the region box.** The region worker reads the
   item crop with OCR once per pass. It stores every line as `item_text_lines` and
   `item_text_tokens` for search (`src/domain/item_text.py:1`, `:38`). On a vehicle item, the
   plate is one of those lines. A mode that hashes only `region_boxes[].text` leaves the same
   characters on the item. Section 4 lists every sink. Phase A turns that list into a test.
2. **Hashing destroys the edit-distance rules.** 6.8's veto ("edit distance >= 2") and near-match
   ("edit distance 1 is a review suggestion") need the text. A plain keyed hash only answers
   "equal or not". Either `hash_only` gives up near-match and distance-aware veto, or it stores
   keyed distance-1 neighbourhoods (section 5.4, J4).
3. **A secret next to the data protects nothing.** A keyed hash of a short code is safe only while
   the key is secret: the space of plate strings is small enough to enumerate. The same is true
   of a salt stored beside the keys in the same OpenSearch cluster, which has no authentication by
   default (`SECURITY.md`, "OpenSearch ships with its security plugin disabled"). The master must
   live outside OpenSearch, and the per-project part must be destroyable.
4. **The API has no authentication.** Anything the API can do, any caller who reaches it can do.
   This covers a mode switch to `raw`, a lookup by plate string, a purge, and serving keys. The
   design therefore puts the escalation (`raw`) behind a deployment env ceiling. It puts the
   destructive action (purge) on the CLI. It refuses a lookup-by-text route in `hash_only`,
   because such a route is a membership oracle.
5. **Confidence today is a mean, not a minimum.** 6.8's `confident` tier wants "the OCR minimum
   per-character confidence above `c_hi`". The OCR pipeline returns one score per line: the
   **mean** of the per-character max softmax (`models/ocr_pipeline/1/ocr_postprocess.py:232`).
   One unreadable character hides inside a high mean. The tier needs a new per-line minimum from
   the pipeline, which sits in B5c's file (section 5.3).
6. **Public data cannot test linking.** Same-vehicle identity labels exist only on the private set
   (makemodel plan 5.4) or under research licences. E-9 can measure read quality and key precision
   on public Open Images. Link precision is owner-run (`research_clustering_plan.md` E-9: "linking
   is owner-run on the private set").

---

## 2. What is actually there today (measured, not estimated)

### 2.1 Region text pipeline

| Piece | Where | Fact |
|---|---|---|
| Example plate profile | the plate example under `examples/region_profiles/` | segmenter-only (SAM 3 with a text prompt) plus the OCR text-hint re-pass. `text_reader: "both"`, `text_uppercase: true`, `text_charset: "[A-Z0-9]"`, `text_len_min: 2`, `text_len_max: 10`, `text_min_confidence: 0.6`, `text_reject_sequences: true`, stopwords (US state names and codes), placeholders. Parent classes by **name**: car, truck, bus, motorcycle. No detector ships (no permissively licensed plate detector was found) |
| Profile fields | `src/config/detection_profile.py:90` to `:135` | `text_reader`, `text_crop_margin`, `text_crop_min_height`, `text_min_height_ratio`, `text_border_margin`, `text_uppercase`, `text_charset`, `text_join`, `text_len_min`/`max`, `text_stopwords`, `text_min_confidence`, **`text_format`** (full-match regex, `:133`), `text_placeholders`, `text_reject_sequences`, `region_class_name` |
| Dominant-text reader | `src/services/detection/region_text.py` | tallest-line selection, border band, normalizer, stopwords; `confidence = min(line scores)` (`:288`); pure and unit-testable |
| Validity rules | `src/services/detection/region_text_rules.py` | placeholder, stock run, charset, length, `format` full match (`:160`) |
| Reader choice | `resolve_region_text` in `region_text.py` | writes `text`, `text_raw`, `text_source`, `text_engine_version`, `text_confidence`, `text_vlm`, `text_ocr`, `text_choice`, `text_vlm_invalid`, `text_disagreement` |
| Box element | `src/domain/region_boxes.py:42` `RegionBox` | fixed keys including all `text*` fields (`:59` onward) |
| Worker stage | `scripts/curation/worker/region_text_stage.py` | region text plus item text in one OCR pass; `_drop_region_text` (`:148`) already strips text fields for a text-free profile. **This is the hook where a mode transform belongs** |
| Item OCR lines | `src/domain/item_text.py:38` | `item_text_lines` (object array: text, box, confidence) and `item_text_tokens` (keyword) |
| OCR line score | `models/ocr_pipeline/1/ocr_postprocess.py:217` to `:232` | per-character `preds.max(axis=2)`, then `np.mean` per line: **mean, not minimum** |
| Human text | `src/services/curation/region_box_edits.py:155` `human_text_fields` | `text_source='human'`, `text_choice='human'`; readers never overwrite it |
| Text repair | `src/services/curation/region_text_repair.py:1` | re-derives the chosen text from the stored per-reader readings `text_vlm` and `text_ocr` |
| Review filter | `src/services/curation/review_tab_clauses.py:54`, `:298` | case-insensitive substring search on `region_boxes.text` in the regions tab |
| Item text search | `GET {P}/crops?item_text=` (`src/domain/item_text.py` docstring) | token match over `item_text_tokens` |
| Wire | `src/services/curation/wire.py:406` (`item_text_lines`), `:411` (`region_boxes_to_wire`) | one serializer for every item endpoint; the `crop.region_verified` SSE event carries status and count only (`region_event_payload`) |
| Exports | `src/services/curation/export_single_class_rows.py:367` | YOLO export writes accepted box geometry only, **no text**. This holds today, and a test should keep it that way (section 8) |
| Combine | `src/services/projects/combine/regions.py` | copies region boxes between projects |
| Clone settings | `src/services/projects/clone_settings.py:19` | copies named documents (strategy defaults, ingest policy, VLM policy), not the whole settings doc |
| Research judge | `scripts/research/judge_protocol.py:23` | `HOSTED_FORBIDDEN_KINDS = {'face', 'identity_text', 'plate'}` refuses hosted calls by the sheet's declared kind |
| VLM locality | `src/services/labeling/vlm_url_policy.py:349` `compute_locality`, `:381` `sends_images_externally` | the existing primitive that says whether an endpoint is outside the deployment |
| External policy | `src/services/config_store/vlm_validation.py:48` `external_policy` (`OP_VLM_EXTERNAL_POLICY` = `ack` or `deny`) | deployment-level policy for endpoints outside the deployment |
| Logging | `src/core/logging.py` | structlog with shared processors (`configure_logging`), stdlib bridge, a filter precedent (`OutageLogFilter`) |
| Secrets precedent | `src/services/labeling/vlm_endpoints.py:187` `secrets_dir()` (`OP_VLM_SECRETS_DIR`, default `/run/secrets/op_vlm`); mounts at `docker-compose.yml:184`, `:514`, `:637`, `:690` | secrets are files in a read-only mount, referenced by slug, never values in OpenSearch |
| Per-project policy precedent | `src/services/curation/vlm_policy.py` (`VlmPolicyBody`, `extra='forbid'`), `src/services/curation/vlm_policy_store.py:24` `put_vlm_policy(expected_revision=...)` | revisioned per-project document with OCC writes |
| Purge precedent | `scripts/purge_face_data.py`, `src/services/face_privacy.py`, CLI `scripts/lib/cli_core.sh:130` | counts-only dry run; refuses while the feature is on; counts-only audit `data/audit/faces_purge.jsonl`; exit 1 if data is left behind |
| Plate sample | `scripts/datasets/manifests/oi_plates_300.json`, `scripts/datasets/fetch_openimages_plates.py` | 300 Open Images images (validation and test splits) with a "Vehicle registration plate" box, CC BY 2.0 only, class resolved by **display name**, not a hardcoded MID |

### 2.2 What does not exist

| Thing | Evidence |
|---|---|
| Any keyed hash in the product | `rg -n -i "hmac" src` returns nothing |
| A privacy mode for region text | no field in `DetectionProfile`, no policy document |
| A per-character minimum OCR confidence | `ocr_postprocess.py:232` averages |
| An identity-text flag on a profile | no `identity_text` field; the example plate profile is a generic text profile |
| A purge for region text | none |
| `cryptography` package | not in `requirements*.txt` and not importable in the dev venv; stdlib `hmac` and `hashlib` cover everything this plan needs |

### 2.3 Program position

- #401 is mapped "defer, stays parked; needs a privacy design"
  (`docs/design/program/09_github_actions.md:644`). Research is outside the 0.7.0 program
  (`docs/design/program/00_index.md` section 10).
- `research_clustering_plan.md` OQ-6 recommends "keep parked; run E-9 read quality only if the
  owner wants the feature this year". Stop rule: "owner does not approve the privacy design, or
  confident-key precision < 0.99".
- B5c ("crops at model size, OCR and faces on resident crops") changes
  `models/ocr_pipeline/1/model.py::_call_recognition` (GPU-7, `docs/design/program/07c_b5_crops_ocr_faces.md:28`).
  Any change to the pipeline's outputs lands after B5c and bumps `MODEL_VERSION`.

---

## 3. Privacy model

### 3.1 What is protected, and from whom

- **Protected data:** the identity text (the characters), anything that lets someone recover it
  (an unkeyed hash, a keyed hash plus its key), and the equality classes between items ("these two
  photos show the same plate"). The plate **pixels** stay visible in crops, thumbnails and the
  region close-up (`GET {P}/crops/{id}/region_thumbnail`). Modes govern text, not images. The docs
  must say so plainly.
- **Threats in scope:**
  - (a) a reader of the OpenSearch indexes or a snapshot;
  - (b) a reader of container or Loki logs;
  - (c) a caller of the unauthenticated API;
  - (d) a hosted model provider;
  - (e) a committed file or a public issue;
  - (f) an export handed to someone else.
- **Out of scope** (documented, like the face section): authentication itself (#369), OpenSearch
  snapshots and backups taken by the operator, logs written by older versions, and copies a client
  already downloaded.

### 3.2 Modes

| Mode | Reads identity regions | Stores | Serves on the wire | Edges |
|---|---|---|---|---|
| `off` | no: the identity profile behaves as `text_reader='none'` for identity regions | nothing new; existing data stays until purge | nothing new | refused (fail closed) |
| `hash_only` (default) | yes, OCR on a local engine; a VLM only when its endpoint is local (3.4) | tier, `text_key`, optional neighbourhood keys (J4), `text_source`, `text_choice`, `text_disagreement`, `text_confidence`, engine version; **no characters** in any field | the tier and a boolean "has key"; **never** the key or text | equality must-link; veto and near-match per J4 |
| `raw` | yes, as `hash_only` | everything `hash_only` stores, plus the text fields as today | text fields as today | all 6.8 rules |

- **Ceiling:** `OP_IDENTITY_TEXT_MAX_MODE` in `.env`. Values are `off`, `hash_only` or `raw`; the
  default is `hash_only`. It is read live on each call, like `faces_enabled()`. A malformed value
  raises `EnvError` at startup. It goes into `env.template`, `tests/test_env_surface.py` and the
  docs in the same commit.
- **Project mode:** a revisioned `identity_text_policy` document (mode, salt, `key_version`,
  revision, timestamps) on the `vlm_policy` store pattern, with OCC `expected_revision` writes.
  - A write above the ceiling is refused with 422 `identity_text_mode_above_ceiling`.
  - An effective mode is `min(project mode, ceiling)`. A lowered ceiling therefore takes effect
    without touching any project.
  - The salt is never serialised to the wire, never copied by `clone_settings`, never exported.
- **Fail closed:**
  - An identity-text profile active with no policy document refuses identity reads (no reading is
    stored) and reports `identity_text_policy_missing` in the project status. Project creation
    writes the document explicitly with `hash_only` (create-only first write, the B0 pattern), so
    new projects are never in this state.
  - `hash_only` with the master secret missing or unreadable also refuses identity reads, and
    `/health` reports it as a count, never a path or value.

### 3.3 Key derivation and hashing

All primitives are Python stdlib, with documented behaviour:

- `secrets.token_bytes` (https://docs.python.org/3/library/secrets.html);
- `hmac.new(key, msg, hashlib.sha256)` and `hmac.compare_digest`
  (https://docs.python.org/3/library/hmac.html);
- HMAC as defined in RFC 2104 (https://www.rfc-editor.org/rfc/rfc2104);
- HMAC used as a PRF for key derivation, as in NIST SP 800-108r1 (KDF in counter mode, one block)
  (https://csrc.nist.gov/pubs/sp/800/108/r1/upd1/final).

```text
master       = 32 bytes, file <secrets mount>/master   (generated once by the CLI; mode 600 on the host)
salt         = secrets.token_bytes(32)                  (per project; in identity_text_policy; replaced on purge)
project_key  = HMAC-SHA256(master, b"op/identity-text/project/v1\x00" + salt)
folded       = confusable_fold(normalize(reading))      (section 5.1)
text_key     = HMAC-SHA256(project_key, b"k\x00" + key_version + b"\x00" + folded.encode("utf-8")).hexdigest()
```

- **Why two parts:** an attacker needs the master (host secrets mount) **and** the salt
  (OpenSearch). Purge replaces the salt, so a backup of the index plus the master still cannot
  reproduce or test old keys.
- **Why not a per-project secret file in the project state directory** (the parked plan's
  choice): the state directory is project data that tools copy and back up with the project. It
  is also mounted into every container that handles the project, so the secret would travel with
  the data it protects.
- **Why not `cryptography`'s HKDF:** it would add a dependency for two HMAC calls. The single-block
  HMAC-PRF derivation above is the documented SP 800-108 construction (J3 lets the owner prefer
  HKDF from RFC 5869 via `cryptography` instead).
- **Key versioning:** `key_version` is part of the hashed message. A change to normalization,
  confusables or the derivation bumps it. Old keys then never compare equal to new ones, which is
  the safe failure, and a rekey job (5.6) recomputes keys where raw text exists.
- **Master rotation:** a new master invalidates every project's keys at once. This is a
  deployment action (`./openprocessor identity-text rotate-master`, CLI only): it re-keys `raw`
  projects from their stored text and purges keys in `hash_only` projects (they cannot be
  recomputed). It is rarely needed and documented as destructive.
- **Containers that mount the master:** only the API and the curation detection worker (the
  writer). Not OpenSearch, Triton, the trainer, the exporters or the VLM. This is fewer mounts
  than the VLM secrets have (four).

### 3.4 Hosted models and identity regions

Forward constraint on the new code (the read path for identity regions):

- An identity region is read by a VLM **only** if that endpoint's locality, recomputed at read time
  with `compute_locality`, is `compose`, `host` or `private`. `sends_images_externally(locality)`
  must be false.
- The rule applies to **every** endpoint source, the built-in env endpoint included. It is
  independent of `allow_external` and of `OP_VLM_EXTERNAL_POLICY`.
- An external or unknown locality means OCR only. The reason is recorded as a non-content code
  (`text_vlm_invalid = 'external_endpoint'`). It fails closed: `unknown` counts as external, as
  `vlm_url_policy.py:356` already defines.
- Contact sheets: the research judge's refusal (`judge_protocol.py:23`) keys on the declared kind.
  Forward constraint: a sheet builder must refuse a hosted backend when **any** member item
  carries an identity region, whatever kind the caller declares.
- Item-level labeling by an external VLM in a project with an identity-text profile sends vehicle
  crops in which plates are legible. Text modes cannot prevent that, so it is J8.

### 3.5 Logs

- **Precondition before any identity-text code lands:** audit every log call on the region-text,
  OCR and VLM paths for logged reading content. Phase A turns the audit into a test.
- **Primary control:** never log reading content on any identity-text path. Log counts, codes and
  ids only.
- **Defence in depth:** a structlog processor `redact_identity_text` in the shared chain of
  `configure_logging` (`src/core/logging.py`). Structlog processors are the documented extension
  point (https://www.structlog.org/en/stable/processors.html). The processor removes or masks
  values under known keys (`text`, `text_raw`, `text_vlm`, `text_ocr`, and any key that carries model output). It must
  run for both structlog-native and foreign stdlib records, so it goes in `shared_processors`,
  ahead of the renderer.
- **Test:** a canary string (made up, for example `QZ7XK41`) is pushed through a full region pass
  in a unit harness. The OCR and VLM stubs return the canary on every reply shape the parsers handle.
  Every captured log line is scanned (`structlog.testing.capture_logs` plus pytest `caplog`). The
  test fails if the canary, or its folded form, appears anywhere.

### 3.6 Retention, deletion, access

| Concern | Rule |
|---|---|
| Item or box deleted | its keys go with the box element (keys live on the element) |
| Project deleted | indexes, the policy document and its salt go with the project (`src/services/projects/delete_paths.py`) |
| Purge (CLI) | section 6 |
| Raw-mode retention | J6: recommended `raw_retention_days` required when choosing `raw` (no "forever" default). A periodic task (after B2's periodic worker) re-keys and drops raw text older than N days, so the `hash_only` information remains |
| Access, `hash_only` | no route returns text or keys; no route accepts a plate string to look up (a membership oracle on an unauthenticated API); the regions-tab text filter answers 400 `identity_text_hashed` for identity regions |
| Access, `raw` | as today; `SECURITY.md` says that in `raw` mode anyone who can reach the API can read identity text |
| Exports | never carry identity fields in any mode in v1 (no opt-in export is built; J9) |
| Combine | identity text and keys never cross projects (keys are per-project by construction). Combine drops them from copied boxes; the target re-reads through reprocess if it wants keys |
| Clone settings | copies the mode only; never the salt (a fresh salt is generated in the target) |
| Committed files and issues | tests use made-up strings only; research transcriptions stay in `artifacts_local/`; aggregate numbers only |

---

## 4. Sink inventory (the Phase A deliverable)

Every place a reading of an identity region can land. Phase A turns this table into a test. A new
sink added later without a row fails CI.

| # | Sink | Today | `hash_only` rule |
|---|---|---|---|
| S1 | `region_boxes[].text`, `text_raw`, `text_vlm`, `text_ocr` | stored as read | never written for identity regions |
| S2 | `region_boxes[].text_vlm_invalid` | a reason code; check that no writer puts content in it | codes only, asserted by test |
| S3 | `item_text_lines`, `item_text_tokens` on the parent item | every OCR line on the item crop | drop every line whose box overlaps an identity box on that item (containment of at least 0.5 of the line box), and drop its tokens. Recompute tokens from the kept lines only |
| S4 | human-typed text (`human_text_fields`) | stored | keyed then discarded; `text_source='human'` and `text_choice='human'` kept |
| S5 | region text repair inputs | reads S1 | repair skips identity boxes in `hash_only` (nothing to re-derive); it never reconstructs text |
| S6 | logs (API, detection worker, VLM worker) | see 3.5 | canary test |
| S7 | job files under the project state directory (region and reprocess job records, progress, error messages) | to inventory | canary scan of the job directory after a region pass |
| S8 | SSE events | `region_event_payload` carries status and count only | test pins the key set |
| S9 | VLM request payloads | the region crop pixels go to the VLM | locality rule 3.4 |
| S10 | exports (YOLO, single-class, classification) | no text today | test that iterates every export writer with a canary-bearing fixture |
| S11 | combine target | copies boxes | strip identity fields |
| S12 | core OCR index (`/ocr`, core ingest) | stores every line read on core-ingested images | out of curation scope: modes are a curation feature. Documented as a scope boundary in `SECURITY.md` |
| S13 | Triton or VLM server logs | outside the app | documented: the local vLLM server may log prompts at debug level; the compose default stays non-debug |
| S14 | research artefacts | `artifacts_local/` only | never committed; aggregates only |

---

## 5. Key design (Phase B)

### 5.1 Normalization and folding

1. Reuse the profile's normalizer (`TextNormalizer`: uppercase, charset) and stopword and
   placeholder rules (`region_text_rules.py`). These are the same rules that choose the reading
   today. No second normalizer.
2. **Format gate: reuse `text_format`** (`detection_profile.py:133`, a full-match regex applied in
   `region_text_rules.py:160`). The parked plan's new `identity_text_formats` list is unnecessary.
   A format list is a regex alternation in `text_format` (correction C4).
3. **Confusable folding, for matching only:** a new profile field `identity_text_confusables`, a
   map from character to canonical character. The example profile maps O to 0, I to 1, B to 8,
   S to 5 and Z to 2. Folding is applied to produce the hashed string; the stored raw text (in
   `raw` mode) is never folded. The fold must be idempotent and must map within the charset,
   checked at profile validation (`src/services/config_store/profile_field_checks.py`).
4. New profile field `identity_text: bool` (default `false`). True marks the profile's regions as
   identity regions, so modes, keys and the locality rule apply. The example plate profile sets it.
   Activation keys off the profile's `region_class_name` **name** and its `parent_classes` names.

### 5.2 Tiers

| Tier | Rule | Stored in `hash_only` |
|---|---|---|
| `confident` | format passes, **and** either the OCR and a local VLM agree after folding, **or** the OCR per-line minimum character confidence is at least `c_hi` (calibrated, 5.3) | tier, `text_key`, neighbourhood keys (J4) |
| `partial` | a reading exists but fails the confident rule (short, occluded, low character confidence) | tier only; no key (a partial pattern with unknown positions cannot be hashed for matching) |
| `none` | no valid reading, or more than one candidate identity box dominates (5.5) | tier only |

### 5.3 Per-character confidence (pipeline change, after B5c)

- Add an output tensor `rec_char_min` (float32, shape `[max_texts]`) to `ocr_pipeline`. It holds
  the minimum per-character max probability over the decoded characters of each line, computed in
  `CTCLabelDecode` next to the existing mean (`ocr_postprocess.py:222` to `:232`). The existing
  `rec_scores` is unchanged, so nothing else moves.
- Thread it through `src/clients/triton_ocr.py` and
  `src/services/detection/cascade_detect/ocr_recognizer.py` (`OcrLine` gains `char_min`). The
  dominant-text reading's `confidence_char_min` is the minimum over the kept lines.
- Bump `MODEL_VERSION` (the sync replaces the installed copy, as `docs/design/program/07c_b5_crops_ocr_faces.md` section 8 describes). Add a
  parity test: `rec_scores` and `texts` are byte-identical before and after on a fixed fixture.
- `c_hi` is calibrated on the E-9 dev half: the smallest threshold whose confident-key precision on
  dev is at least 0.99. It is project data in the profile. The example profile carries the
  calibrated value with the calibration id in a `_comment`.

### 5.4 Matching in `hash_only`: equality, and optionally keyed distance-1 neighbourhoods (J4)

**Equality** is a term match on `text_key`.

**Distance 1 without text:** store keyed hashes of the single-deletion variants, in two families:

```text
sub_i = HMAC(project_key, b"s\x00" + key_version + b"\x00" + str(i) + b"\x00" + delete_at(folded, i))   for i in 0..len-1
del_i = HMAC(project_key, b"d\x00" + key_version + b"\x00" + delete_at(folded, i))                      for i in 0..len-1
self_d = HMAC(project_key, b"d\x00" + key_version + b"\x00" + folded)
```

- **Substitution at distance 1** (equal lengths) holds if and only if the two keys share a
  `sub_i` with the same `i`.
- **Insertion or deletion at distance 1** (lengths differ by 1) holds if and only if the shorter
  key's `self_d` is among the longer key's `del_i`.
- This is the symmetric-deletion idea behind SymSpell (https://github.com/wolfgarbe/SymSpell),
  with keyed hashes in the manner of privacy-preserving record linkage (Schnell, Bachteler and
  Reiher 2009, BMC Medical Informatics and Decision Making,
  https://doi.org/10.1186/1472-6947-9-41).
- At most 10 characters gives at most 21 keys per box, stored as a keyword array.
- A property test proves the equivalence with a reference Levenshtein distance of at most 1
  (test-only DP) on random strings over the charset, transpositions included. An adjacent swap is
  distance 2 in Levenshtein; it must **not** be reported as a near match, and the test pins that.
- **Leakage:** a reader of the index without the keys learns which boxes are within distance 1 of
  each other. Near-match review suggestions reveal that anyway. Without the keys, no characters
  are recoverable.
- **Without neighbourhoods (J4 option b):** a `veto` in `hash_only` would be "two confident keys
  differ", and every distance-1 misread becomes a false veto. In that case the veto is not built
  in `hash_only` at all; must-link only.

### 5.5 Ambiguity rules (from 6.8, kept)

- **Several identity boxes in one item:** only the box inside the item that dominates by area (at
  least 2x the next) counts; otherwise the tier is `none`.
- **The same confident key on two items of one image:** flag for review (a reflection or a double
  detection). No edge.
- **The same key on visually incompatible objects** (embedding similarity below the floor): no
  edge, flagged. This covers dealer plates, moved plates and misreads.
- **A key whose box is human-rejected or a false positive:** no key is kept. The false-positive
  bucket (`src/routers/curation/regions_fp.py`) clears it.
- **The lock rule:** automated writers never overwrite a human- or import-validated box
  (`src/clients/occ_locks.py`). Keys are written through the OCC read-merge-write helpers, never a
  blind update.

### 5.6 Mode transitions

| From | To | Action |
|---|---|---|
| `raw` | `hash_only` | a rekey job (`op.jobs.identity_text_rekey`, queue `jobs`, lease `identity_text_rekey` per project, B2 `JobTask` wrapper) computes keys from stored text, then removes S1, S3 and S4 content through OCC; dry-run counts first; resumable (idempotent per box) |
| `hash_only` | `raw` | forward only: new reads store text; old boxes keep keys only (no text to restore). Allowed only if the ceiling is `raw` |
| any | `off` | stop reading identity regions; existing data stays; status reports counts (never values) |
| `off` | `hash_only` or `raw` | resume; a reprocess re-reads if keys are wanted for old items |
| any | purge | requires effective mode `off` (section 6) |

### 5.7 Edges (Phase C; consumer is #400's `src/domain/attribute_edges.py`)

| Edge | Rule | Level |
|---|---|---|
| must-link | two `confident` equal keys, **and** embedding similarity above a calibrated low floor, **and** compatible coarse class by **name** | identity; at class level the edge joins the two crops' view prototypes |
| cannot-link | two `confident` keys at distance at least 2 (`hash_only` needs J4 option a), hard in the merge validator, overridden only by a person | identity only, never class |
| review | distance 1, or a `partial` tier against a `confident` one (`raw` only for partial) | suggestion, never an edge |
| refused | effective mode `off` | `ValueError` from the edge builder: fail closed |

---

## 6. Purge (CLI only)

```text
./openprocessor identity-text purge --project <slug> --dry-run   # counts only
./openprocessor identity-text purge --project <slug> --yes       # delete; refuses unless effective mode is off
```

- **Script:** `scripts/purge_identity_text.py` (new), modelled line for line on
  `scripts/purge_face_data.py`. It uses `make_script_opensearch` and binds the project with
  `--project` (the project guard refuses any index the bound project does not own).
- **Refusals** (exit 2): effective mode is not `off`, in `.env` **and** in the running API (as the
  face purge checks both: `scripts/lib/cli_core.sh:150` to `:166`); the project's region lease or
  rekey lease is held (a writer is running); the project is unknown.
- **Deletes:**
  - S1 to S4 content and every key and neighbourhood key on every box of the project's identity
    class;
  - item text lines overlapping identity boxes (S3);
  - `identity_text` edge evidence once Phase C exists.
- **Rotates the salt:** writes a new `secrets.token_bytes(32)` with an OCC write on the policy
  document, then deletes the old one.
- **Audit:** appends a counts-only record to `data/audit/identity_text_purge.jsonl` (project slug,
  counts before and after, time, key version). Never values.
- **Verifies:** counts after must all be 0, else exit 1 with the remaining counts. This follows the
  face purge's `FacePurgeIncompleteError`.
- **Docs:** `SECURITY.md` gets a "Privacy: identity text" section next to the face section. It
  lists what the purge does not reach (snapshots, backups, Loki logs, older-version logs,
  downloaded copies, `artifacts_local/`).

---

## 7. Research gate (E-9) and stop rule

### 7.1 What E-9 measures (public data only)

- Detection recall and precision of the identity region at IoU 0.5, per input regime (full frame
  against vehicle crop) and per rotation stratum.
- Read quality: exact-match accuracy, character error rate.
- Confident-key precision (target at least 0.99) and coverage (share of vehicle crops with a
  confident key).
- Region-stage cost per crop.

### 7.2 Data, and why the existing pin is too small

- **The bar needs a sample:**
  - with zero errors in n confident keys, the 95 % upper bound on the error rate is about 3/n
    (rule of three), so n must be at least 300 to show precision at least 0.99;
  - with one error, the Clopper-Pearson bound (`statsmodels proportion_confint(method='beta')`)
    needs about 475;
  - the 300-image pin yields fewer confident keys than images (coverage below 100 %, some images
    with no legible plate).

  **Register n at least 500 confident keys on the test half** before any read.
- **Extend the fetcher:** add Open Images V7 **train** split support, with CC BY 2.0 only and the
  class resolved by display name as today. Count eligible boxes before downloading images. This is
  the same fetcher extension Z5 (#398) needs, so build it once.
- **Transcription:** a person transcribes the sample locally. The transcription stays in
  `artifacts_local/` and is never committed. Committed text holds aggregates only.
- **Optional local reading sets:** OpenALPR benchmarks (https://github.com/openalpr/benchmarks,
  licence to verify), UFPR-ALPR (research use on request, https://arxiv.org/abs/1802.09567) and
  CCPD (https://github.com/detectRecog/CCPD, research use). These are local cross-checks only.
- **Linking (link precision of at least 0.99, same-vehicle recall by view):** owner-run on the
  private set, numbers only (makemodel plan O7: plate text stays in gitignored local artifacts and
  never reaches a hosted model).

### 7.3 Stop rule

| Stop when | Then |
|---|---|
| the owner does not approve this privacy design (J2) | #401 stays parked; Phase A may still ship (J1) |
| E-9 confident-key precision below 0.99 on the registered n, or coverage below 10 % of vehicle crops | Phase B does not land; plates stay generic region text; the outcome note records it |
| owner-run link precision below 0.99 | Phase C does not land |
| RQ-5 (constraint machinery) never lands | Phase C has no consumer; Phase B keys still serve review ("same plate as") only if the owner wants that (J10) |

### 7.4 Revisit triggers (what un-parks #401)

1. The owner answers OQ-6 with "yes, this year" and approves this design (J2).
2. B5c is merged, so the OCR pipeline is stable for the `rec_char_min` change.
3. The Open Images train-split fetcher exists (shared with #398's Z5).

Phase C also needs the #400 edge interface and an approved RQ-5 consumer.

---

## 8. How we prove it: verification plan

All tests offline (`-m 'not live'`), with made-up strings only. Each is watched failing before the
fix (red), against a stub that breaks exactly the rule under test.

### 8.1 Phase A: privacy baseline

| Requirement | Test (proposed) | Red looks like |
|---|---|---|
| No reading reaches logs on any identity path, failure paths included | `tests/test_identity_text_canary.py::test_region_pass_logs_no_canary`: stub OCR and VLM return the canary in every reply shape the parsers branch on; scan `capture_logs` and `caplog` | the canary appears in a captured line |
| The redaction processor covers foreign stdlib records | `...::test_redactor_on_stdlib_record` | a stdlib `logging.warning(..., extra={'text': canary})` renders the canary |
| Job files carry no reading | `...::test_job_dir_has_no_canary` (scan every file a region job writes in a tmp state dir) | the canary appears in a progress or error file |
| SSE payload key set | `...::test_region_event_keys_pinned` | an extra key |
| Exports carry no identity fields | `...::test_every_export_writer_omits_identity_text`, iterating every export writer registered in `src/services/curation/` (assert the list is non-empty, so a registry rename cannot make it pass vacuously) | a writer emits the canary |
| Identity regions never go to an external endpoint | `tests/test_identity_text_locality.py`: endpoint locality `external`, `unknown`, and the built-in env endpoint pointing outside; assert no VLM call, OCR only, `text_vlm_invalid='external_endpoint'` | a VLM call is made |
| Sheet builder refuses hosted backends for identity items | extend `tests/test_contact_sheet.py` | a `kind='object'` sheet with an identity item is accepted |
| Ceiling parse | `tests/test_env_surface.py` plus `tests/test_identity_text_config.py`: malformed value raises; default `hash_only`; live read | a bad value defaults to `raw` |

### 8.2 Phase B: keys and modes

| Requirement | Test | Red looks like |
|---|---|---|
| `hash_only` stores no characters | write path with a canary reading; assert no field of the written doc contains the canary or its folded form (walk the whole body, nested lists included) | `text_raw` written |
| Item text lines overlapping identity boxes are dropped | fixture with a plate line inside the box and a sign line outside | the plate line survives |
| Human text keyed then discarded | `human_text_fields` in an identity project | text stored |
| Keys differ across projects | the same reading in two projects with two salts | equal keys |
| Purge rotation makes old keys unreproducible | key before purge is not equal to the key after, for the same reading and the same master | equal |
| `key_version` in the hash | the same reading at two versions | equal keys |
| Fold idempotence and charset closure | profile validation test | a fold mapping into a char outside the charset is accepted |
| Tiers | confident by agreement; confident by `char_min`; partial on low `char_min`; none on two dominant boxes | wrong tier |
| Neighbourhood equivalence (J4 a) | property test against a reference Levenshtein (at most 1), 10,000 random pairs, adjacent swaps included | a swap reported as near |
| `off` refuses edges and reads | edge builder raises; worker writes no reading | an edge or a reading |
| Mode above ceiling refused | policy PUT with `raw` under ceiling `hash_only` → 422 | accepted |
| Salt never on the wire, never cloned | serialise the policy; run `clone_settings` | salt present, or equal salts |
| Combine strips identity fields | combine fixture | keys copied |
| Lookup oracle refused | regions-tab text filter on identity regions in `hash_only` → 400 `identity_text_hashed` | results returned |
| Rekey job idempotent | run twice; the second run writes 0 docs | rewrites |
| Pipeline parity | `rec_scores` and `texts` identical with and without `rec_char_min` on a fixed fixture | any byte differs |

### 8.3 Purge

| Requirement | Test | Red looks like |
|---|---|---|
| Refuses while on | mode `hash_only` → exit 2, nothing deleted | deletes |
| Refuses while a writer holds the lease | held lease fixture → exit 2 | deletes |
| Dry run changes nothing | counts equal before and after | counts change |
| Leaves 0 | counts after are 0; else exit 1 | exit 0 with leftovers |
| Audit counts only | audit line contains no canary and no key | key present |

### 8.4 Dynamic and empirical

- One live check on the dev stack under its own compose project name (`docker compose -p <name>`,
  never the default project). Ingest 50 Open Images plate images into a throwaway project in
  `hash_only`, then:
  - scan every document in the project's indexes for the transcribed strings (from
    `artifacts_local/`) and expect 0 hits;
  - scan `docker compose -p <name> logs` the same way;
  - run the purge dry run, the purge, and the post-count;
  - delete the throwaway project.
- E-9 on GPU in a research window (RW-3 in the research plan), after the privacy review (J2):
  numbers in `docs/research/identity_text_eval.md` (aggregates only).

### 8.5 Static gates

- `make contracts` in the same commit as each wire change, then `make contracts-check`; the docs
  check (`scripts/docs/check_docs_vs_code.py`). Proposed routes are written without a leading
  slash until they exist.
- `scripts/codegen/check_naming_leaks.py`: field and module names say `identity_text`, never a
  domain word. The scan bans domain field literals; read its Scan B before naming anything.

---

## 9. Corrections to the issue's premises (and to the parked plan 6.8 and W12)

| # | Premise | Correction | Evidence |
|---|---|---|---|
| C1 | Purge is a route (`{P}/identity_text/purge`) | Purge is CLI-only, following the face precedent: the API has no authentication | `SECURITY.md` face section; `scripts/purge_face_data.py:1` |
| C2 | The per-project secret lives in the project state directory | Split: a deployment master in a read-only secrets mount, plus a per-project salt in the policy document. A secret stored with the data it protects travels with that data | 3.3; `vlm_endpoints.py:187` precedent |
| C3 | `hash_only` supports the 6.8 veto (edit distance >= 2) and near match (distance 1) | Not with a plain hash. Either keyed neighbourhoods (5.4) or must-link only | 5.4 |
| C4 | New profile field `identity_text_formats` | `text_format` (full-match regex) already exists and is enforced on every reading | `detection_profile.py:133`, `region_text_rules.py:160` |
| C5 | The confident tier uses "the OCR minimum per-character confidence" | The pipeline exposes only a per-line mean of per-character max probabilities; a new output is needed, after B5c | `ocr_postprocess.py:232` |
| C6 | Raw text is "not stored" in `hash_only` if the region fields are hashed | The item OCR lines (`item_text_lines`, `item_text_tokens`) hold the same characters; the mode must cover them | `item_text.py:38`; S3 |
| C7 | The 300-plate pin can measure confident-key precision at least 0.99 | It cannot read that bar; register n of at least 500 confident keys and extend the fetcher to the train split | 7.2 |
| C8 | "Fail closed when unset" and "`hash_only` default" | Both hold once project creation writes the policy explicitly; "unset" then only means a pre-existing project (J7) | 3.2 |
| C9 | Plate linking can be evaluated on public data | Only read quality and key precision can; linking is owner-run | `research_clustering_plan.md` E-9 |
| C10 | The issue's rationale cites `docs/design/improvement_backlog.md` | That file was deleted when the program plan set landed (`1b0d4e29f`); its rows moved to `docs/design/program/12_prior_plans_reconciliation.md` 2.14 and the research plan | `git log -- docs/design/improvement_backlog.md` |

---

## 10. Judgment calls

| J | Question | Recommendation | Trade-off |
|---|---|---|---|
| J1 | Ship Phase A (privacy baseline: sink inventory, canary tests, local-only identity reads, log redaction, the ceiling env) now, in the v0.8.0 hardening milestone, independent of whether #401 is ever built? | **Yes.** It protects readings the generic text-region feature already stores. It has no research dependency and is small (S to M) | One more v0.8.0 item; the ceiling env exists before the modes do (it then gates only `raw` versus not) |
| J2 | Approve this privacy design (modes, ceiling, key split, CLI purge, no lookup route) as the precondition the triage comment asked for? | **Yes**, with J4 to J9 decided | Without it, #401 stays parked by its own stop rule |
| J3 | Key derivation: stdlib HMAC as a single-block SP 800-108 PRF, or HKDF (RFC 5869) through the `cryptography` package? | **stdlib HMAC**: no new dependency, documented construction, RFC 2104 test vectors in the unit test | `cryptography` gives a named KDF and is widely audited, but it is a heavy native dependency for two HMAC calls |
| J4 | In `hash_only`: (a) keyed distance-1 neighbourhoods (near match and veto work), or (b) equality only (must-link only, no veto, no near match)? | **(a)** if E-9 shows distance-1 misreads among confident keys above 1 %; otherwise **(b)**. Decide on the E-9 numbers | (a) leaks within-distance-1 structure to an index reader and stores up to 21 keys per box; (b) loses the veto |
| J5 | Default ceiling `hash_only` (raw needs an env edit and restart), or `off` (any identity reading needs an env edit)? | **`hash_only`**, per makemodel O9 | `off` is stricter but makes the example plate profile read nothing out of the box |
| J6 | Raw-mode retention: require `raw_retention_days` with a periodic re-key-and-drop, or keep raw text until purge? | **Require it** (no default; refuse `raw` without it). The periodic task runs on B2's periodic worker | One more periodic job; operators who want raw text forever must set a large number deliberately |
| J7 | Existing projects with an identity-text profile and stored readings when Phase B lands: (a) refuse identity reads until the operator picks a mode with the CLI, (b) default to `hash_only` and run the rekey job automatically (destroys the stored text), or (c) default to `raw` (keeps behaviour)? | **(a)**, with the status naming the counts and the command. No silent destruction and no silent `raw` | Projects stop reading identity text until someone acts |
| J8 | In a project with an identity-text profile, may an external VLM label items at all (vehicle crops show legible plates)? | **Refuse activation of an external endpoint for such a project unless the operator acknowledges it per project** (a project-level acknowledgement on the VLM policy document, recorded with time); identity regions themselves are never read externally (3.4) | Friction for operators who rely on a hosted VLM; the alternative is pixels with plates leaving the deployment with only a deployment-wide acknowledgement |
| J9 | An opt-in export of identity text (raw mode only)? | **Not in v1.** Build only on a concrete request | None today; no export carries text |
| J10 | If RQ-5 never lands, should Phase B keys still drive a "same identity text as" review view (no clustering)? | **Owner's call after E-9**; the keys support it without new storage | Feature creep versus a useful review aid |
| J11 | The research judge: also refuse hosted sheets by content (any item with an identity region), not only by declared kind? | **Yes** (3.4); research tooling, small | None |

---

## 11. File-by-file change list

### Phase A: privacy baseline (no research dependency)

| Subsystem | File | Change |
|---|---|---|
| Config | `src/config/identity_text.py` (new) | `identity_text_max_mode()` reads `OP_IDENTITY_TEXT_MAX_MODE` live; malformed raises `EnvError`; `effective_mode(project_mode)`; modeled on `src/config/faces.py` |
| Config | `env.template`, `tests/test_env_surface.py`, docs-site configuration page | the new variable, same commit |
| Logging | `src/core/logging.py` | `redact_identity_text` processor in `shared_processors`; unit tests |
| Logging | every log call on region-text, OCR and VLM-reply paths found by the audit (3.5) | log counts and codes, not content |
| Labeling | the region-text VLM read path (`scripts/curation/worker/region_text_stage.py` and the verify call it makes) | identity regions: VLM only when `compute_locality` is local, for every endpoint source; else OCR only with a reason code |
| Research | `scripts/research/judge_protocol.py` and the sheet builder `src/services/labeling/contact_sheet.py` | refuse hosted backends when any member item carries an identity region |
| Profile | `src/config/detection_profile.py`, `src/routers/curation/_region_profile_schema.py`, `src/services/config_store/profile_field_checks.py` | `identity_text: bool`; validation |
| Example | the example plate profile under `examples/region_profiles/` | `"identity_text": true` |
| Tests | section 8.1 files | canary, locality, exports, SSE, sheets, env |
| Docs | `SECURITY.md` | "Privacy: identity text" section (what is stored, the local-only rule, scope boundary with the core OCR index) |

### Phase B: keys, modes, purge (after E-9 PASS and J2)

| Subsystem | File | Change |
|---|---|---|
| Domain | `src/domain/identity_text.py` (new, pure) | normalize-and-fold (reusing `TextNormalizer` and the rules), tiers, neighbourhood variants (J4), ambiguity rules; imports only `config` and `utils` |
| Service | `src/services/curation/identity_text_keys.py` (new) | master read from the secrets mount (`OP_IDENTITY_TEXT_SECRETS_DIR`, default `/run/secrets/op_identity_text`, a regular non-symlink file, as `_secret_file` checks for VLM keys, of exactly 32 bytes), project key derivation, `text_key` and neighbourhood keys |
| Service | `src/services/curation/identity_text_policy.py`, `identity_text_policy_store.py` (new) | policy body (`extra='forbid'`), OCC get/put, salt generation, never serialised |
| Worker | `scripts/curation/worker/region_text_stage.py` | the mode transform at the `_drop_region_text` hook: in `hash_only` compute keys, drop content from S1, S3 and S4 |
| Domain | `src/domain/region_boxes.py` | `RegionBox` gains `text_tier`, `text_key`, `text_key_version`, `text_key_nbr` (fixed keys) |
| Mapping | `src/clients/curation_opensearch/` (a new `identity_text_fields.py` on the `assist_fields.py` pattern) | keyword fields for the key and the neighbourhood array, inside the `region_boxes` nested mapping |
| Edits | `src/services/curation/region_box_edits.py` | human text in identity projects is keyed then discarded |
| Repair | `src/services/curation/region_text_repair.py` | skip identity boxes in `hash_only` |
| Review | `src/services/curation/review_tab_clauses.py` | 400 `identity_text_hashed` for a text filter on identity regions in `hash_only` |
| Combine, clone | `src/services/projects/combine/regions.py`, `src/services/projects/clone_settings.py` | strip identity fields; copy the mode, fresh salt in the target |
| Projects | project creation (`src/services/projects/lifecycle.py`) | create-only first write of the policy with `hash_only` for a project whose profile is identity-text |
| Jobs | new `op.jobs.identity_text_rekey` (B2 `JobTask`, lease per project) | `raw` to `hash_only` transition and the J6 retention sweep |
| OCR | `models/ocr_pipeline/1/ocr_postprocess.py`, `model.py`, `config.pbtxt`, `src/clients/triton_ocr.py`, `src/services/detection/cascade_detect/ocr_recognizer.py`, `src/services/detection/region_text.py` | `rec_char_min` output and its use; `MODEL_VERSION` bump; after B5c |
| Purge | `scripts/purge_identity_text.py` (new), `scripts/lib/cli_core.sh`, `openprocessor` help text | CLI purge; `rotate-master` |
| Compose | `docker-compose.yml` | `./secrets/identity_text:/run/secrets/op_identity_text:ro` on `api` and `curation-detection-worker` only |
| Installer | `setup-openprocessor.sh`, `scripts/setup.sh` | generate the master once (32 random bytes, mode 600), never print it |
| Wire | `src/services/curation/wire.py`, `src/routers/curation/_item_models.py`, contracts | tier and "has key" only in `hash_only`; `make contracts` |
| Routes | policy GET/PUT (proposed `curation/projects/{project}/identity_text/policy`); no purge route; no lookup route | wire models, contracts, `docs/design/curation_api_contract.md` |
| Docs | `SECURITY.md`, `docs/CURATION.md`, docs-site privacy guide, `CHANGELOG.md` | modes, ceiling, purge, what it does not reach |

### Phase C: edges (after #400's interface and an RQ-5 consumer)

| File | Change |
|---|---|
| `src/domain/attribute_edges.py` (from #400) | register `identity_text` with roles per 5.7 |
| the RQ-5 clustering job | consume the edges; evidence `attribute:identity_text` on suggestions |
| Cropwright | show "same identity text" or "near match" evidence (never the text in `hash_only`) |

### Research (any time after the tag; GPU in RW-3)

| File | Change |
|---|---|
| `scripts/datasets/fetch_openimages_plates.py` | train split, eligible-box count before download (shared with #398 Z5's fetcher work) |
| `scripts/research/identity_text_eval.py` (new) | read accuracy, CER, tiers, `c_hi` calibration on dev, coverage; reads transcriptions from `artifacts_local/` |
| `docs/research/identity_text_eval.md` (new) | R0 registration (n of at least 500 confident keys), then aggregates only |

---

## 12. Edge cases and traps (the failure register)

1. **A canary test that covers only the happy path.** If the stubs only return well-formed
   replies, most code paths are never exercised. The stub must cover every reply shape the parser
   branches on, and the test asserts each branch was hit (a counter on the stub).
2. **A redaction processor that only sees structlog records.** Foreign stdlib records reach the
   formatter by a different route. Put the processor in `shared_processors` (used for both), and
   test both kinds.
3. **Hashing the region fields while the item OCR lines keep the plate.** The single most likely
   way the feature looks done and is not (C6). The "no characters" test walks the **whole**
   written body, not only `region_boxes`.
4. **Token search still finds the plate.** Dropping `item_text_lines` without recomputing
   `item_text_tokens` leaves the plate searchable through `?item_text=`. Test the search, not only
   the field.
5. **The ceiling read once at import.** A module constant ignores a changed `.env` after restart in
   tests and in the CLI's "running API still on" check. Read it live, like `faces_enabled()`.
6. **`effective_mode` compares strings with `==`.** Use an ordered enum (`off` < `hash_only` <
   `raw`) and `min()`. An exact-string check silently disarms the ceiling for any new value.
7. **The salt served by a generic serialiser.** A policy GET that dumps the model includes the
   salt. Serialise named fields only, and pin the key set in a test.
8. **`clone_settings` copying the salt.** Two projects would then share keys, which is a
   cross-project linkage. Test equal readings in source and clone produce different keys.
9. **Combine copying keys.** The same cross-project linkage through another door. Strip, and test.
10. **Edges from different key versions.** After a normalization change, an old and a new key for
    the same plate differ. That is safe (no edge), but a near-match on neighbourhoods across
    versions is meaningless. Compare only within one `key_version`.
11. **Folding that maps outside the charset or is not idempotent.** A fold of O to 0 and 0 to O
    makes keys order-dependent. Validate fold(fold(x)) == fold(x) and closure at profile save.
12. **Confusable folding applied to stored raw text.** In `raw` mode a person reviews the text as
    read. Folding is for the hashed form only.
13. **Neighbourhood keys that count a transposition as distance 1.** "AB" and "BA" share
    untagged deletion variants. The position-tagged substitution family is what makes the test
    exact. A property test against a reference Levenshtein catches it.
14. **A purge that runs while the worker writes.** It leaves new keys keyed with the new salt
    mixed with nothing, or old keys after the count. Refuse on mode not `off` **and** on a held
    lease, then count after.
15. **A purge that "succeeds" with leftovers.** Exit 0 only when every post-count is 0. This is
    the face purge rule.
16. **Rotation that writes the new salt before deleting keys.** A crash in between leaves keys that
    no salt explains, which is harmless but uncountable as "keyed". Order: delete keys, verify
    0, then rotate the salt, then audit.
17. **The locality check done once at activation.** DNS can change. Recompute per read (the
    existing 30-second cache in `acompute_locality` is acceptable), and treat `unknown` as
    external.
18. **The built-in env endpoint treated as trusted for identity reads.** The identity rule applies
    to every endpoint source. Test the env source pointing outside.
19. **Human-typed text in `hash_only` echoed back.** The edit route's response must not return the
    typed text after keying, or the typed value would travel in the wire.
20. **The regions-tab text filter as an oracle.** An exact-match filter on keys in `hash_only`
    lets any caller test whether a string is present. The filter is refused, not converted.
21. **Tests that use realistic strings.** Never use a real registration. Use obviously made-up
    canaries (for example mixed patterns that no jurisdiction issues), and never commit research
    transcriptions.
22. **A test that passes on an empty export registry.** Assert the iterated writer list is
    non-empty and names the known writers.
23. **`rec_char_min` added but `rec_scores` changed.** Every other consumer of OCR confidence
    would silently move. The parity test pins `rec_scores` and `texts` byte for byte.
24. **The pipeline output padded with 0.0.** Padded slots (`model.py:446` pads `rec_scores` with
    0.0) must pad `rec_char_min` the same way and be ignored by count, never read as "very
    unconfident real lines".
25. **Class compatibility by id in the must-link.** Compare coarse-class names (casefolded), never
    ids. After a combine or an import, ids differ.
26. **A key computed on an item whose class a human owns, written blindly.** Keys go through OCC
    read-merge-write and the lock rule, like every automated region write.
27. **Status or health that prints a value.** Counts only. A status that names "the plate read on
    item X" is a leak through the one place operators look first.

---

## 13. Suggested phasing

| Phase | When | What | Exit |
|---|---|---|---|
| A | v0.8.0 hardening if J1 = yes; otherwise with B | sink inventory and canary tests, log redaction, local-only identity reads, sheet refusal by content, ceiling env, `identity_text` profile flag, `SECURITY.md` section | 8.1 green; `SECURITY.md` updated |
| R0 | any time (docs) | register E-9 bars and n in `docs/research/identity_text_eval.md` | committed before any read |
| R1 | after the v0.7.0 tag and J2; GPU in RW-3 | train-split fetcher, local transcription, E-9 read quality and `c_hi` calibration (research script reads per-character minimums directly from the recogniser output, no product change) | PASS or FAIL against the registered bar |
| B | after R1 PASS and B5c merged | keys, modes, policy, rekey job, purge CLI, `rec_char_min`, wire | 8.2 and 8.3 green; one live check (8.4); contracts and docs checks clean |
| C | after #400's edge interface and an approved RQ-5 consumer | edges, owner-run link precision on the private set | owner-run numbers at least 0.99; G3 review in Cropwright |

---

## 14. Sources

- Python `hmac`: https://docs.python.org/3/library/hmac.html. Python `hashlib`:
  https://docs.python.org/3/library/hashlib.html. Python `secrets`:
  https://docs.python.org/3/library/secrets.html
- HMAC: RFC 2104, https://www.rfc-editor.org/rfc/rfc2104. HKDF: RFC 5869,
  https://www.rfc-editor.org/rfc/rfc5869
- NIST SP 800-108r1, Recommendation for Key Derivation Using Pseudorandom Functions:
  https://csrc.nist.gov/pubs/sp/800/108/r1/upd1/final
- structlog processors: https://www.structlog.org/en/stable/processors.html. structlog testing
  (`capture_logs`): https://www.structlog.org/en/stable/testing.html
- SymSpell symmetric deletion: https://github.com/wolfgarbe/SymSpell
- Privacy-preserving record linkage with keyed q-gram encodings: Schnell, Bachteler, Reiher 2009,
  https://doi.org/10.1186/1472-6947-9-41
- statsmodels `proportion_confint` (Clopper-Pearson `beta`, Wilson):
  https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
- Rule of three for zero-event upper bounds: Hanley and Lippman-Hand 1983, JAMA 249(13)
- PaddleOCR CTC decoding (mean confidence per line): https://github.com/PaddlePaddle/PaddleOCR
  (`ppocr/postprocess/rec_postprocess.py`, `CTCLabelDecode`)
- Open Images V7: https://storage.googleapis.com/openimages/web/factsfigures_v7.html
- Reading sets (local only): https://github.com/openalpr/benchmarks, https://arxiv.org/abs/1802.09567,
  https://github.com/detectRecog/CCPD
- Repo: `SECURITY.md` (face privacy), `scripts/purge_face_data.py`, `src/config/faces.py`,
  `docs/design/research_clustering_plan.md` (RQ-7, E-9, 7.1, OQ-6),
  `docs/design/cluster_search_makemodel_plan.md` (6.8, W12, O7, O9).
