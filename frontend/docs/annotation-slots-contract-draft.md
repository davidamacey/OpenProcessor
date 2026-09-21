# `annotation_slots` wire contract — draft (H3 opening offer)

> **Status: DRAFT / PROPOSAL. Not a commitment by either side.** This is
> Cropwright's opening offer for the `annotation_slots` wire contract
> (H3). Nothing described here is implemented on either side: the
> frontend's tier-2 and tier-3 profile loading are both explicitly "not
> yet wired" (`src/lib/annotations/registry.ts:4-9`), and the backend has
> not added the field. It exists so the backend has something concrete to
> amend or reject rather than a blank page.

This document is Cropwright's (formerly legacy-labeler's) half of
handoff point **H3** from
`/data/repos/wt-oss-hardening/docs/design/cropwright_backend_integration_plan.md`
and from this repo's own
`docs/genericization-plan-2026-09-13.md` §4.4 item 3. It is written from
the shipped code, not from that plan's original §2.5 sketch, which has
drifted from what actually landed — see the file citations throughout.

---

## 1. The problem

Two independent mapping tables describe one wire format, with no shared
source of truth:

- The backend (OpenProcessor, historically openprocessor) has its own
  field-mapping indirection, `src/config/region_fields.py`
  (`RegionFields`), and intends to keep it regardless of whether the
  `plate_*` → generic field rename ever runs.
- The frontend has `src/lib/annotations/types.ts`'s capability model,
  where each capability names the wire field(s) it reads —
  `SubBoxCapability.bboxField`, `TextCapability.valueField`,
  `ProvenanceCapability.detectorField`, and so on — and
  `src/lib/annotations/profiles/licensePlate.ts` is the one deployed
  instance, decomposing the ~30 `plate_*` fields on a crop into five
  capability blocks.

Two hand-maintained tables between two indirections drift by
construction, not by accident. This repo already has direct evidence of
that failure mode: `HUMAN_PLATE_STATUS_VALUES` is defined twice in the
backend's `src/routers/legacy/_common.py`, and this repo separately
hand-copies the same four human-writable plate-status values in two more
places (`docs/genericization-plan-2026-09-13.md` Finding C.4). Four
copies, two repos, no shared source, for one four-value list. A field
the size of the full annotation-slots contract will not do better left
to hand-maintenance.

---

## 2. The three-tier resolution model

`src/lib/annotations/registry.ts:4-9` documents the intended merge order,
of which only tier 1 is wired today:

1. **Built-in profiles** (tier 1, **the only one shipped**). Compiled
   into the app — `src/lib/annotations/profiles/licensePlate.ts` is the
   sole real, deployed instance today.
2. **Deployment override**, loaded from `static/annotation-profiles.json`
   (tier 2, **wired as of 2026-09-20** —
   `src/lib/annotations/deploymentProfiles.ts` fetches it in the root
   layout's `load()` and `src/lib/annotations/config/parseSlotConfig.ts`
   is the hardened §4/§5 validator that feeds `resolveSlotRegistry`'s
   `deployment` param. See
   `docs/design/tier2-annotation-profile-config-plan-2026-09-20.md` for
   the full design and §9 below for what this closes).
3. **Server-declared**, `OpClass.annotation_slots` on the class registry
   response (tier 3, **not wired** — the field does not exist on either
   side).

`resolveSlotRegistry()` in `registry.ts` already implements the merge
mechanics ahead of tier 2/3 being wired, so this document is describing
a function that exists and is exercised in tests
(`src/lib/annotations/registry.ts`), just not yet fed by anything beyond
the built-in array.

**Merging is per-slot-key REPLACE, not deep-merge.** `resolveSlotRegistry`
keys a `Map<SlotKey, SlotSpec>` by `spec.key` and later tiers overwrite
the whole entry for that key — never merge field-by-field. This is
deliberate: a deep-merge would let a partial, buggy override silently
combine with the built-in default and produce a spec that was never
actually validated as a whole. A REPLACE either takes the full override
or falls back to the full built-in.

**An absent or invalid entry degrades to the built-in with a warning,
never a throw.** `validateSlot()` in `registry.ts` checks for a
non-empty string `key`, an object `bind` with at least one of
`className`/`classId`, and an object `capabilities` block; on any
failure it pushes a human-readable string onto a `warnings: string[]`
array and returns `false`, and the caller skips that candidate instead of
installing it. A backend or deployment-JSON bug in one slot's
declaration therefore cannot break every other slot, or crash the
registry.

---

## 3. The proposed wire shape

An optional field on `GET {API_PREFIX}/classes`:

```ts
annotation_slots?: SlotSpec[]  // see §4 for SlotSpec
```

**Consumed when present, ignored when absent** — the same
capability-gating idiom `src/lib/strategies.ts` already uses for the
`/methods` overlay list (`FALLBACK_METHODS`, gated per-overlay on a
`status` the backend reports). A backend that has not shipped the field
yet, or a backend intentionally choosing not to declare per-class slots,
produces no error and no missing functionality beyond what tier 1
already provides — `resolveSlotRegistry({ builtins })` with no
`deployment` argument is already a fully-valid, tested call.

---

## 4. The `SlotSpec` schema

Full TypeScript source: `src/lib/annotations/types.ts`. Reproduced here
as JSON Schema for the wire shape, followed by the TypeScript it
mirrors.

### 4.1 JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "SlotSpec",
  "type": "object",
  "required": ["key", "bind", "label", "capabilities", "endpoints"],
  "properties": {
    "key": { "type": "string", "minLength": 1 },
    "bind": {
      "type": "object",
      "properties": {
        "className": { "type": "string" },
        "classId": { "type": "integer" }
      },
      "minProperties": 1
    },
    "label": {
      "type": "object",
      "required": ["singular", "plural", "title"],
      "properties": {
        "singular": { "type": "string" },
        "plural": { "type": "string" },
        "title": { "type": "string" }
      }
    },
    "capabilities": {
      "type": "object",
      "properties": {
        "subBox": { "$ref": "#/$defs/subBoxCapability" },
        "text": { "$ref": "#/$defs/textCapability" },
        "provenance": { "$ref": "#/$defs/provenanceCapability" },
        "lifecycle": { "$ref": "#/$defs/lifecycleCapability" },
        "queue": { "$ref": "#/$defs/queueCapability" },
        "trainingCohorts": { "$ref": "#/$defs/trainingCohortsCapability" }
      }
    },
    "endpoints": { "$ref": "#/$defs/slotEndpoints" },
    "stats": {
      "type": "object",
      "properties": {
        "key": { "type": "string" },
        "panelTitle": { "type": "string" },
        "coverageTitle": { "type": "string" }
      }
    },
    "extras": { "type": "object" }
  },
  "$defs": {
    "wireField": {
      "type": "string",
      "description": "A field name on the raw crop JSON."
    },
    "templatePath": {
      "type": "string",
      "description": "A path template relative to API_PREFIX, e.g. '/crops/{cropId}/region_thumbnail?size={size}'. Placeholders are a closed allow-list — see §5.1."
    },
    "shapeEnvelope": {
      "type": "object",
      "properties": {
        "aspectMin": { "type": "number" },
        "aspectMax": { "type": "number" },
        "maxWidthFrac": { "type": "number" },
        "maxHeightFrac": { "type": "number" },
        "maxAreaFrac": { "type": "number" }
      }
    },
    "subBoxCapability": {
      "type": "object",
      "required": ["bboxField", "storedFrame", "ring", "editor"],
      "properties": {
        "bboxField": { "$ref": "#/$defs/wireField" },
        "storedFrame": { "enum": ["source", "parent"] },
        "frameField": { "$ref": "#/$defs/wireField" },
        "scoreField": { "$ref": "#/$defs/wireField" },
        "visibleField": { "$ref": "#/$defs/wireField" },
        "envelope": { "$ref": "#/$defs/shapeEnvelope" },
        "thumbnail": {
          "type": "object",
          "required": ["path", "aspect", "defaultSize"],
          "properties": {
            "path": { "$ref": "#/$defs/templatePath" },
            "aspect": { "type": "string" },
            "defaultSize": { "type": "integer" }
          }
        },
        "ring": {
          "description": "Amended 2026-09-20 (docs/design/tier2-annotation-profile-config-plan-2026-09-20.md §2.3): a JSON config may supply EITHER a named preset string OR the explicit three-string object — never a third, ad hoc shape.",
          "oneOf": [
            {
              "type": "string",
              "enum": ["default", "neutral"],
              "description": "A name from src/lib/annotations/config/allowLists.ts's RING_PRESETS."
            },
            {
              "type": "object",
              "required": ["confirmed", "proposed", "rejected"],
              "properties": {
                "confirmed": { "type": "string" },
                "proposed": { "type": "string" },
                "rejected": { "type": "string" }
              },
              "description": "Every one of the three values must additionally be a member of RING_CLASS_ALLOWLIST — see §5.4."
            }
          ]
        },
        "editor": {
          "type": "object",
          "required": ["thumbSize", "viewPadding", "nudgeStep"],
          "properties": {
            "thumbSize": { "type": "integer" },
            "viewPadding": { "type": "number" },
            "nudgeStep": { "type": "number" }
          }
        }
      }
    },
    "textCapability": {
      "type": "object",
      "required": ["valueField", "label"],
      "properties": {
        "valueField": { "$ref": "#/$defs/wireField" },
        "rawField": { "$ref": "#/$defs/wireField" },
        "sourceField": { "$ref": "#/$defs/wireField" },
        "confidenceField": { "$ref": "#/$defs/wireField" },
        "engineVersionField": { "$ref": "#/$defs/wireField" },
        "label": { "type": "string" },
        "placeholder": { "type": "string" },
        "transform": { "enum": ["none", "uppercase", "lowercase", "trim"] },
        "pattern": {
          "type": "object",
          "properties": {
            "source": { "type": "string" },
            "flags": { "type": "string" }
          },
          "description": "String + optional flags on the wire, never a serialized RegExp (see §5.3)."
        },
        "maxLength": { "type": "integer" },
        "monospace": { "type": "boolean" },
        "vocabulary": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["value", "label"],
            "properties": {
              "value": { "type": "string" },
              "label": { "type": "string" },
              "description": { "type": "string" }
            }
          }
        }
      }
    },
    "provenanceCapability": {
      "type": "object",
      "required": ["detectorField", "showChainOnCard"],
      "properties": {
        "detectorField": { "$ref": "#/$defs/wireField" },
        "detectorVersionField": { "$ref": "#/$defs/wireField" },
        "chainField": { "$ref": "#/$defs/wireField" },
        "verifierField": { "$ref": "#/$defs/wireField" },
        "verifierVersionField": { "$ref": "#/$defs/wireField" },
        "verifiedAtField": { "$ref": "#/$defs/wireField" },
        "detectedAtField": { "$ref": "#/$defs/wireField" },
        "showChainOnCard": { "type": "boolean" }
      }
    },
    "lifecycleCapability": {
      "type": "object",
      "required": ["statusField", "states", "confirmState", "rejectState"],
      "properties": {
        "statusField": { "$ref": "#/$defs/wireField" },
        "verifiedField": { "$ref": "#/$defs/wireField" },
        "rejectionReasonField": { "$ref": "#/$defs/wireField" },
        "states": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["value", "label", "humanWritable"],
            "properties": {
              "value": { "type": "string" },
              "label": { "type": "string" },
              "humanWritable": { "type": "boolean" },
              "role": {
                "enum": [
                  "proposed",
                  "confirmed",
                  "rejected",
                  "falsePositive",
                  "absent",
                  "pending"
                ]
              },
              "dim": { "type": "boolean" },
              "badge": { "type": "string" },
              "aliases": {
                "type": "array",
                "items": { "type": "string" },
                "maxItems": 8,
                "description": "Additional raw status values that resolve to this state on read. `value` is always what the UI writes; `aliases` is what it accepts — the read-tolerance half of a wire-vocabulary migration (see docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §8.4(ii))."
              }
            }
          }
        },
        "confirmState": { "type": "string" },
        "rejectState": { "type": "string" },
        "falsePositiveState": { "type": "string" }
      }
    },
    "queueCapability": {
      "type": "object",
      "required": [
        "endpointId",
        "urlId",
        "tabLabel",
        "browsePath",
        "keymap",
        "alwaysVisible"
      ],
      "properties": {
        "endpointId": { "type": "string" },
        "urlId": { "type": "string" },
        "tabLabel": { "type": "string" },
        "browsePath": { "type": "string" },
        "keymap": {
          "type": "object",
          "description": "Partial<Record<SlotAction, string[]>>. See §5.4 for the reserved-hotkey collision rule.",
          "additionalProperties": { "type": "array", "items": { "type": "string" } }
        },
        "textFilter": {
          "type": "object",
          "properties": {
            "param": { "type": "string" },
            "label": { "type": "string" },
            "placeholder": { "type": "string" }
          }
        },
        "alwaysVisible": { "type": "boolean" }
      }
    },
    "trainingCohortsCapability": {
      "type": "object",
      "properties": {
        "cohorts": { "type": "array", "items": { "$ref": "#/$defs/cohortSpec" } },
        "suppressDerived": { "type": "array", "items": { "type": "string" } }
      }
    },
    "cohortSpec": {
      "type": "object",
      "required": ["id", "label", "description", "query", "rowKind"],
      "properties": {
        "id": { "type": "string" },
        "label": { "type": "string" },
        "description": { "type": "string" },
        "query": {
          "oneOf": [
            {
              "type": "object",
              "required": ["kind", "path", "params"],
              "properties": {
                "kind": { "const": "endpoint" },
                "path": { "$ref": "#/$defs/templatePath" },
                "params": { "type": "object" }
              }
            },
            {
              "type": "object",
              "required": ["kind", "filters"],
              "properties": {
                "kind": { "const": "predicate" },
                "filters": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "required": ["field", "op"],
                    "properties": {
                      "field": { "$ref": "#/$defs/wireField" },
                      "op": { "enum": ["exists", "eq", "lt", "containsAll"] },
                      "value": {}
                    }
                  }
                },
                "excludeTestHoldout": { "const": true }
              }
            }
          ]
        },
        "rowKind": { "enum": ["slot", "crop"] },
        "reviewTarget": { "enum": ["slotQueue", "all"] }
      }
    },
    "slotEndpoints": {
      "type": "object",
      "properties": {
        "setBox": { "$ref": "#/$defs/templatePath" },
        "clearBox": { "$ref": "#/$defs/templatePath" },
        "patchMeta": { "$ref": "#/$defs/templatePath" },
        "batchStatus": { "$ref": "#/$defs/templatePath" }
      }
    }
  }
}
```

### 4.2 The TypeScript it mirrors

The JSON Schema above is a wire projection of `src/lib/annotations/types.ts`'s
`SlotSpec` (and `src/lib/annotations/cohorts.ts`'s `TrainingCohortsCapability`/
`CohortSpec`). Reading the real file is more reliable than a second copy
drifting here, but the shape at a glance:

```ts
interface SlotSpec {
  key: SlotKey; // string
  bind: { className?: string; classId?: number }; // classId wins; >=1 required
  label: { singular: string; plural: string; title: string };
  capabilities: {
    subBox?: SubBoxCapability;
    text?: TextCapability;
    provenance?: ProvenanceCapability;
    lifecycle?: LifecycleCapability;
    queue?: QueueCapability;
    trainingCohorts?: TrainingCohortsCapability;
  };
  endpoints: SlotEndpoints;
  stats?: { key: string; panelTitle: string; coverageTitle: string };
  extras?: Record<string, unknown>; // profile-private escape hatch, not part of the contract
}
```

Every one of the six capability blocks is optional — `defectCodeSlot`
(`src/lib/annotations/profiles/defectCode.ts`) has no `subBox` at all,
proving the model doesn't assume every slot has geometry.

---

## 5. Serialization caveats

These are the parts a naive JSON-ification of the TypeScript types gets
wrong, and they are the highest-value content in this document.

### 5.1 Functions become allow-listed template strings

`SubBoxCapability.thumbnail.path` and every field on `SlotEndpoints`
(`setBox`, `clearBox`, `patchMeta`, `batchStatus`) are TypeScript
**functions** — e.g.
`(cropId: string) => \`/crops/${encodeURIComponent(cropId)}/plate\``in`licensePlate.ts:272`. A function cannot cross the wire. Over JSON they
must become template strings with a **closed placeholder set**
(`{cropId}`, `{size}`today), mirroring`cohorts.ts`'s `CohortTemplate`comment at`:22-25`: _"no arbitrary expressions, no field access."_

Say plainly why this matters: a server-supplied profile that could ship
executable path logic — even something as innocuous-looking as a
`{cropId}.toUpperCase()` — is a code-injection surface on every client
that resolves it. The resolver must reject any placeholder outside the
allow-list rather than attempt to interpret it.

### 5.2 Paths are relative to `API_PREFIX`, never absolute

`cohorts.ts:36` states the rule for `CohortEndpointQuery.path`: _"Relative
to API_PREFIX, joined by apiFetch — never absolute."_ The same rule must
hold for every template path in `SlotEndpoints` and
`SubBoxCapability.thumbnail.path`. `licensePlate.ts`'s own declarations
already follow it — `endpoints.setBox`, `clearBox`, `patchMeta`,
`batchStatus`, and `capabilities.subBox.thumbnail.path` are all
prefix-relative (`/crops/{id}/plate`, not `/curation/crops/{id}/plate}`) even
though none of them is consumed by application code yet.

Why it matters concretely: a profile that hardcoded `/curation/...` would
break the moment the prefix moved — which is exactly what is happening
in this repo's own Phase B (`docs/design/backend-integration-phase-b-plan-2026-09-20.md`,
T-E2), where `API_PREFIX`'s default flips from `/curation` to `/curation`. A
server-declared profile with a baked-in `/curation` would silently 404 on that
day. Relative-only paths are not a style preference; they are what makes
the prefix flip a one-line config change instead of a coordinated
data migration.

### 5.3 Regex becomes a string + flags, not a serialized `RegExp`

`TextCapability.pattern` is a `RegExp` in TypeScript
(`aircraftTailNumber.ts:57`: `/^[A-Z]\d{1,5}[A-Z]{0,2}$/`). On the wire
it must be `{ source: string, flags: string }` (see the JSON Schema
above), and the client must be free to reject or sandbox the pattern
before compiling it — an unbounded, server-supplied regex is a ReDoS
surface on every keystroke of the text-slot editor.

### 5.4 Palette / ring values are static strings, never built at runtime

`Palette` and `SubBoxRing`'s `confirmed`/`proposed`/`rejected` values are
Tailwind utility-class strings (`types.ts:69-76`), and
`types.ts:69-71`'s comment is explicit: they "may never be built with
template literals at runtime" because Tailwind's JIT compiler statically
scans source for class names it can see — a class name assembled from
server-supplied fragments at runtime is invisible to that scan and
silently renders unstyled. A server-declared profile's ring/palette
values must therefore be complete, literal class strings, exactly as
`licensePlate.ts` and `aircraftTailNumber.ts` already write them, not
composed from parts (e.g. not `{color}-400` for a server-supplied
`color`).

**Implemented 2026-09-20 (tier 2):** the object form is additionally
constrained to a closed allow-list, not merely "any literal string" —
`src/lib/annotations/config/allowLists.ts`'s `RING_CLASS_ALLOWLIST`,
derived from the same `RING_PRESETS` a config may reference by name (§4.1's
amendment). This is stricter than the paragraph above technically
requires (a literal string that happened not to be pre-declared would
still satisfy Tailwind's JIT scan, since it's written literally
somewhere in `allowLists.ts`) — but a config's object-form ring is
validated against a vocabulary the parser controls, not "any string that
looks like a Tailwind class," because there is no way for the parser to
confirm a class name it has never seen is even valid Tailwind syntax,
let alone one Tailwind's build-time scan will find. Reusing the named
presets' own literal values as the allow-list is what keeps the two from
drifting.

### 5.5 `QueueCapability.keymap` can silently steal a hotkey

`keymap`'s letters feed the reserved-hotkey set computed in
`src/lib/classHotkey.ts` (`reservedHotkeyLetters()`, layered on top of
the base `RESERVED_HOTKEY_LETTERS` — `g n d z x u a m /` plus whatever
each active slot's queue keymap adds). A server-supplied `annotation_slots`
entry that declares, say, `confirm: ['g']` would collide with the global
"accept Gemma suggestion" binding the moment that slot's queue becomes
active, and `setClassHotkey` would then refuse to let any class bind
`g` — a confusing failure mode traced back to a server payload, not a
local misconfiguration.

**Collision rule for the contract:** the client computes the reserved
set from all currently-active queues' keymaps (base set ∪ every bound
slot's keymap letters) and a server-declared slot whose keymap collides
with an already-reserved letter must be rejected (skipped, with a
warning — see §2's degrade-not-throw rule) rather than silently
overriding a global action.

**Implemented 2026-09-20 (tier 2), the exact rule:**
`src/lib/annotations/config/allowLists.ts`'s `FORBIDDEN_SLOT_COMBOS` —
`n` (skip) and `z` (undo last) are registered by `/review` unconditionally,
including while a slot tab is active
(`src/routes/review/+page.svelte:1165-1166`, outside the `if (activeSlot)`
branch), so a slot may never claim either. `enter` and `arrowright` are
emitted unconditionally by `buildSlotKeymap`
(`src/lib/review/slotKeymap.ts:65,91`) — re-declaring either would
double-register the same combo. `escape` is edit-mode cancel, likewise
unconditional. `confirm` is a special case, not merely forbidden:
`buildSlotKeymap` hardcodes it to `enter` and never reads
`keymap.confirm` at all, so a config declaring anything other than
exactly `['enter']` for `confirm` is a silent no-op rather than an
override — the parser rejects it outright instead of accepting-and-
ignoring, so the operator sees why their `confirm` binding "didn't do
anything" instead of discovering it by trial and error. Cross-slot: two
resolved slots may bind the SAME letter to the SAME action (e.g. both
declaring `markFalsePositive: ['f']`) — they never run concurrently, so
this is not a real collision — but binding the same letter to two
DIFFERENT actions across two slots rejects the second slot's keymap
entry.

---

## 6. A worked example — `license_plate` as a server would emit it

The following is `src/lib/annotations/profiles/licensePlate.ts` rendered
as the JSON a server implementing this contract would send, annotated
against today's ~30 `plate_*` wire fields already live on a raw crop
row. This is the concrete artifact the backend can diff against
`region_fields.py`'s `RegionFields`.

```json
{
  "key": "license_plate",
  "bind": { "className": "license_plate" },
  "label": { "singular": "plate", "plural": "plates", "title": "Plate" },
  "capabilities": {
    "subBox": {
      "bboxField": "plate_bbox_norm",
      "storedFrame": "source",
      "frameField": "plate_bbox_frame",
      "scoreField": "plate_score",
      "visibleField": "plate_visible",
      "envelope": {
        /* PLATE_SHAPE_ENVELOPE, see src/lib/shapeGate.ts */
      },
      "thumbnail": {
        "path": "/crops/{cropId}/region_thumbnail?size={size}",
        "aspect": "2 / 1",
        "defaultSize": 160
      },
      "ring": {
        "confirmed": "border-green-400 shadow-[0_0_0_1px_rgba(34,197,94,0.45)]",
        "proposed": "border-yellow-400 shadow-[0_0_0_1px_rgba(250,204,21,0.45)]",
        "rejected": "border-zinc-600 shadow-none"
      },
      "editor": { "thumbSize": 512, "viewPadding": 2.5, "nudgeStep": 0.001953125 }
    },
    "text": {
      "valueField": "plate_text",
      "rawField": "plate_text_raw",
      "sourceField": "plate_text_source",
      "confidenceField": "plate_text_confidence",
      "engineVersionField": "plate_text_engine_version",
      "label": "Plate text",
      "placeholder": "ABC123",
      "transform": "uppercase",
      "monospace": true
    },
    "provenance": {
      "detectorField": "plate_detector",
      "detectorVersionField": "plate_detector_version",
      "chainField": "plate_detector_chain",
      "verifierField": "plate_verifier",
      "verifierVersionField": "plate_verifier_version",
      "verifiedAtField": "plate_verified_at",
      "detectedAtField": "plate_detected_at",
      "showChainOnCard": true
    },
    "lifecycle": {
      "statusField": "plate_status",
      "verifiedField": "plate_verified",
      "rejectionReasonField": "plate_rejection_reason",
      "states": [
        {
          "value": "pending_detection",
          "label": "pending detection",
          "humanWritable": false,
          "role": "pending"
        },
        {
          "value": "pending_verification",
          "label": "pending verification",
          "humanWritable": false,
          "role": "pending"
        },
        {
          "value": "detected",
          "label": "detected (plate visible)",
          "humanWritable": true,
          "role": "proposed"
        },
        {
          "value": "verify_rejected",
          "label": "rejected (bad detection)",
          "humanWritable": true,
          "role": "rejected"
        },
        {
          "value": "no_plate_box",
          "label": "no box found",
          "humanWritable": false,
          "role": "absent"
        },
        {
          "value": "no_plate_visible",
          "label": "no plate visible",
          "humanWritable": true,
          "role": "absent"
        },
        {
          "value": "detection_failed",
          "label": "detection failed",
          "humanWritable": false,
          "role": "pending"
        },
        {
          "value": "false_positive",
          "label": "false positive (keep box)",
          "humanWritable": true,
          "role": "falsePositive",
          "dim": true,
          "badge": "false pos"
        }
      ],
      "confirmState": "detected",
      "rejectState": "no_plate_visible",
      "falsePositiveState": "false_positive"
    },
    "queue": {
      "endpointId": "plates",
      "urlId": "plates",
      "tabLabel": "Plates",
      "browsePath": "/plates",
      "keymap": {
        "confirm": ["enter"],
        "reject": ["d"],
        "markFalsePositive": ["f"],
        "editBox": ["e"],
        "back": ["arrowleft", "b"]
      },
      "textFilter": { "param": "text", "label": "Text", "placeholder": "e.g. S14" },
      "alwaysVisible": true
    },
    "trainingCohorts": {
      "suppressDerived": ["blind_spots", "low_conf"],
      "cohorts": [
        {
          "id": "lpr_blind_spots",
          "label": "LPR blind spots",
          "description": "SAM3 found the plate, Gemma confirmed, LPR missed — high-signal training examples",
          "query": {
            "kind": "endpoint",
            "path": "/plates/training_candidates",
            "params": { "mode": "lpr_blind_spots", "class_id": "{classId}" }
          },
          "rowKind": "slot",
          "reviewTarget": "slotQueue"
        }
        /* ...4 more modes, see licensePlate.ts:177-224 verbatim */
      ]
    }
  },
  "endpoints": {
    "setBox": "/crops/{cropId}/plate",
    "clearBox": "/crops/{cropId}/plate",
    "patchMeta": "/crops/{cropId}/plate_meta",
    "batchStatus": "/plates/batch_status"
  },
  "stats": {
    "key": "plates",
    "panelTitle": "Plate detections",
    "coverageTitle": "Plate coverage"
  }
}
```

Note what's deliberately absent: `extras.datasetExport` (the LPR export
panel config, `licensePlate.ts:235-269`). It's a profile-private escape
hatch typed `unknown` on purpose — a genuine non-goal for
generalization, not part of this contract. It stays client-side.

---

## 7. The field-mapping table question (D3)

**Recommendation: the contract doc owns the field-mapping table, and it
should be generated, not hand-maintained.** A hand-written table sitting
between two independent indirections (`region_fields.py` on the backend,
the capability model here) will drift the same way
`HUMAN_PLATE_STATUS_VALUES` already has — four hand copies of one status
whitelist, across two repos, already exist as proof (§1 above). Whoever
owns generation — backend, since `region_fields.py` is presumably closer
to the source of truth, or a small script fed by both repos' committed
`SlotSpec`/`RegionFields` definitions — is an open question for the
backend team; this document does not presume an answer.

---

## 8. The codegen-ownership question (D2)

`export_plate_status_to_ts.py` (lives in the backend repo, per its
generated header: `// AUTO-GENERATED by openprocessor/scripts/codegen/export_plate_status_to_ts.py`)
writes `src/lib/plateStatus.ts` into **this** repo by a hardcoded
absolute path, exports the full 8-value `PlateStatus` enum (not the
4-value human-writable subset the UI actually needs), and **has zero
importers** — `grep -rn "from '\$lib/plateStatus'" src/` (and every
import-form variant) finds nothing. Meanwhile the human-writable subset
is hand-duplicated in at least two more places in this repo
(`docs/genericization-plan-2026-09-13.md` Finding C.4), and the backend
independently defines `HUMAN_PLATE_STATUS_VALUES` twice in
`_common.py`. Four copies, two repos, one generator pointed at none of
them.

**Cropwright's position: retire `export_plate_status_to_ts.py` and
`plateStatus.ts`.** `LifecycleCapability.states` (with each state's
`humanWritable: boolean`) in the slot profile is already the single
source of truth this contract needs — `licensePlateSlot.capabilities.lifecycle.states`
carries exactly the same 8 values with the human-writable flag inline,
today, with a real importer (`readSlot.ts`, `SlotCard.svelte`, and
everything reading `SlotData.lifecycle`). A generated enum with zero
consumers is strictly worse than the capability that already exists and
is used.

**Until the owner rules on this, do not rename `plateStatus.ts`** — it
costs nothing to leave in place, unused, until H3 formally closes this
question.

**Update, 2026-09-21 (Wave 2 C13,
docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §8.3):** the
frontend deleted `src/lib/plateStatus.ts` outright rather than
renaming its `NO_PLATE_*` members to `NO_REGION_*` in step with
OpenProcessor's `no_plate_box`/`no_plate_visible` ->
`no_region_box`/`no_region_visible` rename (merged at `b3f928d`) —
renaming a still-zero-importer generated file just produces a
generated file that is still imported by nothing. The frontend's
single source of truth for these values remains
`licensePlateSlot.capabilities.lifecycle.states`. If the backend still
runs `export_plate_status_to_ts.py`, it can be retired; it has no
frontend consumer to write to.

---

## 9. What Cropwright is explicitly NOT asking for yet (D1b)

**Do not add `annotation_slots` to `GET /classes` until tier 2 (static
deployment-JSON profile loading, `static/annotation-profiles.json`) is
wired here first.** A server field no client can consume is a frozen
wire commitment with zero users and no way to falsify the design before
committing to it.

Sequencing:

1. **Tier 2 first** — wire the fetch of `static/annotation-profiles.json`
   and feed it into `resolveSlotRegistry`'s existing `deployment` param.
   This exercises the identical parse/validate/merge path tier 3 will
   need, at zero cross-repo cost (a static JSON file this repo controls,
   not a live backend contract).
2. **Prove the schema** — use tier 2 to configure a second real slot
   (beyond the `aircraftTailNumber`/`defectCode` proof-of-concept
   profiles, which are not bound to any real class today) and confirm
   the JSON Schema in §4 round-trips through `validateSlot()` without a
   schema change.
3. **Tier 3 becomes near-mechanical** — once tier 2's parse path is
   proven, adding `OpClass.annotation_slots` on the backend and fetching
   it into the same `deployment` param is close to a one-line change on
   this side.

**Steps 1 and 2: done, 2026-09-20.** See
`docs/design/tier2-annotation-profile-config-plan-2026-09-20.md` for the
full design.

- Step 1's evidence: `src/lib/annotations/deploymentProfiles.ts` fetches
  `/annotation-profiles.json` in the root layout's `load()` (bounded at
  2 s, absent-or-malformed degrades silently to tier 1) and feeds the
  parsed result into `resolveSlotRegistry`'s existing `deployment` param
  via `installDeploymentSlots()` (`registeredSlots.ts`) — `registry.ts`
  itself received zero edits, exactly as this document's original
  sequencing intended.
- Step 2's evidence: `static/annotation-profiles.example.json` ships a
  second real slot (`pallet_label`, bound to `wooden_pallet`) that
  exercises every capability in §4's schema, proven end to end
  (registry merge, review tab, keymap, reserved hotkeys, training
  cohorts, `readSlot()`) by
  `src/lib/annotations/config/exampleProfile.test.ts`. Separately,
  `src/lib/annotations/config/roundTrip.test.ts` serializes the existing
  `aircraftTailNumberSlot` to JSON
  (`config/__fixtures__/aircraftTailNumber.profile.json`) and asserts the
  reparsed spec is structurally and behaviorally identical to the
  hand-written TypeScript — **no schema change was needed**; the JSON
  Schema in §4 (as amended by §4.1's `ring` `oneOf` above) round-trips
  losslessly, aside from the pre-existing, already-documented §5
  serialization caveats (functions → template strings, `RegExp` →
  `{source, flags}`).
- **Tier 3 is now legitimately askable** — the parser tier 3 will need
  (`src/lib/annotations/config/parseSlotConfig.ts`) already exists and
  already accepts the bare-array document shape
  `OpClass.annotation_slots` will hand it, unchanged.

---

## 10. Open questions for the backend

Numbered for inline reply:

1. Does `region_fields.py`'s `RegionFields` already have (or plan) a
   JSON-serializable form close to §4's `SlotSpec`, or is this shape
   entirely new to that side?
2. Who generates the shared field-mapping table from §7 — backend,
   frontend, or a third script fed by both repos' committed specs?
3. `export_plate_status_to_ts.py` (§8) — retire it, repoint it at the
   human-writable subset, or leave it as an unused artifact indefinitely?
4. Is `region_fields.py` intentionally staying even after Chunk 8's
   `plate_*` → generic field rename (currently leaning skip per
   `docs/genericization-plan-2026-09-13.md` §4.1.2-3), or would that
   rename fold `RegionFields` into whatever emits `annotation_slots`?
5. For §5.1's template-path allow-list: does the backend want to define
   the placeholder vocabulary (`{cropId}`, `{size}`, …) once, shared by
   both `SlotEndpoints`/`thumbnail.path` here and `CohortTemplate` in
   `cohorts.ts`, or does each capability define its own closed set?

   **Cropwright's own answer, decided 2026-09-20 for tier 2 (this is a
   frontend-only decision until the backend weighs in — happy to
   reconsider):** one shared registry, per-site subsets.
   `src/lib/annotations/config/allowLists.ts` is the single place a
   placeholder name is declared, but it exports two distinct constant
   sets rather than one flat vocabulary — `PATH_PLACEHOLDERS`
   (`cropId`, `size`) for `SlotEndpoints`/`thumbnail.path`, and
   `COHORT_PLACEHOLDERS` (`classId`, `slotKey`) for a training cohort's
   `query.path`/`query.params`, mirroring `cohorts.ts`'s existing
   `CohortTemplate` comment ("no arbitrary expressions, no field
   access"). The two sets are deliberately disjoint today — a path
   template never needs `{classId}` and a cohort query never needs
   `{cropId}` — so `validatePathTemplate()` takes the applicable set as
   a parameter rather than exposing one placeholder namespace a
   capability could accidentally use out of context.

6. Timeline expectation for H2 (backend publishes its new route
   prefix/index names)? This repo's `API_PREFIX` (Phase B,
   `docs/design/backend-integration-phase-b-plan-2026-09-20.md`) is
   already the unilateral half of that handoff and is unblocked on this
   document's answers.
