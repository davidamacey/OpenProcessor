---
sidebar_position: 6
title: Open-vocabulary sets
---

# Open-vocabulary sets (`/p/<project>/settings/open-vocab`)

An **open-vocabulary set** describes things to find in words, and has the
backend's SAM 3 segmenter find them in your images. A set is a named,
revisioned list of **targets**. Each target has a `prompt` (what to look for)
and, optionally, the class its hits are stored as. A target with no class is
in *discovery mode*: its hits are stored as unlabeled proposals named by the
prompt. One set is active per project.

The pages appear only when the backend serves open-vocabulary sets;
otherwise **Settings** has no Open-vocabulary card. Whether the segmenter is
configured and reachable is shown on both pages. A set can be prepared
before the segmenter is ready, so the editor is never disabled by it.

## The list

`/settings/open-vocab` lists the project's sets and the read-only templates
the backend ships.

- **Create** starts a new set; the backend fills in the defaults.
- **Clone** copies a set or template into a new editable set, or the set of
  the same name from another project ("Copy from another project").
- **Delete** removes a set (confirmed).
- The **active set** panel offers **Rollback** and **Turn off**, both
  confirmed.

## The editor

`/settings/open-vocab/<name>` has one row per target, with the common fields
inline and the advanced ones in a per-row expander. The class cell suggests
the project's classes but accepts any text. Set-level and gating fields are
drawn from the backend's own schema. Nothing is validated in the browser: a
moment after you stop typing, the draft goes to the backend, and each problem
it finds appears next to the cell it names.

- **Save** writes a new revision. If the set changed since you opened it you
  are offered **Reload** or **Keep my edits**.
- **Revisions.** Open an older revision read-only, **Restore as new
  revision**, or **Activate** a specific one.
- The backend's own limit on enabled targets is shown as a fact; the
  validator, not the page, refuses a set with too many.
- **Check the draft for activation** runs the activation checks without
  activating.

<Screenshot name="cropwright/open-vocab-editor-1600.png" alt="Open-vocabulary set editor with the segmenter notice and a list of targets" caption="Open-vocabulary editor — the segmenter notice, set fields and one row per target" />

## Activating

Activating pins the revision you are looking at and asks for confirmation.
**Activate anyway** appears only when the backend's report says the check can
be overridden.

## Test on one image

The **Test** panel runs one target (any row of your draft, or the saved
revision) on one image, without storing anything. The image is a stored
crop's source image or a file you upload. You can tick the optional VLM
pre-check. The result shows the gate decision and each hit with its score,
whether it was selected, and why a dropped hit was dropped. Hits are drawn
over the image, with dropped ones dimmed. A segmenter error is shown as an
error, never as "no hits".

<Screenshot name="cropwright/open-vocab-test-1600.png" alt="Open-vocabulary test panel with a hits table of scores, selected and dropped reasons, and the hits drawn over the image" caption="Test a target — every hit with its score and why dropped hits were dropped, drawn over the image" />

## Run on existing images

While a set is active, the list page offers **re-run** buttons that apply it
to images already in the project: all images, or only those whose
open-vocabulary status is `pending`, `skipped_gate` or `failed`. Each opens
the same dialog as [Reprocess](./dataset-import.md): a dry run first, then
the real run behind a confirmation.

## Where the hits show up

An item found by a set carries its prompt, the set and revision, and a
"Show matching items" link on its detail panel. That link opens
[Matching items](./item-filter.md) filtered to that set and prompt.

The region stage's own **pause**, **resume** and **re-run gate-skipped**
controls are on the [Ingest](./ingest.md) page.
