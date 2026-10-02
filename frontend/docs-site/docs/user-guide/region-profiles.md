---
sidebar_position: 6
title: Region-profile editor
---

# Region profiles (`/p/<project>/settings/region-profiles`)

A **region profile** defines the region type this project works with: what to
detect inside an item crop (a tag on a part, a plate on a vehicle, a defect
zone), which models find and verify it, and whether text is read from it. The
profile that is **active** drives every region screen (the region review tab,
the region gallery, the box editor), so changing it is a deliberate act with
an impact preview. The pages exist only when the backend serves region
profiles.

## The list

`/settings/region-profiles` lists the project's profiles and the shipped
templates, with the **active** profile at the top:

- **Rollback** and **Turn off** (both confirmed; off means region detection
  is off for the project).
- **Show impact** reads what activating or keeping the profile means for
  existing data.
- **Clone** copies a profile or template, and **Delete** removes one
  (confirmed).
- A collapsed, read-only **Models and sources** panel lists the detectors,
  segmenters, readers and other choices the backend knows about.

A profile that has no name and no activation shows as "None: no region
profile configured"; an explicit off shows "off: region detection is off".

## The editor

`/settings/region-profiles/<name>` builds the form from the backend's schema:
fields are grouped, typed (text, number, toggle, choice, list, pair, colour)
and checked by the backend as you edit. Advanced fields are behind a toggle;
a field that doesn't currently apply (given the other values) is dimmed,
never disabled. Model fields offer the backend's choices, including an
"empty" choice where that is allowed, and **Include other projects' shared
models** adds detectors other projects have shared. The segmenter section
shows the backend's served cap and floor beside the related fields.

- **Check the draft for activation** runs the stricter activation checks and
  shows them separately from ordinary validation.
- **Save**, **revisions**, **restore** and the conflict handling work as in
  the [prompt-pack editor](./prompt-packs.md#the-editor).

## Activation

Activating pins the revision on screen and asks for confirmation; **Activate
anyway** appears only if the backend's report allows forcing. A successful
activation shows the backend's **impact** and validation reports. If the
backend suggests a reprocess to bring existing items in line, **Re-run** runs
exactly the suggested request: a dry run first, then a confirmed apply (see
[Reprocess](./dataset-import.md#reprocess)).

Region screens don't hot-swap. After an activation, rollback or turn-off,
Cropwright re-reads the backend's health and shows the usual **reload to
apply** notice; reload the page to see the new region screens.

## Test on a crop

A **Test on a crop** panel runs the profile on one crop without saving
anything. Choose the crop, whether to use your unsaved draft or a saved
revision (read-only profiles and old revisions offer the saved version only),
an optional segmenter prompt, and whether to **Verify with the VLM** (with
the same optional endpoint picker as the pack test). The result shows each
stage of the pipeline with its status, reason and time; a table of candidates
with score, whether it was selected, why it was dropped, and which detector
made it (dropped ones greyed out); the candidates drawn over the source image
(boxes and mask outlines, dropped ones dimmed) and in the crop's own frame;
and a preview of the item under either **Selection (not verified)** or **VLM
verdicts**. A crop the profile wouldn't process reads "not eligible".

<Screenshot name="region-profile-editor-1600.png" alt="Region-profile editor with grouped, typed fields and model pickers" caption="Region-profile editor — schema-driven form with model pickers" />

<Screenshot name="region-profile-test-1600.png" alt="Region-profile test panel with candidate boxes drawn over the source image" caption="Test on a crop — candidates over the source image and in the crop's frame" />
