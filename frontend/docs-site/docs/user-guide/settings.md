---
sidebar_position: 10
title: Settings
---

# Settings (`/p/<project>/settings`)

Per-project curation defaults, one place to pin:

- The clustering method.
- The review-queue default sort.
- The VLM prompt pack.
- The region (detection) profile.
- The VLM endpoint, including **off**.

Each control appears only when the backend's `/methods` response marks that
axis `settable` — a control never appears for an axis the backend hasn't
opened up. Every save goes through an explicit confirm dialog, since this
is a shared setting for the current project, not a personal preference. A
value the backend reports as currently set but doesn't offer (such as
**off**) is shown as a disabled option so you can see it. An axis the
backend does not allow changing is listed read-only under "Not settable on
this backend".

The prompt-pack, region-profile and VLM axes are **activations**: choosing one
here activates it for the project, the same as in their editors. For the
full editors, see [Prompt packs](./prompt-packs.md),
[Region profiles](./region-profiles.md) and [VLM models](./vlm-models.md);
**Settings** links to each when the backend serves it. An external VLM that
hasn't been acknowledged is disabled in the dropdown with a link to
**Settings → Models**.

## Curation scores card

Per-scorer coverage, with confirm-gated "Compute all" / "Compute selected"
actions, a progress poll while a compute job runs, and Cancel. Absent
entirely when the backend hasn't enabled scores.

## Related cards

When the backend serves them, **Settings** also links to the Prompt packs,
Region profiles and Models pages described above.

## Keyboard shortcuts card

Only appears when the backend serves a keymap for the current project. See
[Keyboard shortcuts](./keyboard-shortcuts.md#customizing-shortcuts) for what
it lets you change.

<Screenshot name="settings-1600.png" alt="Cropwright Settings page" caption="Settings — project defaults and curation scores" />
