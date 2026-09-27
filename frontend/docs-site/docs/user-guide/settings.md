---
sidebar_position: 10
title: Settings
---

# Settings (`/p/<project>/settings`)

Per-project curation defaults, one place to pin:

- The clustering method.
- The review-queue default sort.
- The VLM prompt pack.

Each control appears only when the backend's `/methods` response marks that
axis `settable` — a control never appears for an axis the backend hasn't
opened up. Every save goes through an explicit confirm dialog, since this
is a shared setting for the current project, not a personal preference.

A read-only section shows `detection_profile` — the backend's own startup
config, never editable here.

## Curation scores card

Per-scorer coverage, with confirm-gated "Compute all" / "Compute selected"
actions, a progress poll while a compute job runs, and Cancel. Absent
entirely when the backend hasn't enabled scores.

## Keyboard shortcuts card

Only appears when the backend serves a keymap for the current project. See
[Keyboard shortcuts](./keyboard-shortcuts.md#customizing-shortcuts) for what
it lets you change.

<Screenshot name="settings-1600.png" alt="Cropwright Settings page" caption="Settings — project defaults and curation scores" />
