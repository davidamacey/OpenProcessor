---
sidebar_position: 5
title: Classes
---

# Class management (`/p/<project>/classes`)

Add, rename, merge, deprecate and restore classes, and bind a per-class
hotkey letter (reserved action keys are rejected server-side and
client-side).

- A merge dry-run shows how many human validations carry over.
- Deprecating a class that's still referenced offers the merge dialog
  instead, preselected to the class as the merge source.
- A deprecated class that was merged shows what it was merged into, rather
  than an always-disabled Restore button.

## Proposals

Lists VLM-proposed terms the class registry doesn't have yet: create a new
class from one, or map it onto an existing class — either resolves every
pending item that proposed the term (a dry-run shows the real count first),
undoable with `Z`. Flagged terms (an existing class, a generic parent, or a
non-object) get no create action, only a served reason and, for an
existing-class match, a one-click map.

<Screenshot name="classes-1600.png" alt="Cropwright Class management" caption="Classes — registry, hotkeys and proposals" />
