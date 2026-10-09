---
sidebar_position: 3
title: Runtime settings
---

# Runtime settings

Cropwright keeps no configuration of its own beyond the
[environment variables](./environment-variables.md) that wire the container
to a backend. What you actually tune while using it lives in one of three
places.

## 1. Per-project settings, stored by the backend

Each project has its own settings, which the backend keeps and Cropwright
displays and edits. They are scoped to the project, so two projects on the
same deployment can differ.

| Setting | Where to edit it | Notes |
| --- | --- | --- |
| Clustering method, review-queue default sort | [Settings](../user-guide/settings.md) | Only the axes the backend marks settable get a control. |
| VLM prompt pack | [Prompt packs](../user-guide/prompt-packs.md) | Edited and tested as revisions; one revision is active at a time. |
| Region profile | [Region profiles](../user-guide/region-profiles.md) | Revisions, an impact preview and a reload-to-apply notice. |
| Active VLM endpoint | [VLM models](../user-guide/vlm-models.md) | The endpoint registry is shared by every project; the choice is per project. |
| Keyboard shortcuts | [Keyboard shortcuts](../user-guide/keyboard-shortcuts.md#customizing-shortcuts) | Per project, with per-page overrides. |
| Class list and class hotkeys | [Classes](../user-guide/classes.md) | |

Prompt packs, region profiles and VLM endpoints share one model. They are
**revisioned**: saving creates a new revision and never edits history. A
revision is **activated** explicitly (activation records which revision is
pinned and is confirmation-gated), and the previous one can be rolled back
to. The backend reports whether each worker has picked up the active
revision. Saves and activations carry the revision you started from, so two
people editing at once get a conflict message instead of silently overwriting
each other. The [Copy settings](../user-guide/projects.md) action on
`/projects` copies chosen groups between projects.

## 2. Deployment-wide settings, stored by the backend

The VLM endpoint **registry** (which endpoints exist, their schema and the
local model catalog) is shared by every project. A VLM API key is a secret
held on the backend host and referenced by name; Cropwright never sees it.
Backend startup configuration (for example, which feature flags are on, see
[Backend feature flags](./backend-feature-flags.md)) is changed on the
backend, not here.

## 3. Cropwright deployment files

- Environment variables, read at container start: see
  [Environment variables](./environment-variables.md).
- `annotation-profiles.json` placed next to the built app to add an
  annotation slot: see [Annotation profiles](./annotation-profiles.md).

Nothing Cropwright needs is stored in the browser: the active project is the
URL, and everything else is read from the backend.
