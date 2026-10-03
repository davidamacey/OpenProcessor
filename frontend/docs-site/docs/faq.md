---
title: FAQ
---

# FAQ

### Does Cropwright work without OpenProcessor?

No. Cropwright is a pure frontend with no database or mock mode — it has
nothing to show without a running OpenProcessor backend.

### Can I use Cropwright for [some other domain]?

Yes — Cropwright ships with no domain built in. What you label is defined
by the backend's served region profile plus, optionally, an
`annotation-profiles.json` file. See
[Annotation profiles](./configuration/annotation-profiles.md).

### Is there authentication?

No, not in the curation API Cropwright talks to. Read
[Security](./operations/security.md) before deploying anywhere but a
trusted network.

### Why don't I see [some feature/tab/button]?

Almost every optional feature is gated on the backend advertising it at
runtime. See [Backend feature flags](./configuration/backend-feature-flags.md)
— absent is the expected behavior for a backend that hasn't enabled it.

### Can I run two Cropwright instances against two different backends?

Yes — see [Running a second instance](./configuration/second-instance.md).

### Where do I report a bug or request a feature?

Open an issue on [GitHub](https://github.com/davidamacey/OpenProcessor/issues).
Cropwright is part of the OpenProcessor project.

### Can I keep different datasets or domains separate?

Yes, with [projects](./user-guide/projects.md): each has its own classes,
data, settings and keyboard shortcuts, on one deployment. You can
[combine](./user-guide/combine-projects.md) projects into a new one later,
and share trained models between them.

### Can I import a dataset that is already labeled?

Yes, if the backend supports it: see
[Dataset import and Reprocess](./user-guide/dataset-import.md). Labels a
human has set are locked against machine overwrite.

### Does the VLM see my images? Where do API keys go?

Only the VLM endpoint you activate sees crops. An endpoint outside your
deployment is flagged and needs an explicit acknowledgement. Keys never pass
through Cropwright: the backend host holds the secret and the app shows only
its reference name. See [VLM models](./user-guide/vlm-models.md).

### Can I change the keyboard shortcuts?

Yes, per project, in Settings. See
[Keyboard shortcuts](./user-guide/keyboard-shortcuts.md#customizing-shortcuts).
