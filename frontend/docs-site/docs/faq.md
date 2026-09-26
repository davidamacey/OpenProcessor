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

Open an issue on [GitHub](https://github.com/davidamacey/OpenProcessor/issues)
using the provided templates.
