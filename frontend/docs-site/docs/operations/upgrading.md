---
sidebar_position: 3
title: Upgrading
---

# Upgrading

Cropwright does not yet maintain release branches — fixes land on `main`
and ship in the next tagged image. To upgrade:

```bash
git pull
docker compose up -d --build
```

Check `CHANGELOG.md` for anything that needs a matching backend version —
Cropwright's vendored API contract (`contracts/openprocessor/`) is checked
against the backend's own OpenAPI snapshot in CI, so a genuine wire-shape
mismatch fails a build rather than shipping a silent blank page. If you
change `PUBLIC_API_PREFIX`, it must still equal the backend's own
`OP_API_PREFIX` after the upgrade.

There is no data migration on the Cropwright side — it holds no state of
its own.
