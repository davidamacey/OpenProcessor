---
sidebar_position: 3
title: Upgrading
---

# Upgrading

Until Cropwright ships as an image with OpenProcessor 0.5.0, upgrade from a
source checkout: pull the new code, compare `.env.example` with your `.env`
when the changelog mentions an environment change, and rebuild:

```bash
git pull
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build
```

Check `CHANGELOG.md` for anything that needs a matching backend version —
Cropwright's vendored API contract (`contracts/openprocessor/`) is checked
against the backend's own OpenAPI snapshot in CI, so a genuine wire-shape
mismatch fails a build rather than shipping a silent blank page. If you
change `PUBLIC_API_PREFIX`, it must still equal the backend's own
`OP_API_PREFIX` after the upgrade.

There is no data migration on the Cropwright side — it holds no state of
its own.
