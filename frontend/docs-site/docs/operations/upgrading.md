---
sidebar_position: 3
title: Upgrading
---

# Upgrading

Fixes land on `main` and ship in the next tagged image. To upgrade, set
`CROPWRIGHT_TAG` in `.env` to the new version (or leave it at `latest`),
then:

```bash
docker compose pull && docker compose up -d
```

Re-download `docker-compose.yml` and compare `.env.example` when the
release notes mention a compose or environment change. If you build from
source, `git pull` and rerun
`docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build`.

Check `CHANGELOG.md` for anything that needs a matching backend version —
Cropwright's vendored API contract (`contracts/openprocessor/`) is checked
against the backend's own OpenAPI snapshot in CI, so a genuine wire-shape
mismatch fails a build rather than shipping a silent blank page. If you
change `PUBLIC_API_PREFIX`, it must still equal the backend's own
`OP_API_PREFIX` after the upgrade.

There is no data migration on the Cropwright side — it holds no state of
its own.
