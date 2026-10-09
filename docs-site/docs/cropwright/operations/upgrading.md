---
sidebar_position: 3
title: Upgrading
---

# Upgrading

Cropwright upgrades together with OpenProcessor: pull the new code, compare `env.template`
with your `.env` when the changelog mentions an environment change, and rebuild:

```bash
git pull
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile cropwright up -d --build cropwright
```

Check `CHANGELOG.md` for anything that needs a matching backend version —
the frontend's tests read the backend's generated API contract (`contracts/`) in the same
repository, so a genuine wire-shape mismatch fails a build rather than shipping a silent blank page. If you
change `PUBLIC_API_PREFIX`, it must still equal the backend's own
`OP_API_PREFIX` after the upgrade.

There is no data migration on the Cropwright side — it holds no state of
its own.
