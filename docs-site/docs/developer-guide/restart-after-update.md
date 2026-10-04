---
sidebar_position: 2
title: Restart after pulling or updating
---

# Restart after pulling or updating

The API and worker containers mount the Python source, but the running
processes import it once at startup. After you pull, merge or switch
branches, restart them:

```bash
make dev-restart
```

`make dev-restart` restarts the code-mounting containers (the API and the
workers). It does not recreate them or re-read `.env`; for a changed `.env`
or a changed Dockerfile or `requirements.txt`, recreate or rebuild the
affected service instead.

Skipping the restart can look like a bug. A process that already imported
an older module and then lazily imports a newer one from the updated files
can fail with an `ImportError` or a transient `500` (for example when
activating a region profile) until the API restarts.
