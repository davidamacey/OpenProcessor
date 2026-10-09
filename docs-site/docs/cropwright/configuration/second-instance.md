---
sidebar_position: 4
title: Running a second instance
---

# Running a second, independent instance

Give it its own compose project name, container name and host port so it
never collides with a running instance:

```bash
CROPWRIGHT_PORT=5190 docker compose -p second --profile cropwright up -d
```

This is the same image and compose file, pointed at a different backend
(or a different docker network) with a different host port — useful for
running two datasets or deployments side by side on one host.
