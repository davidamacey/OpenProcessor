---
name: Bug report
about: Report a problem with OpenProcessor
title: "[Bug] "
labels: bug
assignees: ''
---

**Describe the bug**
A clear, concise description of what went wrong.

**To reproduce**
Steps to reproduce, including the exact request (curl/Python snippet) if
this is an API bug:

1. ...
2. ...

**Expected behavior**
What you expected to happen instead.

**Environment**
- OpenProcessor version/commit: `curl http://localhost:4603/health | jq .version`
- Deployment profile (`minimal`/`standard`/`full`, `docker-compose.a6000.yml` vs default, etc.):
- GPU(s):
- Docker / Docker Compose version:
- Is this on the `curation` subsystem (experimental) or the core API?

**Logs**
Relevant output from `docker compose logs <service>` (redact anything
sensitive — API keys, internal paths, etc.).

**Additional context**
Anything else that might help.
