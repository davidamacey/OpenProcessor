---
sidebar_position: 6
title: Releasing
---

# Releasing

Fixes and features land on `main`; a release is a tagged, multi-arch image
built from a known commit, published to Docker Hub as
`davidamacey/cropwright`, plus a GitHub release whose notes are that
version's `CHANGELOG.md` section.

Releases are cut locally with `./scripts/release.sh`, a small orchestrator
over the stages in `scripts/release/`:

| Stage       | What it does                                                                                             |
| ----------- | -------------------------------------------------------------------------------------------------------- |
| `preflight` | Clean worktree, version matches `package.json`, a `CHANGELOG.md` section exists, the tag is free, the buildx builder has amd64 and arm64, Docker Hub login present |
| `verify`    | `npm ci`, lint, type check, unit tests, build, the stubbed end-to-end suite                              |
| `build`     | Builds `linux/amd64` and `linux/arm64` images (arm64 natively on a remote builder node, no emulation)   |
| `scan`      | Trivy on both images; fails on a CRITICAL/HIGH finding that has a fix                                    |
| `smoke`     | Starts each image and checks health, security headers, the non-root user, assets and SPA routing        |
| `tag`       | Creates and pushes the annotated `vX.Y.Z` tag                                                            |
| `publish`   | Pushes one multi-arch manifest as `X.Y.Z`, `X.Y` and `latest`, with an SBOM attestation                 |
| `finish`    | Creates the GitHub release from the `CHANGELOG.md` section                                              |

Each stage can be run on its own and a failed release resumes where it
stopped (a local ledger under `.release/<version>/`). `tag`, `publish` and
`finish` are the only stages that reach outside the machine, and each asks
for confirmation (or `--yes`).

```bash
./scripts/release.sh preflight 0.2.0
./scripts/release.sh run 0.2.0                  # all stages, confirming external ones
./scripts/release.sh run 0.2.0 --from build     # resume
./scripts/release.sh status 0.2.0
```

The build needs a multi-arch buildx builder with an arm64 node and a docker
context for that node; point the stages at them with `CROPWRIGHT_BUILDER`
and `CROPWRIGHT_REMOTE_ARM64_CONTEXT`.

See [Upgrading](../operations/upgrading.md) for what a consumer should
check when moving to a new version.
