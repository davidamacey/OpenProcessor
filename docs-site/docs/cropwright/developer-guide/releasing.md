---
sidebar_position: 6
title: Releasing
---

# Releasing

Fixes and features land on `main`; a release is a tagged, multi-arch image
built from a known commit, published to Docker Hub as
`davidamacey/cropwright`, plus a GitHub release whose notes are that
version's `CHANGELOG.md` section.

Cropwright ships with the OpenProcessor release. `scripts/release/build_and_publish.sh`
at the repository root builds the multi-arch image and publishes it, and
`docs/releases/RELEASE_CHECKLIST.md` lists the steps in order. The frontend
`CHANGELOG.md` carries the Cropwright notes for each version.

See [Upgrading](../operations/upgrading.md) for what a consumer should
check when moving to a new version.
