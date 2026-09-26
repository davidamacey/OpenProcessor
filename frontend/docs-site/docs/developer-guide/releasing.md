---
sidebar_position: 6
title: Releasing
---

# Releasing

Cropwright does not yet maintain release branches — fixes and features land
on `main`, and a release is a tagged image built from a known commit.

The maintainers' internal export/publish tooling that promotes a private
working tree into the public `davidamacey/OpenProcessor` history is
intentionally not described here — it's not part of the day-to-day
contributor loop, and the specifics of a private-to-public export process
aren't useful to a contributor working directly against the public repo.
For anyone working from a clone of the public repo, "releasing" just means:
tag a commit, build the image from it, publish.

See [Upgrading](../operations/upgrading.md) for what a consumer should
check when moving to a new tag.
