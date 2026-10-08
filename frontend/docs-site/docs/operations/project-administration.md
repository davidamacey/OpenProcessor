---
sidebar_position: 5
title: Project administration
---

# Project administration

Operator notes for running several projects on one deployment. The
step-by-step screens are in the [Projects](../user-guide/projects.md) user
guide; this page is about what to check and when.

## Capacity

`/projects` shows a capacity banner when the backend reports search-shard
usage against its recommended and hard limits. **New project** is disabled
only when the backend says creation is blocked. Treat the warning level as
the cue to archive or delete projects you no longer need.

## Pausing the pipeline

Pause a project's pipeline before maintenance that must not race with
background work, such as a backend restart or a bulk edit, and resume afterwards. Pausing is per project and takes
effect on the backend, not in the browser, so it holds when nobody has the
page open. The paused state and who paused it are visible in the project
switcher. It needs the project to be writable; an archived project can't be
paused or resumed.

## Archive before delete

Archiving makes a project read-only but keeps all its data and keeps it
selectable, so it is the safe way to retire one. **Delete** is destructive:
the backend runs a dry run, the report and any blocking reasons (such as a
running job) are shown, and you must type the project's slug. Combined
projects show **Undo combine** instead, which is the same guarded delete.
Deleting a project does not touch other projects or the projects it was
combined from.

## Shared models

A model one project shared is visible to others. Before un-sharing, read the
in-use list in the dialog. **Unshare anyway** can break another project's
active region profile, so contact its owners first.

## Containers

Cropwright itself is one stateless container (see
[Deployment](./deployment.md)); restarting it loses nothing. The data lives
in the backend, so back up and restart the backend per its own
documentation. After the backend restarts, the status chip returns to green
once its health endpoint answers, and open pages re-read what they show.
If you run a second Cropwright against another backend, see
[Running a second instance](../configuration/second-instance.md).

## Upgrading the backend

Match the backend release to the Cropwright version: Cropwright's wire
contract is vendored from a specific backend release. See
[Upgrading](./upgrading.md). Features that a backend predates simply don't
appear, so a missing page after a backend change is usually a version
mismatch, not a bug.
