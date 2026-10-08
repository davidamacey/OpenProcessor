---
sidebar_position: 1
title: Projects
---

# Projects (`/projects`, `/p/<project>/...`)

A **project** is its own isolated dataset and its own settings on the same
Cropwright/OpenProcessor deployment — its own classes, crops, clusters,
review queues, exports, training runs and curation defaults. Nothing about
one project (classes, keyboard shortcuts, curation settings, cached data)
leaks into another.

## URLs

Every page except `/projects` itself lives under a project's slug:
`/p/<project>/dashboard`, `/p/<project>/review`, and so on. The active
project is the URL — there is nothing else to reconstruct it from, so
reloading, sharing a link, or opening a bookmark always lands you back in
the same project.

An old bare URL with no `/p/<project>` prefix (including plain `/`)
redirects to the deployment's default project on the matching page — `/`
goes to `/p/<default-project>/dashboard`, `/review?tab=all` goes to
`/p/<default-project>/review?tab=all`, and so on. There is no unscoped
"global" version of any of these pages anymore.

## Switching projects

The top bar has a project switcher listing every selectable project (plus
whichever one is currently active), each with its display name and, for
anything other than an ordinary active project, a status label. Picking one
takes you to the same page/section in the new project, dropping anything
that doesn't carry across projects (a cluster id, a deep-linked crop id, a
class filter). "Manage projects…" in the switcher opens `/projects`.

A project you're not allowed to write to (an archived one) shows a
read-only banner under the top bar for as long as it's active.

## The `/projects` page

This one page is **not** scoped to any project — it manages the list
itself. It shows every project (with a "show archived" toggle) and, when
the deployment tracks it, a capacity banner naming how many search shards
are in use against the deployment's recommended and hard limits, so you
can see a capacity warning before it turns into a blocked create.

Available actions, each shown only when the project's own served status
allows it:

- **New project** — create a project with its own data and settings, from
  empty. Disabled only when the deployment reports it's at its shard
  capacity.
- **Open** / **Edit** — open a project, or rename it and change its
  description. An edit is saved against the revision you loaded; if someone
  else changed the project in the meantime you are offered a reload that
  keeps what you typed.
- **Archive** / **Unarchive** — archiving a project makes it read-only
  (still viewable, still selectable, nothing writable); unarchiving
  reverses it. Each button follows the backend's own "can archive" /
  "can unarchive" answer for that project.
- **Copy settings** — copy chosen setting groups from another project into
  this one, without copying any data. The groups offered are whatever the
  deployment says can be cloned (they can include the region profile,
  prompt pack and VLM activation). No source project is preselected, and
  Copy stays disabled until you pick one. If a copied keymap overlaps a
  class hotkey in the destination, you are told which keys clashed.
- **Pause pipeline** / **Resume pipeline** — see below.
- **Delete** — guarded. Before anything is deleted, the backend runs a dry
  run and Cropwright shows its report along with any blocking reasons
  (for example, a project that still has running jobs). While anything
  blocks the delete, there's no confirmation field to fill in at all —
  you have to resolve the blocker first. Once nothing blocks it, deleting
  requires typing the project's own slug to confirm.
- **Combine projects…** — merges several projects into a new one; see
  [Combine projects](./combine-projects.md). Shown only when the backend
  supports it. A project that was created by a combine links back to its
  job, and its delete button reads **Undo combine**.
- A row whose status is still transient (building or deleting) refreshes
  itself every couple of seconds until it settles, so a finished delete
  disappears without a reload.

Every one of these actions is gated purely on what the backend says about
that project (selectable, writable, archivable, deletable) — there's no
client-side guess about whether an action is safe to offer. A refused
action shows the backend's own message, and any warnings the backend
attaches to a successful action appear as toasts.

## Pausing a project's pipeline

A project's backend pipeline can be paused, so the backend stops starting
automatic work for that project while you do something that shouldn't race
with it. What exactly stops is the backend's decision. A paused
project shows a **paused** chip on `/projects` and in the project switcher.
On a writable project, **Pause pipeline** and **Resume pipeline** each ask
for confirmation first. Pausing affects only that project. If something
else on the deployment paused it (for example a training claim on the GPU),
the switcher's tooltip shows who paused it and why, as served.

## Sharing models between projects

A promoted model belongs to one project. Its owner can share it, and other
projects then see it on their [Models](./models.md) page, with classes
matched to the project's own by name. Sharing, un-sharing and the
force-unshare confirmation are described on that page.

<Screenshot name="projects-1600.png" alt="Cropwright projects page" caption="Projects — list, capacity banner, and lifecycle actions" />

<Screenshot name="projects-delete-dry-run-1600.png" alt="Guarded project delete showing the dry-run report and a blocking reason" caption="Guarded delete — the dry-run report and any blocking reason come before the confirmation field" />

<Screenshot name="projects-copy-settings-1600.png" alt="Copy settings dialog listing the setting groups that can be cloned" caption="Copy settings — choose a source project and the setting groups to copy" />
