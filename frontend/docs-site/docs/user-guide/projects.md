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

- **Create** — new project, its own data and settings from empty. Disabled
  only when the deployment reports it's at its shard capacity.
- **Open** / **Edit** — rename or otherwise edit a project you can still
  select.
- **Archive** / **Unarchive** — archiving a project makes it read-only
  (still viewable, still selectable, nothing writable); unarchiving
  reverses it.
- **Copy settings** — clone specific setting groups (whichever the
  deployment allows cloning) from one project onto another, without
  copying the underlying data.
- **Delete** — guarded. Before anything is deleted, the backend runs a dry
  run and Cropwright shows its report along with any blocking reasons
  (for example, a project that still has running jobs). While anything
  blocks the delete, there's no confirmation field to fill in at all —
  you have to resolve the blocker first. Once nothing blocks it, deleting
  requires typing the project's own slug to confirm.

Every one of these actions is gated purely on what the backend says about
that project (selectable, writable, deletable) — there's no client-side
guess about whether an action is safe to offer. A refused action shows the
backend's own message.

<Screenshot name="projects-1600.png" alt="Cropwright projects page" caption="Projects — list, capacity banner, and lifecycle actions" />
