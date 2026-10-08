---
sidebar_position: 2
title: Combine projects
---

# Combine projects (`/projects/combine`)

Merge several existing projects into a **new** one. The sources are left
untouched; the new project gets their images, items and a class list you
define during the combine. Like `/projects`, these pages are not scoped to a
project, since the target doesn't exist yet. The **Combine projects…**
button on `/projects` appears only when the backend supports combining;
otherwise the page says so and nothing else is requested.

## The wizard

1. **Sources and target.** Pick the source projects (only active, selectable
   projects are offered) and put them in priority order with the up/down
   buttons. The first source wins when two sources disagree about the same
   image, and donates the image when a duplicate is found. Each source can
   keep **all** its labels or **validated only**. Give the target a slug, a
   display name and a description; the backend's slug rules are shown as a
   hint and its preview errors are the real gate.
2. **Class mapping.** One table per source, listing that source's classes
   and how many items each has. Each class is mapped by **name**: create a
   class in the new project, map to a class another row creates, skip it, or
   (where offered) treat it as a region. Rows you haven't touched are filled
   in from the backend's suggestions after each preview; a row you changed is
   never overwritten, and **Reset to suggestions** re-copies them.
3. **Options.** Duplicate handling, how close boxes must be to count as the
   same, how the frozen test holdout carries over, and which source the new
   project copies its settings from. Anything you don't touch is left to the
   backend's default.
4. **Preview.** After every change (a short moment later) the backend
   re-computes a preview: errors and warnings, a per-source table, what the
   target will contain, duplicate and conflict counts, and how many bytes
   will be linked versus copied. **Start** stays disabled while a preview is
   in flight, out of date, or reporting an error.
5. **Start.** A confirmation lists the target's counts and the request is
   tied to the exact preview you saw. If anything changed since, the backend
   refuses and Cropwright re-previews rather than starting a different
   combine than the one you confirmed.

## The job view (`/projects/combine/<job>`)

Shows the job's status, phase, progress, the sources and target, and the
backend's report. While it runs the page refreshes every couple of seconds
(and immediately when the backend announces progress).

- **Cancel** while it is queued or running; **Resume** when it was
  interrupted or cancelled. Both ask for confirmation.
- When it completes: **Open project**, **Review flagged conflicts** (the new
  project's All review queue filtered to items where sources disagreed), and
  one confirmed button per **next step** the backend suggests, such as
  reclustering the new project. The backend's answer to a next step stays on
  the page under "Last next step".
- When it fails: the error and report, plus **Undo combine**.

You can find a job again later from the combined project's row on
`/projects`.

## Undo

Undoing a combine means deleting the project it created. **Undo combine**
(on the job view, or in place of Delete on the project's row) opens the same
guarded delete as any project: a dry-run report first, then typing the
project's slug. The source projects are not modified.

## Provenance in item details

In the **Details** panel, an item from a combined project shows where it came
from: origin project, item, image and split. An amber "Conflict between
sources" chip lists the projects that disagreed, and merged origins are
listed when several sources contributed the same item.

<Screenshot name="combine-wizard-1600.png" alt="Combine projects wizard with source ordering, class mapping and a preview" caption="Combine wizard — ordered sources, class mapping and the served preview" />

<Screenshot name="combine-job-1600.png" alt="Combine job view with progress and next steps" caption="Combine job — progress, conflicts link and next steps" />
