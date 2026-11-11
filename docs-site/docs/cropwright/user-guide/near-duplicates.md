---
sidebar_position: 6
title: Near-duplicates
---

# Near-duplicates

**Near-duplicates** shows which images (and which crops) are almost the same
picture, so a test split does not leak and an export does not repeat itself.
The grouping, the cuts and every count come from the backend; this page only
shows them and lets you decide pairs. The page reads
`/curation/projects/{project}/near_duplicates/status`, `/runs`, `/groups` and
`/decisions`, and the thresholds from `/near_dup_profiles`.

Open it from **Near-duplicates** in the top bar.

## Status and runs

Each level has a card:

- **Image level**: whole frames, compared by their image vectors.
- **Item level**: crops inside the image-level leak groups.

A card shows the current run (when it ran, how many groups it made) and, when
the groups are out of date, why: **has items not yet grouped**, **needs a full
rebuild** (the thresholds changed), or **has never been built**.

- **Group new items** appears when only new items are pending. It runs the
  incremental pass.
- **Full rebuild** regroups the whole level.

Both start a job and follow it; the card shows the job's status. If the server
refuses (a run is already active, or the item level needs a current image-level
run first), its message appears under the title and nothing starts.

**Run history** lists the last runs of both levels with their counts, the
profile revision they used, and how many items the test holdout absorbed.

## Groups

Choose the level (**Images** or **Items**) and the kind:

- **Duplicates**: members at or above the duplicate cut. Keep the
  representative and drop the rest, for example in an export that collapses
  near-duplicates.
- **Leak groups**: members at or above the leak cut. They must stay on one
  train/test side, so freezing the test holdout moves a whole group together.

Groups are listed largest first with a picture of the representative, the
lowest cosine between linked members and the lowest cosine to the
representative. **In test holdout** marks a group with a held-out member.
**Load more** pages on.

An empty list means nothing is at or above the cut, or the level has no current
run yet; the status card says which.

## Deciding a pair

Click a group to see every member with its cosine to the representative.

- **Not a duplicate** pins the member apart from the representative.
- **Keep together** pins it to the representative.

A decision is kept for every later run and applies at the next run of that
level (the card then shows items pending). **Undo** appears on a member you
decided in this dialog and removes the decision. The API serves no list of
older decisions, so they cannot be undone from here.

## Advanced: thresholds

The **Advanced: thresholds** panel edits the project's near-duplicate profile:
the duplicate cosine, the leak cosine and the largest group per level, the
neighbour count for images, and "only group crops of the same detector class"
for items. The seeded defaults (0.98 cosine) are shown beside each field, and
**Use the seeded defaults** puts them in the form.

**Save and activate** writes the next revision of the active profile and makes
it active. Both levels then need a **Full rebuild**; nothing regroups on its
own. The server checks the values (the leak cut cannot exceed the duplicate
cut) and its reasons are shown as sent. Few projects need to change them.
