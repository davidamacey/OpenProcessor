---
sidebar_position: 4
title: Assist (VLM)
---

# Assist (VLM)

**Assist…** asks the VLM about a part of your data and shows you what it
suggests. It is on a cluster page (the whole cluster or the open sub-cluster,
or the crops you selected), on one crop's details panel, on one box in
[Review](./review.md), and on [Matching items](./item-filter.md) (everything
a filter matches). The concept, the sample sizes and the API are in
[Selective VLM and segmenter assist](../../guides/selective-assist.mdx).

The VLM only suggests. A label a human or an import set, a validated label
and a test-holdout item are never overwritten; what the VLM said about them
is shown next to the label as "VLM said". Everything on the page, the counts
and the refusals included, is what the backend answered.

## The dialog

<Screenshot name="cropwright/assist-dialog-1600.png" alt="The Assist dialog on a cluster page: the sample choice, the actions, the VLM and the server's dry run" caption="The Assist dialog: nothing is written until you press Start run" />

1. **What the VLM sees.**
   - **Selected**: the crops (or boxes) you picked. Shown only when you
     selected some.
   - **All members**: every member of the cluster, sub-cluster or filter the
     VLM may see.
   - **Random representative sample**: a seeded sample that grows with the
     size of the scope and is spread over the sub-clusters. Leave the size
     empty for the server's default, or type one; the seed makes it
     repeatable.
2. **What to do.** **Classify**, **Attributes**, **Verify boxes** and
   **Segment**; only the actions that fit the target are offered (a box is
   not classified, a cluster is not verified). With **Segment**, choose how
   the segmenter's new boxes are verified: by the project's active VLM, or
   accepted without one. Ticking **Verify boxes** together with **Segment**
   judges the new boxes with the endpoint you chose for this run.
3. **Which VLM.** The project's active endpoint, or any registered endpoint
   for this run only, and a prompt pack. An endpoint that sends crops outside
   the deployment shows its warning and needs your acknowledgement. See
   [VLM models](./vlm-models.md#choosing-a-vlm-per-run).

Every change asks the server for a **dry run**, shown at the bottom: how many
of the scope's members go to the VLM, the estimated number of VLM calls, the
sub-cluster spread, what is left out (holdout, ignored, locked) and the
endpoint and pack the run will pin. If the server refuses (no VLM is
configured, another run is live, the scope is too large, the pack declares no
attributes), its reason replaces the figures and **Start run** stays disabled.

**Start run** posts the same request for real. The dialog follows the run:
`Queued`, `Running` with progress, and for a **Segment** run
`Waiting for the region worker` until the worker has written the boxes. When
it ends, the served counts are listed (answered, written, kept as suggestions
on locked items, failed; boxes judged; segmenter items done and pending). You
can close the dialog while it runs; **Cancel run** stops it after the current
chunk and what it wrote stays. Only one run is live per project.

## The proposal for a cluster or sub-cluster

A **Classify** run over a cluster or sub-cluster reads the sampled answers as
a label. On the cluster page a folded line, "VLM proposal for cluster #…:
class, 83% of 41 answers", opens the proposal card:

- the **proposed class** and the **agreement**: how many sampled answers
  named it, and the gate it had to pass. If no class reached the gate the card
  says so and shows only the votes; **Use** on a row picks a class from them.
- the **votes** per class, and the new class names the VLM proposed that the
  registry lacks (with **Add class…** to create one).
- the **attribute suggestions**, labelled "suggestions, not labels".
- **Apply {class} to {N} members**: the explicit button. N is the server's
  dry-run count of members it would label; the members it leaves alone
  (locked, validated, held-out) are counted next to it. Applying labels them
  as you would by hand (validated, trainable).

The card marks a proposal **stale** when the scope has changed since the run,
for example after **Refine (AHC)** (it shows how many of the members now were
in the run). Apply then refuses with the backend's `scope_changed` message;
run again on the current members. **Count the current members anyway** is
offered only after that refusal, and still counts first.

## Attribute suggestions

Attributes a [prompt pack](./prompt-packs.mdx) declares appear as dashed
**VLM suggests** chips with the VLM's confidence, on the crop card, in the crop
details panel and in Review, and per box on the Review box row. They are never
labels and never change the class or the boxes. When the VLM's class answer
differs from a locked label, the card shows **VLM said: {class}** beside it.

## Region verification on the VLM scope panel

On [Settings](./settings.md), the VLM scope panel has a **Region
verification** choice: **Verify with the VLM** (the region worker asks the
project's VLM about each box the segmenter proposes) or **Accept without a
VLM** (boxes are accepted as they are, even with a VLM active). It is saved
with the scope and applies to the whole project; a **Segment** run can choose
otherwise for its own items.

## Sub-cluster tabs

After **Refine (AHC)** the cluster page lists the sub-clusters as tabs. A tab
asks the server for that sub-cluster's members (`cluster_subid`), so a
sub-cluster larger than a page is listed whole and **Assist…** on it targets
exactly it. The tab list is built from the sub-clusters seen while the whole
cluster is shown; scroll the **all** view to load the rest.
