---
sidebar_position: 5
title: Prompt-pack editor
---

# Prompt packs (`/p/<project>/settings/prompt-packs`)

A **prompt pack** is the set of instructions and reply formats the vision
language model (VLM) uses when it suggests or verifies labels in this
project. The editor lets you read, edit, test and activate them without
touching the backend's files. The pages appear only when the backend serves
prompt packs; otherwise **Settings** has no Prompt packs card.

## The list

`/settings/prompt-packs` lists the project's packs and the read-only
templates the backend ships, with a panel for the **active pack**:

- **Rollback** returns to the previously active revision (confirmed).
- **Clone** copies any pack or template into a new editable pack, or the pack
  of the same name from another project ("Copy from another project").
- **Delete** removes a pack (confirmed).

The panel also shows whether the running workers have picked up the active
revision yet (a lagging worker is flagged) and whether the active revision
is stale.

## The editor

`/settings/prompt-packs/<name>` shows one field for every entry in the
backend's own schema, grouped by the part of the VLM workflow it belongs to,
with the backend's help text, placeholders and the reply keys each prompt
should produce. Map-style fields edit as key and value rows. Nothing is
validated in the browser: a moment after you stop typing the draft goes to
the backend, and each problem it finds appears under the field it names (or
at the top).

- **Save** writes a new revision. If the pack changed since you opened it you
  are offered **Reload** or **Keep my edits** (which rebases your draft on
  the latest revision).
- **Revisions.** Open any older revision read-only, **Restore as new
  revision**, or **Activate** a specific one.
- **Read-only packs** (templates and built-ins) offer **Clone to edit**.
- If someone else changes the active pack while you are editing, you get a
  notice instead of losing your draft.

## Activating

Activating a revision pins that exact revision for the project and asks for
confirmation. If the backend's validation report says the pack shouldn't be
activated, it is shown in full; **Activate anyway** appears only when the
backend says forcing is allowed. The `/settings` page's prompt-pack dropdown
is a shortcut that activates a pack's latest revision.

## Test on a crop

For each VLM call the schema marks testable, a **Test on a crop** panel runs
that call on real crops without saving anything. Choose crops (and, where it
applies, whether to send the region box), optionally pick which VLM endpoint
to use, and run it. The result shows the pack and VLM used (as
`name@revision`, endpoint and model, with "draft" for an unsaved draft), the
latency and parse state, the exact prompt, the raw reply, and for each crop
its thumbnail and either the parsed answer or the reason it was skipped, plus
a preview of the item as the pack would label it. A crop the backend can't
find is named in the message.

The endpoint picker offers **Active endpoint** (the project's own) plus any
registered endpoint. Choosing an endpoint that sends crops outside your
deployment requires ticking the same acknowledgement used elsewhere; see
[VLM models](./vlm-models.md#sending-images-outside-your-deployment).

<Screenshot name="prompt-pack-editor-1600.png" alt="Prompt-pack editor with grouped fields and validation issues" caption="Prompt-pack editor — schema-driven fields with live validation" />

<Screenshot name="prompt-pack-test-1600.png" alt="Prompt-pack test panel showing the prompt, raw reply and parsed answer for a crop" caption="Test on a crop — the prompt, raw reply and parsed answer" />
