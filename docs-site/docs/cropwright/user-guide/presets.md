---
sidebar_position: 7
title: Presets
---

# Presets (`/p/<project>/settings/presets/<kind>`)

Four kinds of settings are documents with revisions, like
[prompt packs](./prompt-packs.md) and [region profiles](./region-profiles.md). **Settings** has a
**Presets and embedding spaces** card that opens each one. The routes, the body fields and the
refusals are in the [Presets guide](../../guides/presets.mdx).

| Kind | Scope | What it sets |
|---|---|---|
| Serving profiles | The whole deployment | Which Triton servers run, which models stay loaded, instances, image preparation, priorities |
| Cluster profiles | This project | How clusters are built, active for items and for region boxes separately |
| Assist-loop presets | This project | The cluster-sampled auto-label loop: gate, recursion, budget, concurrency |
| Near-duplicate profiles | This project | The cosine cuts that decide duplicates and leaks, per level |

## The list

Each kind has a list with its documents, the revision, when it was last saved and what is
active. A document that is active shows a chip (for a cluster profile, which target); when the
active revision is older than the saved one the row says "saved r3, active r2".

- **New** asks for a name, a description and the body as JSON. The server checks the body when
  you create it and names each field it refuses; the dialog stays open with those lines.
- **Clone** copies a document under a new name and opens it.
- **Delete** removes a document (confirmed). It needs the revision the list showed.

## The editor

The body is JSON text. The only check the page makes itself is that the text is a JSON object;
every rule on the fields is the server's and appears when you press **Save**, one line per field.
Beside the text, **Fields** lists every field of the body as the server describes it: its type,
whether it is required, the default, the choices, the limits and what it does, at every depth
(`refresh.max_deferral_s` for a field inside `refresh`). If the server cannot describe the fields,
that is said there and you can still edit and save.

- **Save** writes a new revision and never changes what is active. If someone saved a newer
  revision first, the server's answer is shown with **Reload** (drop your edits) and **Keep my
  edits** (your next Save replaces the revision the server named).
- **Activate** asks what to activate (a cluster profile asks for items or region boxes), shows
  what is active now and what will be, and sends that as the guard. If the active document
  changed meanwhile, the server refuses, the page reads it again, and the dialog shows the
  refusal. A cluster profile whose target is items is refused on the region-box target and the
  other way round; the dialog shows the server's message and nothing is activated.

The targets Activate offers are the ones the server lists for the kind. The body fields are
also in the [Presets guide](../../guides/presets.mdx).
