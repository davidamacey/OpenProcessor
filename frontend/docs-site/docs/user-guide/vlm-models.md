---
sidebar_position: 7
title: VLM models
---

# VLM models (`/p/<project>/settings/models`)

The vision language model (VLM) suggests labels and verifies regions. The
**VLM registry** lets you register more than one VLM endpoint, test each, and
choose which one each project uses. The registry is **shared by every
project**; the choice of active endpoint is **per project**. The pages appear
only when the backend serves the registry; without it there is no Models card
on Settings and no link from Models.

## Secrets stay on the host

An endpoint that needs an API key stores only a **reference** to a secret
held by the backend host. Cropwright shows the reference and whether the host
actually has that secret; it never reads, displays or asks you to type the
key itself.

## The Models page

- **Active endpoint for this project**, with **Rollback**, **Turn off**
  (both confirmed) and the VLM's current health.
- **Endpoints.** Each registered endpoint with its status, whether it is
  local or external, and where it came from. An endpoint that sends crops
  outside your deployment carries a red chip with the backend's warning.
  Actions: **Activate here**, **Probe** (test it now), **Clone**, and
  **Delete** for stored endpoints (refused with the project names if some
  project still uses it). Endpoints defined by the backend's environment are
  read-only; clone one to edit a copy.
- **Local model.** When the backend runs a local VLM, the catalog of models
  it can serve, with a confirmed **Switch**. A switch may need a restart: the
  page then shows the backend's reason and the exact command to run, and
  follows the backend until the new model is serving. If a model doesn't fit
  the hardware you are asked to confirm before forcing it.
- **All model choices.** A read-only table of every model the project can be
  pointed at, linking to the editor for the role it plays.

## Creating and editing an endpoint

`/settings/models/new-endpoint` and `/settings/models/vlm/<name>` show a form
built from the backend's schema, validated as you type. **Test connection**
probes the unsaved draft, **Probe saved** re-reads the saved endpoint, and
revisions can be viewed and restored. If a probe is already running you get
the backend's "busy" message.

## Sending images outside your deployment

Activating an endpoint that sends crops outside the deployment (a hosted
service, say) shows the backend's warning and an **I understand** checkbox.
The acknowledgement is sent only if you tick it, never by default. The same
rule applies to every place a VLM can be chosen.

## Choosing a VLM per run

A **Project default** option sends nothing and uses the active endpoint.
Alternatively, choose a specific endpoint for one run in:

- the dashboard's assist bar (a pick implies running the VLM);
- the **Run VLM** buttons on the dashboard and on a cluster page;
- the prompt-pack and region-profile **test on a crop** panels.

Each shows the endpoint's status; an external endpoint shows the warning and
needs the acknowledgement for that run. If a run is refused, the backend's
reason is shown, and an unknown endpoint names the valid ones.

On [Settings](./settings.md) the **VLM** default is a dropdown of the same
endpoints, including **off**; an external endpoint without a recorded
acknowledgement is disabled with a link back to this page.

## Where the VLM shows up

The [Models](./models.md) page lists one row per registered endpoint with its
resolved model and an **active** chip. An item's **Details** panel shows the
VLM endpoint (`name@revision`), model and prompt pack that labeled it, when
that is recorded.

<Screenshot name="settings-models-1600.png" alt="Settings Models page with the active endpoint, endpoint table and local model panel" caption="Settings → Models — active endpoint, endpoint registry and local model" />

<Screenshot name="vlm-endpoint-editor-1600.png" alt="VLM endpoint editor showing a key reference and no key value" caption="Endpoint editor — a secret reference, never the key itself" />

<Screenshot name="vlm-run-picker-1600.png" alt="Per-run VLM picker on the dashboard assist bar" caption="Per-run VLM picker — project default, served endpoints and off" />
