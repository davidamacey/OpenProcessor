# Cropwright

Cropwright is a keyboard-first, cluster-assisted web app for labeling
image crops at scale — the human-in-the-loop frontend for
[OpenProcessor](https://github.com/example-org/openprocessor)

<!-- TODO(coordinator): confirm the public OpenProcessor repo URL. -->.

It is a **pure frontend** — no database of its own, no offline/mock
mode — and is domain-agnostic: what you're labeling (vehicles, defects,
aircraft tail numbers, anything with a region of interest or none at
all) is entirely defined by the OpenProcessor backend's served region
profile and, optionally, a small JSON config you drop in at deploy time
(see "Configuring your own domain" below). Cropwright ships with **no
domain built in**.

Built with SvelteKit 2 + Svelte 5 runes + TypeScript strict + Tailwind
v4. Pointer-event drag-and-drop via `svelte-dnd-action`. Dark theme,
Apple system colors, keyboard-first UX.

## Screenshots

<!-- TODO(F10): add screenshots captured from the public-data fresh-start run
     (COCO val2017 + Open Images plates). The existing docs/screenshots*/
     images show non-public imagery and must never be linked here. -->

Full walkthrough of every route with explanations: **[docs/FEATURES.md](docs/FEATURES.md)**.

## Security — read this before deploying

**The curation API has no request authentication.** Anyone who can
reach Cropwright's nginx origin can label, discard, ingest, export and
train — there is no login and no token, by design (the backend owns
that decision; see its own docs). **Never expose Cropwright or the
OpenProcessor API it proxies to the public internet.** Run it only on a
trusted LAN/VPN, or put it behind an authenticating reverse proxy
(`oauth2-proxy`, nginx basic auth). See [SECURITY.md](SECURITY.md).

## Requirements

- Docker with Compose v2 (the supported path), **or** Node 20+ for a
  source build.
- A reachable **OpenProcessor** backend, started with `OP_API_PREFIX=/curation`
  (the default) and its own docker network — note that network's name,
  you'll need it below.

Cropwright does nothing useful without a running OpenProcessor backend:
it has no data of its own to show.

## Quick start (Docker)

```bash
git clone https://github.com/davidamacey/OpenProcessor && cd cropwright
cp .env.example .env
```

Edit `.env` and set, at minimum:

- `API_UPSTREAM` — where nginx proxies API calls, by container name over
  the shared docker network (default `http://op-api:8000`, matching
  OpenProcessor's own default Compose service name/port).
- `PUBLIC_API_PREFIX` — must equal the backend's own `OP_API_PREFIX`
  (default `/curation`).
- `OP_DOCKER_NETWORK` — the OpenProcessor backend's docker network name
  (`docker network ls`; defaults to OpenProcessor's own default network
  name).
- `CROPWRIGHT_PORT` — host port to publish (default `5184`).

Then:

```bash
docker compose up -d --build
```

Open `http://localhost:5184` (or whatever `CROPWRIGHT_PORT` you set).

**Verify it's connected:**

- Look at the top-bar status chip — it polls `{PUBLIC_API_PREFIX}/health`
  and shows green when the backend answers.
- Or from the shell: `curl http://localhost:5184/curation/health`
  (swap in your own port/prefix).

### Running a second, independent instance

Give it its own compose project name, container name and host port so
it never collides with a running instance:

```bash
CROPWRIGHT_PORT=5190 CROPWRIGHT_CONTAINER_NAME=cw-second \
  OP_DOCKER_NETWORK=some_other_openprocessor_net \
  docker compose -p cw-second up -d --build
```

This is the same image and compose file, pointed at a different
backend (or a different network) with a different host port — useful
for running two datasets/deployments side by side on one host.

## What each feature needs from the backend

Every feature below degrades to **absent, not broken**, when the
backend hasn't enabled it — the nav link/page section simply doesn't
render, and Cropwright never fires a request against a route the
backend hasn't mounted.

| Feature                                                      | Backend requirement                                                                                                    |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Region review tab, `/clusters` region gallery, sub-box edit  | A region profile served on `GET {API_PREFIX}/health` (`region_profile`) — absent, no region UI anywhere                |
| Score chips, mistakenness/uniqueness sort options            | `OP_SCORES_ENABLED`                                                                                                    |
| Diverse-selection overlay on `/clusters/[id]`                | `OP_SELECT_DIVERSE_ENABLED`                                                                                            |
| Embedding-plot lasso tool on `/clusters`                     | `OP_VIZ_PROJECTION_ENABLED`                                                                                            |
| Semantic (embedding) search box                              | `OP_SEMANTIC_SEARCH_ENABLED`                                                                                           |
| `/train` (preflight, launch, promote)                        | The backend's trainer container/worker running                                                                         |
| `/bakeoff`                                                   | The backend's on-demand evaluator container, and its `/bakeoff/*` router mounted                                       |
| `/ingest` browser upload persisting bytes for later browsing | The backend's ingest-upload persistence (a pre-persistence backend shows a banner explaining uploads aren't browsable) |

All of these are read from the backend's own `/methods`, `/health` and
availability-probe endpoints at runtime — nothing here is a Cropwright
build flag.

## End-to-end workflow

`/ingest` (bring images in) → `/review` + `/clusters` (label and
triage) → `/classes` (manage the class registry) → `/export` (freeze a
test holdout, export YOLO) → `/train` (launch, promote) → `/bakeoff`
(compare models) → back to `/review` for the next cycle
(Model Disagreements surfaces where a newly promoted model and the
human label diverge).

**Trying it with sample data.** For a new install with no images yet,
OpenProcessor ships a small script to fetch a public COCO val2017
subset as sample data (never bundled with Cropwright itself) — see the
OpenProcessor README for the exact command

<!-- TODO(coordinator): the backend's sample-dataset script/command is still being finalized; fill in the exact invocation once its runbook lands. --> —

then ingest it through `/ingest`'s server-path panel, pointing at the
sample folder the backend mounts under its configured batch source
roots (`/ingest` only shows this panel when the backend advertises at
least one source root).

## Configuring your own domain

A domain (what class of object you're labeling, and whether it has a
sub-region like a license plate or a defect zone) is **backend
configuration, not a Cropwright rebuild**:

1. The backend's served region profile (if any) drives the built-in
   region tab, gallery and sub-box editor automatically.
2. A deployment can further customize labels/keymap/cohorts, or add a
   second annotation slot the backend doesn't know about, by dropping
   an `annotation-profiles.json` file next to the built app (no fork,
   no rebuild) — see `static/annotation-profiles.example.json` for a
   worked example, `examples/annotation-profiles/` for three more
   (license plate, aircraft tail number, defect code), and
   `docs/annotation-slots-contract-draft.md` for the schema. A missing
   or invalid file is always silently ignored.

## Routes and keyboard shortcuts

See [docs/FEATURES.md](docs/FEATURES.md) for a full route-by-route walkthrough with
screenshots, and [CLAUDE.md](CLAUDE.md)'s Routes and Keyboard shortcuts
sections for the authoritative, current tables (kept in lockstep with
the code, not duplicated here to avoid drift). Highlights: `/dashboard`
(pipeline health), `/ingest` (bring images in), `/clusters` +
`/clusters/[id]` (cluster-based triage, drag-and-drop), `/review`
(keyboard-driven queues), `/classes`, `/export`, `/train`, `/models`,
`/bakeoff`, `/settings`.

There is a single class-assignment scheme: a per-class `hotkey_letter`
configured on `/classes` or the `` ` `` shortcut overlay. Reserved
action keys (`g n d z x u a m /`, plus whatever a registered region
slot's own keymap declares) can never be bound to a class — rejected
both client- and server-side.

## Configuration

| Var                                    | Default                    | Used by       | Purpose                                                                                                                                                                     |
| -------------------------------------- | -------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PUBLIC_API_PREFIX`                    | `/curation`                | both          | Path prefix the backend serves its curation endpoints under. Must equal the backend's own `OP_API_PREFIX`.                                                                  |
| `PUBLIC_TRITON_API_URL`                | _(empty)_                  | `npm run dev` | Legacy name (predates the OpenProcessor rename). Points the dev server straight at a backend origin, bypassing the nginx proxy Docker gets for free. Leave empty in Docker. |
| `API_UPSTREAM`                         | `http://op-api:8000`       | Docker        | Where nginx proxies `PUBLIC_API_PREFIX/*` to, by container name over the shared docker network.                                                                             |
| `OP_DOCKER_NETWORK`                    | `openprocessor_triton_net` | Docker        | The OpenProcessor backend's docker network name (must already exist).                                                                                                       |
| `CROPWRIGHT_PORT`                      | `5184`                     | Docker        | Host port nginx is published on.                                                                                                                                            |
| `CROPWRIGHT_CONTAINER_NAME`            | `cropwright`               | Docker        | Container name — set uniquely for a second instance (see above).                                                                                                            |
| `CROPWRIGHT_INGEST_MAX_REQUEST_MB`     | `256`                      | Docker        | Upload cap for `/ingest` (nginx `client_max_body_size`); kept in lockstep with the client-side chunk planner.                                                               |
| `PUBLIC_APP_NAME` / `PUBLIC_APP_BADGE` | `Cropwright` / `CW`        | both          | Top-bar wordmark/badge, for a white-labeled deployment.                                                                                                                     |

The app is a pure SPA consumer of the OpenProcessor API — there is
**no** local database. State is reconstructed from API calls;
`localStorage` only caches transient UI state (sidebar collapse, last
cluster viewed).

## Development

```bash
npm install
npm run dev      # http://localhost:5173 — set PUBLIC_TRITON_API_URL in .env
                  # first if your backend isn't proxied
npm run check    # svelte-check + tsc
npm run build    # SvelteKit static adapter → build/
npm test         # vitest, jsdom
npm run test:e2e # Playwright against an in-browser stub backend (own venv, auto-provisioned)
```

**Live, read-only tests** against a real deployed backend:
`CROPWRIGHT_LIVE_URL=http://localhost:5184 npm run test:live` — skipped
entirely unless that env var is set; structurally read-only (aborts any
non-GET/HEAD request to the API and fails the test if one was even
attempted).

**Mutation testing** (`npm run test:mutation`, ~15-20 min) checks
whether the test suite actually notices a deliberately broken line, not
just that it passes.

**Vendored API contract:** `npm run contract:sync` / `contract:check`
pull the wire-format snapshot from a local OpenProcessor checkout
(`OPENPROCESSOR_REPO`, default `../openprocessor`; `OPENPROCESSOR_REF`,
default `main`) and diff it against what's checked in, so a backend
rename fails a frontend test instead of silently rendering blanks.

## Troubleshooting

| Symptom                                               | Likely cause                                                                                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Top-bar status chip red / "API unavailable"           | Backend not reachable at `API_UPSTREAM` (Docker) or `PUBLIC_TRITON_API_URL` (dev); or `{API_PREFIX}/health` returns 5xx              |
| A write returns 409 "no region profile is configured" | The backend has no region profile set — region-specific actions (confirm/reject a sub-box, etc.) are unavailable until it does       |
| Upload to `/ingest` fails with 413                    | File(s) exceed `CROPWRIGHT_INGEST_MAX_REQUEST_MB` (nginx) or the backend's own configured upload cap — raise the env var, rebuild    |
| Port already in use                                   | Another service already publishes that host port — set `CROPWRIGHT_PORT` (and give the instance its own `CROPWRIGHT_CONTAINER_NAME`) |
| Empty cluster grid / review queues                    | The backend's index isn't populated yet — ingest some images first                                                                   |
| Build fails: `Cannot find module 'svelte-dnd-action'` | `npm install` wasn't run, or `node_modules` is stale — `rm -rf node_modules && npm install`                                          |

## License

MIT. See [LICENSE](LICENSE). Cropwright is the frontend companion to
[OpenProcessor](https://github.com/example-org/openprocessor)

<!-- TODO(coordinator): same URL as above. -->, which handles inference,

search and clustering server-side.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev workflow and
[SECURITY.md](SECURITY.md) for reporting vulnerabilities.
