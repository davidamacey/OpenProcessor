# Security Policy

## No authentication — do not expose this service to the internet

**OpenProcessor's API (port 4603 by default) has no authentication,
authorization, or rate limiting of any kind.** It is designed to run on
a trusted internal network behind your own reverse proxy / auth layer,
not to be reachable directly from the internet.

This matters because several routes are genuinely dangerous without
access control:

- `DELETE /query/image/{id}` — deletes indexed data with no confirmation.
- `DELETE /curation/projects/{project}/models/{model_name}` — unloads and
  removes a promoted model from Triton.
- `DELETE /curation/projects/{project}` — deletes a project's indexes and
  directories. A real delete needs only `?confirm=<slug>`.
- `POST /ingest/directory` — performs an arbitrary server-side
  filesystem path read; anyone who can reach this endpoint can make the
  server read any path it has filesystem access to.
- `POST /curation/projects/{project}/ingest/batch` and
  `POST /curation/projects/{project}/datasets/preview` — read server-side paths
  too, limited to the configured source roots (`OP_SOURCE_ROOT_HOST`
  mount and the upload root).

The `curation` subsystem (see [`docs/CURATION.md`](docs/CURATION.md), opt-in
behind the `curation` compose profile) adds about 237 operations, about 125 of
them writes, all under the same no-authentication model. Projects isolate data
from each other inside the application (an OpenSearch request guard refuses
cross-project access), but that is a correctness boundary, not an access
control: any caller can address any project.

**If you deploy this service, put it behind a reverse proxy that
enforces authentication, and do not map its ports directly to a public
interface.**

### LAN access from the one-line installer

**Cropwright is reachable on your LAN by default, for homelab or
small-business use. The API itself stays bound to 127.0.0.1. There is no login
on Cropwright — a warning is shown. Pass `--local-only` to opt out and keep
everything on 127.0.0.1.**

`setup-openprocessor.sh` targets one computer **and** its local network
(homelab, small office). With the `cropwright` tier, the Cropwright web UI
is published on all interfaces (`CROPWRIGHT_BIND_ADDRESS=0.0.0.0`) so other
computers on the LAN can open it; its nginx proxies to the API over the
Docker network. Every backend port (API, Triton, OpenSearch, ...) stays
bound to `127.0.0.1` unless you pass `--bind` and confirm it. A specific
`--bind` address (say `10.0.0.5`) also narrows Cropwright to that one
interface; `--local-only` always keeps it on `127.0.0.1`.

The UI has **no login**, and it can reach every API route above. Use it only
on a network you trust, never port-forward it to the public internet, and put
a reverse proxy with authentication in front for anything wider. Install with
`--local-only` (or answer "no" to the LAN question) to keep it on this
computer.

### Release checksums are integrity, not authenticity

The installer checks every downloaded file against the release's
`SHA256SUMS`, pins images by digest from `images.lock`, pins Cropwright's
files through `cropwright.lock`, and cross-checks each `images.lock` entry's
image repo against `scripts/lib/image_keys.sh`. These catch truncated or
corrupted downloads and an internally inconsistent release. They do **not**
prove who published the release: `SHA256SUMS` comes from the same origin as
the files it covers. Release signing is follow-up work.

### Other default-open components in the shipped `docker-compose.yml`

- **Grafana** ships with the default credentials `admin` / `admin`
  (`GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD` in
  `docker-compose.yml`). Change these before exposing Grafana to
  anyone but yourself.
- **OpenSearch** ships with its security plugin disabled by default
  (`DISABLE_SECURITY_PLUGIN=true` / `DISABLE_SECURITY_DASHBOARDS_PLUGIN=true`
  in `docker-compose.yml`, commented "Disable security for dev (enable
  in production)"). There is no authentication on the OpenSearch or
  OpenSearch Dashboards ports either. Enable OpenSearch's security
  plugin before any production or multi-tenant deployment.

### VLM endpoints and server-side requests

Any caller can register a VLM endpoint URL (there is no authentication) and
make the API send requests to it (a test call, a probe, and every labeling
call once it is activated). Endpoint URLs are therefore checked before use:
this stack's own services, link-local and cloud-metadata addresses, and
anything that resolves to them are refused for every spelling of the
address, redirects are never followed, keys are only ever
references to files on the host, and an endpoint outside this deployment
needs an explicit acknowledgement (`OP_VLM_EXTERNAL_POLICY=deny` refuses
them).

The residual risk is deliberate: a URL that resolves to another host on
your private network (a router, a NAS, an internal admin page) is a valid
VLM endpoint as far as the API can tell, so a caller who can reach the API
can make it send synthetic probe requests and OpenAI-shaped chat requests
there. DNS is checked when an endpoint is saved, tested, activated and
built, and again by every labeler before it sends, at most every 30 seconds
(a refusal is immediate and stays until the host is acceptable again). So a
name that changes its answer is caught within about 30 seconds, but a
request already sent, or sent inside that window, goes to whatever the name
resolved to then; the connection itself is not pinned to the checked
address. The probe of an endpoint outside the deployment is only made once
the endpoint carries `allow_external`, because the probe sends its key. A
credential written into `OP_VLM_URL` is dropped, never served or logged.
Put the API behind your own network controls, as the rest of this document
already says.

None of this is a bug to be reported — it's the current, deliberate
state of the project, tracked internally as follow-up work.
Authentication is explicitly out of scope for this release: this is a
local tool meant to run on a trusted machine or private network behind
your own reverse proxy / auth layer, not to be exposed to a network you
don't trust.

## Reporting a vulnerability

If you find a security issue that is **not** one of the known gaps
above (for example, a bug that lets an *authenticated, intended* caller
escalate privileges, corrupt data outside their own request, or execute
arbitrary code), please report it privately rather than opening a
public issue:

- Preferred: use [GitHub's private vulnerability reporting](https://github.com/davidamacey/OpenProcessor/security/advisories/new)
  for this repository.
- Alternative: open a GitHub Security Advisory draft, or contact the
  maintainer via the email listed on the maintainer's GitHub profile.

Please include reproduction steps, the affected version/commit, and
your assessment of impact. We aim to acknowledge reports within a few
business days. There is no bug-bounty program at this time.

## Supported versions

This project does not yet maintain parallel supported release branches.
Security fixes land on `main`; use the latest tagged release.
