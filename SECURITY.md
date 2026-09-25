# Security Policy

## ⚠️ No authentication — do not expose this service to the internet

**OpenProcessor's API (port 4603 by default) has no authentication,
authorization, or rate limiting of any kind.** It is designed to run on
a trusted internal network behind your own reverse proxy / auth layer,
not to be reachable directly from the internet.

This matters because several routes are genuinely dangerous without
access control:

- `DELETE /query/image/{id}` — deletes indexed data with no confirmation.
- `DELETE /curation/models/{model_name}` — unloads/removes a promoted
  model from Triton.
- `POST /ingest/directory` — performs an arbitrary server-side
  filesystem path read; anyone who can reach this endpoint can make the
  server read any path it has filesystem access to.

The `curation` subsystem (see [`docs/CURATION.md`](docs/CURATION.md),
shipped experimental and opt-in for this release) adds a further ~109
routes, ~47 of them write endpoints, all under the same
no-authentication model.

**If you deploy this service, put it behind a reverse proxy that
enforces authentication, and do not map its ports directly to a public
interface.**

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
