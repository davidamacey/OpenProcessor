# Security Policy

## Reporting a vulnerability

This repo is currently private; if you have access and find a security
issue, please **do not open a public issue**. Instead, report it
privately to the maintainers (David Macey / example-org LLC) so it can be
assessed and fixed before disclosure.

Include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (minimal repro if possible).
- Which version/commit you tested against.

## Scope notes

- Cropwright is a pure frontend SPA with **no local database and
  no authentication of its own** — see `README.md`'s "Limits" section.
  It is designed to be deployed behind a trusted network boundary (LAN)
  or fronted by an auth proxy (oauth2-proxy, nginx basic auth) before
  exposure beyond that. "Anyone who can reach the URL has full edit
  rights" is documented, expected behavior, not a vulnerability to
  report on its own — but a way to bypass an operator's chosen auth
  front-end would be.
- The actual data/model backend is the OpenProcessor API (separate repo)
  — most security-relevant surface (auth, data validation, injection
  risks) lives there, not in this frontend.

## Supported versions

This project does not yet follow semantic versioning with maintained
release branches — security fixes land on `master` and should be
pulled from there.
