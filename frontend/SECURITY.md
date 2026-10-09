# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it
privately through GitHub's private vulnerability reporting: on the
repository's **Security** tab, choose **Report a vulnerability**. The
maintainers (OpenProcessor Contributors) will assess it and coordinate a fix before
disclosure.

Include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (a minimal repro if possible).
- The version or commit you tested against.

## The API has no authentication

**The curation API that Cropwright talks to has no request
authentication.** Anyone who can reach Cropwright's nginx origin can
label, discard, ingest (including server-path ingest), export, freeze a
test holdout and start training. There is no login and no token.

- **Never expose Cropwright, or the OpenProcessor API it proxies, to the
  public internet.** Run it only on a trusted LAN or VPN, or put it
  behind an authenticating reverse proxy (for example `oauth2-proxy` or
  nginx basic auth).
- "Anyone who can reach the URL has full edit rights" is documented,
  expected behavior, not a vulnerability to report on its own. A way to
  bypass an operator's chosen authenticating front end would be.

## Scope notes

- Cropwright is a static single-page app served by nginx, with no
  database and no authentication of its own. The container runs nginx as
  a non-root user (uid 101).
- The data and model backend is [OpenProcessor](https://github.com/davidamacey/OpenProcessor),
  a separate project. Most security-relevant surface (authentication,
  input validation, file access) lives there; report backend issues to
  that project.

## Supported versions

Cropwright does not yet maintain release branches. Security fixes land on
`main` and ship in the next release.
