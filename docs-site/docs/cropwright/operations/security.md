---
sidebar_position: 2
title: Security
---

# Security

**The curation API that Cropwright talks to has no request
authentication.** Anyone who can reach Cropwright's nginx origin can label,
discard, ingest (including server-path ingest), export, freeze a test
holdout, and start training. The API has no login and no token.

Cropwright's nginx can add one layer in front: set `OP_CROPWRIGHT_BASIC_AUTH=true`
with an htpasswd file and every route except `/nginx-health` asks for a login
(`./openprocessor cropwright password` sets it; the installer turns it on for a LAN
bind). It protects this origin only, not the API port. See
[Environment variables](../configuration/environment-variables.md).

- **Never expose Cropwright, or the OpenProcessor API it proxies, to the
  public internet.** Run it only on a trusted LAN or VPN, or put it behind
  an authenticating reverse proxy (`oauth2-proxy`, nginx basic auth).
- "Anyone who can reach the URL has full edit rights" is documented,
  expected behavior — not a vulnerability to report on its own. A way to
  bypass an operator's chosen authenticating front end would be.

## Scope

- Cropwright is a static single-page app served by nginx, with no database
  and no application-level authentication of its own (only the optional nginx basic auth above). The container runs nginx as a non-root
  user (uid 101).
- The data and model backend is OpenProcessor, a separate project. Most
  security-relevant surface (authentication, input validation, file access)
  lives there.

## Reporting a vulnerability

Do not open a public issue. Use GitHub's private vulnerability reporting on
the repository's Security tab.

## Optional backend auth

Backend-side, opt-in request authentication is a tracked but not-yet-built
ask (see the [Roadmap](/roadmap)). Until it lands, network isolation is the
only real control.
