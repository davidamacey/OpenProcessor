Vendored API-docs assets served at `/docs-assets` by `src/routers/api_docs.py`
(self-hosted so `/docs` and `/redoc` work offline).

- `swagger-ui-bundle.js`, `swagger-ui.css`, `favicon-32x32.png`: swagger-ui-dist 5.33.1 (Apache-2.0, `LICENSE.swagger-ui`)
- `redoc.standalone.js`: redoc 2.5.1 (MIT, `LICENSE.redoc`)

To update: `npm pack swagger-ui-dist@<v> redoc@<v>` and copy the same files.
- `redoc.standalone.js` is patched: the footer logo URL (`cdn.redoc.ly/.../logo-mini.svg`) points at the local `redoc-logo-mini.svg` placeholder. Re-apply after upgrading.
