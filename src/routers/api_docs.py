"""Self-hosted API docs: ``/docs`` (Swagger UI), ``/redoc`` and
``/docs/oauth2-redirect`` render from vendored bundles under
``src/static/docs`` served at ``/docs-assets``, so they work offline and
air-gapped (FastAPI's defaults load JS/CSS/fonts from public CDNs and
render a blank page without them). ``/openapi.json`` stays FastAPI's own.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI  # noqa: TC002
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import HTMLResponse  # noqa: TC002 - runtime return annotation
from fastapi.staticfiles import StaticFiles


ASSETS_DIR = Path(__file__).resolve().parent.parent / 'static' / 'docs'
ASSETS_URL = '/docs-assets'


def install_api_docs(app: FastAPI) -> None:
    """Call on an app created with ``docs_url=None, redoc_url=None``."""
    openapi_url = app.openapi_url or '/openapi.json'
    title = app.title

    @app.get('/docs', include_in_schema=False)
    async def swagger_ui() -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url=openapi_url,
            title=f'{title} - Swagger UI',
            oauth2_redirect_url='/docs/oauth2-redirect',
            swagger_js_url=f'{ASSETS_URL}/swagger-ui-bundle.js',
            swagger_css_url=f'{ASSETS_URL}/swagger-ui.css',
            swagger_favicon_url=f'{ASSETS_URL}/favicon-32x32.png',
        )

    @app.get('/docs/oauth2-redirect', include_in_schema=False)
    async def swagger_oauth2_redirect() -> HTMLResponse:
        return get_swagger_ui_oauth2_redirect_html()

    @app.get('/redoc', include_in_schema=False)
    async def redoc() -> HTMLResponse:
        return get_redoc_html(
            openapi_url=openapi_url,
            title=f'{title} - ReDoc',
            redoc_js_url=f'{ASSETS_URL}/redoc.standalone.js',
            redoc_favicon_url=f'{ASSETS_URL}/favicon-32x32.png',
            with_google_fonts=False,
        )

    app.mount(ASSETS_URL, StaticFiles(directory=ASSETS_DIR), name='docs-assets')
