"""Wire models for `GET/PUT /settings`."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Runtime import: pydantic resolves the Literal annotation from module globals.
from src.services.resource_links import ResourceLink, resource_links_from_config


class CurationSettingsResponse(BaseModel):
    """``GET,PUT /curation/settings`` response envelope.

    ``defaults`` is deliberately ``dict[str, str]`` (an OPEN map keyed by
    axis id), not a fixed set of named fields (``cluster``/``sort``/etc.)
    -- a future axis must not require a wire-format change. Missing key =
    no shared override for that axis; the caller falls back to
    ``GET /methods``'s own hardcoded-default resolution (see
    ``src.services.curation.strategy_registry.resolve_effective_default``).
    """

    defaults: dict[str, str] = Field(default_factory=dict)
    updated_at: str | None = None
    updated_by: str | None = None
    revision: int = Field(
        default=0,
        description='Send it back as `expected_revision` on the next PUT; 0 while never written.',
    )
    resource_links: list[ResourceLink] = Field(default_factory=resource_links_from_config)


class CurationSettingsUpdateRequest(BaseModel):
    """``PUT /curation/settings`` body -- partial by design. Only the axes
    present here are validated + merged into the stored document; axes
    already set are left untouched (see
    ``src.services.curation.settings_store.update_curation_settings``).

    A value of ``null`` for an axis clears that axis's shared override
    (falls back to that endpoint's own hardcoded default / each /review
    tab's own tuned sort) -- otherwise a pinned override was permanently
    unreachable once set, since every id must be currently-advertised and
    there was no way to express "go back to no override" (raised by the
    Cropwright settings-UI integration pass)."""

    defaults: dict[str, str | None] = Field(default_factory=dict)
    expected_revision: int = Field(
        ge=0,
        description='The `revision` of the settings this edit was made against (GET /settings); '
        'a stale one is a 409 `revision_conflict`.',
    )
