"""Wire models for the per-project keymap surface (W2b). See
``docs/design/openprocessor_internal/any_domain_plan.md`` W2b and CW-K
§4."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.routers.curation._config_common_models import (
    ValidationIssue,  # noqa: TC001 - pydantic field type, needed at runtime
)


class KeymapGrammarWire(BaseModel):
    modifiers: list[str]
    named_keys: list[str]
    printable: str
    max_combos_per_action: int
    locked_keys: list[str]
    browser_reserved: list[str]


class KeymapContextWire(BaseModel):
    id: str
    label: str
    description: str
    includes: list[str]
    class_hotkeys_live: bool


class KeymapActionWire(BaseModel):
    id: str
    context: str
    group: str | None
    label: str
    description: str
    default: list[str]
    keys: list[str]
    modifiable: bool
    available: bool
    locked_keys: list[str] = Field(default_factory=list)


class KeymapGetResponse(BaseModel):
    scope: Literal['project'] = 'project'
    project: str
    revision: int
    etag: str
    is_default: bool
    updated_at: str | None
    grammar: KeymapGrammarWire
    contexts: list[KeymapContextWire]
    actions: list[KeymapActionWire]
    overrides: dict[str, list[str]]
    reserved_hotkeys: list[str]
    issues: list[ValidationIssue] = Field(default_factory=list)


class KeymapPutRequest(BaseModel):
    # M5: an ``If-Match: "keymap:N"`` header is also accepted (CW-K §4.3);
    # ``expected_revision`` is optional here so a caller may supply either.
    # At least one of the two must resolve, else 422.
    expected_revision: int | None = None
    overrides: dict[str, list[str]]
    unbind_conflicting_class_hotkeys: bool = False


class KeymapPutResponse(KeymapGetResponse):
    unbound_class_hotkeys: list[dict[str, Any]] = Field(default_factory=list)


class KeymapValidateRequest(BaseModel):
    overrides: dict[str, list[str]]


class KeymapValidateResponse(BaseModel):
    ok: bool
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]
    force_allowed: bool
    resolved: dict[str, list[str]]
    reserved_hotkeys: list[str]
    class_conflicts: list[dict[str, Any]] = Field(default_factory=list)


class KeymapResetRequest(BaseModel):
    expected_revision: int | None = None
    action_ids: list[str] | None = None
    unbind_conflicting_class_hotkeys: bool = False


__all__ = [
    'KeymapActionWire',
    'KeymapContextWire',
    'KeymapGetResponse',
    'KeymapGrammarWire',
    'KeymapPutRequest',
    'KeymapPutResponse',
    'KeymapResetRequest',
    'KeymapValidateRequest',
    'KeymapValidateResponse',
]
