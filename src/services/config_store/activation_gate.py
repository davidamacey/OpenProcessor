"""The ONE `for_activation` gate every activation-writing entry point must
call (W3/W4 round-4 review, R4-1).

Round 3 fixed the gate bypass for `PUT /settings` (N1) by copying the
`/activate` route's checks into ``settings.py``. Round 4 found the SAME
class of bug a fourth time: ``POST /prompt_packs/active/rollback`` (and
its region-profile twin) re-activate a revision with **no** gate at all,
so a pack/profile that ``/activate`` correctly 422s (even with
``force: true``) can still go live through rollback.

The round-3 reviewer's own diagnosis: "the lasting fix is one gate
function that every activation path calls, plus a single test that walks
all of them with a revision that should fail." This module is that
function. Every writer of an ``activation:*`` doc must call
:func:`run_activation_gate` before writing:

- ``POST /prompt_packs/{name}/activate``, ``POST /region_profiles/{name}/activate``
- ``PUT /settings`` (the config-store-backed axes)
- ``POST /prompt_packs/active/rollback``, ``POST /region_profiles/active/rollback``

Deactivating (``name is None``) never needs the gate -- there is nothing
to validate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from src.routers.curation._config_common_models import ValidationReport
    from src.services.config_store.index import ConfigAxis


# Sentinel distinguishing "no override supplied -- use the OTHER axis's
# currently-STORED value" from "override supplied, even if that override
# is itself None/off" (R5-1 fix, W3/W4 round-5 review). A real `None`
# override means "the other axis is being deactivated in this same
# request" and must NOT fall back to the stored value.
_UNSET: Any = object()


async def run_activation_gate(
    axis: ConfigAxis,
    name: str,
    revision: int | None,
    *,
    force: bool = False,
    client: Any = None,
    pending_sibling: Any = _UNSET,
    pending_vlm: Any = _UNSET,
    body: dict[str, Any] | None = None,
) -> ValidationReport:
    """Run the exact `for_activation` validation the dedicated `/activate`
    routes run, against whichever axis is being activated.

    ``client`` (an OpenSearch client) is optional but should be passed
    whenever the caller has one: rollback's ``previous`` ref can name a
    revision that is no longer the STORED CURRENT one (e.g. pack was
    active at rev 1, a later, never-activated PUT moved current to rev
    2) -- ``build_record`` only resolves the current revision, so without
    a client-backed fallback to the immutable ``<kind>:<name>@<rev>``
    revision-copy doc, a legitimate rollback to an old-but-real revision
    would incorrectly 404 here. Callers with no client (none exist today)
    keep the current-revision-only behavior the dedicated activate routes
    already had (round-1..3 Minor 2: activating a genuinely past revision
    through those routes is a separate, pre-existing, non-blocking gap).

    ``pending_sibling`` (R5-1 fix, W3/W4 round-5 review): the OTHER
    axis's PENDING (about-to-be-applied) value, for a caller that is
    activating BOTH axes in the same request (``PUT /settings`` with
    both ``prompt_pack`` and ``detection_profile`` set). Left unset, the
    gate validates against the other axis's currently-STORED value --
    correct for every single-axis caller, but wrong for a combined
    request: each half could individually pass against the OTHER axis's
    OLD value while the NEW pairing (both values applied together) is
    invalid, defeating the never-bypassable multi-box check. When
    ``axis == 'prompt_pack'``, ``pending_sibling`` is the pending
    ``DetectionProfile`` (or ``None`` for a pending profile
    deactivation). When ``axis == 'detection_profile'``, it is the
    pending active ``PromptPack`` (never ``None`` -- a pack axis always
    resolves to *some* pack, env/file default included).

    ``pending_vlm`` (W9): the same idea for the VLM axis -- the endpoint a
    combined ``PUT /settings`` is about to activate (``None`` = about to be
    switched off). Left unset, the pack/profile is paired with the
    project's stored active VLM.

    ``body`` (R5-3 fix, W3/W4 round-5 review, Major -- project clone):
    when supplied, skips the ``build_record``/``get_revision_record``
    name lookup (and the template 403 check) entirely and validates
    THIS body under ``name``/``revision`` instead. For a caller
    validating a body that does not live in the CURRENTLY BOUND
    project's own config store -- e.g. project clone checking whether
    the SOURCE project's activated pack/profile would still pass
    ``for_activation`` validation in the TARGET project's context
    (detector-sharing, class registry, Triton reachability all differ
    per project) -- ``name``/``revision`` resolve nothing there; the
    caller must already have the body from the source project's own
    store.

    Raises the router-shaped ``HTTPException`` (via ``api_error``) for:
    - 404 ``not_found`` -- unknown name/revision;
    - 403 ``read_only`` -- a template can't be activated directly;
    - 422 ``validation_failed`` -- blocking `for_activation` errors, i.e.
      every error not in ``BYPASSABLE_CODES`` when ``force`` is set.

    Returns the ``ValidationReport`` on success (the caller may want it,
    e.g. to echo back in the response).
    """
    from src.routers.curation._config_common_models import api_error

    if axis == 'prompt_pack':
        from src.routers.curation.prompt_packs import _registry_class_names, _resolve_profile
        from src.services.config_store.pack_validation import BYPASSABLE_CODES, validate_pack
        from src.services.config_store.packs import (
            build_record as build_pack_record,
            get_revision_record as get_pack_revision,
        )

        if body is None:
            record = build_pack_record(name, revision=revision)
            if record is None and revision is not None and client is not None:
                record = await get_pack_revision(client, name, revision)
            if record is None:
                raise api_error(404, 'not_found', f'{name!r} is not a known pack')
            if record.read_only and record.source == 'template':
                raise api_error(403, 'read_only', f'{name!r} is a template; clone it first')
            pack_body = record.body
        else:
            pack_body = body
        profile = _resolve_profile(None) if pending_sibling is _UNSET else pending_sibling
        paired_profile = profile
        paired_pack = _decode_pack(name, pack_body)
        report = validate_pack(
            None,
            pack_body,
            profile=profile,
            for_activation=True,
            class_names=_registry_class_names(),
        )
    else:
        from src.routers.curation.region_profiles import (
            _project_slug,
            _registry_class_names,
            _segmenter_health_fn,
        )
        from src.services.config_store.profile_validation import BYPASSABLE_CODES, validate_profile
        from src.services.config_store.profiles import (
            build_record as build_profile_record,
            get_revision_record as get_profile_revision,
        )
        from src.services.labeling.vlm_prompts import active_prompt_pack

        if body is None:
            profile_record = build_profile_record(name, revision=revision)
            if profile_record is None and revision is not None and client is not None:
                profile_record = await get_profile_revision(client, name, revision)
            if profile_record is None:
                raise api_error(404, 'not_found', f'{name!r} is not a known region profile')
            if profile_record.read_only and profile_record.source == 'template':
                raise api_error(403, 'read_only', f'{name!r} is a template; clone it first')
            profile_body = profile_record.body
        else:
            profile_body = body
        active_pack = active_prompt_pack() if pending_sibling is _UNSET else pending_sibling
        paired_pack = active_pack
        paired_profile = _decode_profile(name, profile_body)
        report = await validate_profile(
            None,
            profile_body,
            for_activation=True,
            segmenter_health=_segmenter_health_fn,
            active_pack=active_pack,
            class_names=_registry_class_names(),
            project_slug=_project_slug(),
        )

    # W9.5: a change to one side of the pack <-> profile <-> VLM triple is
    # checked against the active VLM here, in the one gate every activation
    # path calls (never a per-route copy).
    from src.routers.curation._config_common_models import ValidationReport
    from src.services.config_store.vlm_gate import active_vlm_pairing_issues
    from src.services.config_store.vlm_validation import BYPASSABLE_CODES as VLM_BYPASSABLE

    vlm_issues = await active_vlm_pairing_issues(
        client,
        paired_pack,
        paired_profile,
        **({} if pending_vlm is _UNSET else {'pending_vlm': pending_vlm}),
    )
    vlm_errors = [i for i in vlm_issues if i.severity == 'error']
    if vlm_issues:
        errors = [*report.errors, *vlm_errors]
        report = ValidationReport(
            ok=not errors,
            errors=errors,
            warnings=[*report.warnings, *[i for i in vlm_issues if i.severity != 'error']],
            force_allowed=bool(errors)
            and all(e.code in {*BYPASSABLE_CODES, *VLM_BYPASSABLE} for e in errors),
        )
    blocking = [
        e
        for e in report.errors
        if not (force and e.code in (VLM_BYPASSABLE if e in vlm_errors else BYPASSABLE_CODES))
    ]
    if blocking:
        raise api_error(
            422,
            'validation_failed',
            f'{name!r} has {len(blocking)} blocking error(s)',
            report=report,
        )
    return report


def _decode_pack(name: str, body: dict[str, Any]) -> Any:
    """The body as a :class:`PromptPack`, or ``None`` when it does not decode
    (the pack validator already reports that)."""
    from src.services.labeling.vlm_prompts import PromptPack

    try:
        return PromptPack.from_dict({**body, 'name': name})
    except (TypeError, ValueError, KeyError):
        return None


def _decode_profile(name: str, body: dict[str, Any]) -> Any:
    from src.services.detection.profile_registry import region_profile_from_dict

    try:
        return region_profile_from_dict({**body, 'name': name}, source='validate')
    except (TypeError, ValueError, KeyError):
        return None


__all__ = ['run_activation_gate']
