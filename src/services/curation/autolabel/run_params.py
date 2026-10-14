"""Per-run options of an auto-label run that need a decision before it is queued:
the prompt pack (pinned to a revision), the rejected ``detection_profile``, and
which pack the VLM stage resolves against.
"""

from __future__ import annotations

from typing import Any

from src.core.exceptions import RequestRejectedError
from src.services.curation.strategy_defaults import UnknownStrategyError, resolve_strategy_selection


CLUSTER_SCOPED_SKIP: dict[str, object] = {
    'skipped': True,
    'reason': 'cluster-scoped run: index-wide stage not run',
}

# detection_profile is NOT a per-run option: region detection runs in the
# detection worker on the active region profile, and no auto-label stage
# uses it. The param stays declared (hidden) only so an old client still
# sending it gets a clear 422 instead of FastAPI silently ignoring an
# unknown param.
# Reworded off OP_REGION_PROFILE (the config store made
# the active profile a config-store activation, not just an env var) to
# name the real source of truth and how to change it.
_UNRESOLVABLE_REVISION = (
    'revision {revision} of prompt_pack {resolved!r} is not resolvable '
    'in this process (only the current saved revision or the '
    'currently-activated revision can be pinned per run)'
)
DETECTION_PROFILE_REJECTED = (
    'detection_profile is not a per-run auto_label option: no auto_label stage '
    'runs region detection, and region detection runs in the detection worker '
    'on the active region profile (POST /region_profiles/{name}/activate). '
    'Remove the parameter.'
)


def reject_detection_profile(detection_profile: Any) -> None:
    """422 for any explicit ``detection_profile`` value (see
    :data:`DETECTION_PROFILE_REJECTED`). Non-``str`` means omitted."""
    if isinstance(detection_profile, str):
        raise RequestRejectedError(
            status_code=422,
            detail={
                'error': DETECTION_PROFILE_REJECTED,
                'message': DETECTION_PROFILE_REJECTED,
                'param': 'detection_profile',
            },
        )


async def resolve_run_prompt_pack(
    opensearch: Any, prompt_pack: Any
) -> tuple[str | None, int | None]:
    """Resolve the per-run ``prompt_pack`` id, and PIN a revision:
    ``"name@<revision>"`` pins that exact
    revision; a bare ``"name"`` resolves to ``(name, None)`` -- "latest
    saved" at the time the job reads the labeler, unaffected by an edit
    made after the job starts (the job dict stores the returned revision
    so ``resolve_prompt_pack``/``get_prompt_pack`` calls made later in the
    same job pass it explicitly).

    Non-``str`` values (``None``, or an unfilled FastAPI ``Query`` default
    when the endpoint function is called directly) mean "omitted" and
    resolve to the settings default. An unknown id is a 422 listing
    the valid ids — never a silent fallback.
    """
    requested = prompt_pack if isinstance(prompt_pack, str) else None
    revision: int | None = None
    name_only = requested
    if requested is not None and '@' in requested:
        name_only, _, rev_str = requested.rpartition('@')
        try:
            revision = int(rev_str)
        except ValueError:
            raise RequestRejectedError(
                status_code=422,
                detail={
                    'error': f'invalid revision in {requested!r}',
                    'message': f'invalid revision in {requested!r}',
                    'axis': 'prompt_pack',
                },
            ) from None
    try:
        resolved = await resolve_strategy_selection('prompt_pack', name_only, opensearch)
    except UnknownStrategyError as exc:
        raise RequestRejectedError(
            status_code=422,
            detail={
                'error': str(exc),
                'message': str(exc),
                'axis': exc.axis,
                'requested': exc.requested,
                'valid_ids': exc.valid,
            },
        ) from exc
    if revision is not None and opensearch is not None:
        # validate + pin the exact
        # revision at REQUEST time, not job-start time -- an unresolvable
        # `name@rev` (including the currently-*activated*-but-superseded
        # revision right after a PUT moves "current" forward, §3.7's own
        # motivating case) used to be accepted with a 202 and only fail
        # once the background job called `get_vlm_labeler`. §3.7 requires
        # a 422 here instead. `get_prompt_pack` resolves `name@rev` against
        # both the current doc and the store's pinned-active-revision copy, so this covers exactly the revision the job
        # will resolve later. `opensearch=None` (no client -- unit tests
        # exercising this resolver in isolation, same contract as
        # `resolve_effective_default`) skips the store lookup entirely,
        # same as before this fix -- there is nothing to validate against.
        from src.services.config_store import get_config_store
        from src.services.labeling.vlm_prompts import get_prompt_pack

        await get_config_store().ensure_fresh(opensearch)
        # `revision is not None` only when `requested` carried an explicit
        # `name@rev` (checked above), so `resolve_strategy_selection` was
        # called with a non-`None` `name_only` and, having not raised,
        # always returns that same non-`None` name.
        assert resolved is not None
        if get_prompt_pack(resolved, revision=revision) is None:
            # Nit: reworded off "unknown revision"
            # -- revision `1` of `resolved` can genuinely EXIST (it may
            # even be an earlier activation's `previous`), so that wording
            # is misleading. This process only ever resolves a per-run
            # pin against its own process-cached current doc plus the
            # store's currently-*activated*-revision pin (§3.7) -- not a
            # full historical lookup of every revision the name ever had.
            raise RequestRejectedError(
                status_code=422,
                detail={
                    'error': _UNRESOLVABLE_REVISION.format(revision=revision, resolved=resolved),
                    'message': _UNRESOLVABLE_REVISION.format(revision=revision, resolved=resolved),
                    'axis': 'prompt_pack',
                    'requested': requested,
                },
            )
    elif requested is not None and opensearch is not None:
        # an explicit per-run bare `name`
        # (no `@rev`) means "latest saved revision" per any_domain_plan.md
        # §3.7 ("Per-run `?prompt_pack=` ... accepts `name` (latest saved
        # revision)"), NOT the config-store's *active* revision --
        # that's the separate "Default (active) pack" behavior, which
        # applies only when `prompt_pack` is omitted entirely (`requested
        # is None` here, `revision` stays `None` and `get_vlm_labeler`
        # falls through to the pinned-active body, unchanged). Pin the
        # exact current revision now so the job resolves the same body
        # `get_vlm_labeler`/`get_prompt_pack(name, revision=N)` will
        # serve later, instead of letting `revision=None` ride through to
        # `get_prompt_pack`'s active-pack redirect (the pinning,
        # which must stay untouched for the omitted-param path).
        from src.services.config_store import get_config_store

        await get_config_store().ensure_fresh(opensearch)
        assert resolved is not None
        stored = get_config_store().current.packs.get(resolved)
        if stored is not None:
            revision = stored.revision
    # NOTE: `resolved` here still echoes the
    # active pack's NAME for the omitted-param case (`requested is None`)
    # -- several callers (job summaries, `/start`'s own response) read
    # this purely for display and must keep seeing it. The job/labeler
    # resolution side of the omitted-pack fix (never resolve the VLM
    # labeler against this echoed name once it can go stale) lives in
    # ``pipeline.py``'s ``prompt_pack_omitted`` plumbing instead, not
    # here -- see ``pipeline_auto_label_start`` and ``_run_auto_label``.
    return resolved, revision


def omitted_pack_is_store_active(resolved_name: str | None) -> bool:
    """True when ``resolved_name`` (the echo for an omitted per-run
    ``prompt_pack``) came from the config store's own activation, not a
    legacy settings-doc/env/file default.

    ``labeler_resolution_args``'s ``(None, None)`` re-resolution exists
    only to dodge the store-active-pack TOCTOU: the echoed name
    can go STALE between request time and the VLM stage actually running,
    once a *different* pack is activated. An env/file default
    has no such staleness -- nothing "activates" out
    from under it -- so forcing it through ``(None, None)`` instead just
    makes the job silently run ``active_prompt_pack()`` (the env/file
    default) while the summary/echo keeps reporting the settings-doc
    name: the job runs a DIFFERENT pack than the one it reports and the
    one ``8bed60f3`` ran. Scoping the omitted signal to "the echo really
    is the store's current active pack" keeps the TOCTOU fix for the one
    path that needs it and restores echo == run everywhere else.
    """
    from src.services.config_store import get_config_store

    ref = get_config_store().current.active_pack
    if ref is None or ref == 'off':
        return False
    return ref[0] == resolved_name


def labeler_resolution_args(
    prompt_pack: str | None, prompt_pack_revision: int | None, *, prompt_pack_omitted: bool
) -> tuple[str | None, int | None]:
    """What ``get_vlm_labeler`` must be called with for the VLM stage.

    ``(prompt_pack, prompt_pack_revision)`` are the RESOLVED echo values
    (kept for ``summary``/job-status display -- existing callers read
    those directly). When the original request omitted ``prompt_pack``
    entirely, the labeler must instead resolve against ``(None, None)``
    -- always "whatever is active right now" -- because the echoed name
    can go stale between request time and whenever the VLM stage actually
    runs (a window spanning the whole pre-VLM pipeline, not a tight
    race): once a different pack is activated, ``get_prompt_pack(name,
    revision=None)`` no longer redirects that OLD name to its pinned
    body and instead silently serves its un-activated CURRENT draft
    (B1, reachable again once a separate process's config-store
    snapshot is warm). Split out as its own function so it is
    directly unit-testable without driving the whole pipeline.
    """
    if prompt_pack_omitted:
        return None, None
    return prompt_pack, prompt_pack_revision
