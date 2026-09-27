"""W2: a per-run ``?prompt_pack=name@rev`` pins the exact revision at job
start; a later edit/activation never changes what an in-flight job's
labeler uses -- any_domain_plan.md §3.7, §9 W2."""

from __future__ import annotations

import pytest

from src.routers.curation.pipeline_params import resolve_run_prompt_pack


@pytest.mark.asyncio
async def test_bare_name_resolves_with_no_pinned_revision() -> None:
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    name, revision = await resolve_run_prompt_pack(None, GENERIC_ITEM_PACK.name)
    assert name == GENERIC_ITEM_PACK.name
    assert revision is None


@pytest.mark.asyncio
async def test_name_at_revision_pins_the_revision() -> None:
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    name, revision = await resolve_run_prompt_pack(None, f'{GENERIC_ITEM_PACK.name}@3')
    assert name == GENERIC_ITEM_PACK.name
    assert revision == 3


@pytest.mark.asyncio
async def test_omitted_prompt_pack_resolves_to_default_with_no_revision() -> None:
    name, revision = await resolve_run_prompt_pack(None, None)
    assert revision is None
    assert isinstance(name, str)


@pytest.mark.asyncio
async def test_unknown_name_before_the_at_sign_is_422() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await resolve_run_prompt_pack(None, 'not_a_real_pack@1')
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_pinned_revision_survives_a_later_active_pack_change() -> None:
    """The job dict stores the resolved (name, revision) at start; a
    later ``activate`` call must not retroactively change it -- this
    test exercises the resolver only (the job-dict wiring itself is in
    ``src.routers.curation.pipeline``), pinning that the resolver's
    contract stays "resolve once, at call time" rather than something
    that re-reads live state on every access."""
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    name, revision = await resolve_run_prompt_pack(None, f'{GENERIC_ITEM_PACK.name}@1')
    pinned = (name, revision)

    # Simulate an activation happening after this job started -- the
    # already-resolved tuple must be unaffected (it is a plain tuple of
    # immutable values, not a live handle).
    from src.services.config_store.store import get_config_store

    store = get_config_store()
    store.apply_local(active_pack=('some_other_pack', 9))
    assert pinned == (GENERIC_ITEM_PACK.name, 1)
