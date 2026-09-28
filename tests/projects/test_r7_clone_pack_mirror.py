"""R7-4 fix (Major, pre-existing, W3/W4 round-7 review): mirror of R6-2.

R6-2 fixed clone skipping PROFILE validation when the source has no real
stored profile (falls back to the target's own default profile, unchecked
against the pack that WILL land). This is the mirror: clone correctly
copies the PROFILE, but used to skip PACK validation when the source's
pack is an env/file default (not a real stored revision) -- so the
target ends up running its OWN env-default pack under the CLONED
profile, with nobody checking that pairing.

Deployment here: ``OP_PROMPT_PACK_PATH`` (target's env/file default) is a
multi-box-STRIPPED pack (no multi-box reply-key support);
``OP_PROMPT_PACK_PATHS`` has a full pack ``filegood``. Source has
``filegood`` active as its env/file default (``revision=None`` -- never
written to the store, so it will NOT be cloned) plus a real STORED
3-region profile ``mp@1`` (WILL be cloned). That pairing is valid in the
source (filegood supports multi-box). It is NOT valid once ``mp`` lands
paired with the TARGET's own env-default pack (``envsingle``, stripped)
instead of ``filegood`` -- exactly the gap
``check_activation_pair_in_target_context``'s ``profile_will_be_cloned``
branch used to miss (its ``pending_sibling`` was the SOURCE's pack body,
not what actually lands in the target).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from curation.test_r4_probes import _body, _stripped
from curation.test_r5_probes import ready_state  # noqa: F401
from curation.test_r6_probes import _live_pair, multi_default_registry  # noqa: F401


pytestmark = pytest.mark.unbound


def test_r7_clone_rejects_profile_with_targets_own_stripped_pack(
    ready_state,  # noqa: F811 - pytest fixture param shadows the cross-module import
    multi_default_registry,  # noqa: F811 - pytest fixture param shadows the cross-module import
):
    from curation._fake_config_opensearch import FakeConfigOpenSearch

    import src.config.curation as curation_mod
    from src.config.curation import base_curation_config
    from src.config.project_context import bind_project
    from src.config.projects import ProjectRecord, resources_for_new
    from src.services.config_store import get_config_store
    from src.services.config_store.index import activate, save_config
    from src.services.config_store.store import reset_config_stores

    class _Fake(FakeConfigOpenSearch):
        async def count(self, index, body=None):  # noqa: ARG002
            return {'count': 0}

        async def bulk(self, body, refresh=False):  # noqa: ARG002
            return {'errors': False, 'items': []}

    def _rec(slug):
        now = datetime.now(UTC).isoformat()
        return ProjectRecord(
            slug=slug,
            display_name=slug,
            description='',
            status='active',
            revision=1,
            created_at=now,
            updated_at=now,
            origin=None,
            resources=resources_for_new(slug, base_curation_config()),
        )

    with tempfile.TemporaryDirectory() as tmp:
        single = {**_stripped(), 'name': 'envsingle'}
        good = {**_body('FILEGOOD'), 'name': 'filegood'}
        with open(f'{tmp}/single.json', 'w') as f:
            f.write(json.dumps(single))
        with open(f'{tmp}/good.json', 'w') as f:
            f.write(json.dumps(good))
        keys = (
            'OP_STATE_DIR',
            'OP_PROJECTS_DATA_ROOT',
            'OP_PROMPT_PACK_PATH',
            'OP_PROMPT_PACK_PATHS',
        )
        old = {k: os.environ.get(k) for k in keys}
        os.environ['OP_STATE_DIR'] = f'{tmp}/state'
        os.environ['OP_PROJECTS_DATA_ROOT'] = f'{tmp}/pd'
        os.environ['OP_PROMPT_PACK_PATH'] = f'{tmp}/single.json'
        os.environ['OP_PROMPT_PACK_PATHS'] = f'{tmp}/good.json'
        curation_mod._default_curation_config = None
        reset_config_stores()
        try:
            from src.services.labeling.vlm_prompts import resolve_prompt_pack

            assert resolve_prompt_pack().name == 'envsingle'
            client = _Fake()
            source, target = _rec('r7alpha'), _rec('r7beta')
            with bind_project(source):
                from src.config import get_curation_config

                idx = get_curation_config().configs_index
                asyncio.run(
                    save_config(
                        client,
                        idx,
                        kind='region_profile',
                        name='mp',
                        body={
                            'detector_model': 'wheel_detector',
                            'text_reader': 'none',
                            'max_regions_per_item': 3,
                        },
                        expected_revision=None,
                    )
                )
                # Source's pack is the env/file default (filegood) --
                # `revision=None` is `activate()`'s own invariant for
                # "never written to the store" -- this axis will NOT be
                # cloned (`pack_will_be_cloned` is False).
                asyncio.run(
                    activate(
                        client,
                        idx,
                        axis='prompt_pack',
                        name='filegood',
                        revision=None,
                        expected_active=None,
                    )
                )
                # The profile IS a real stored activation -- WILL be
                # cloned.
                asyncio.run(
                    activate(
                        client,
                        idx,
                        axis='detection_profile',
                        name='mp',
                        revision=1,
                        expected_active=None,
                    )
                )
                asyncio.run(get_config_store().refresh(client))
                sp, sprof = _live_pair()
                assert sp.name == 'filegood'
                assert sp.class_system != 'STRIPPED'
                assert sprof is not None
                assert sprof.max_regions_per_item == 3

            from src.services.projects import lifecycle as lm, registry as rm
            from src.services.projects.clone import clone_settings_into

            saved = (lm._resolve_existing, lm._get_mutable_record, lm.write_record)
            saved_reg = rm.get_project_registry
            lm._resolve_existing = AsyncMock(return_value=source)
            lm._get_mutable_record = AsyncMock(return_value=(target, 1, 1))
            lm.write_record = AsyncMock(return_value=None)
            reg = MagicMock()
            reg.ensure_fresh = AsyncMock(return_value=None)
            rm.get_project_registry = lambda: reg
            outcome = 'ok'
            try:
                asyncio.run(
                    clone_settings_into(
                        client,
                        slug=target.slug,
                        from_slug=source.slug,
                        axes=None,
                        expected_revision=1,
                    )
                )
            except Exception as exc:
                outcome = repr(exc)[:400]
            finally:
                lm._resolve_existing, lm._get_mutable_record, lm.write_record = saved
                rm.get_project_registry = saved_reg

            # R7-4 fix: the gate must reject this clone -- pairing 'mp'
            # (3 regions) with the TARGET's own stripped pack
            # (`envsingle`, no multi-box support) fails
            # `check_multi_region_keys`. Before the fix, `pending_
            # sibling` was the SOURCE's `filegood` body (valid), so this
            # landed a 200 and copied `mp` into the target's store --
            # exactly the bug this test guards against.
            assert outcome != 'ok', f'clone should have been rejected, got: {outcome}'
            assert '422' in outcome or 'HTTPException' in outcome
            assert 'cannot clone active detection_profile' in outcome

            # And the store itself was never mutated -- no partial write
            # landed the profile axis before the rejection.
            with bind_project(target):
                asyncio.run(get_config_store().refresh(client))
                assert get_config_store().current.active_profile is None
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            curation_mod._default_curation_config = None
            reset_config_stores()
