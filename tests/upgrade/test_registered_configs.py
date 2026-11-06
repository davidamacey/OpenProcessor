"""Built-ins a 0.6.0 project activated by name alone load as stored revisions; an unresolvable one
stops the upgrade."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from src.services.config_store.activation_apply import registered_for
from src.services.labeling.vlm_prompts import GENERIC_REGION_PACK
from src.services.upgrade.from_060.model import UnresolvedBuiltinError


def rows(pg, sql: str):
    with pg.engine('super').connect() as conn:
        return conn.execute(text(sql)).all()


@pytest.mark.db
@pytest.mark.asyncio
async def test_a_builtin_pack_and_the_env_region_profile_load_as_stored_revisions(
    make_runner, clean_db
) -> None:
    await make_runner().run()
    got = rows(
        clean_db,
        'SELECT a.axis, d.kind, d.name, d.project_id = p.id, a.revision, r.created_by, r.description, r.body '
        'FROM config_activations a JOIN projects p ON p.id = a.project_id '
        'JOIN config_documents d ON d.id = a.config_id '
        'JOIN config_revisions r ON r.config_id = d.id AND r.revision = a.revision '
        "WHERE p.slug = 'lean' ORDER BY a.axis",
    )
    pack = registered_for('prompt_pack', GENERIC_REGION_PACK.name)
    profile = registered_for('detection_profile', 'region')
    assert pack is not None
    assert profile is not None
    assert [(g[0], g[1], g[2], g[3], g[4], g[5]) for g in got] == [
        ('detection_profile', 'region_profile', 'region', True, 1, 'upgrade'),
        ('prompt_pack', 'prompt_pack', GENERIC_REGION_PACK.name, True, 1, 'upgrade'),
    ]
    assert got[0][6:] == (profile.description, profile.body)
    assert got[1][6:] == (pack.description, pack.body)
    # each choice is also an event pointing at the stored revision
    assert rows(
        clean_db,
        'SELECT e.axis, e.to_revision FROM config_activation_events e JOIN projects p ON p.id = e.project_id '
        "WHERE p.slug = 'lean' ORDER BY e.axis",
    ) == [('detection_profile', 1), ('prompt_pack', 1)]


@pytest.mark.db
@pytest.mark.asyncio
async def test_an_unresolvable_builtin_stops_the_upgrade_naming_it(
    make_runner, clean_db, v060
) -> None:
    configs = v060.os.docs('op_prj_lean__configs')
    configs['activation:prompt_pack']['name'] = 'ghost_pack'
    configs['activation_event:00000000-0000-4000-8000-000000000008']['name'] = 'ghost_pack'
    with pytest.raises(UnresolvedBuiltinError, match=r"'ghost_pack'.*prompt_pack"):
        await make_runner().run()
    assert rows(clean_db, 'SELECT count(*) FROM projects') == [(0,)]
