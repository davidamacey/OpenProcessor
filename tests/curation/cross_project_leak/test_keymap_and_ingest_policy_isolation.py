"""Config-store isolation gaps: ingest policy, keymap and region actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi.testclient import TestClient

from curation.cross_project_leak._constants import SCOPED


if TYPE_CHECKING:
    import pytest

    from curation.cross_project_leak._transport import LeakEnv


# --------------------------------------------------------------------------
# Config-store isolation gaps, through the real leak_env guard.
# --------------------------------------------------------------------------


def test_ingest_policy_alpha_invisible_to_beta(leak_env: LeakEnv) -> None:
    """Alpha's ingest policy is never read from beta, and a beta write does not
    advance alpha's revision."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    alpha = f'{SCOPED.format(project="alpha")}/ingest/policy'
    beta = f'{SCOPED.format(project="beta")}/ingest/policy'

    put = client.put(
        alpha,
        json={
            'expected_revision': 0,
            'detect': {'exclude_classes': ['person']},
            'embedding': {'mode': 'lazy'},
        },
    )
    assert put.status_code == 200, put.text

    beta_get = client.get(beta).json()
    assert beta_get['revision'] == 0
    assert beta_get['embedding']['mode'] == 'all'
    assert beta_get['detect']['exclude_classes'] == []
    assert client.get(alpha).json()['embedding']['mode'] == 'lazy'

    stale = client.put(alpha, json={'expected_revision': 0, 'embedding': {'mode': 'all'}})
    assert stale.status_code == 409
    assert client.put(beta, json={'expected_revision': 0}).status_code == 200
    assert client.get(alpha).json()['revision'] == 1


def test_keymap_alpha_invisible_to_beta(leak_env: LeakEnv) -> None:
    """M7 isolation gap: alpha's keymap override never leaks into beta's
    GET, and the reserved letter it introduces is reserved in alpha only."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    alpha = f'{SCOPED.format(project="alpha")}/keymap'
    beta = f'{SCOPED.format(project="beta")}/keymap'

    put = client.put(alpha, json={'expected_revision': 0, 'overrides': {'cluster.ignore': ['k']}})
    assert put.status_code == 200, put.text

    beta_get = client.get(beta).json()
    assert beta_get['is_default'] is True
    assert beta_get['overrides'] == {}
    assert 'k' not in beta_get['reserved_hotkeys']

    alpha_get = client.get(alpha).json()
    assert alpha_get['is_default'] is False
    assert 'k' in alpha_get['reserved_hotkeys']


def test_keymap_class_hotkey_checks_use_only_the_bound_registry(leak_env: LeakEnv) -> None:
    """M7 isolation gap: a keymap write against a class hotkey only 409s
    against the *bound* project's class -- beta's class 'q' never blocks
    alpha's write, and alpha's write never blocks beta's later one."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    alpha = SCOPED.format(project='alpha')
    beta = SCOPED.format(project='beta')

    bound_q = client.put(f'{beta}/classes/2', json={'hotkey_letter': 'q'})
    assert bound_q.status_code == 200, bound_q.text

    alpha_put = client.put(
        f'{alpha}/keymap', json={'expected_revision': 0, 'overrides': {'cluster.move': ['q']}}
    )
    assert alpha_put.status_code == 200, alpha_put.text

    beta_put = client.put(
        f'{beta}/keymap', json={'expected_revision': 0, 'overrides': {'cluster.move': ['q']}}
    )
    assert beta_put.status_code == 409, beta_put.text
    assert beta_put.json()['detail']['error'] == 'class_hotkey_conflict'


def test_put_keymap_unbind_rolls_back_class_hotkey_if_save_fails(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: if the keymap save fails after the class-hotkey unbind already
    wrote, the class hotkey must be rolled back, not left cleared. Seen
    red on the pre-fix code (409 revision_conflict + hotkey_letter -> None)."""
    from src.routers.curation import keymap as keymap_route_module
    from src.services.curation.keymap import RevisionConflictError

    client = TestClient(leak_env.app, raise_server_exceptions=False)
    beta = SCOPED.format(project='beta')

    bound = client.put(f'{beta}/classes/2', json={'hotkey_letter': 'i'})
    assert bound.status_code == 200, bound.text

    real_save = keymap_route_module.save_keymap_doc

    async def _fail_save(*_args: Any, **_kwargs: Any) -> Any:
        raise RevisionConflictError(current_revision=999)

    monkeypatch.setattr(keymap_route_module, 'save_keymap_doc', _fail_save)
    try:
        resp = client.put(
            f'{beta}/keymap',
            json={
                'expected_revision': 0,
                'overrides': {'cluster.ignore': ['i']},
                'unbind_conflicting_class_hotkeys': True,
            },
        )
    finally:
        monkeypatch.setattr(keymap_route_module, 'save_keymap_doc', real_save)

    assert resp.status_code == 409, resp.text
    assert resp.json()['detail']['error'] == 'revision_conflict'

    entry = client.get(f'{beta}/classes/2').json()
    assert entry['hotkey_letter'] == 'i', 'the class hotkey must survive a failed keymap save'


def test_clone_keymap_axis_all_or_nothing_on_kept_override_collision(leak_env: LeakEnv) -> None:
    """B2 probe 1: alpha has {cluster.ignore:['m'], cluster.move:['q']};
    beta has a class on 'q'. cluster.move is dropped for the conflict, but
    writing {cluster.ignore:['m']} alone would collide with cluster.move's
    default 'm' -- the whole clone must be a no-op, not a partial write
    that leaves an invalid keymap."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    alpha = SCOPED.format(project='alpha')
    beta = SCOPED.format(project='beta')

    put = client.put(
        f'{alpha}/keymap',
        json={
            'expected_revision': 0,
            'overrides': {'cluster.ignore': ['m'], 'cluster.move': ['q']},
        },
    )
    assert put.status_code == 200, put.text

    bound_q = client.put(f'{beta}/classes/2', json={'hotkey_letter': 'q'})
    assert bound_q.status_code == 200, bound_q.text

    before = client.get(f'{beta}/keymap').json()
    project_record = client.get(beta).json()

    resp = client.post(
        f'{beta}/clone_settings',
        json={
            'from': 'alpha',
            'axes': ['keymap'],
            'expected_revision': project_record['revision'],
        },
    )
    # clone_settings acts through the global (non-project-scoped) router
    # under the project path, per src/routers/curation/projects.py.
    assert resp.status_code == 200, resp.text
    conflicts = resp.json().get('keymap_clone_conflicts', [])
    assert any(c['action_id'] == 'cluster.move' for c in conflicts)

    after = client.get(f'{beta}/keymap').json()
    assert after['overrides'] == before['overrides'] == {}
    assert after['revision'] == before['revision']

    # The persisted state (nothing changed) must still validate clean.
    validate = client.post(f'{beta}/keymap/validate', json={'overrides': after['overrides']})
    assert validate.json()['ok'] is True


def test_clone_keymap_axis_all_or_nothing_on_defaults_source_class_clash(
    leak_env: LeakEnv,
) -> None:
    """B2 probe 2: the source is at defaults ({}); the target moved
    cluster.ignore off its default 'x' to free that letter for a class.
    Cloning an empty override map must not silently reintroduce 'x' on
    cluster.ignore over the class -- the clone reports the conflict and
    leaves the target's keymap unchanged."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    beta = SCOPED.format(project='beta')

    # 'x' is the default for both cluster.ignore *and* clusters_search.ignore
    # (different, non-overlapping active sets) -- both must move to free
    # the letter deployment-wide.
    moved = client.put(
        f'{beta}/keymap',
        json={
            'expected_revision': 0,
            'overrides': {'cluster.ignore': ['k'], 'clusters_search.ignore': ['k']},
        },
    )
    assert moved.status_code == 200, moved.text

    bound_x = client.put(f'{beta}/classes/2', json={'hotkey_letter': 'x'})
    assert bound_x.status_code == 200, bound_x.text

    before = client.get(f'{beta}/keymap').json()
    assert before['overrides'] == {'cluster.ignore': ['k'], 'clusters_search.ignore': ['k']}
    project_record = client.get(beta).json()

    resp = client.post(
        f'{beta}/clone_settings',
        json={
            'from': 'alpha',
            'axes': ['keymap'],
            'expected_revision': project_record['revision'],
        },
    )
    assert resp.status_code == 200, resp.text

    after = client.get(f'{beta}/keymap').json()
    assert after['overrides'] == before['overrides'], (
        "cloning alpha's empty overrides must not reintroduce cluster.ignore's "
        "default 'x' over beta's class"
    )


def test_region_actions_available_with_env_registered_profile(
    leak_env: LeakEnv, reference_region_profile: None
) -> None:
    """B3: with no config-store activation but an env-registered default
    region profile (get_active_region_profile() falls back to it), every
    review.region.*/box_edit.* action must be available."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    body = client.get(f'{SCOPED.format(project="beta")}/keymap').json()
    by_id = {a['id']: a for a in body['actions']}
    assert by_id['review.region.accept_box']['available'] is True
    assert by_id['box_edit.next_box']['available'] is True
