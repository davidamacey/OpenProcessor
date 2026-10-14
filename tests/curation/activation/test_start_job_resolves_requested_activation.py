"""POST /pipeline/auto_label/start resolves the prompt pack once at request time; the job
runs exactly what the request resolved, even in a separate worker process."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from curation.activation.constants import PREFIX, pack_body
from src.services.curation.auto_label_run import run_auto_label


pytestmark = pytest.mark.usefixtures('vlm_env')


def test_start_job_args_are_accepted_by_pipeline_fn():
    """The worker calls pipeline_fn(opensearch=, progress=, **trigger_args);
    /pipeline/auto_label/start puts 'prompt_pack_revision' in those args.

    The worker's target is ``run_auto_label``, not the public
    ``pipeline_auto_label`` route wrapper: the route does not accept these
    params at all (they would otherwise be reachable from an HTTP request)."""
    import inspect

    from src.services.curation.auto_label_run import run_auto_label

    assert 'prompt_pack_revision' in inspect.signature(run_auto_label).parameters


def test_job_keeps_request_pin_without_reresolving(app_client, monkeypatch):
    """A bare per-run name resolves to the LATEST saved revision by design, so
    re-running the resolver on a bare name is *supposed* to move forward. The bug
    to guard against is ``pipeline_auto_label`` (the job body) calling the
    resolver a second time at all once ``/start`` had already resolved
    and pinned ``(name, revision)``. So: with
    ``prompt_pack_revision`` supplied (exactly as ``/start`` -> the
    worker -> ``pipeline_fn(**trigger_args)`` supplies it),
    ``pipeline_auto_label`` must use the pin as-is and must NOT call
    ``resolve_run_prompt_pack`` again."""
    from curation.test_pipeline import _FakeOpenSearch
    from src.services.curation import auto_label_run
    from src.services.curation.autolabel.run_params import resolve_run_prompt_pack

    c = app_client
    assert c.post(PREFIX, json={'name': 'my_pack', 'body': pack_body('A')}).status_code == 201
    assert c.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(
            f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': pack_body('B')}
        ).status_code
        == 200
    )

    # What /start does at request time: resolve-and-pin once.
    name, revision = asyncio.run(resolve_run_prompt_pack(c.fake_os, 'my_pack@1'))
    assert (name, revision) == ('my_pack', 1)

    def _must_not_be_called(*_a, **_kw):  # pragma: no cover - failure path
        raise AssertionError(
            'pipeline_auto_label re-resolved a pinned prompt_pack instead of using the pin'
        )

    monkeypatch.setattr(auto_label_run, 'resolve_run_prompt_pack', _must_not_be_called)
    fake_os = _FakeOpenSearch({})
    summary = asyncio.run(
        run_auto_label(
            opensearch=fake_os,
            train_clusters=False,
            promote_min_purity=0.85,
            promote_min_members=4,
            vlm_batch_size=32,
            vlm_concurrency=8,
            max_vlm_crops=0,
            classifier_confidence_skip_vlm=0.80,
            clustering_method=None,
            run_vlm=False,
            recluster_unvalidated=False,
            reassign_only=False,
            run_auto_promote=False,
            gate_max_rank=None,
            gate_min_blur_ratio=None,
            n_clusters=None,
            class_id=None,
            cluster_id=None,
            prompt_pack=name,
            prompt_pack_revision=revision,
            # "Already resolved" is its own explicit flag, not inferred from
            # `prompt_pack_revision is not None`: `/start` sets it unconditionally
            # (pinned, bare-name, AND omitted alike), as a real trigger dict carries it.
            prompt_pack_resolved=True,
        )
    )
    print('pinned job summary ->', summary['prompt_pack'], summary['prompt_pack_revision'])
    assert summary['prompt_pack'] == 'my_pack'


def _run_job(c, monkeypatch, query):
    from curation.test_pipeline import _FakeOpenSearch
    from src.services.curation import auto_label_run
    from src.services.curation.autolabel import job as auto_label_job
    from src.services.curation.autolabel.run_params import resolve_run_prompt_pack as real_resolve

    captured = {}

    def fake_start(fn, args):
        captured.update(args)
        return {'job_id': 'x'}

    monkeypatch.setattr(auto_label_job, 'start_job', fake_start)
    r = c.post(f'/curation/projects/default/pipeline/auto_label/start{query}')
    assert r.status_code in (200, 202), r.text

    async def _resolve(_os, pp):
        return await real_resolve(c.fake_os, pp)

    monkeypatch.setattr(auto_label_run, 'resolve_run_prompt_pack', _resolve)
    args = {k: v for k, v in captured.items() if k != 'opensearch'}
    args['train_clusters'] = False
    args['run_vlm'] = False
    summary = asyncio.run(run_auto_label(opensearch=_FakeOpenSearch({}), **args))
    return captured, summary


@pytest.mark.parametrize(
    ('query', 'want_rev'),
    [('', 1), ('?prompt_pack=my_pack@1', 1), ('?prompt_pack=my_pack', 2)],
)
def test_start_job_runs_what_the_request_resolved(app_client, monkeypatch, query, want_rev):
    from src.services.labeling.run_labeler import get_vlm_labeler
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    c = app_client
    assert c.post(PREFIX, json={'name': 'my_pack', 'body': pack_body('A')}).status_code == 201
    assert c.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(
            f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': pack_body('B DRAFT')}
        ).status_code
        == 200
    )
    captured, summary = _run_job(c, monkeypatch, query)
    lab = get_vlm_labeler(summary['prompt_pack'], summary['prompt_pack_revision'])
    stamp = prompt_pack_stamp(lab._pack, revision=summary['prompt_pack_revision'])
    print(
        repr(query),
        'trigger',
        (captured['prompt_pack'], captured['prompt_pack_revision']),
        '| job',
        (summary['prompt_pack'], summary['prompt_pack_revision']),
        '| serves',
        lab._pack.class_system,
        stamp,
    )
    want_body = 'A' if want_rev == 1 else 'B DRAFT'
    assert lab._pack.class_system == want_body
    assert stamp == f'my_pack@{want_rev}'


def _start_args(c, monkeypatch, query):
    from src.services.curation.autolabel import job as auto_label_job

    captured: dict[str, Any] = {}

    def fake_start(fn, args):
        captured.update(args)
        return {'job_id': 'x'}

    monkeypatch.setattr(auto_label_job, 'start_job', fake_start)
    r = c.post(f'/curation/projects/default/pipeline/auto_label/start{query}')
    assert r.status_code in (200, 202), r.text
    return {k: v for k, v in captured.items() if k != 'opensearch'}


def _pack_setup(c):
    assert c.post(PREFIX, json={'name': 'my_pack', 'body': pack_body('A')}).status_code == 201
    assert c.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code == 200
    assert (
        c.put(
            f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': pack_body('B DRAFT')}
        ).status_code
        == 200
    )


@pytest.mark.parametrize(('query', 'want'), [('', 'A'), ('?prompt_pack=my_pack@1', 'A')])
def test_worker_process_with_cold_store_serves_requested_pack(app_client, monkeypatch, query, want):
    """auto_label_worker is a separate container: its config store starts
    empty and nothing on the job path refreshes it once
    prompt_pack_resolved=True skips resolve_run_prompt_pack.

    Uses ``c.fake_os`` (not a fresh, unrelated fake) as the job's
    ``opensearch`` client -- this is the one piece of state that IS
    shared with a real cross-process worker (the same OpenSearch
    cluster); a genuinely different, empty backing store would never be
    able to resolve the pack regardless of any fix and would not be
    testing what this is about (a stale in-memory snapshot, not a
    disconnected client)."""
    from src.services.config_store import get_config_store
    from src.services.config_store.store import reset_config_stores
    from src.services.labeling.run_labeler import get_vlm_labeler

    c = app_client
    _pack_setup(c)
    args = _start_args(c, monkeypatch, query)
    reset_config_stores()  # a different process: cold store
    args['train_clusters'] = False
    args['run_vlm'] = False
    summary = asyncio.run(run_auto_label(opensearch=c.fake_os, **args))
    print('store loaded after job stages:', get_config_store().current.loaded_at)
    # Exactly what `run_auto_label`'s VLM stage does: `prompt_pack_omitted`, not the
    # echoed `summary['prompt_pack']`, decides whether the labeler resolves against
    # `None`/`None`.
    omitted = args.get('prompt_pack_omitted', False)
    try:
        pack_arg = None if omitted else summary['prompt_pack']
        rev_arg = None if omitted else summary['prompt_pack_revision']
        lab = get_vlm_labeler(pack_arg, rev_arg)
        got = lab._pack.class_system
    except Exception as exc:
        got = repr(exc)
    print(repr(query), args['prompt_pack'], args['prompt_pack_revision'], omitted, '->', got)
    assert got == want


def test_omitted_job_after_active_switch_never_serves_draft(app_client, monkeypatch):
    """/start omitted (my_pack@1 active, r2 draft). Before the VLM stage the
    operator activates another pack. The job must serve either the pin
    (A) or the new active (OTHER), never my_pack's un-activated draft."""
    from src.services.labeling.run_labeler import get_vlm_labeler

    c = app_client
    _pack_setup(c)
    assert c.post(PREFIX, json={'name': 'other', 'body': pack_body('OTHER')}).status_code == 201
    args = _start_args(c, monkeypatch, '')
    r = c.post(
        f'{PREFIX}/other/activate', json={'expected_active': {'name': 'my_pack', 'revision': 1}}
    )
    assert r.status_code == 200, r.text
    # The job resolves against `None`/`None` when `prompt_pack_omitted` is set, exactly
    # like `run_auto_label`'s VLM stage, not the echoed `(prompt_pack, prompt_pack_revision)`.
    assert args['prompt_pack_omitted'] is True
    lab = get_vlm_labeler(None, None)
    print(
        'trigger', args['prompt_pack'], args['prompt_pack_revision'], '->', lab._pack.class_system
    )
    assert lab._pack.class_system in ('A', 'OTHER')


def test_hidden_resolved_flag_is_not_client_settable(app_client):
    """A direct POST /pipeline/auto_label caller can set the hidden flag and
    skip the unknown-pack 422."""
    c = app_client
    r = c.post(
        '/curation/projects/default/pipeline/auto_label',
        params={'prompt_pack': 'nope', 'prompt_pack_resolved': 'true', 'train_clusters': 'false'},
    )
    print('direct+resolved flag ->', r.status_code, r.text[:200])
    assert r.status_code == 422
