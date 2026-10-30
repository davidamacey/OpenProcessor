"""acceptance_run.sh 0.7.0 phases against the fake HTTP server: health, clusters, label ops, GPU route, loop."""

from __future__ import annotations

import json

import pytest

from tests.release.test_acceptance_run import P, _happy_routes, by_name, run


def _health(models=None, triton_ok=True):
    services = {'triton': {'ok': triton_ok}}
    if models is not None:
        services['models'] = models
    return (200, {'status': 'ready', 'version': '1', 'services': services})


def test_health_fails_when_required_models_not_ok(fake, tmp_path):
    routes = _happy_routes()
    routes[('GET', '/health')] = _health({'ok': False, 'detail': 'yolo MISSING'})
    proc, data = run(fake(routes), tmp_path, '--only', 'health')
    assert proc.returncode == 1
    assert 'yolo MISSING' in json.dumps(by_name(data)['health']['evidence'])


def test_health_models_absent_only_passes_for_control_plane_runs(fake, tmp_path):
    routes = _happy_routes()
    routes[('GET', '/health')] = _health(None, triton_ok=False)
    proc, data = run(fake(routes), tmp_path, '--only', 'health')
    assert by_name(data)['health']['status'] == 'fail'
    proc, data = run(fake(routes), tmp_path, '--only', 'health', '--control-plane-only')
    assert by_name(data)['health']['status'] == 'pass', proc.stdout


@pytest.mark.parametrize('path', ['/live', '/'])
def test_health_requires_live_and_root(fake, tmp_path, path):
    routes = _happy_routes()
    routes[('GET', path)] = (503, {})
    _proc, data = run(fake(routes), tmp_path, '--only', 'health')
    assert by_name(data)['health']['status'] == 'fail'


def _card(cid, size, flags=(), **extra):
    return {
        'cluster_id': cid, 'size': size, 'flags': list(flags),
        'label_agreement': 0.9, **extra,
    }  # fmt: skip


def _cluster_routes(cards, facets_ok=True):
    """A fake clusters + crops pair whose counts agree (or, with facets_ok False, not)."""
    routes = _happy_routes()
    routes[('POST', f'{P}/classes/seed_from_detector')] = (200, {'created': []})
    routes[('POST', f'{P}/pipeline/auto_label')] = (200, {})
    routes[('GET', f'{P}/pipeline/auto_label/status')] = (200, {'status': 'idle'})

    def listing(_p, q):
        out = list(cards)
        if q.get('flag'):
            out = [c for c in out if q['flag'][0] in c['flags']]
        if q.get('sort') == ['size_asc']:
            out.sort(key=lambda c: c['size'])
        return 200, {
            'items': out, 'total': len(out), 'truncated': False,
            'sort': (q.get('sort') or ['size_desc'])[0],
            'sort_options': [{'id': 'size_desc'}, {'id': 'size_asc'}],
            'flag_vocabulary': [{'id': 'mixed'}],
        }  # fmt: skip

    def facets(_p):
        return 200, {
            'cluster_id': 1, 'total': 10,
            'validation': {'validated': 4, 'unvalidated': 6 if facets_ok else 5},
            'detector_classes': [{'value': 'car', 'count': 3}],
            'class_sources': [{'value': 'vlm', 'count': 7}],
            'review_tabs': {'all': 10},
        }  # fmt: skip

    def crops(_p, q):
        total = 10
        if q.get('label_validated') == ['true']:
            total = 4
        elif q.get('label_validated') == ['false']:
            total = 6
        elif q.get('detector_class_name'):
            total = 3
        elif q.get('class_source'):
            total = 7
        return 200, {'total': total, 'crops': []}

    routes[('GET', f'{P}/clusters')] = (200, listing)
    routes[('GET', f'{P}/clusters/1/facets')] = (200, facets)
    routes[('GET', f'{P}/crops')] = (200, crops)
    return routes, listing, crops


def _cluster_run(fake, tmp_path, cards, facets_ok=True):
    routes, _listing, _crops = _cluster_routes(cards, facets_ok)
    f = fake(routes)
    return run(f, tmp_path, '--only', 'cluster'), f


GOOD_CARDS = [_card(1, 30, ['mixed']), _card(2, 5), _card(3, 12, ['mixed'])]


def test_cluster_checks_sort_flag_and_facets(fake, tmp_path):
    (proc, data), f = _cluster_run(fake, tmp_path, GOOD_CARDS)
    ph = by_name(data)['cluster']
    assert ph['status'] == 'pass', proc.stdout + json.dumps(ph)
    ev = json.dumps(ph['evidence'])
    assert 'sort=size_asc served in order' in ev
    assert 'flag=mixed' in ev
    assert 'facets equal' in ev
    assert any('sort=size_asc' in r[1] for r in f.raw)
    assert any('flag=mixed' in r[1] for r in f.raw)


def test_cluster_fails_when_a_card_serves_label_purity(fake, tmp_path):
    cards = [*GOOD_CARDS, _card(4, 9, label_purity=0.5)]
    (_proc, data), _f = _cluster_run(fake, tmp_path, cards)
    assert by_name(data)['cluster']['status'] == 'fail'
    assert 'label_purity' in json.dumps(by_name(data)['cluster']['evidence'])


def test_cluster_fails_when_a_card_lacks_label_agreement(fake, tmp_path):
    cards = [*GOOD_CARDS, {'cluster_id': 4, 'size': 9, 'flags': []}]
    (_proc, data), _f = _cluster_run(fake, tmp_path, cards)
    assert by_name(data)['cluster']['status'] == 'fail'


def test_cluster_fails_when_facets_disagree_with_crops(fake, tmp_path):
    (_proc, data), _f = _cluster_run(fake, tmp_path, GOOD_CARDS, facets_ok=False)
    ph = by_name(data)['cluster']
    assert ph['status'] == 'fail'
    assert 'facets validation' in json.dumps(ph['evidence'])


def test_cluster_fails_when_sort_is_not_applied(fake, tmp_path):
    routes, listing, _crops = _cluster_routes(GOOD_CARDS)

    def unsorted(_p, q):  # ignores sort: size_asc comes back descending
        code, body = listing(_p, q)
        body['items'] = sorted(body['items'], key=lambda c: -c['size'])
        return code, body

    f = fake(routes)
    f.routes[('GET', f'{P}/clusters')] = (200, unsorted)
    _proc, data = run(f, tmp_path, '--only', 'cluster')
    assert by_name(data)['cluster']['status'] == 'fail'
    assert 'size_asc' in json.dumps(by_name(data)['cluster']['evidence'])


def _label_ops_world(*, undo_works=True, dry_run_writes=False, holdout_moves=False):
    """A small stateful project: 6 unvalidated crops, one holdout, one validated."""
    items = {
        f'c{i}': {'crop_id': f'c{i}', 'class_id': 0, 'label_validated': False,
                  'class_source': 'vlm', 'test_holdout': False}
        for i in range(6)
    }  # fmt: skip
    items['h'] = {'crop_id': 'h', 'class_id': 0, 'label_validated': False,
                  'class_source': 'vlm', 'test_holdout': True}  # fmt: skip
    items['v'] = {'crop_id': 'v', 'class_id': 0, 'label_validated': True,
                  'class_source': 'human', 'test_holdout': False}  # fmt: skip
    world = {'items': items, 'ops': [], 'undone': False, 'prior': {}}

    def crops(_p, q):
        if q.get('ids'):
            ids = q['ids'][0].split(',')
            return 200, {'crops': [items[i] for i in ids if i in items], 'total': len(ids)}
        rows = list(items.values())
        if q.get('label_validated') == ['false']:
            rows = [r for r in rows if not r['label_validated']]
        return 200, {'crops': rows, 'total': len(rows)}

    def batch(payload):
        body = json.loads(payload)
        world['prior'] = {i: dict(items[i]) for i in body['crop_ids']}
        for i in body['crop_ids']:
            items[i].update(class_id=body['class_id'], label_validated=True, class_source='human')
        if holdout_moves:
            items['h']['class_id'] = 5
        world['ops'].append('op1')
        return 200, {'updated': len(body['crop_ids']), 'op_id': 'op1', 'conflicts': []}

    def listing(_p):
        return 200, {'items': [{'op_id': 'op1', 'undoable': not world['undone'],
                                'undone': world['undone'], 'current': 4}] if world['ops'] else [],
                     'limit': 50, 'retention': 200}  # fmt: skip

    def undo(payload):
        dry = json.loads(payload).get('dry_run', True)
        if dry and not dry_run_writes:
            return 200, {'op_id': 'op1', 'dry_run': True, 'restored': len(world['prior']),
                         'matched': len(world['prior'])}  # fmt: skip
        if world['undone']:
            return 409, {'detail': {'error': 'already_undone'}}
        world['undone'] = True
        if undo_works:
            for i, old in world['prior'].items():
                items[i].update(old)
        return 200, {'op_id': 'op1', 'dry_run': dry, 'restored': len(world['prior']),
                     'kept_changed': 0, 'kept_holdout': 0}  # fmt: skip

    routes = _happy_routes()
    routes[('GET', f'{P}/classes')] = (200, {'classes': [{'class_id': 7}]})
    routes[('GET', f'{P}/crops')] = (200, crops)
    routes[('PUT', f'{P}/crops/batch_label')] = (200, batch)
    routes[('GET', f'{P}/label_ops')] = (200, listing)
    routes[('POST', f'{P}/label_ops/op1/undo')] = (200, undo)
    return routes, world


def _label_ops_run(fake, tmp_path, **kw):
    routes, world = _label_ops_world(**kw)
    f = fake(routes)
    proc, data = run(f, tmp_path, '--only', 'label_ops', '--label-ops-crops', '4')
    return proc, by_name(data)['label_ops'], f, world


def test_label_ops_batch_dry_run_undo_and_second_undo_409(fake, tmp_path):
    proc, ph, f, world = _label_ops_run(fake, tmp_path)
    assert ph['status'] == 'pass', proc.stdout + json.dumps(ph)
    undos = [json.loads(r[2]) for r in f.raw if r[1].endswith('/undo')]
    assert undos == [{}, {'dry_run': False}, {'dry_run': False}]
    assert world['items']['c0']['class_id'] == 0  # back
    assert world['items']['h']['class_id'] == 0
    assert world['items']['v']['class_id'] == 0
    assert 'already_undone' in json.dumps(ph['evidence'])


def test_label_ops_fails_when_dry_run_writes(fake, tmp_path):
    _proc, ph, _f, _w = _label_ops_run(fake, tmp_path, dry_run_writes=True)
    assert ph['status'] == 'fail'


def test_label_ops_fails_when_undo_does_not_restore(fake, tmp_path):
    _proc, ph, _f, _w = _label_ops_run(fake, tmp_path, undo_works=False)
    assert ph['status'] == 'fail'
    assert 'not back after undo' in json.dumps(ph['evidence'])


def test_label_ops_fails_when_holdout_item_changes(fake, tmp_path):
    _proc, ph, _f, _w = _label_ops_run(fake, tmp_path, holdout_moves=True)
    assert ph['status'] == 'fail'
    assert 'holdout' in json.dumps(ph['evidence'])


def test_label_ops_skips_when_route_absent(fake, tmp_path):
    routes = _happy_routes()  # GET label_ops -> 404
    _proc, data = run(fake(routes), tmp_path, '--only', 'label_ops')
    assert by_name(data)['label_ops']['status'] == 'skip'


def test_label_ops_requires_an_op_id(fake, tmp_path):
    routes, _world = _label_ops_world()
    routes[('PUT', f'{P}/crops/batch_label')] = (200, {'updated': 4, 'op_id': None})
    f = fake(routes)
    _proc, data = run(f, tmp_path, '--only', 'label_ops')
    assert by_name(data)['label_ops']['status'] == 'fail'


def _metrics(decode: str) -> str:
    base = (
        'op_ingest_images_total{project="acc-test",outcome="ok"} 3\nop_queue_depth{queue="embed"} 0\n'
        'op_embedding_state_items{project="acc-test",state="embedded"} 3\n'
    )
    return base + decode


def _metrics_run(fake, tmp_path, decode, triton_body=None, *extra):
    routes = _happy_routes()
    routes[('GET', '/metrics')] = (200, _metrics(decode))
    f = fake(routes)
    args = ['--only', 'metrics']
    if triton_body is not None:
        f.routes[('GET', '/triton')] = (200, triton_body)
        args += ['--triton-metrics-url', f.url + '/triton']
    _proc, data = run(f, tmp_path, *args, *extra)
    return by_name(data)['metrics']


STAGE = 'op_gpu_prep_stage_seconds{stage="decode"} 1\nop_gpu_prep_stage_observations{stage="decode"} 9\n'


def test_metrics_pass_when_decode_counters_absent_with_note(fake, tmp_path):
    ph = _metrics_run(fake, tmp_path, '')
    assert ph['status'] == 'pass'
    assert 'op_ingest_decode_total absent' in json.dumps(ph['evidence'])


def test_metrics_gpu_route_within_fallback_bound(fake, tmp_path):
    decode = 'op_ingest_decode_total{route="gpu"} 996\nop_ingest_decode_total{route="cpu_call_failed"} 4\nop_ingest_decode_total{route="cpu_small"} 50\n'
    ph = _metrics_run(fake, tmp_path, decode, STAGE)
    assert ph['status'] == 'pass', ph
    ev = json.dumps(ph['evidence'])
    assert 'gpu=996 fallbacks=4' in ev  # cpu_small is policy, not a fallback
    assert 'Triton metrics carry' in ev


def test_metrics_fail_when_fallbacks_exceed_bound(fake, tmp_path):
    decode = 'op_ingest_decode_total{route="gpu"} 90\nop_ingest_decode_total{route="cpu_call_failed"} 10\n'
    ph = _metrics_run(fake, tmp_path, decode, STAGE)
    assert ph['status'] == 'fail'
    assert 'exceed' in json.dumps(ph['evidence'])


def test_metrics_fallback_bound_is_configurable(fake, tmp_path):
    decode = 'op_ingest_decode_total{route="gpu"} 90\nop_ingest_decode_total{route="cpu_call_failed"} 10\n'
    ph = _metrics_run(fake, tmp_path, decode, STAGE, '--decode-fallback-max', '15')
    assert ph['status'] == 'pass', ph


def test_metrics_fail_when_stage_counters_missing_on_active_gpu_route(fake, tmp_path):
    decode = 'op_ingest_decode_total{route="gpu"} 100\n'
    ph = _metrics_run(fake, tmp_path, decode, 'nv_inference_count 1\n')
    assert ph['status'] == 'fail'
    assert 'op_gpu_prep_stage_seconds' in json.dumps(ph['evidence'])


def test_metrics_cpu_only_decode_skips_gpu_checks(fake, tmp_path):
    decode = 'op_ingest_decode_total{route="cpu_small"} 100\n'
    ph = _metrics_run(fake, tmp_path, decode, 'nothing 1\n')
    assert ph['status'] == 'pass'
    assert 'GPU decode route took no images' in json.dumps(ph['evidence'])


def _loop_routes(*, loops_status=200, vlm='env'):
    routes = _happy_routes()
    routes[('GET', f'{P}/vlm/endpoints/active')] = (200, {'active': {'name': vlm}})

    def post(_p):
        return 200, {
            'loop_id': None,
            'status': 'dry_run',
            'request': {},
            'plan': {'fits_budget': True},
        }

    routes[('GET', f'{P}/assist/loops')] = (200, {'loops': []})
    if loops_status != 200:
        routes[('GET', f'{P}/assist/loops')] = (loops_status, {})
    routes[('POST', f'{P}/assist/loops')] = (200, post)
    return routes


def test_loop_dry_run_passes_and_queues_nothing(fake, tmp_path):
    f = fake(_loop_routes())
    _proc, data = run(f, tmp_path, '--only', 'loop')
    ph = by_name(data)['loop']
    assert ph['status'] == 'pass', ph
    body = json.loads(
        next(r for r in f.raw if r[0] == 'POST' and r[1].endswith('/assist/loops'))[2]
    )
    assert body['budget']['max_vlm_crops'] > 0
    assert 'dry_run' not in body or body['dry_run'] is not False


def test_loop_skips_when_routes_404_or_no_vlm(fake, tmp_path):
    _proc, data = run(fake(_loop_routes(loops_status=404)), tmp_path, '--only', 'loop')
    assert by_name(data)['loop']['status'] == 'skip'
    _proc, data = run(fake(_loop_routes(vlm=None)), tmp_path, '--only', 'loop')
    assert by_name(data)['loop']['status'] == 'skip'


def test_loop_fails_when_dry_run_returns_a_loop_id(fake, tmp_path):
    routes = _loop_routes()
    routes[('POST', f'{P}/assist/loops')] = (
        200, {'loop_id': 'L1', 'status': 'queued', 'request': {}, 'plan': {}}
    )  # fmt: skip
    _proc, data = run(fake(routes), tmp_path, '--only', 'loop')
    assert by_name(data)['loop']['status'] == 'fail'
