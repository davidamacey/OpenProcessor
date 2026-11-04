"""The dataset_import and combine acceptance phases against a fake HTTP server."""

from __future__ import annotations

from tests.release.test_acceptance_run import SLUG, P, _happy_routes, by_name, run


def _classes(*names: str, deprecated: bool = False) -> dict:
    return {
        'classes': [
            {'class_id': i, 'class_name': n, 'deprecated': deprecated, 'hotkey_letter': None}
            for i, n in enumerate(names)
        ]
    }


def _import_routes(own: dict, *, fresh_classes: list[str]) -> dict:
    """A project API where the run's registry never changes on import and a fresh project gets
    ``fresh_classes`` (then loses them on undo)."""
    routes = _happy_routes()
    state = {'fresh_undone': False}
    routes[('GET', f'{P}/export/datasets')] = (
        200,
        {'datasets': [{'kind': 'yolo', 'export_dir': '/exports/v1'}], 'count': 1},
    )
    routes[('GET', f'{P}/classes')] = (200, own)
    for slug in (SLUG, f'{SLUG}-imp'):
        base = f'/curation/projects/{slug}/datasets'
        routes[('POST', f'{base}/preview')] = (
            200,
            {'blocking': False, 'import_key': 'k1', 'classes': [{'dataset_class': 'a'}]},
        )
        routes[('POST', f'{base}/imports')] = (
            202,
            {'import_id': f'imp-{slug}', 'status': 'running'},
        )
        routes[('GET', f'{base}/imports/imp-{slug}')] = (200, {'status': 'completed'})
    routes[('POST', f'{P}-imp/datasets/imports/imp-{SLUG}-imp/undo')] = (
        200,
        lambda payload: (state.update(fresh_undone='false' in payload) or 200, {'dry_run': True}),
    )
    routes[('GET', f'/curation/projects/{SLUG}-imp/classes')] = (
        200,
        lambda _p: (
            200,
            _classes(*fresh_classes, deprecated=state['fresh_undone']),
        ),
    )
    routes[('GET', f'/curation/projects/{SLUG}-imp')] = (404, {})
    routes[('POST', '/curation/projects')] = (201, {'project': {}})
    routes[('DELETE', f'/curation/projects/{SLUG}-imp')] = (200, {})
    return routes


def test_dataset_import_keeps_ids_and_creates_fresh_classes_by_name(fake, tmp_path):
    f = fake(_import_routes(_classes('car', 'person'), fresh_classes=['car', 'person']))
    proc, data = run(f, tmp_path, '--only', 'dataset_import')
    assert by_name(data)['dataset_import']['status'] == 'pass', proc.stdout + proc.stderr
    assert f.count('DELETE', f'/curation/projects/{SLUG}-imp') == 1, 'the throwaway is removed'
    assert 'WARNING: cleanup' not in proc.stderr, proc.stderr


def test_dataset_import_fails_when_a_class_the_export_never_had_appears(fake, tmp_path):
    f = fake(_import_routes(_classes('car', 'person'), fresh_classes=['car', 'bicycle']))
    _proc, data = run(f, tmp_path, '--only', 'dataset_import')
    ph = by_name(data)['dataset_import']
    assert ph['status'] == 'fail', data
    assert 'export' in ph['message']


def test_dataset_import_fails_when_importing_changes_the_runs_registry(fake, tmp_path):
    seen = {'n': 0}

    def classes(_payload):
        seen['n'] += 1
        return 200, _classes('car', 'person') if seen['n'] == 1 else _classes('car', 'person', 'x')

    routes = _import_routes(_classes('car'), fresh_classes=['car'])
    routes[('GET', f'{P}/classes')] = (200, classes)
    _proc, data = run(fake(routes), tmp_path, '--only', 'dataset_import')
    ph = by_name(data)['dataset_import']
    assert ph['status'] == 'fail', data
    assert 'changed its classes' in ph['message']


def test_dataset_import_skips_without_an_export(fake, tmp_path):
    routes = _import_routes(_classes('car'), fresh_classes=['car'])
    routes[('GET', f'{P}/export/datasets')] = (200, {'datasets': [], 'count': 0})
    _proc, data = run(fake(routes), tmp_path, '--only', 'dataset_import')
    assert by_name(data)['dataset_import']['status'] == 'skip', data


def _combine_routes(*, target_classes: list[str]) -> dict:
    second = f'{SLUG}-src2'
    routes = _import_routes(_classes('car', 'person'), fresh_classes=[])
    routes[('GET', f'/curation/projects/{second}')] = (404, {})
    routes[('GET', f'/curation/projects/{second}/classes')] = (200, _classes('car', 'person'))
    base = f'/curation/projects/{second}/datasets'
    routes[('POST', f'{base}/preview')] = (
        200,
        {'blocking': False, 'import_key': 'k2', 'classes': []},
    )
    routes[('POST', f'{base}/imports')] = (202, {'import_id': 'imp2'})
    routes[('GET', f'{base}/imports/imp2')] = (200, {'status': 'completed'})
    routes[('POST', '/curation/projects/combine/preview')] = (
        200,
        {'ok': True, 'preview_sha': 's1', 'suggested_mapping': {'a': []}},
    )
    routes[('POST', '/curation/projects/combine')] = (202, {'job_id': 'cj', 'target': 't'})
    routes[('GET', '/curation/projects/combine/cj')] = (200, {'status': 'completed'})
    routes[('GET', f'/curation/projects/{SLUG}-combined/classes')] = (
        200,
        _classes(*target_classes),
    )
    routes[('GET', f'/curation/projects/{SLUG}-combined/stats')] = (200, {'counts': {'items': 4}})
    for slug in (second, f'{SLUG}-combined'):
        routes[('DELETE', f'/curation/projects/{slug}')] = (200, {})
    return routes


def test_combine_unions_the_sources_classes_by_name_and_leaves_them_alone(fake, tmp_path):
    f = fake(_combine_routes(target_classes=['car', 'person']))
    proc, data = run(f, tmp_path, '--only', 'combine')
    assert by_name(data)['combine']['status'] == 'pass', proc.stdout + proc.stderr
    assert f.count('DELETE', f'/curation/projects/{SLUG}-combined') == 1
    assert f.count('DELETE', f'/curation/projects/{SLUG}-src2') == 1


def test_combine_fails_when_the_target_misses_a_source_class(fake, tmp_path):
    f = fake(_combine_routes(target_classes=['car']))
    _proc, data = run(f, tmp_path, '--only', 'combine')
    ph = by_name(data)['combine']
    assert ph['status'] == 'fail', data
    assert 'union' in ph['message']
