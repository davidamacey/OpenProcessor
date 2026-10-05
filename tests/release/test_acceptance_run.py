"""acceptance_run.sh against a fake HTTP server: runner, report, flags, cleanup trap."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(
    os.environ.get('ACC_SCRIPT_UNDER_TEST')
    or REPO_ROOT / 'scripts' / 'release' / 'acceptance_run.sh'
)
SLUG = 'acc-test'
P = f'/curation/projects/{SLUG}'

GOOD_DOCS = '<html><script src="/docs-assets/swagger-ui-bundle.js"></script></html>'


def _happy_routes() -> dict[tuple[str, str], tuple[int, object]]:
    return {
        ('GET', '/health'): (200, {'status': 'healthy', 'version': '9.9.9'}),
        ('GET', '/curation/health'): (200, {'status': 'ok'}),
        ('GET', '/curation/projects/default/settings'): (
            200,
            {'resource_links': [{'name': 'API docs', 'kind': 'docs', 'url': '/docs'}]},
        ),
        ('GET', '/docs'): (200, GOOD_DOCS),
        ('GET', '/redoc'): (
            200,
            '<html><script src="/docs-assets/redoc.standalone.js"></script></html>',
        ),
        ('GET', '/openapi.json'): (200, {'openapi': '3.1.0', 'paths': {'/health': {}}}),
        ('GET', '/docs-assets/swagger-ui-bundle.js'): (200, 'x'),
        ('GET', '/docs-assets/redoc.standalone.js'): (200, 'x'),
        ('POST', '/curation/projects'): (201, {'project': {'slug': SLUG}}),
        ('DELETE', P): (200, {'deleted': True}),
    }


class Fake:
    def __init__(self, routes: dict[tuple[str, str], tuple[int, object]]) -> None:
        self.routes = routes
        self.requests: list[tuple[str, str]] = []
        self.raw: list[tuple[str, str, str]] = []  # method, path?query, body
        fake = self

        class H(BaseHTTPRequestHandler):
            def _do(self) -> None:
                n = int(self.headers.get('Content-Length') or 0)
                payload = self.rfile.read(n).decode(errors='replace') if n else ''
                path = self.path.split('?')[0]
                fake.requests.append((self.command, path))
                fake.raw.append((self.command, self.path, payload))
                code, body = fake.routes.get((self.command, path), (404, {'detail': 'nf'}))
                if callable(body):
                    code, body = body(payload)
                raw = body if isinstance(body, str) else json.dumps(body)
                self.send_response(code)
                self.send_header('Content-Length', str(len(raw.encode())))
                self.end_headers()
                self.wfile.write(raw.encode())

            do_GET = do_POST = do_PUT = do_DELETE = _do  # noqa: N815

            def log_message(self, *a: object) -> None:
                pass

        self.srv = ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.srv.server_address[1]}'
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def count(self, method: str, path: str) -> int:
        return sum(1 for r in self.requests if r == (method, path))


@pytest.fixture
def fake():
    made: list[Fake] = []

    def make(routes=None) -> Fake:
        f = Fake(routes if routes is not None else _happy_routes())
        made.append(f)
        return f

    yield make
    for f in made:
        f.srv.shutdown()


def run(fake: Fake, tmp_path: Path, *args: str) -> tuple[subprocess.CompletedProcess, dict]:
    report = tmp_path / 'r.json'
    proc = subprocess.run(
        [
            'bash', str(SCRIPT), '--base-url', fake.url, '--project', SLUG,
            '--report', str(report), '--workdir', str(tmp_path / 'work'), *args,
        ],
        capture_output=True, text=True, timeout=120, check=False,
        env={'PATH': '/usr/bin:/bin:/usr/local/bin', 'ACC_POLL_S': '0.05', 'HOME': str(tmp_path)},
    )  # fmt: skip
    data = json.loads(report.read_text()) if report.exists() else {}
    return proc, data


def by_name(data: dict) -> dict[str, dict]:
    return {p['name']: p for p in data['phases']}


def test_report_shape_and_happy_path(fake, tmp_path):
    f = fake()
    proc, data = run(
        f, tmp_path, '--only', 'health,resource_links,docs_selfhosted,project_create,teardown'
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert {
        'schema_version',
        'base_url',
        'project',
        'started_at',
        'finished_at',
        'summary',
        'phases',
    } <= set(data)
    ph = by_name(data)
    for name in ('health', 'resource_links', 'docs_selfhosted', 'project_create', 'teardown'):
        assert ph[name]['status'] == 'pass', ph[name]
        assert isinstance(ph[name]['seconds'], (int, float))
        assert ph[name]['evidence'], name
    assert ph['ingest']['status'] == 'skip'
    assert data['summary']['required_failed'] == 0
    assert f.count('DELETE', P) == 1
    assert 'PASS' in proc.stdout


def test_failed_required_phase_exits_nonzero(fake, tmp_path):
    routes = _happy_routes()
    routes[('GET', '/health')] = (500, {'detail': 'boom'})
    proc, data = run(fake(routes), tmp_path, '--only', 'health,docs_selfhosted')
    assert proc.returncode == 1
    ph = by_name(data)
    assert ph['health']['status'] == 'fail'
    assert '500' in json.dumps(ph['health']['evidence'])
    assert ph['docs_selfhosted']['status'] == 'pass'
    assert data['summary']['required_failed'] == 1


def test_skip_flag_marks_phase_skipped(fake, tmp_path):
    proc, data = run(
        fake(), tmp_path, '--only', 'health,resource_links', '--skip', 'resource_links'
    )
    assert proc.returncode == 0, proc.stdout
    ph = by_name(data)
    assert ph['health']['status'] == 'pass'
    assert ph['resource_links']['status'] == 'skip'


def test_external_url_in_docs_fails(fake, tmp_path):
    routes = _happy_routes()
    routes[('GET', '/docs')] = (200, '<script src="https://cdn.jsdelivr.net/x.js"></script>')
    proc, data = run(fake(routes), tmp_path, '--only', 'docs_selfhosted')
    assert proc.returncode == 1
    assert 'cdn.jsdelivr.net' in json.dumps(by_name(data)['docs_selfhosted'])


def test_cleanup_trap_deletes_project_after_failure(fake, tmp_path):
    f = fake()  # ingest route is absent -> 404 -> phase fails; teardown not selected
    proc, data = run(f, tmp_path, '--only', 'project_create,ingest')
    assert proc.returncode == 1
    assert by_name(data)['ingest']['status'] == 'fail'
    assert f.count('DELETE', P) == 1, f.requests


def test_cleanup_trap_runs_on_sigterm(fake, tmp_path):
    import signal
    import time

    f = fake()
    f.routes[('POST', f'{P}/ingest/upload')] = (200, {'ingested': 1})
    f.routes[('GET', f'{P}/ingest/status')] = (200, {'total': 0})  # never drains: runner hangs
    imgs = tmp_path / 'imgs'
    imgs.mkdir()
    (imgs / 'a.jpg').write_bytes(b'\xff\xd8\xff\xd9')
    proc = subprocess.Popen(
        ['bash', str(SCRIPT), '--base-url', f.url, '--project', SLUG,
         '--report', str(tmp_path / 'r.json'), '--workdir', str(tmp_path / 'w'),
         '--only', 'project_create,ingest', '--images-dir', str(imgs)],
        env={'PATH': '/usr/bin:/bin', 'ACC_POLL_S': '0.2', 'HOME': str(tmp_path)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )  # fmt: skip
    deadline = time.time() + 20
    while time.time() < deadline and f.count('GET', f'{P}/ingest/status') < 2:
        time.sleep(0.1)
    assert f.count('GET', f'{P}/ingest/status') >= 2, f.requests
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=30)
    assert proc.returncode != 0
    assert f.count('DELETE', P) == 1, f.requests


def test_installer_phases_skip_without_install_dir(fake, tmp_path):
    proc, data = run(fake(), tmp_path, '--only', 'installer_rerun,installer_repair')
    assert proc.returncode == 0, proc.stdout
    ph = by_name(data)
    assert ph['installer_rerun']['status'] == 'skip'
    assert 'install-dir' in ph['installer_rerun']['message']


def test_list_prints_every_phase():
    out = subprocess.run(
        ['bash', str(SCRIPT), '--list'], capture_output=True, text=True, check=True
    ).stdout
    for name in (
        'health',
        'resource_links',
        'docs_selfhosted',
        'project_create',
        'ingest',
        'detect_embed',
        'cluster',
        'vlm_label',
        'confirm_labels',
        'holdout',
        'export',
        'train',
        'bakeoff',
        'promote',
        'infer',
        'model_delete',
        'metrics',
        'log_noise',
        'vlm_switch',
        'installer_rerun',
        'installer_repair',
        'installer_upgrade',
        'installer_uninstall',
        'teardown',
    ):
        assert name in out.split(), name


def test_train_promote_infer_delete_chain_forces_after_gate_failure(fake, tmp_path):
    routes = _happy_routes()
    job = 'job1'
    name = 'acc_acc_test'
    routes[('POST', f'{P}/train/preflight')] = (
        200,
        {'blocked': False, 'summary': 'ok', 'checks': []},
    )
    routes[('POST', f'{P}/train/start')] = (201, {'job_id': job, 'preflight': {}})
    routes[('GET', f'{P}/train/status/{job}')] = (
        200,
        {'state': 'finished', 'eval': {'map50': 0.1}},
    )

    def promote(payload: str):
        if json.loads(payload).get('force'):
            return 202, {'promote_id': 'p1', 'status': 'queued'}
        return 422, {'detail': {'error': 'gate_failed'}}

    routes[('POST', f'{P}/train/promote/{job}')] = (200, promote)
    routes[('GET', f'{P}/train/promote/{job}/jobs/p1')] = (200, {'status': 'done'})
    routes[('POST', '/detect')] = (200, {'detections': []})
    routes[('DELETE', f'{P}/models/{name}')] = (200, {'unloaded': True})
    routes[('GET', '/metrics')] = (
        200,
        'op_ingest_images_total{project="acc-test",outcome="ok"} 3\nop_queue_depth{queue="embed"} 0\n'
        'op_embedding_state_items{project="acc-test",state="embedded"} 3\n',
    )
    f = fake(routes)
    proc, data = run(f, tmp_path, '--only', 'train,promote,infer,model_delete,metrics,teardown')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ph = by_name(data)
    assert all(
        ph[n]['status'] == 'pass' for n in ('train', 'promote', 'infer', 'model_delete', 'metrics')
    ), ph
    assert 'force=true' in json.dumps(ph['promote']['evidence'])
    assert f.count('POST', f'{P}/train/promote/{job}') == 2
    # deleted by the model_delete phase, so cleanup does not delete it again
    assert f.count('DELETE', f'{P}/models/{name}') == 1


# --- shapes observed against a live dev stack (v0.4.1) -----------------------


def _write_images(tmp_path: Path, n: int) -> Path:
    imgs = tmp_path / 'imgs'
    imgs.mkdir()
    for i in range(n):
        (imgs / f'{i:03d}.jpg').write_bytes(b'\xff\xd8\xff\xd9')
    return imgs


def test_detect_embed_survives_many_images(fake, tmp_path):
    # `collect_images | head -1` died of SIGPIPE once there were enough images
    routes = _happy_routes()
    routes[('POST', '/detect')] = (200, {'detections': []})
    routes[('POST', '/embed/image')] = (200, {'embedding': [0.0] * 512})
    routes[('GET', f'{P}/crops')] = (200, {'total': 3, 'crops': [], 'n_unembedded': None})
    imgs = _write_images(tmp_path, 60)
    proc, data = run(
        fake(routes), tmp_path, '--only', 'detect_embed', '--images-dir', str(imgs),
        '--image-count', '60',
    )  # fmt: skip
    assert by_name(data)['detect_embed']['status'] == 'pass', proc.stdout


def test_cluster_seeds_classes_for_real(fake, tmp_path):
    # seed_from_detector is a dry run unless dry_run=false is sent
    routes = _happy_routes()
    routes[('POST', f'{P}/classes/seed_from_detector')] = (200, {'created': [{'class_id': 0}]})
    routes[('POST', f'{P}/pipeline/auto_label')] = (200, {'stages': {}})
    routes[('GET', f'{P}/pipeline/auto_label/status')] = (200, {'status': 'idle'})
    routes[('GET', f'{P}/clusters')] = (200, {'clusters': []})
    f = fake(routes)
    proc, data = run(f, tmp_path, '--only', 'cluster')
    assert by_name(data)['cluster']['status'] == 'pass', proc.stdout
    seed = [r for r in f.raw if r[1].endswith('/classes/seed_from_detector')]
    assert json.loads(seed[0][2]) == {'dry_run': False}


def test_teardown_accepts_async_202_delete_and_waits_for_404(fake, tmp_path):
    # live DELETE /projects/{slug} answers 202 (status "deleting"), then 404
    routes = _happy_routes()
    state = {'gets': 0, 'deleted': False}

    def delete(_payload: str):
        state['deleted'] = True
        return 202, {'project': {'slug': SLUG, 'status': 'deleting'}}

    def get(_payload: str):
        if not state['deleted']:
            return 404, {'detail': 'nf'}  # project_create precheck
        state['gets'] += 1
        return (200, {'project': {'status': 'deleting'}}) if state['gets'] < 3 else (404, {})

    routes[('DELETE', P)] = (200, delete)
    routes[('GET', P)] = (200, get)
    proc, data = run(fake(routes), tmp_path, '--only', 'project_create,teardown')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert by_name(data)['teardown']['status'] == 'pass'
    assert data['cleanup_complete'] is True


def test_confirm_one_class_then_train_single_class_subset(fake, tmp_path):
    # the trainer blocks any class under 20 validated crops, so train a 1-class subset
    routes = _happy_routes()
    crops = [{'crop_id': f'c{i}'} for i in range(60)]
    routes[('GET', f'{P}/crops')] = (200, {'total': 40, 'crops': crops})
    routes[('GET', f'{P}/classes')] = (200, {'classes': [{'class_id': 7, 'class_name': 'x'}]})
    for c in crops:
        routes[('PUT', f'{P}/crops/{c["crop_id"]}/label')] = (200, {'class_id': 7})
    routes[('POST', f'{P}/train/preflight')] = (200, {'blocked': False, 'checks': []})
    routes[('POST', f'{P}/train/start')] = (201, {'job_id': 'j'})
    routes[('GET', f'{P}/train/status/j')] = (200, {'state': 'finished'})
    routes[('POST', f'{P}/vlm/label_batch')] = (200, {'predicted': 1})
    f = fake(routes)
    proc, data = run(
        f, tmp_path, '--only', 'vlm_label,confirm_labels,train', '--confirm-crops', '40'
    )  # fmt: skip
    ph = by_name(data)
    assert ph['confirm_labels']['status'] == 'pass', proc.stdout
    assert ph['train']['status'] == 'pass', proc.stdout
    puts = [r for r in f.raw if r[0] == 'PUT']
    assert len(puts) == 40
    spec = json.loads(next(r for r in f.raw if r[1].endswith('/train/start'))[2])
    assert spec['include_classes'] == [7]
    assert spec['single_cls'] is True


def test_promote_uses_the_namespaced_triton_name(fake, tmp_path):
    # live API serves the model as <project>__<requested name>
    routes = _happy_routes()
    served = f'{SLUG}__acc_acc_test'
    routes[('POST', f'{P}/train/preflight')] = (200, {'blocked': False, 'checks': []})
    routes[('POST', f'{P}/train/start')] = (201, {'job_id': 'j'})
    routes[('GET', f'{P}/train/status/j')] = (200, {'state': 'finished'})
    routes[('POST', f'{P}/train/promote/j')] = (
        202,
        {'promote_id': 'p1', 'triton_name': served, 'status': 'building'},
    )
    routes[('GET', f'{P}/train/promote/j/jobs/p1')] = (
        200,
        {'promote_id': 'p1', 'triton_name': served, 'status': 'done'},
    )
    routes[('POST', '/detect')] = (200, {'detections': []})
    routes[('DELETE', f'{P}/models/{served}')] = (200, {})
    f = fake(routes)
    proc, _data = run(f, tmp_path, '--only', 'train,promote,infer,model_delete,teardown')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert any(r[1] == f'/detect?model_name={served}' for r in f.raw), f.raw
    assert f.count('DELETE', f'{P}/models/{served}') == 1


def test_bakeoff_posts_run_and_dataset_refs(fake, tmp_path):
    routes = _happy_routes()
    routes[('POST', f'{P}/train/preflight')] = (200, {'blocked': False, 'checks': []})
    routes[('POST', f'{P}/train/start')] = (201, {'job_id': 'j'})
    routes[('GET', f'{P}/train/status/j')] = (200, {'state': 'finished'})
    routes[('GET', f'{P}/bakeoff/trained_models')] = (200, {'models': [{'run_id': 'j'}]})
    routes[('POST', f'{P}/bakeoff/run')] = (202, {'job_id': 'b1'})
    routes[('GET', f'{P}/bakeoff/status/b1')] = (200, {'state': 'done'})
    f = fake(routes)
    proc, data = run(f, tmp_path, '--only', 'train,bakeoff,teardown')
    assert by_name(data)['bakeoff']['status'] == 'pass', proc.stdout
    body = json.loads(next(r for r in f.raw if r[1].endswith('/bakeoff/run'))[2])
    assert body == {'models': [{'source': 'run', 'run_id': 'j'}], 'datasets': [{'id': 'run:j'}]}
