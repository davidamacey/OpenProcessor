#!/usr/bin/env python3
"""Cases for the Rust upload receiver (plan 18a, slice R1u, crate ``op_ingest``).

Writes ``contracts/json/upload_validation_cases.json``: multipart requests to
``POST /curation/projects/{project}/ingest/upload`` and the status and JSON body the Python route
answered, plus the pure helpers the Rust receiver ports (the extension rule, the disk-guard
classification and its 507 body, the batch-response merge). Nothing is typed as an expectation:
the Python route and helpers are the definition, and ``native/op_ingest`` is held to them case by
case.

The request cases run the real route function (``curation_ingest_upload``) behind a small FastAPI
app that wires it exactly as ``src/app/factory.py`` does for this route: the app-level
``enforce_file_cap`` dependency, the ``DISK_GUARD`` route dependency, the ceiling check of the
performance middleware, and the ``UploadTooLargeError`` handler. The ingest service is a stub: a
request whose photos are all refused before the pipeline (empty, wrong extension) or refused as a
whole (413, 422, 400, 507) never reaches it, which is every case here. A photo the pipeline
would take is the Rust sink's business, not a Python behaviour to copy.

A truncated body (the client stopped sending, or a body shorter than its boundary) is not a case:
Python accepts it and answers 422 "images missing" for the part that never ended; the Rust
receiver refuses it with 400 instead, so a half sent photo is never taken for a whole one.

Left out because Python itself is not the reference there: the text after ``image_paths is not
JSON: `` (it is the message of Python's ``json`` module, which no other parser repeats; the
cases keep it for the record and the Rust test compares the prefix), and ``request_id`` (the
case runs with none bound, which Python writes as ``-``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterator


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contracts' / 'json' / 'upload_validation_cases.json'
BOUNDARY = '----OpenProcessorCaseBoundary'
MIB = 1024 * 1024

# Small limits so a case that crosses one stays a few bytes in the case file.
SMALL = {
    'OP_UPLOAD_MAX_IMAGES_PER_REQUEST': 3,
    'OP_UPLOAD_MAX_BYTES_PER_REQUEST': 100,
    'MAX_FILE_SIZE_MB': 1,
}
EXTENSIONS = ('.jpg', '.jpeg', '.png')


def _file(filename: str | None, size: int, name: str = 'images') -> dict[str, Any]:
    return {'name': name, 'filename': filename, 'size': size}


def _text(name: str, value: str) -> dict[str, Any]:
    return {'name': name, 'text': value}


def _paths(*ids: str) -> dict[str, Any]:
    return _text('image_paths', json.dumps(list(ids)))


def encode_body(parts: list[dict[str, Any]]) -> bytes:
    """The multipart body of a case: ``size`` bytes of ``x`` for a file, the text for a field."""
    out = bytearray()
    for part in parts:
        out += f'--{BOUNDARY}\r\n'.encode()
        head = f'Content-Disposition: form-data; name="{part["name"]}"'
        if 'text' in part or 'text_repeat' in part:
            text = part['text'] if 'text' in part else 'x' * part['text_repeat']
            out += f'{head}\r\n\r\n'.encode() + text.encode() + b'\r\n'
            continue
        if part['filename'] is not None:
            head += f'; filename="{part["filename"]}"'
        out += f'{head}\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
        out += b'x' * part['size'] + b'\r\n'
    return bytes(out) + f'--{BOUNDARY}--\r\n'.encode()


def _route_cases() -> list[dict[str, Any]]:
    big = MIB + 1
    cases: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = [
        # --- request level ---
        ('no_parts', [], {}),
        ('only_text_fields', [_text('source', 'x')], {}),
        ('images_sent_as_text', [_text('images', 'not a file')], {}),
        ('four_images_over_the_count', [_file(f'{i}.jpg', 1) for i in range(4)], {}),
        (
            'count_beats_bad_paths',
            [*(_file(f'{i}.jpg', 1) for i in range(4)), _text('image_paths', 'nope')],
            {},
        ),
        (
            'count_limit_from_env',
            [_file('a.jpg', 1), _file('b.jpg', 1)],
            {'OP_UPLOAD_MAX_IMAGES_PER_REQUEST': 1},
        ),
        ('file_over_the_cap', [_file('a.jpg', big)], {}),
        ('file_over_the_cap_unsupported_type', [_file('a.gif', big)], {}),
        ('file_over_the_cap_other_field', [_file('a.jpg', big, name='extra')], {}),
        ('file_at_the_cap', [_file('a.gif', MIB)], {'OP_UPLOAD_MAX_BYTES_PER_REQUEST': 2 * MIB}),
        ('cap_beats_count', [*(_file(f'{i}.jpg', 1) for i in range(3)), _file('x.jpg', big)], {}),
        (
            'cap_beats_total',
            [_file('a.jpg', 60), _file('b.jpg', 60), _file('c.jpg', big)],
            {},
        ),
        ('two_images_as_text', [_text('images', 'a'), _text('images', 'b')], {}),
        (
            'text_images_between_files',
            [_file('a.gif', 1), _text('images', 'b'), _file('c.gif', 1)],
            {},
        ),
        (
            'text_images_beat_count',
            [*(_file(f'{i}.jpg', 1) for i in range(4)), _text('images', 'b')],
            {},
        ),
        ('text_images_empty_value', [_text('images', '')], {}),
        ('paths_sent_as_file', [_file('a.gif', 1), _file('p.json', 2, name='image_paths')], {}),
        ('source_sent_as_file', [_file('a.gif', 1), _file('p.json', 2, name='source')], {}),
        ('too_many_files', [_file('f.gif', 0, name='extra') for _ in range(1001)], {}),
        (
            'thousand_files',
            [_file('f.gif', 0, name='extra') for _ in range(1000)] + [_file('a.gif', 0)],
            {},
        ),
        ('too_many_fields', [_text(f'f{i}', '') for i in range(1001)], {}),
        # --- image_paths ---
        ('paths_not_json', [_file('a.jpg', 1), _text('image_paths', '[')], {}),
        ('paths_not_json_text', [_file('a.jpg', 1), _text('image_paths', 'nope')], {}),
        ('paths_an_object', [_file('a.jpg', 1), _text('image_paths', '{"a": 1}')], {}),
        ('paths_a_string', [_file('a.jpg', 1), _text('image_paths', '"a.jpg"')], {}),
        ('paths_a_number_in_the_list', [_file('a.jpg', 1), _text('image_paths', '[1]')], {}),
        (
            'paths_an_empty_string_in_the_list',
            [_file('a.jpg', 1), _text('image_paths', '[""]')],
            {},
        ),
        ('paths_null_in_the_list', [_file('a.jpg', 1), _text('image_paths', '[null]')], {}),
        ('paths_nested_list', [_file('a.jpg', 1), _text('image_paths', '[["a.jpg"]]')], {}),
        ('paths_too_few', [_file('a.jpg', 1), _file('b.jpg', 1), _paths('a.jpg')], {}),
        ('paths_too_many', [_file('a.jpg', 1), _paths('a.jpg', 'b.jpg', 'c.jpg')], {}),
        ('paths_empty_list_one_image', [_file('a.jpg', 1), _text('image_paths', '[]')], {}),
        (
            'paths_first_then_too_many_images',
            [_paths('a.jpg'), _file('a.jpg', 1), _file('b.jpg', 1), _file('c.jpg', 1)],
            {},
        ),
        (
            'paths_mismatch_beats_total',
            [_file('a.jpg', 60), _file('b.jpg', 60), _paths('a.jpg')],
            {},
        ),
        ('paths_blank_is_the_default', [_file('a.gif', 1), _text('image_paths', '  \t ')], {}),
        ('paths_empty_is_the_default', [_file('a.gif', 1), _text('image_paths', '')], {}),
        ('paths_whitespace_unicode', [_file('a.gif', 1), _text('image_paths', '\u2003')], {}),
        ('paths_whitespace_separator', [_file('a.gif', 1), _text('image_paths', '\x1f')], {}),
        # --- per photo refusals (the route answers 200) ---
        ('empty_photo', [_file('a.jpg', 0)], {}),
        ('empty_beats_extension', [_file('a.gif', 0)], {}),
        ('wrong_extension', [_file('a.gif', 1)], {}),
        ('wrong_extension_from_identifier', [_file('a.jpg', 1), _paths('x/a.webp')], {}),
        (
            'extension_from_filename_when_identifier_has_none',
            [_file('a.gif', 1), _paths('x/a')],
            {},
        ),
        ('upper_case_extension', [_file('A.GIF', 1)], {}),
        ('extension_with_quote', [_file("a.j'pg", 1)], {}),
        ('extension_with_both_quotes', [_file('a.j\'p"g', 1), _paths('a.j\'p"g')], {}),
        ('extension_with_unicode', [_file('a.\u00e9\u4e2d', 1), _paths('a.\u00e9\u4e2d')], {}),
        ('extension_with_control_character', [_file('a.x', 1), _paths('a.\u0001')], {}),
        ('extension_with_no_break_space', [_file('a.x', 1), _paths('a.\u00a0x')], {}),
        ('extension_with_astral_character', [_file('a.x', 1), _paths('a.\U0001f600')], {}),
        ('identifier_dotfile', [_file('a.x', 1), _paths('.jpg')], {}),
        ('identifier_trailing_dot', [_file('a.x', 1), _paths('a.')], {}),
        ('identifier_trailing_slash', [_file('a.x', 1), _paths('dir.gif/')], {}),
        ('identifier_double_extension', [_file('a.x', 1), _paths('a.tar.gif')], {}),
        ('identifier_backslash', [_file('a.x', 1), _paths('c:\\pics\\a.gif')], {}),
        ('two_refusals_in_order', [_file('a.gif', 1), _file('b.jpg', 0)], {}),
        (
            'identifiers_listed_in_failures',
            [_file('a.gif', 1), _file('b.gif', 1), _paths('first', 'second')],
            {},
        ),
        ('filename_used_as_identifier', [_file('only.gif', 1)], {}),
        ('image_without_filename_gets_a_number', [_file(None, 0), _file('', 0)], {}),
        # --- total bytes of the accepted photos ---
        ('total_over_the_limit', [_file('a.jpg', 60), _file('b.jpg', 60)], {}),
        ('total_at_the_limit', [_file('a.gif', 50), _file('b.gif', 50)], {}),
        ('total_excludes_refused_photos', [_file('a.gif', 90), _file('b.gif', 90)], {}),
        (
            'total_over_after_a_refused_photo',
            [_file('a.gif', 90), _file('b.jpg', 60), _file('c.jpg', 60)],
            {},
        ),
        (
            'total_limit_from_env',
            [_file('a.jpg', 6), _file('b.jpg', 6)],
            {'OP_UPLOAD_MAX_BYTES_PER_REQUEST': 10},
        ),
        # --- extensions from the environment ---
        (
            'extension_list_from_env',
            [_file('a.jpg', 1), _file('b.jpeg', 1)],
            {'OP_UPLOAD_ACCEPTED_EXTENSIONS': ' .GIF , ,.png'},
        ),
        # --- fields ---
        (
            'duplicate_paths_field_last_wins',
            [_file('a.gif', 1), _paths('x.gif'), _paths('y.gif')],
            {},
        ),
        ('unknown_text_field_ignored', [_text('other', 'v'), _file('a.gif', 1)], {}),
        (
            'empty_run_id_and_source',
            [_text('source', ''), _text('run_id', ''), _file('a.gif', 1)],
            {},
        ),
        (
            'text_field_over_one_mib',
            [{'name': 'image_paths', 'text_repeat': MIB + 1}, _file('a.gif', 1)],
            {},
        ),
        (
            'text_field_of_one_mib',
            [{'name': 'other', 'text_repeat': MIB}, _file('a.gif', 1)],
            {},
        ),
    ]
    return [{'name': n, 'parts': p, 'env': e} for n, p, e in cases]


def _raw_cases() -> list[dict[str, Any]]:
    """Requests whose body is not a form the case encoder writes."""
    cases: list[tuple[str, dict[str, Any]]] = [
        ('not_multipart_json', {'content_type': 'application/json', 'body': '{}'}),
        ('multipart_without_boundary', {'content_type': 'multipart/form-data', 'body': ''}),
        (
            'declared_length_over_the_ceiling',
            {
                'content_type': f'multipart/form-data; boundary={BOUNDARY}',
                'body': f'--{BOUNDARY}--\r\n',
                'content_length': 2 * MIB + 1,
                'env': {'MAX_FILE_SIZE_MB': 1, 'OP_UPLOAD_MAX_BYTES_PER_REQUEST': 100},
            },
        ),
        (
            'declared_length_at_the_ceiling',
            {
                'content_type': f'multipart/form-data; boundary={BOUNDARY}',
                'body': f'--{BOUNDARY}--\r\n',
                'content_length': 2 * MIB,
                'env': {'MAX_FILE_SIZE_MB': 1, 'OP_UPLOAD_MAX_BYTES_PER_REQUEST': 100},
            },
        ),
    ]
    return [{'name': n, **spec} for n, spec in cases]


@contextmanager
def _python_env() -> Iterator[None]:
    """Make ``src`` importable and keep the case run from touching real state."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    with tempfile.TemporaryDirectory() as tmp:
        sys.path.insert(0, tmp)
        try:
            yield
        finally:
            sys.path.remove(tmp)


def _swap(module: object, name: str, value: object) -> None:
    """Replace a module attribute for the rest of the run (the generator is a short process)."""
    setattr(module, name, value)


@contextmanager
def _environ(env: dict[str, Any]) -> Iterator[None]:
    """Set the variables for the block and put the process environment back after it."""
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update({key: str(value) for key, value in env.items()})
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _Wired:
    """The real route behind the wiring of ``src/app/factory.py``, with settings swapped per case."""

    def __init__(self) -> None:
        from fastapi import APIRouter, Depends, FastAPI, Request
        from fastapi.testclient import TestClient

        import src.config as config_module
        import src.core.upload_limits as limits
        import src.routers.curation.ingest_upload as route_module
        from src.core import disk_space
        from src.core.wire_json import WireRoute
        from src.routers.curation._common import _raw_opensearch_dep, _registry_dep
        from src.services.curation import upload_ingest

        self.limits = limits
        self.route_module = route_module
        self.config_module = config_module
        self.disk_space = disk_space

        app = FastAPI(dependencies=[Depends(limits.enforce_file_cap)])
        app.exception_handler(limits.UploadTooLargeError)(limits.upload_too_large_handler)

        @app.middleware('http')
        async def ceiling(request: Request, call_next: Any) -> Any:
            if request.method == 'POST' and limits.request_exceeds_ceiling(request):
                return limits.too_large_response()
            return await call_next(request)

        router = APIRouter(route_class=WireRoute)
        router.post(
            '/ingest/upload',
            dependencies=[disk_space.DISK_GUARD],
            response_model=route_module._BatchIngestResponse,
        )(route_module.curation_ingest_upload)
        app.include_router(router)
        app.dependency_overrides[_raw_opensearch_dep] = lambda: None
        app.dependency_overrides[_registry_dep] = lambda: None
        self.client = TestClient(app, raise_server_exceptions=True)

        class NoPipeline:
            """The ingest service of a case: any call means a photo reached the pipeline."""

            def __getattr__(self, name: str) -> Any:
                msg = f'a case reached the ingest service ({name})'
                raise AssertionError(msg)

        async def no_service(*_args: Any, **_kwargs: Any) -> Any:
            return NoPipeline()

        _swap(route_module, 'get_ingest_service', no_service)

        def no_disk(_payload: bytes, **_kwargs: Any) -> Path:
            return Path('/nowhere')

        # A photo the pipeline would take is persisted and then ingested; a case that gets that far
        # with the total still under the limit is not a request-edge case, and a case that goes on
        # to cross the limit must not write the first photo to the host.
        _swap(upload_ingest, 'persist_uploaded_bytes', no_disk)

    def configure(self, env: dict[str, Any], free_ratio: float | None) -> None:
        """Build the config and settings the route reads from ``env``, as the process would."""
        from src.config.curation import CurationConfig
        from src.config.settings import Settings

        with _environ({**SMALL, **env}):
            cfg = CurationConfig.from_env()
            settings = Settings()
        _swap(self.config_module, 'get_curation_config', lambda: cfg)
        _swap(self.limits, 'get_curation_config', lambda: cfg)
        _swap(self.limits, 'get_settings', lambda: settings)
        self.free(0.5 if free_ratio is None else free_ratio)

    def free(self, ratio: float) -> None:
        """Make the guard see a data volume with ``ratio`` of it free."""
        _swap(self.disk_space, 'data_volume', lambda: self.disk_space.measure('data', ratio))

    def post(self, content_type: str, body: bytes, content_length: int | None) -> dict[str, Any]:
        headers = {'content-type': content_type}
        if content_length is not None:
            headers['content-length'] = str(content_length)
        reply = self.client.post('/ingest/upload', content=body, headers=headers)
        try:
            payload: Any = reply.json()
        except ValueError:
            payload = reply.text
        return {'status': reply.status_code, 'body': payload}


def _scrub(body: Any) -> Any:
    """Drop what a client must not be held to: the request id (none is bound here, Python writes
    ``-``) and, for a file sent where text belongs, Python's ``repr`` of its upload object."""
    if not isinstance(body, dict):
        return body
    body = {k: v for k, v in body.items() if k != 'request_id'}
    detail = body.get('detail')
    if isinstance(detail, list):
        body['detail'] = [
            {
                k: v
                for k, v in item.items()
                if not (k == 'input' and item.get('type') == 'string_type')
            }
            for item in detail
        ]
    return body


def _request_results(wired: _Wired) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case in _route_cases():
        wired.configure(case['env'], None)
        body = encode_body(case['parts'])
        answer = wired.post(f'multipart/form-data; boundary={BOUNDARY}', body, None)
        out.append({**case, 'answer': {**answer, 'body': _scrub(answer['body'])}})
    for case in _raw_cases():
        wired.configure(case.get('env', {}), None)
        body = case['body'].encode()
        answer = wired.post(case['content_type'], body, case.get('content_length'))
        out.append({**case, 'answer': {**answer, 'body': _scrub(answer['body'])}})
    return out


def _disk_cases(wired: _Wired) -> list[dict[str, Any]]:
    """The guard's verdict and 507 body for a free fraction and a pair of thresholds."""
    from src.core.disk_space import classify, require_disk_space
    from src.core.exceptions import RequestRejectedError

    out: list[dict[str, Any]] = []
    grid = [(None, None), ('20', '5'), ('10', '3'), ('0', '0'), ('100', '100'), ('2.5', '0.5')]
    ratios = [0.0, 0.01, 0.029999, 0.03, 0.0300001, 0.05, 0.0999, 0.1, 0.5, 0.999, 1.0]
    for low, critical in grid:
        env = {} if low is None else {'OP_DISK_LOW_PCT': low, 'OP_DISK_CRITICAL_PCT': critical}
        with _environ(env):
            for ratio in ratios:
                wired.free(ratio)
                body: Any = None
                status = 200
                try:
                    asyncio.run(require_disk_space())
                except RequestRejectedError as exc:
                    status = exc.status_code
                    body = {'detail': exc.detail}
                out.append(
                    {
                        'env': env,
                        'free_ratio': ratio,
                        'level': classify(ratio),
                        'answer': {'status': status, 'body': body},
                    }
                )
    return out


def _disk_threshold_errors() -> list[dict[str, Any]]:
    """Settings the guard refuses: critical above low, and values outside 0 to 100."""
    from src.config.env import EnvError
    from src.core.disk_space import thresholds

    out = []
    for low, critical in [('3', '10'), ('10.5', '10.6'), ('101', '3'), ('10', '-1'), ('x', '3')]:
        env = {'OP_DISK_LOW_PCT': low, 'OP_DISK_CRITICAL_PCT': critical}
        with _environ(env):
            try:
                thresholds()
                refused = False
            except (ValueError, EnvError):
                refused = True
        out.append({'env': env, 'refused': refused})
    return out


def _batch_merge_cases() -> list[dict[str, Any]]:
    """``_batch_response`` over a pipeline result and the photos refused before it."""
    from types import SimpleNamespace

    from src.routers.curation._common_models import IngestImageResponse
    from src.routers.curation.ingest import _batch_response

    def result(**kw: Any) -> Any:
        base = {
            'status': 'success',
            'image_id': 'id',
            'image_path': '/p',
            'imohash': 'h',
            'n_crops': 2,
            'n_region_queued': 1,
            'n_embedded': 2,
            'n_not_embedded': 0,
            'n_embed_failed': 0,
            'n_filtered': 1,
            'error': None,
            'error_kind': None,
            'source_identifier': 'a.jpg',
            'secondary_detector_error': None,
        }
        return SimpleNamespace(**{**base, **kw})

    def summary(**kw: Any) -> Any:
        base = {
            'successful': 0,
            'duplicates': 0,
            'failed': 0,
            'crops_indexed': 0,
            'secondary_detector_failures': 0,
            'n_embedded': 0,
            'n_not_embedded': 0,
            'n_embed_failed': 0,
            'n_filtered': 0,
        }
        return SimpleNamespace(**{**base, **kw})

    early = IngestImageResponse(
        status='failed',
        image_path='bad.gif',
        error='empty upload',
        error_kind='empty',
        source_identifier='bad.gif',
    )
    ok = result()
    dup = result(status='duplicate', image_id='d', source_identifier='d.jpg')
    failed = result(status='failed', error='decode failed', error_kind='decode_failed', n_crops=0)
    shapes: list[tuple[str, list[Any], Any]] = [
        ('only_early_failures', [early], None),
        ('nothing_at_all', [], None),
        (
            'all_successful',
            [],
            SimpleNamespace(
                results=[ok, ok],
                summary=summary(successful=2, crops_indexed=4, n_embedded=4, n_filtered=2),
            ),
        ),
        (
            'early_failure_and_success',
            [early],
            SimpleNamespace(
                results=[ok], summary=summary(successful=1, crops_indexed=2, n_embedded=2)
            ),
        ),
        (
            'duplicates_only',
            [],
            SimpleNamespace(results=[dup], summary=summary(duplicates=1)),
        ),
        (
            'pipeline_failure_only',
            [],
            SimpleNamespace(results=[failed], summary=summary(failed=1)),
        ),
        (
            'mixed',
            [early, early],
            SimpleNamespace(
                results=[ok, dup, failed],
                summary=summary(
                    successful=1,
                    duplicates=1,
                    failed=1,
                    crops_indexed=2,
                    secondary_detector_failures=1,
                    n_embedded=2,
                    n_not_embedded=1,
                    n_embed_failed=1,
                    n_filtered=1,
                ),
            ),
        ),
        (
            'secondary_detector_error_kept',
            [],
            SimpleNamespace(
                results=[result(secondary_detector_error='timeout')],
                summary=summary(successful=1, secondary_detector_failures=1),
            ),
        ),
    ]
    out = []
    for name, early_list, batch in shapes:
        response = _batch_response(batch, list(early_list))
        out.append(
            {
                'name': name,
                'early': [e.model_dump() for e in early_list],
                'batch': None
                if batch is None
                else {
                    'results': [vars(r) for r in batch.results],
                    'summary': vars(batch.summary),
                },
                'response': response.model_dump(),
            }
        )
    return out


def build() -> dict[str, Any]:
    with _python_env():
        wired = _Wired()
        return {
            'about': 'Generated by scripts/codegen/upload_validation_cases.py; do not edit.',
            'boundary': BOUNDARY,
            'small_limits': SMALL,
            'accepted_extensions': list(EXTENSIONS),
            'requests': _request_results(wired),
            'disk': _disk_cases(wired),
            'disk_threshold_errors': _disk_threshold_errors(),
            'batch_merge': _batch_merge_cases(),
        }


def main() -> int:
    # The disk guard logs a warning for every low reading the cases feed it.
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__.split('\n', maxsplit=1)[0])
    parser.add_argument('--check', action='store_true', help='fail when the file is stale')
    args = parser.parse_args()
    text = json.dumps(build(), indent=1, ensure_ascii=False, sort_keys=True) + '\n'
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding='utf-8') != text:
            sys.stderr.write(
                f'{OUT.relative_to(ROOT)} is stale: run scripts/codegen/upload_validation_cases.py\n'
            )
            return 1
        return 0
    OUT.write_text(text, encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
