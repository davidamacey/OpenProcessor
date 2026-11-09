"""Characterization of ``download_model`` flow: skip-if-present, source fallback, verification."""

import sys
from pathlib import Path

import pytest


EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import download_paddleocr as dl  # noqa: E402


@pytest.fixture
def calls(monkeypatch, tmp_path):
    monkeypatch.setattr(dl, 'OUTPUT_DIR', tmp_path)
    log: dict[str, list] = {'download': [], 'verify': []}
    state = {'verify': True}

    def fake_download(url, path, _name):
        log['download'].append(url)
        path.write_bytes(b'x' * 2048)
        return True

    def fake_verify(path):
        log['verify'].append(path.name)
        return state['verify']

    monkeypatch.setattr(dl, 'download_with_progress', fake_download)
    monkeypatch.setattr(dl, 'verify_onnx_model', fake_verify)
    log['state'] = state  # type: ignore[assignment]
    return log


def _cfg(filename='m.onnx', urls=('http://a/m.onnx',), **extra):
    return {'filename': filename, 'description': 'd', 'urls': list(urls), **extra}


def test_existing_onnx_verified_is_skipped(calls, tmp_path):
    (tmp_path / 'm.onnx').write_bytes(b'x' * 300_000)
    assert dl.download_model('m', _cfg(size_mb=1)) is True
    assert calls['download'] == []


def test_existing_dictionary_is_skipped(calls, tmp_path):
    (tmp_path / 'd.txt').write_text('abc' * 100_000)
    assert dl.download_model('d', _cfg('d.txt', size_mb=1)) is True
    assert calls['download'] == []


def test_existing_without_size_mb_skipped_when_nonempty(calls, tmp_path):
    (tmp_path / 'd.txt').write_text('abc' * 100_000)
    assert dl.download_model('d', _cfg('d.txt')) is True
    assert calls['download'] == []


def test_force_redownloads(calls, tmp_path):
    (tmp_path / 'm.onnx').write_bytes(b'x' * 300_000)
    assert dl.download_model('m', _cfg(), force=True) is True
    assert calls['download'] == ['http://a/m.onnx']


def test_tar_source_skipped_then_next_used(calls):
    cfg = _cfg(urls=('http://a/m.tar', 'http://b/m.onnx'))
    assert dl.download_model('m', cfg) is True
    assert calls['download'] == ['http://b/m.onnx']


def test_failed_verification_falls_through_and_removes_file(calls, tmp_path):
    calls['state']['verify'] = False
    cfg = _cfg(urls=('http://a/m.onnx', 'http://b/m.onnx'))
    assert dl.download_model('m', cfg) is False
    assert calls['download'] == ['http://a/m.onnx', 'http://b/m.onnx']
    assert not (tmp_path / 'm.onnx').exists()


def test_dictionary_download_succeeds(calls, tmp_path):
    (tmp_path / 'd.txt').unlink(missing_ok=True)
    assert dl.download_model('d', _cfg('d.txt', urls=('http://a/d.txt',))) is True
    assert calls['verify'] == []
