"""The compare CLI end to end on stored synthetic captures, and the environment record helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from scripts.bench import parity, parity_env, parity_store as ps


if TYPE_CHECKING:
    from pathlib import Path


def _store(tmp_path: Path, make_capture, tag: str, **kw) -> Path:
    path = tmp_path / tag
    ps.save_capture(path, make_capture(tag, **kw))
    return path


def test_compare_passes_and_writes_the_summary(make_capture, tmp_path: Path, capsys) -> None:
    ref = _store(tmp_path, make_capture, 'ref', engine='fp32')
    cand = _store(tmp_path, make_capture, 'cand', engine='fp16')
    out = tmp_path / 'summary'
    code = parity.main(['compare', str(ref), str(cand), '--out', str(out)])
    assert code == 0
    assert 'PASS' in capsys.readouterr().out
    doc = json.loads((out / 'parity_summary.json').read_text())
    assert doc['evaluation']['passed'] is True
    assert doc['reference']['engine']['label'] == 'fp32'
    assert doc['metrics']['boxes']['images_identical_share'] == 1.0
    assert (out / 'parity_summary.md').is_file()


def test_compare_exits_one_on_a_failed_gate(make_capture, tmp_path: Path) -> None:
    def scramble(a: dict[str, np.ndarray]) -> None:
        a['item_vec'] = np.random.default_rng(0).normal(size=a['item_vec'].shape).astype(np.float16)

    ref = _store(tmp_path, make_capture, 'ref')
    cand = _store(tmp_path, make_capture, 'cand', mutate=scramble)
    assert parity.main(['compare', str(ref), str(cand)]) == 1


def test_compare_exits_two_when_inputs_cannot_be_compared(
    make_capture, tmp_path: Path, capsys
) -> None:
    ref = _store(tmp_path, make_capture, 'ref')
    other = _store(tmp_path, make_capture, 'other', manifest='different')
    assert parity.main(['compare', str(ref), str(other)]) == 2
    assert 'manifest' in capsys.readouterr().err
    assert parity.main(['compare', str(ref), str(tmp_path / 'nope')]) == 2


def test_compare_refuses_captures_of_a_different_image_count(make_capture, tmp_path: Path) -> None:
    ref = _store(tmp_path, make_capture, 'ref')
    cand_dir = _store(tmp_path, make_capture, 'cand')
    meta = json.loads((cand_dir / 'capture.json').read_text())
    meta['manifest']['count'] = 10
    (cand_dir / 'capture.json').write_text(json.dumps(meta))
    assert parity.main(['compare', str(ref), str(cand_dir)]) == 2


def test_compare_strict_fails_on_skips(make_capture, tmp_path: Path) -> None:
    ref = _store(tmp_path, make_capture, 'ref')
    cand = make_capture('cand')
    cand.meta['sections'] = ['items', 'frames']
    ps.save_capture(tmp_path / 'cand', cand)
    assert parity.main(['compare', str(ref), str(tmp_path / 'cand')]) == 0
    assert parity.main(['compare', str(ref), str(tmp_path / 'cand'), '--strict']) == 1


def test_capture_refuses_to_overwrite_a_tag(tmp_path: Path, capsys) -> None:
    (tmp_path / 'fp16').mkdir()
    (tmp_path / 'fp16' / 'capture.json').write_text('{}')
    code = parity.main(
        [
            'capture', '--tag', 'fp16', '--images', str(tmp_path), '--api-url', 'http://x',
            '--triton-url', 'http://x', '--opensearch-url', 'http://x', '--compose-project', 'p',
            '--engine-label', 'fp16', '--out', str(tmp_path),
        ]
    )  # fmt: skip
    assert code == 2
    assert 'already holds a capture' in capsys.readouterr().err


def test_vlm_crops_need_classes(tmp_path: Path, capsys) -> None:
    code = parity.main(
        [
            'capture', '--tag', 't', '--images', str(tmp_path), '--api-url', 'http://x',
            '--triton-url', 'http://x', '--opensearch-url', 'http://x', '--compose-project', 'p',
            '--engine-label', 'fp16', '--vlm-crops', '5',
        ]
    )  # fmt: skip
    assert code == 2
    assert '--vlm-classes' in capsys.readouterr().err


def test_engine_record_hashes_the_plan_and_parses_pairs(tmp_path: Path) -> None:
    plan = tmp_path / 'model.plan'
    plan.write_bytes(b'abc')
    rec = parity_env.engine_record('fp16', plan, ['precision=fp16', 'trt=11.1'])
    assert rec['plan_bytes'] == 3
    assert rec['plan_sha256'] == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert rec['trt'] == '11.1'
    assert parity_env.engine_record('fp32', None, []) == {'label': 'fp32'}


def test_engine_record_rejects_a_bare_word(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='KEY=VALUE'):
        parity_env.engine_record('x', None, ['oops'])


def test_parse_inspect_reads_name_image_and_id() -> None:
    text = '/w0par-api repo/api:1 sha256:aa\n/w0par-triton triton:2 sha256:bb\nnoise\n'
    assert parity_env.parse_inspect(text) == {
        'w0par-api': {'image': 'repo/api:1', 'image_id': 'sha256:aa'},
        'w0par-triton': {'image': 'triton:2', 'image_id': 'sha256:bb'},
    }


def test_image_digests_filters_by_compose_label(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake(cmd: list[str], timeout: float = 30.0) -> str:
        seen.append(cmd)
        return 'w0par-api\n' if cmd[1] == 'ps' else '/w0par-api img sha256:cc\n'

    monkeypatch.setattr(parity_env, 'run_cmd', fake)
    out = parity_env.image_digests('w0par')
    assert 'label=com.docker.compose.project=w0par' in seen[0]
    assert out == {'w0par-api': {'image': 'img', 'image_id': 'sha256:cc'}}


def test_http_errors_are_reported_with_the_response_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    import httpx

    def boom(_args) -> int:
        request = httpx.Request('POST', 'http://x/ingest')
        raise httpx.HTTPStatusError(
            'bad',
            request=request,
            response=httpx.Response(503, text='opensearch unavailable', request=request),
        )

    monkeypatch.setattr(parity, 'cmd_capture', boom)
    code = parity.main(
        [
            'capture', '--tag', 't', '--images', str(tmp_path), '--api-url', 'http://x',
            '--triton-url', 'http://x', '--opensearch-url', 'http://x', '--compose-project', 'p',
            '--engine-label', 'fp16',
        ]
    )  # fmt: skip
    assert code == 2
    assert '503: opensearch unavailable' in capsys.readouterr().err
