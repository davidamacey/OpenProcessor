"""judge_protocol: verdict schema, privacy refusal, write-ahead budget, cache and the run loop."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.research import judge_backends as jb, judge_protocol as jp, judge_run as jr


if TYPE_CHECKING:
    from pathlib import Path


def _manifest(
    sheet_id: str = 's1', kind: str = 'object', layout: str = 'cluster'
) -> dict[str, Any]:
    cells = [
        {'cell': n, 'label': str(n), 'item_id': f'i{n}', 'box_id': 'b', 'role': 'random'}
        for n in range(1, 7)
    ]
    return {
        'sheet_id': sheet_id,
        'kind': kind,
        'layout': layout,
        'png_sha256': f'sha-{sheet_id}',
        'grid': [3, 2],
        'cells': cells,
    }


def _request(sheet_id: str = 's1', kind: str = 'object') -> jp.JudgeRequest:
    m = _manifest(sheet_id, kind)
    return jp.JudgeRequest(
        manifest=m,
        png=b'png-' + sheet_id.encode(),
        prompt=jp.render_prompt(m, level='object class'),
    )


class FakeBackend:
    def __init__(self, replies: list[str | None], *, hosted: bool = False) -> None:
        self.hosted = hosted
        self.model = 'fake-1'
        self.name = 'fake'
        self.calls: list[str] = []
        self._replies = replies
        self.on_call: Any = None

    def judge(self, request: jp.JudgeRequest) -> jp.RawReply | None:
        self.calls.append(request.manifest['sheet_id'])
        if self.on_call:
            self.on_call()
        reply = self._replies[len(self.calls) - 1]
        return (
            None if reply is None else jp.RawReply(text=reply, input_tokens=100, output_tokens=20)
        )


GOOD = json.dumps({'same': False, 'odd_cells': [2, 5], 'name': 'car', 'confidence': 'high'})


def test_prompt_shows_no_labels_and_names_the_level_and_cell_count() -> None:
    prompt = jp.render_prompt(_manifest(), level='car model')
    assert 'car model' in prompt
    assert 'JSON' in prompt
    assert 'i1' not in prompt
    assert 'role' not in prompt
    assert '6 numbered cells' in prompt
    pair = jp.render_prompt({**_manifest(layout='pair'), 'grid': [2, 3]}, level='x')
    assert 'A1' in pair
    assert 'B1' in pair


def test_cluster_verdict_parses_with_fences_and_checks_cells() -> None:
    v = jp.parse_verdict(_manifest(), '```json\n' + GOOD + '\n```')
    assert v == {'same': False, 'odd_cells': [2, 5], 'name': 'car', 'confidence': 'high'}
    with pytest.raises(jp.VerdictError, match='odd_cells'):
        jp.parse_verdict(
            _manifest(),
            json.dumps({'same': False, 'odd_cells': [9], 'name': 'a', 'confidence': 'low'}),
        )
    with pytest.raises(jp.VerdictError, match='same'):
        jp.parse_verdict(_manifest(), json.dumps({'odd_cells': []}))
    with pytest.raises(jp.VerdictError, match='JSON'):
        jp.parse_verdict(_manifest(), 'I think they match')


def test_a_same_verdict_with_odd_cells_is_inconsistent() -> None:
    bad = json.dumps({'same': True, 'odd_cells': [1], 'name': 'a', 'confidence': 'high'})
    with pytest.raises(jp.VerdictError, match='consistent'):
        jp.parse_verdict(_manifest(), bad)


def test_pair_and_search_verdicts() -> None:
    pair = _manifest(layout='pair')
    ok = json.dumps(
        {
            'rows': [
                {'row': 1, 'answer': 'yes'},
                {'row': 2, 'answer': 'unsure'},
                {'row': 3, 'answer': 'no'},
            ]
        }
    )
    assert [r['answer'] for r in jp.parse_verdict(pair, ok)['rows']] == ['yes', 'unsure', 'no']
    with pytest.raises(jp.VerdictError, match='rows'):
        jp.parse_verdict(pair, json.dumps({'rows': [{'row': 1, 'answer': 'maybe'}]}))
    with pytest.raises(jp.VerdictError, match='rows'):
        jp.parse_verdict(pair, json.dumps({'rows': [{'row': 7, 'answer': 'yes'}]}))
    search = _manifest(layout='search')
    assert jp.parse_verdict(search, json.dumps({'matching_cells': [2, 3]})) == {
        'matching_cells': [2, 3]
    }
    with pytest.raises(jp.VerdictError, match='matching_cells'):
        jp.parse_verdict(search, json.dumps({'matching_cells': [0]}))


def test_verdict_cells_map_back_to_item_ids_through_the_manifest() -> None:
    v = jp.parse_verdict(_manifest(), GOOD)
    assert jp.odd_items(_manifest(), v) == [('i2', 'b'), ('i5', 'b')]


@pytest.mark.parametrize('kind', ['face', 'identity_text', 'plate'])
def test_hosted_backend_refuses_sensitive_kinds_before_any_call(kind: str, tmp_path: Path) -> None:
    backend = FakeBackend([GOOD], hosted=True)
    budget = jr.Budget(tmp_path / 'b.json', max_calls=5, max_input_tokens=10_000)
    with pytest.raises(jp.PrivacyRefusalError, match=kind):
        jr.run_requests(
            [_request(kind=kind)], backend, budget, None, tmp_path / 'r.jsonl', run_id='r'
        )
    assert backend.calls == []
    assert (
        not (tmp_path / 'b.json').exists()
        or json.loads((tmp_path / 'b.json').read_text())['reserved'] == {}
    )


def test_local_backend_may_judge_faces() -> None:
    jp.assert_may_send('face', hosted=False)
    jp.assert_may_send('object', hosted=True)


def test_budget_is_reserved_write_ahead_and_never_exceeded(tmp_path: Path) -> None:
    path = tmp_path / 'b.json'
    budget = jr.Budget(path, max_calls=3, max_input_tokens=1_000_000)
    backend = FakeBackend([GOOD] * 10)
    seen: list[int] = []
    backend.on_call = lambda: seen.append(len(json.loads(path.read_text())['reserved']))
    reqs = [_request(f's{k}') for k in range(10)]
    with pytest.raises(jr.BudgetExceededError, match='max_calls'):
        jr.run_requests(reqs, backend, budget, None, tmp_path / 'r.jsonl', run_id='r')
    assert len(backend.calls) == 3
    assert seen == [1, 2, 3]  # the reservation is on disk before each call
    assert len((tmp_path / 'r.jsonl').read_text().splitlines()) == 4  # header + 3 verdicts


def test_token_cap_stops_the_run(tmp_path: Path) -> None:
    budget = jr.Budget(tmp_path / 'b.json', max_calls=100, max_input_tokens=250)
    backend = FakeBackend([GOOD] * 10)
    with pytest.raises(jr.BudgetExceededError, match='max_input_tokens'):
        jr.run_requests(
            [_request(f's{k}') for k in range(5)],
            backend,
            budget,
            None,
            tmp_path / 'r.jsonl',
            run_id='r',
            est_tokens=100,
        )
    assert len(backend.calls) == 2


def test_reserving_the_same_key_twice_costs_once(tmp_path: Path) -> None:
    budget = jr.Budget(tmp_path / 'b.json', max_calls=1, max_input_tokens=1000)
    budget.reserve('k', 100)
    budget.reserve('k', 100)
    with pytest.raises(jr.BudgetExceededError):
        budget.reserve('other', 100)
    assert jr.Budget(tmp_path / 'b.json', max_calls=1, max_input_tokens=1000).calls == 1


def test_cache_reuses_a_verdict_until_model_or_prompt_changes(tmp_path: Path) -> None:
    cache = jr.VerdictCache(tmp_path / 'cache')
    budget = jr.Budget(tmp_path / 'b.json', max_calls=10, max_input_tokens=100_000)
    backend = FakeBackend([GOOD] * 5)
    out = tmp_path / 'r.jsonl'
    jr.run_requests([_request()], backend, budget, cache, out, run_id='a')
    jr.run_requests([_request()], backend, budget, cache, out, run_id='b')
    assert backend.calls == ['s1']  # second run served from the cache
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r.get('cached') for r in rows if r.get('sheet_id')] == [False, True]
    backend.model = 'fake-2'
    jr.run_requests([_request()], backend, budget, cache, out, run_id='c')
    assert backend.calls == ['s1', 's1']
    assert cache.key('sha', 'm', 'p1') != cache.key('sha', 'm', 'p2')


def test_records_carry_the_documented_fields(tmp_path: Path) -> None:
    budget = jr.Budget(tmp_path / 'b.json', max_calls=2, max_input_tokens=100_000)
    out = tmp_path / 'r.jsonl'
    jr.run_requests(
        [_request()],
        FakeBackend([GOOD]),
        budget,
        None,
        out,
        run_id='run1',
        header={'seed': 7, 'git_sha': 'abc'},
    )
    header, rec = (json.loads(x) for x in out.read_text().splitlines())
    assert header['run_id'] == 'run1'
    assert header['seed'] == 7
    assert header['budget']['max_calls'] == 2
    for key in (
        'run_id',
        'model',
        'prompt_version',
        'unit',
        'sheet_id',
        'png_sha256',
        'verdict',
        'odd_cells',
        'name',
        'unsure',
        'input_tokens',
        'output_tokens',
        'latency_ms',
        'request_id',
        'cached',
        'error',
    ):
        assert key in rec
    assert rec['unit'] == 'cluster'
    assert rec['odd_cells'] == [2, 5]
    assert rec['error'] is None


def test_an_invalid_reply_is_recorded_not_dropped(tmp_path: Path) -> None:
    budget = jr.Budget(tmp_path / 'b.json', max_calls=2, max_input_tokens=100_000)
    out = tmp_path / 'r.jsonl'
    jr.run_requests([_request()], FakeBackend(['not json']), budget, None, out, run_id='r')
    rec = json.loads(out.read_text().splitlines()[-1])
    assert rec['verdict'] is None
    assert 'JSON' in rec['error']
    assert rec['raw_text'] == 'not json'


def test_sheet_file_backend_round_trip(tmp_path: Path) -> None:
    backend = jb.SheetFileBackend(tmp_path / 'exchange', model='haiku-subagent')
    assert backend.hosted is True
    budget = jr.Budget(tmp_path / 'b.json', max_calls=5, max_input_tokens=100_000)
    out = tmp_path / 'r.jsonl'
    req = _request('s9')
    pending = jr.run_requests([req], backend, budget, None, out, run_id='r')
    assert pending.pending == ['s9']
    ex = tmp_path / 'exchange'
    assert (ex / 'pending' / 's9.png').read_bytes() == req.png
    assert req.prompt in (ex / 'pending' / 's9.prompt.txt').read_text()
    (ex / 'verdicts').mkdir(exist_ok=True)
    (ex / 'verdicts' / 's9.json').write_text(GOOD)
    done = jr.run_requests([req], backend, budget, None, out, run_id='r')
    assert done.judged == ['s9']
    assert done.pending == []
    assert budget.calls == 1  # exporting and collecting reserved the call once
    assert json.loads(out.read_text().splitlines()[-1])['odd_cells'] == [2, 5]


def test_local_backend_builds_an_openai_request_with_the_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class Resp:
        status_code = 200
        headers = {'x-request-id': 'rid'}

        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, Any]:
            return {
                'choices': [{'message': {'content': GOOD}}],
                'usage': {'prompt_tokens': 900, 'completion_tokens': 30},
            }

    def fake_post(url: str, **kw: Any) -> Resp:
        seen['url'], seen['body'] = url, kw['json']
        return Resp()

    backend = jb.LocalVlmBackend('http://127.0.0.1:4612/v1', model='m', post=fake_post)
    raw = backend.judge(_request())
    assert raw is not None
    assert raw.input_tokens == 900
    assert raw.output_tokens == 30
    assert seen['url'] == 'http://127.0.0.1:4612/v1/chat/completions'
    content = seen['body']['messages'][0]['content']
    assert content[0]['type'] == 'image_url'
    assert content[0]['image_url']['url'].startswith('data:image/png;base64,')
    assert seen['body']['temperature'] == 0


def test_local_backend_refuses_an_external_endpoint() -> None:
    with pytest.raises(jp.PrivacyRefusalError, match='external'):
        jb.LocalVlmBackend('https://api.example.org/v1', model='m')
