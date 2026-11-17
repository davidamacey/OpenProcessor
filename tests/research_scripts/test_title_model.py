"""Run R8: the data, the template table, the truth, the thresholds, the keep rule, the word finder,
the invented-fact check, the title checks and the timing summary, on hand-made inputs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from scripts.research.title_model import (
    fact_sheets as fs,
    memories as mem,
    run_titles as rt,
    scene_words_embed as emb,
    scene_words_mirage as mir,
    score,
)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

VOCAB = mem.load_vocabulary()
NO_WORDS: list[tuple[str, ...]] = [()] * 60
TABLE = rt.load_templates()


def _supers() -> dict[int, set[str]]:
    names = [f'super{i}' for i in range(12)]
    return {pid: {names[pid % 12], names[(pid + 5) % 12]} for pid in range(1, 2401)}


def _sheet(**kw: object) -> fs.FactSheet:
    base = {
        'sheet_id': 0,
        'start': dt.date(2019, 5, 12),
        'end': dt.date(2019, 5, 15),
        'place': 'Lisbon',
        'people': 4,
        'named': (),
        'event': None,
        'scene_words': (),
    }
    return fs.FactSheet(**{**base, **kw})  # type: ignore[arg-type]


# --- the data ---------------------------------------------------------------------------------


def test_a_missing_folder_setting_fails_with_its_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('R8_RUN_DIR', raising=False)
    with pytest.raises(RuntimeError, match='R8_RUN_DIR'):
        mem.run_dir()


def test_split_is_deterministic_disjoint_and_shares_a_supercategory() -> None:
    supers = _supers()
    ids = sorted(supers)
    a, b = mem.build_split(supers, ids), mem.build_split(supers, ids)
    assert a == b
    assert len(a.memories) == 60
    assert all(len(m) == 3 for m in a.memories)
    assert len(a.tuning) == 1000
    in_memories = [p for m in a.memories for p in m]
    assert len(set(in_memories)) == 180
    assert not set(in_memories) & set(a.tuning)
    assert all(set.intersection(*(supers[p] for p in m)) for m in a.memories)
    assert mem.build_split(supers, ids, seed=1).memories != a.memories


def test_sheets_are_deterministic_and_the_first_60_have_no_event() -> None:
    words: list[tuple[str, ...]] = [('beach',)] * 60
    a = fs.make_sheets(words, VOCAB, mem.SEED)
    assert a == fs.make_sheets(words, VOCAB, mem.SEED)
    assert len(a) == 200
    assert all(s.event is None and s.scene_words == ('beach',) for s in a[:60])
    assert any(s.event for s in a[60:])
    assert all(s.start <= s.end and 1 <= s.days <= 21 for s in a)
    assert all(2015 <= s.start.year <= 2025 for s in a)
    assert len(fs.CITIES) == 100
    assert len(fs.LANDMARKS) == 30


def test_holiday_sheets_contain_the_holiday() -> None:
    assert fs.holiday_date('Thanksgiving', 2020) == dt.date(2020, 11, 26)
    assert fs.holiday_date('Easter', 2024) == dt.date(2024, 3, 31)
    assert fs.holiday_date('Memorial Day', 2021) == dt.date(2021, 5, 31)
    for s in fs.make_sheets(NO_WORDS, VOCAB, mem.SEED):
        if s.event in fs.HOLIDAYS:
            assert s.start <= fs.holiday_date(s.event, s.start.year) <= s.end or s.event in (
                "New Year's Eve",
                'Christmas',
                'Christmas Eve',
                'Boxing Day',
                "New Year's Day",
            )


# --- the template table -----------------------------------------------------------------------


def test_every_vocabulary_word_has_a_family_with_a_plain_row_and_a_fallback_chain() -> None:
    members = {w: name for name, f in TABLE['families'].items() for w in f['members']}
    assert {v.word for v in VOCAB} == set(members)
    chains = TABLE['variant_fallback']
    for name, family in TABLE['families'].items():
        assert family['rows']['plain'], name
        for variant in ('plain', 'place', 'person', 'group'):
            reachable = [variant, *chains[variant]]
            assert any(family['rows'].get(v) for v in reachable), (name, variant)
        assert set(family['word_rows']) <= set(family['members'])
    assert set(TABLE['events']) | set(fs.HOLIDAYS) == set(fs.EVENTS)


def test_row_is_picked_by_scene_word_place_and_people() -> None:
    snow = _sheet(scene_words=('snow',), place=None)
    assert rt.render_table(snow, TABLE) in TABLE['families']['snow']['word_rows']['snow']['plain']
    assert rt.render_table(_sheet(scene_words=('snow',)), TABLE).endswith('Lisbon')
    assert 'Ana' in rt.render_table(_sheet(scene_words=('beach',), named=('Ana',)), TABLE)
    group = rt.render_table(_sheet(scene_words=('beach',), named=('Ana', 'Ben', 'Kai')), TABLE)
    assert 'Ana, Ben and Kai' in group
    skiing = rt.render_table(_sheet(scene_words=('skiing',), place=None), TABLE)
    assert skiing != rt.render_table(_sheet(scene_words=('snowboarding',), place=None), TABLE)
    assert rt.render_table(_sheet(event='Christmas', place=None), TABLE) in (
        'Christmas, 2019',
        'Christmas memories',
    )


def test_no_scene_word_gives_the_plain_template() -> None:
    assert rt.render_table(_sheet(), TABLE) == rt.render_plain(_sheet()) == 'Lisbon, May 2019'
    assert rt.render_table(_sheet(place=None), TABLE) == 'May 2019'


def test_every_template_is_true_to_its_sheet_and_short() -> None:
    names = ('Ana', 'Ben', 'Kai')
    longest_place = 'Christ the Redeemer'
    for word in (v.word for v in VOCAB):
        for named in ((), names[:1], names):
            for place in (None, longest_place):
                sheet = _sheet(scene_words=(word,), named=named, place=place)
                title = rt.render_table(sheet, TABLE)
                assert score.title_shape_ok(title), (word, title)
                assert not fs.invented_facts(
                    f'{title} . {rt.template_subtitle(sheet)}', sheet, VOCAB
                ), (word, title)
    for event in fs.EVENTS:
        sheet = _sheet(event=event, named=names, place='Rio de Janeiro')
        title = rt.render_table(sheet, TABLE)
        assert score.title_shape_ok(title), (event, title)
        assert not fs.invented_facts(f'{title} . {rt.template_subtitle(sheet)}', sheet, VOCAB), (
            event,
            title,
        )


# --- the truth, the word finder ---------------------------------------------------------------


def test_truth_from_captions_and_proof_categories() -> None:
    truth = mem.photo_truth(['A man rides a wave.', 'People on a sandy beach!'], {'person'}, VOCAB)
    assert {'beach', 'ocean'} <= truth
    assert 'snow' not in truth
    by_proof = mem.photo_truth(['A person outside.'], {'skis'}, VOCAB)
    assert {'snow', 'skiing'} <= by_proof
    assert mem.photo_truth(['A plain wall.'], {'person'}, VOCAB) == set()


def test_word_finder_is_whole_word_case_blind_and_knows_synonyms() -> None:
    assert 'beach' in mem.find_words('A BEACH, at noon', VOCAB)
    assert 'beach' in mem.find_words('the seashore', VOCAB)
    assert 'beach' not in mem.find_words('a beachfront hotel', VOCAB)
    assert 'beach' not in mem.find_words('a peachy sky', VOCAB)
    assert 'living room' in mem.find_words('in the living-room', VOCAB)
    assert mem.find_words('', VOCAB) == set()


def test_oracle_memory_words_need_two_of_three_photos() -> None:
    truth = {1: {'beach', 'dog'}, 2: {'beach'}, 3: {'dog', 'cat'}, 4: set(), 5: {'snow'}, 6: set()}
    assert mem.oracle_memory_words([[1, 2, 3], [4, 5, 6]], truth) == [('beach', 'dog'), ()]


# --- thresholds and the keep rule -------------------------------------------------------------


def test_threshold_is_the_lowest_value_with_precision_090() -> None:
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    truth = [True, True, True, True, True, True, True, True, True, False]
    assert score.pick_threshold(scores, truth) == 0.05  # 9 of 10 at >= 0.05 is 0.90
    seven = [True] * 7 + [False] * 3
    assert score.pick_threshold(scores, seven) == 0.3  # 7 of 8 at >= 0.2 is 0.875: one step short
    truth2 = [True, True, False, False, False, False, False, False, False, False]
    assert score.pick_threshold(scores, truth2) == 0.8
    assert score.pick_threshold(scores, [False] * 10) is None
    assert score.pick_threshold([0.5, 0.5, 0.4], [True, False, True]) is None  # ties are one step


def test_a_word_that_never_reaches_precision_is_dropped_per_encoder() -> None:
    rng = np.random.default_rng(1)
    vocab = VOCAB[:2]
    scores = rng.random((50, 2))
    truth = np.zeros((50, 2), dtype=bool)
    truth[scores[:, 0] > 0.5, 0] = True  # word 0 is separable, word 1 never true
    found = emb.fit_thresholds(scores, truth, vocab)
    assert found[vocab[1].word] is None
    assert found[vocab[0].word] is not None
    row = np.array([0.99, 0.99])
    assert set(emb.photo_findings(row, found, vocab)) == {vocab[0].word}


def test_word_scores_take_the_best_phrase_of_each_word() -> None:
    text = np.array([[1.0, 0], [0, 1.0], [0.6, 0.8]])
    starts = np.array([0, 2])  # word 0: phrases 0 and 1; word 1: phrase 2
    out = emb.word_scores(np.array([[0.0, 1.0]]), text, starts)
    assert out.tolist() == [[1.0, 0.8]]


def test_keep_rule() -> None:
    per_photo = [{'beach': 0.3, 'dog': 0.7}, {'beach': 0.2, 'cat': 0.9}, {'dog': 0.5, 'beach': 0.4}]
    kept = score.keep_words(per_photo)
    assert kept == [
        'dog',
        'beach',
    ]  # margin: dog (0.7, 0.5) mean 0.6, beach (0.3, 0.2, 0.4) mean 0.3
    assert 'cat' not in kept  # found for 1 photo
    assert score.keep_words(per_photo, rank='count') == ['beach', 'dog']
    many = [dict.fromkeys('abcde', 1.0)] * 3
    assert score.keep_words(many, rank='count') == ['a', 'b', 'c']
    assert score.keep_words([{'x': 0.1}, {'x': 0.9}, {}], rank='margin') == ['x']
    assert score.keep_words([{'x': 1.0}, {}, {}]) == []


def test_keep_rule_orders_by_mean_margin() -> None:
    per_photo = [{'a': 0.1, 'b': 0.5}, {'a': 0.1, 'b': 0.7}, {'a': 0.9}]
    assert score.keep_words(per_photo) == ['b', 'a']  # b mean 0.6, a mean 0.37


def test_scene_summary_counts_wrong_words_and_coverage() -> None:
    truths: list[list[set[str]]] = [
        [{'beach'}, {'beach'}, set()],
        [{'dog'}, set(), set()],
        [set()] * 3,
    ]
    summary = score.scene_summary([['beach'], ['dog'], []], truths)
    assert summary['with_wrong_word'] == 1  # 'dog' is true for 1 of 3 photos
    assert summary['with_a_word'] == 2
    assert summary['coverage_bar_met'] is False
    assert summary['wrong_bar_met'] is True


# --- arm B ------------------------------------------------------------------------------------


def test_mirage_description_text_skips_numbers_and_masks() -> None:
    answer = {
        'scene': {'setting': 'beach', 'lighting': 'warm'},
        'objects': [
            {
                'id': 1,
                'name': 'surfboard',
                'bbox': [1, 2, 3, 4],
                'mask': [1, 2],
                'score': 0.4,
                'description': {'color': 'red'},
            }
        ],
        'text': [{'string': 'EXIT', 'score': -1.0}],
    }
    text = mir.description_text(answer)
    assert 'beach' in text
    assert {'surfboard', 'red', 'EXIT'} <= set(text.replace('.', ' ').split())
    assert '0.4' not in text
    assert mir.found_words(text, VOCAB).keys() >= {'beach', 'surfing'}


class _FakeMirage:
    def __init__(self, answers: list[object]) -> None:
        self.answers = answers

    def describe(self, path: str, **kw: object) -> object:
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_an_invalid_engine_answer_is_an_error_of_that_call_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('R8_COCO_IMAGES', str(tmp_path))
    fake = _FakeMirage(
        [{'scene': {'setting': 'beach'}}, json.JSONDecodeError('Expecting value', 'x', 0)]
    )
    text, error, seconds = mir.describe_once(fake, 1, 'default')
    assert text == 'beach'
    assert error is None
    assert seconds >= 0
    text, error, _ = mir.describe_once(fake, 2, 'default')
    assert text == ''
    assert error is not None
    assert 'JSONDecodeError' in error


# --- the invented-fact check ------------------------------------------------------------------


def test_invented_facts() -> None:
    sheet = _sheet(scene_words=('beach',), people=4)
    ok = rt.render_plain(sheet)
    assert fs.invented_facts(ok, sheet, VOCAB) == []
    assert fs.invented_facts('Madrid, May 2019', sheet, VOCAB) == [
        'madrid'
    ]  # a place not in the sheet
    assert fs.invented_facts('Lisbon with 7 friends', sheet, VOCAB) == [
        '7'
    ]  # a number not in the sheet
    assert fs.invented_facts('Seven friends in Lisbon', sheet, VOCAB) == [
        'seven'
    ]  # a number word it lacks
    assert (
        fs.invented_facts('Four days at the beach', sheet, VOCAB) == []
    )  # 4 days, 4 people; beach is its own
    assert fs.invented_facts('Snow in Lisbon', sheet, VOCAB) == ['snow']  # a scene word it lacks
    assert sorted(fs.invented_facts('Lisbon, March 2018', sheet, VOCAB)) == ['2018', 'march']
    assert (
        fs.invented_facts('A May weekend in spring', sheet, VOCAB) == []
    )  # dates: month, weekend, season
    assert fs.invented_facts('A summer in Lisbon', sheet, VOCAB) == ['summer']
    assert fs.invented_facts('Christmas in Lisbon', sheet, VOCAB) == ['christmas']
    assert fs.invented_facts('Beachfront Lisbon', sheet, VOCAB) == []
    assert sorted(fs.invented_facts('Ana and Ben', sheet, VOCAB)) == ['ana', 'ben']
    named = _sheet(named=('Ana',), place='Lake Louise')
    assert (
        fs.invented_facts('Lake Louise with Ana', named, VOCAB) == []
    )  # a place that holds a scene word


def test_a_month_of_the_range_passes_and_an_event_words_scene_word_is_allowed() -> None:
    wide = _sheet(start=dt.date(2019, 5, 28), end=dt.date(2019, 6, 3), event='hike', place=None)
    assert fs.invented_facts('Hiking from May to June 2019', wide, VOCAB) == []
    assert fs.invented_facts('July 2019', wide, VOCAB) == ['july']


# --- the title checks -------------------------------------------------------------------------


def test_word_count_plain_text_and_json() -> None:
    assert score.title_shape_ok('one two three four five six seven eight')
    assert not score.title_shape_ok('one two three four five six seven eight nine')
    assert not score.title_shape_ok('')
    assert score.title_shape_ok("New Year's Eve in Lisbon")
    assert not score.title_shape_ok('Beach day \U0001f600')
    assert not score.title_shape_ok('The "best" day')
    assert not score.title_shape_ok('**Bold** title')
    ok = '{"title": "Lisbon in May", "subtitle": "Four days"}'
    assert score.parse_answer(ok) == ('Lisbon in May', 'Four days')
    assert score.parse_answer('Sure! ' + ok) is None
    assert score.parse_answer(ok + ' Hope this helps') is None
    assert score.parse_answer('{"title": "x"}') is None
    assert score.parse_answer('{"title": "x", "subtitle": 3}') is None
    assert score.parse_answer('{"title": "x", "subtitle": "y", "extra": "z"}') is None
    assert score.parse_answer('[1, 2]') is None


def test_title_checks_count_each_failure_once() -> None:
    sheets = [_sheet(sheet_id=i) for i in range(4)]
    rows: list[dict[str, object]] = [
        {'title': 'Lisbon, May 2019', 'subtitle': 'Four days'},
        {'title': 'Madrid, May 2019', 'subtitle': 'x'},  # invented place and no Lisbon
        {'title': None, 'subtitle': None},  # invalid JSON
        {'title': 'Slow bright sunny warm lazy happy long quiet days', 'subtitle': 'Lisbon'},
    ]
    out = score.title_checks(rows, sheets, VOCAB)
    assert out['invented_fact'] == 1
    assert out['valid_json_rate'] == 0.75
    assert out['place_rate'] == 0.5  # rows 0 and 3 name Lisbon in title or subtitle
    assert out['shape_rate'] == 0.5  # rows 0 and 1; the invalid one and the nine-word one fail
    assert out['invented_bar_met'] is True
    assert out['shape_bar_met'] is False


# --- timing, ratings, blind files -------------------------------------------------------------


def test_timing_summary_is_nearest_rank_p95_and_tests_the_bar() -> None:
    secs = [0.1] * 94 + [0.4, 0.5, 0.9, 1.5, 3.0, 5.0]  # the 95th of 100 values is 0.4
    assert score.percentile(secs, 95) == 0.4
    ok = score.timing_summary(secs, 0.5, 400.0, 1024.0)
    assert ok['p95_s'] == 0.4
    assert ok['bar_met'] is True
    assert score.timing_summary(secs, 0.3)['bar_met'] is False
    assert score.timing_summary(secs, 1.0, 2000.0, 1024.0)['bar_met'] is False


def _timing_args(tmp_path: Path, payload: Mapping[str, object]) -> argparse.Namespace:
    f = tmp_path / 'timing.json'
    f.write_text(json.dumps(payload))
    return argparse.Namespace(file=str(f))


def test_cmd_timing_reports_a_saved_timing_file_from_its_summary(tmp_path: Path) -> None:
    summary = {'n': 3, 'median_s': 0.2, 'p95_s': 0.3, 'rss_mb': None, 'bar_met': True}
    saved = {'model': 'm', 'reps': [{'rep': 0, 'seconds': [0.1, 0.2, 0.3]}], 'summary': summary}
    assert score.cmd_timing(_timing_args(tmp_path, saved)) == summary


def test_cmd_timing_summarises_a_raw_timing_file(tmp_path: Path) -> None:
    out = score.cmd_timing(_timing_args(tmp_path, {'seconds': [0.1, 0.2, 0.3], 'p95_bar_s': 0.5}))
    assert out['n'] == 3
    assert out['bar_met'] is True


def test_wilson_interval_and_paired_bootstrap() -> None:
    lo, hi = score.wilson(0, 60)
    assert lo == 0.0
    assert 0.05 < hi < 0.07
    lo, hi = score.wilson(30, 60)
    assert lo < 0.5 < hi
    boot = score.paired_bootstrap([4.0, 4.0, 5.0, 4.0], [3.0, 3.0, 4.0, 3.0], resamples=500)
    assert boot['mean_diff'] == 1.0
    assert boot['lo'] == boot['hi'] == 1.0
    rated = score.rating_summary(
        {'plain': [2.0, 3.0, 2.0], 'table': [3.0, 3.5, 2.5]}, 'plain', 'table'
    )
    diff = rated['difference']
    assert isinstance(diff, dict)
    assert rated['bar_met'] is True
    assert diff['mean_diff'] == pytest.approx(0.667, abs=0.001)
    assert score.rating_summary({'a': [3.0, 3.0], 'b': [3.2, 3.2]}, 'a', 'b')['bar_met'] is False


def test_blind_files_hide_the_way_and_pair_ways_on_the_same_sheets() -> None:
    sheets = fs.make_sheets(NO_WORDS, VOCAB, mem.SEED)
    ways = {w: [(f'{w} title {i}', 'sub') for i in range(200)] for w in ('plain', 'table')}
    items, key = score.blind_titles(ways, sheets, n_sheets=50)
    assert len(items) == len(key) == 100
    assert all('way' not in item for item in items)
    assert items == score.blind_titles(ways, sheets, n_sheets=50)[0]
    by_way = score.ratings_by_way(key, {int(str(k['item'])): float(str(k['sheet'])) for k in key})
    assert by_way['plain'] == by_way['table']
    assert len(by_way['plain']) == 50


def test_ratings_command_reads_the_key_and_the_ratings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key = [{'item': i, 'way': 'plain' if i < 3 else 'table', 'sheet': i % 3} for i in range(6)]
    ratings = {str(i): (2.0 if i < 3 else 3.0) for i in range(6)}
    path = tmp_path / 'ratings.json'
    path.write_text(json.dumps({'key': key, 'ratings': ratings}))
    assert score.main(['ratings', str(path), '--reference', 'plain', '--candidate', 'table']) == 0
    out = json.loads(capsys.readouterr().out)
    assert out['bar_met'] is True
    assert out['means'] == {'plain': 2.0, 'table': 3.0}


def test_blind_scenes_skip_empty_arms_hide_the_arm_and_sum_the_wrong(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('R8_COCO_IMAGES', str(tmp_path))
    memories = [[3 * i + 1, 3 * i + 2, 3 * i + 3] for i in range(60)]
    kept = {'a': [['beach'] if i % 2 else [] for i in range(60)], 'b': [['dog']] * 60}
    items, key = score.blind_scenes(kept, memories)
    arms = [k['arm'] for k in key]
    assert arms.count('b') == 30
    assert 0 < arms.count('a') < 30
    assert all('arm' not in item and len(item['photos']) == 3 for item in items)  # type: ignore[arg-type]
    wrong = {int(str(k['item'])): ['dog'] for k in key if k['arm'] == 'b'}
    summary = score.scene_check_summary(key, {i: w for i, w in wrong.items() if i % 2 == 0})
    assert summary['a']['with_wrong_word'] == 0
    assert summary['a']['bar_met'] is True
    assert summary['b']['bar_met'] is False
