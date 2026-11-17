"""Run R8 scoring: scene-word keep rule and checks, title checks, timing and rating summaries.

Every function is pure; the command line at the bottom reads the run's output files and prints one
JSON summary per call.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import unicodedata
from pathlib import Path

from scripts.research.title_model import fact_sheets as fs, memories as mem


MAX_TITLE_WORDS = 8
_MARKUP = re.compile(r'[*#_`\[\]<>{}|\\~^]')
_QUOTES = '"“”„″`'


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson interval of a rate ``k`` of ``n``."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, ``q`` in (0, 100]."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def timing_summary(
    seconds: list[float],
    p95_bar_s: float,
    rss_mb: float | None = None,
    rss_bar_mb: float | None = None,
) -> dict[str, object]:
    p95 = percentile(seconds, 95)
    ok = p95 <= p95_bar_s and (rss_bar_mb is None or rss_mb is None or rss_mb <= rss_bar_mb)
    return {
        'n': len(seconds),
        'median_s': statistics.median(seconds),
        'p95_s': p95,
        'rss_mb': rss_mb,
        'bar_met': ok,
    }


def pick_threshold(scores: list[float], truth: list[bool], precision: float = 0.90) -> float | None:
    """The lowest score value ``t`` whose photos at ``score >= t`` have precision >= ``precision``.

    ``None`` when no value reaches it: the word is dropped for that encoder.
    """
    best = None
    tp = 0
    ranked = sorted(zip(scores, truth, strict=True), key=lambda x: -x[0])
    for i, (score, is_true) in enumerate(ranked):
        seen = i + 1
        tp += is_true
        last_of_value = i + 1 == len(ranked) or ranked[i + 1][0] != score
        if last_of_value and tp / seen >= precision:
            best = score
    return best


def keep_words(
    per_photo: list[dict[str, float]], rank: str = 'margin', min_photos: int = 2, limit: int = 3
) -> list[str]:
    """The keep rule. ``per_photo`` maps each found word to its value (one dict per photo).

    A word found for at least ``min_photos`` photos is kept, at most ``limit``: ranked by the mean
    value over the photos it was found for (``margin``, arm A) or by how many photos (``count``,
    arm B); ties by the word.
    """
    seen: dict[str, list[float]] = {}
    for found in per_photo:
        for word, value in found.items():
            seen.setdefault(word, []).append(value)
    kept = {w: v for w, v in seen.items() if len(v) >= min_photos}
    if rank == 'count':
        return sorted(kept, key=lambda w: (-len(kept[w]), w))[:limit]
    return sorted(kept, key=lambda w: (-statistics.fmean(kept[w]), w))[:limit]


def wrong_words(kept: list[str], photo_truths: list[set[str]], min_photos: int = 2) -> list[str]:
    """Kept words that are not true for ``min_photos`` of the memory's photos."""
    return [w for w in kept if sum(w in t for t in photo_truths) < min_photos]


def scene_summary(
    kept_by_memory: list[list[str]], truths_by_memory: list[list[set[str]]]
) -> dict[str, object]:
    n = len(kept_by_memory)
    wrong = sum(
        bool(wrong_words(k, t)) for k, t in zip(kept_by_memory, truths_by_memory, strict=True)
    )
    covered = sum(bool(k) for k in kept_by_memory)
    return {
        'memories': n,
        'with_wrong_word': wrong,
        'with_wrong_word_ci95': wilson(wrong, n),
        'with_a_word': covered,
        'with_a_word_rate': covered / n if n else 0.0,
        'with_a_word_ci95': wilson(covered, n),
        'wrong_bar_met': wrong <= 3,
        'coverage_bar_met': covered / n >= 0.70 if n else False,
    }


def word_count(title: str) -> int:
    return len(title.split())


def is_plain_text(title: str) -> bool:
    """No emoji or symbols, no markup, no quote characters (an apostrophe is plain text)."""
    if _MARKUP.search(title) or any(q in title for q in _QUOTES):
        return False
    return not any(unicodedata.category(c) in ('So', 'Sk', 'Cs', 'Co') for c in title)


def title_shape_ok(title: str) -> bool:
    return 0 < word_count(title) <= MAX_TITLE_WORDS and is_plain_text(title)


def parse_answer(raw: str) -> tuple[str, str] | None:
    """The answer must be exactly one JSON object with string ``title`` and ``subtitle``."""
    try:
        obj = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(obj, dict) or set(obj) != {'title', 'subtitle'}:
        return None
    if not all(isinstance(obj[k], str) for k in obj):
        return None
    return obj['title'], obj['subtitle']


def place_present(sheet: fs.FactSheet, title: str, subtitle: str) -> bool:
    if not sheet.place:
        return True
    text = f' {mem.normalize(title + " . " + subtitle)} '
    return f' {mem.normalize(sheet.place)} ' in text


def title_checks(
    rows: list[dict[str, object]], sheets: list[fs.FactSheet], vocabulary: list[mem.SceneWord]
) -> dict[str, object]:
    """The script checks of the title rubric for one way. ``rows``: ``raw`` (model answer, or
    ``None``), ``title`` and ``subtitle`` (``None`` when the answer is invalid), one per sheet."""
    n = len(rows)
    valid = [r for r in rows if r['title'] is not None]
    invented = shape = placed = 0
    for row, sheet in zip(rows, sheets, strict=True):
        title, subtitle = str(row['title'] or ''), str(row['subtitle'] or '')
        invented += bool(fs.invented_facts(f'{title} . {subtitle}', sheet, vocabulary))
        shape += bool(row['title'] is not None and title_shape_ok(title))
        placed += place_present(sheet, title, subtitle)
    return {
        'n': n,
        'invented_fact': invented,
        'invented_bar_met': invented <= 2,
        'valid_json_rate': len(valid) / n if n else 0.0,
        'valid_json_ci95': wilson(len(valid), n),
        'valid_json_bar_met': len(valid) / n >= 0.99 if n else False,
        'place_rate': placed / n if n else 0.0,
        'place_ci95': wilson(placed, n),
        'place_bar_met': placed / n >= 0.95 if n else False,
        'shape_rate': shape / n if n else 0.0,
        'shape_ci95': wilson(shape, n),
        'shape_bar_met': shape / n >= 0.95 if n else False,
    }


def paired_bootstrap(
    a: list[float], b: list[float], resamples: int = 10_000, seed: int = mem.SEED
) -> dict[str, float]:
    """Mean of ``a - b`` with a 95% paired bootstrap interval."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.fmean(rng.choices(diffs, k=n)) for _ in range(resamples))
    return {
        'mean_diff': statistics.fmean(diffs),
        'lo': means[int(0.025 * resamples)],
        'hi': means[int(0.975 * resamples) - 1],
    }


def rating_summary(
    ratings: dict[str, list[float]], reference: str, candidate: str, margin: float = 0.5
) -> dict[str, object]:
    """Mean rating per way and the paired difference ``candidate - reference`` against ``margin``."""
    diff = paired_bootstrap(ratings[candidate], ratings[reference])
    return {
        'means': {k: statistics.fmean(v) for k, v in ratings.items()},
        'n': {k: len(v) for k, v in ratings.items()},
        'difference': diff,
        'margin': margin,
        'bar_met': diff['mean_diff'] >= margin,
    }


def blind_titles(
    ways: dict[str, list[tuple[str, str]]],
    sheets: list[fs.FactSheet],
    n_sheets: int = 50,
    seed: int = mem.SEED,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Blinded items for the rating agent and the key that undoes the shuffle.

    ``ways`` maps a way's name to its (title, subtitle) per sheet. The same ``n_sheets`` sheets are
    rated in every way (spread over the 200 by a seeded sample); items are shuffled.
    """
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(sheets)), n_sheets))
    pairs = [(way, i) for way in ways for i in chosen]
    rng.shuffle(pairs)
    items, key = [], []
    for item_id, (way, i) in enumerate(pairs):
        title, subtitle = ways[way][i]
        items.append(
            {
                'item': item_id,
                'facts': describe_sheet(sheets[i]),
                'title': title,
                'subtitle': subtitle,
            }
        )
        key.append({'item': item_id, 'way': way, 'sheet': i})
    return items, key


def blind_scenes(
    kept_by_arm: dict[str, list[list[str]]],
    memories: list[list[int]],
    n_memories: int = 30,
    seed: int = mem.SEED,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Blinded items for the picture check: 3 photo paths and the words an arm kept for them.

    The same ``n_memories`` memories are looked at for every arm; an arm that kept no word for a
    memory has nothing to check there and gets no item.
    """
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(memories)), n_memories))
    pairs = [(arm, i) for arm in kept_by_arm for i in chosen if kept_by_arm[arm][i]]
    rng.shuffle(pairs)
    items, key = [], []
    for item_id, (arm, i) in enumerate(pairs):
        photos = [str(mem.image_path(p)) for p in memories[i]]
        items.append({'item': item_id, 'photos': photos, 'words': kept_by_arm[arm][i]})
        key.append({'item': item_id, 'arm': arm, 'memory': i})
    return items, key


def scene_check_summary(
    key: list[dict[str, object]],
    wrong: dict[int, list[str]],
    n_memories: int = 30,
    max_wrong: int = 2,
) -> dict[str, dict[str, object]]:
    """Per arm: memories (of ``n_memories``) with a wrong word by the picture check, against the bar."""
    counts: dict[str, int] = {}
    for k in key:
        arm = str(k['arm'])
        counts.setdefault(arm, 0)
        counts[arm] += bool(wrong.get(int(str(k['item'])), []))
    return {
        arm: {
            'with_wrong_word': c,
            'of': n_memories,
            'ci95': wilson(c, n_memories),
            'bar_met': c <= max_wrong,
        }
        for arm, c in counts.items()
    }


def describe_sheet(sheet: fs.FactSheet) -> str:
    parts = [f'{sheet.start.isoformat()} to {sheet.end.isoformat()}']
    parts.append(f'place: {sheet.place or "none"}')
    parts.append(
        f'people: {sheet.people}' + (f' (named: {", ".join(sheet.named)})' if sheet.named else '')
    )
    parts.append(f'event: {sheet.event or "none"}')
    parts.append(f'scene: {", ".join(sheet.scene_words) or "none"}')
    return '; '.join(parts)


def ratings_by_way(
    key: list[dict[str, object]], ratings: dict[int, float]
) -> dict[str, list[float]]:
    """Ratings per way, ordered by sheet so ways pair up."""
    out: dict[str, list[tuple[int, float]]] = {}
    for k in key:
        out.setdefault(str(k['way']), []).append(
            (int(str(k['sheet'])), ratings[int(str(k['item']))])
        )
    return {way: [r for _, r in sorted(v)] for way, v in out.items()}


def _read_rows(path: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def cmd_checks(args: argparse.Namespace) -> dict[str, object]:
    """The script checks of the title rubric for each titles file (one JSON line per sheet)."""
    from scripts.research.title_model.run_titles import build_sheets

    sheets = build_sheets(args.memory_words)
    vocab = mem.load_vocabulary()
    return {Path(p).stem: title_checks(_read_rows(p), sheets, vocab) for p in args.rows}


def cmd_blind(args: argparse.Namespace) -> dict[str, object]:
    """Write the blinded items for the rating agent and the key that undoes the shuffle."""
    from scripts.research.title_model.run_titles import build_sheets

    sheets = build_sheets(args.memory_words)
    ways = {}
    for p in args.rows:
        rows = _read_rows(p)
        ways[Path(p).stem] = [(str(r['title'] or ''), str(r['subtitle'] or '')) for r in rows]
    items, key = blind_titles(ways, sheets)
    Path(args.items).write_text(json.dumps(items, indent=1))
    Path(args.key).write_text(json.dumps(key, indent=1))
    return {'items': len(items), 'ways': sorted(ways)}


def cmd_blind_scenes(args: argparse.Namespace) -> dict[str, object]:
    """Blinded picture-check files from the ``kept`` lists of the arm files (one file per arm)."""
    coco = mem.load_coco()
    memories = mem.build_split(coco.supers, coco.all_ids).memories
    kept = {Path(p).stem: json.loads(Path(p).read_text())['kept'] for p in args.arms}
    items, key = blind_scenes(kept, memories)
    Path(args.items).write_text(json.dumps(items, indent=1))
    Path(args.key).write_text(json.dumps(key, indent=1))
    return {'items': len(items), 'arms': sorted(kept)}


def cmd_scene_ratings(args: argparse.Namespace) -> dict[str, object]:
    """``file``: ``key`` and ``wrong`` (item id -> the words the checker found wrong, possibly empty)."""
    data = json.loads(Path(args.file).read_text())
    return scene_check_summary(data['key'], {int(k): v for k, v in data['wrong'].items()})  # type: ignore[return-value]


def cmd_timing(args: argparse.Namespace) -> dict[str, object]:
    """A saved timing file (it holds ``summary``) is reported as saved; a raw one is summarised."""
    data = json.loads(Path(args.file).read_text())
    if 'summary' in data:
        return data['summary']  # type: ignore[no-any-return]
    return timing_summary(
        data['seconds'], data['p95_bar_s'], data.get('rss_mb'), data.get('rss_bar_mb')
    )


def cmd_ratings(args: argparse.Namespace) -> dict[str, object]:
    """``--file``: ``key`` (from ``blind``), ``ratings`` (item id -> 1 to 5) from the rating agent."""
    data = json.loads(Path(args.file).read_text())
    ratings = {int(k): float(v) for k, v in data['ratings'].items()}
    return rating_summary(ratings_by_way(data['key'], ratings), args.reference, args.candidate)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('checks')
    c.add_argument('rows', nargs='+')
    c.add_argument('--memory-words')
    c.set_defaults(func=cmd_checks)
    b = sub.add_parser('blind')
    b.add_argument('rows', nargs='+')
    b.add_argument('--items', required=True)
    b.add_argument('--key', required=True)
    b.add_argument('--memory-words')
    b.set_defaults(func=cmd_blind)
    bs = sub.add_parser('blind-scenes')
    bs.add_argument('arms', nargs='+')
    bs.add_argument('--items', required=True)
    bs.add_argument('--key', required=True)
    bs.set_defaults(func=cmd_blind_scenes)
    sr = sub.add_parser('scene-ratings')
    sr.add_argument('file')
    sr.set_defaults(func=cmd_scene_ratings)
    t = sub.add_parser('timing')
    t.add_argument('file')
    t.set_defaults(func=cmd_timing)
    r = sub.add_parser('ratings')
    r.add_argument('file')
    r.add_argument('--reference', required=True)
    r.add_argument('--candidate', required=True)
    r.set_defaults(func=cmd_ratings)
    args = p.parse_args(argv)
    json.dump(args.func(args), sys.stdout, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
