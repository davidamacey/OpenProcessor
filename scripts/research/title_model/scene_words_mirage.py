"""Run R8 arm B: scene words from Mirage descriptions (cactus-compute, CPU, nothing leaves the host).

    python -m scripts.research.title_model.scene_words_mirage describe --out descriptions.jsonl
    python -m scripts.research.title_model.scene_words_mirage timing --out timing.json
    python -m scripts.research.title_model.scene_words_mirage findings --descriptions D --out findings.json

Environment (the coordinator's command sets it): ``CACTUS_TELEMETRY=0 DO_NOT_TRACK=1
HF_HUB_OFFLINE=1`` and ``XDG_CACHE_HOME`` / ``HF_HOME`` on the folders that hold the fetched weights.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from scripts.research.title_model import memories as mem, score


THREADS = 4
_SKIP_KEYS = frozenset({'bbox', 'mask', 'score', 'id'})


def description_text(answer: Any) -> str:
    """Every string of Mirage's answer (scene setting, object names, attributes), joined by ' . '."""
    parts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key not in _SKIP_KEYS:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(answer)
    return ' . '.join(parts)


def found_words(text: str, vocabulary: list[mem.SceneWord]) -> dict[str, float]:
    """The scene words found in a description, each worth 1 (arm B ranks by how many photos)."""
    return dict.fromkeys(mem.find_words(text, vocabulary), 1.0)


def keep_for_memories(
    descriptions: dict[int, str], memories: list[list[int]], vocabulary: list[mem.SceneWord]
) -> list[list[str]]:
    return [
        score.keep_words([found_words(descriptions[p], vocabulary) for p in photos], rank='count')
        for photos in memories
    ]


VARIANTS = {
    'default': {},  # the published call, describe(file); its text section can hold `-nan`, invalid JSON
    'objects': {'sections': ['scene', 'objects'], 'fields': ['bbox']},
    'scene': {'sections': ['scene']},
}


def _options(variant: str) -> dict[str, Any]:
    return VARIANTS[variant]


def describe_once(mirage: Any, photo: int, variant: str) -> tuple[str, str | None, float]:
    """(description text, error or None, seconds). An engine answer that is not valid JSON is an
    error of that call: the photo has no description and the failure is counted, not hidden."""
    started = time.perf_counter()
    try:
        text, error = (
            description_text(mirage.describe(str(mem.image_path(photo)), **_options(variant))),
            None,
        )
    except (RuntimeError, ValueError) as exc:
        text, error = '', f'{type(exc).__name__}: {exc}'
    return text, error, time.perf_counter() - started


def _mirage() -> Any:
    import cactus

    cactus.set_threads(THREADS)
    return cactus.Mirage()


def _photo_ids(limit: int | None) -> list[int]:
    mem.extract_captions()
    coco = mem.load_coco()
    split = mem.build_split(coco.supers, coco.all_ids)
    ids = [p for photos in split.memories for p in photos]
    return ids[:limit] if limit else ids


def describe(args: argparse.Namespace) -> None:
    mirage = _mirage()
    done = set()
    if Path(args.out).exists():
        done = {json.loads(line)['photo'] for line in Path(args.out).read_text().splitlines()}
    with Path(args.out).open('a') as out:
        for photo in _photo_ids(args.limit):
            if photo in done:
                continue
            text, error, seconds = describe_once(mirage, photo, args.variant)
            out.write(
                json.dumps({'photo': photo, 'seconds': seconds, 'text': text, 'error': error})
                + '\n'
            )
            out.flush()


def timing(args: argparse.Namespace) -> None:
    """3 repetitions over ``--photos`` photos, one at a time, 4 threads, load beside each row."""
    from scripts.research.title_model.run_titles import peak_rss_mb

    mirage = _mirage()
    ids = _photo_ids(args.photos)
    describe_once(mirage, ids[0], args.variant)  # warm-up
    reps: list[dict[str, Any]] = []
    errors = 0
    for rep in range(args.reps):
        load = os.getloadavg()[0]
        calls = [describe_once(mirage, photo, args.variant) for photo in ids]
        errors += sum(c[1] is not None for c in calls)
        reps.append({'rep': rep, 'load_1min': load, 'seconds': [c[2] for c in calls]})
    every = [s for r in reps for s in r['seconds']]
    summary = {**score.timing_summary(every, 0.5, peak_rss_mb(), 500.0), 'calls_with_error': errors}
    Path(args.out).write_text(
        json.dumps({'variant': args.variant, 'reps': reps, 'summary': summary}, indent=1)
    )


def findings(args: argparse.Namespace) -> None:
    vocab = mem.load_vocabulary()
    descriptions = {}
    for line in Path(args.descriptions).read_text().splitlines():
        row = json.loads(line)
        descriptions[row['photo']] = row['text']
    coco = mem.load_coco()
    memories = mem.build_split(coco.supers, coco.all_ids).memories[: args.memories]
    kept = keep_for_memories(descriptions, memories, vocab)
    truths = [
        [mem.photo_truth(coco.captions[p], coco.categories.get(p, set()), vocab) for p in photos]
        for photos in memories
    ]
    out = {'kept': kept, 'summary': score.scene_summary(kept, truths)}
    Path(args.out).write_text(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('describe')
    d.add_argument('--out', required=True)
    d.add_argument('--variant', choices=sorted(VARIANTS), default='default')
    d.add_argument('--limit', type=int)
    d.set_defaults(func=describe)
    t = sub.add_parser('timing')
    t.add_argument('--out', required=True)
    t.add_argument('--variant', choices=sorted(VARIANTS), default='default')
    t.add_argument('--photos', type=int, default=40)
    t.add_argument('--reps', type=int, default=3)
    t.set_defaults(func=timing)
    f = sub.add_parser('findings')
    f.add_argument('--descriptions', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--memories', type=int, default=mem.N_MEMORIES)
    f.set_defaults(func=findings)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
