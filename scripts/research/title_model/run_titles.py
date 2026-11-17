"""Run R8 titles: the plain template, the template table and a text model, on the same fact sheets.

    python -m scripts.research.title_model.run_titles generate --way table --out titles_table.jsonl
    python -m scripts.research.title_model.run_titles generate --way model --model-path X.gguf \\
        --name smollm2-360m --out titles_smollm2-360m.jsonl
    python -m scripts.research.title_model.run_titles timing --model-path X.gguf --out timing.json

Run from the repository root with the run's venv. The text model runs on the CPU through
llama-cpp-python (llama.cpp), imported only when a model is used.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from scripts.research.title_model import fact_sheets as fs, memories as mem, score


TEMPLATES_PATH = mem.HERE / 'title_templates.json'
SYSTEM_PROMPT = (
    'You write short titles for photo memories. Use only the facts given. Never add a place, a name, '
    'a number, a date or an event that is not listed. Answer with one JSON object and nothing else.'
)
USER_PROMPT = (
    'Facts:\nDates: {dates}\nPlace: {place}\nPeople: {people}{named}\nEvent: {event}\n'
    'Scene words: {scene_words}\n\n'
    'Write a title of at most 8 words and a subtitle of at most 12 words. '
    'Answer as {{"title": "...", "subtitle": "..."}}.'
)
SCHEMA = {
    'type': 'object',
    'properties': {
        'title': {'type': 'string', 'maxLength': 50},
        'subtitle': {'type': 'string', 'maxLength': 70},
    },
    'required': ['title', 'subtitle'],
    'additionalProperties': False,
}
MAX_NEW_TOKENS = 48
THREADS = 4
CONTEXT = 1024


def load_templates(path: Path = TEMPLATES_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def dates_text(sheet: fs.FactSheet) -> str:
    a, b = sheet.start, sheet.end
    first = f'{a.day} {fs.MONTHS[a.month - 1]}'
    if a == b:
        return f'{first} {a.year}'
    if (a.year, a.month) == (b.year, b.month):
        return f'{a.day} to {b.day} {fs.MONTHS[b.month - 1]} {b.year}'
    if a.year == b.year:
        return f'{first} to {b.day} {fs.MONTHS[b.month - 1]} {b.year}'
    return f'{first} {a.year} to {b.day} {fs.MONTHS[b.month - 1]} {b.year}'


def render_plain(sheet: fs.FactSheet) -> str:
    month_year = f'{fs.MONTHS[sheet.start.month - 1]} {sheet.start.year}'
    return f'{sheet.place}, {month_year}' if sheet.place else month_year


def template_subtitle(sheet: fs.FactSheet) -> str:
    return f'{sheet.place}, {dates_text(sheet)}' if sheet.place else dates_text(sheet)


def variant_for(sheet: fs.FactSheet) -> str:
    if 2 <= len(sheet.named) <= 3:
        return 'group'
    if len(sheet.named) == 1:
        return 'person'
    return 'place' if sheet.place else 'plain'


def choose_family(sheet: fs.FactSheet, table: dict[str, Any]) -> tuple[str | None, str | None]:
    """(family, word) of the sheet: its event first, else its first kept scene word."""
    if sheet.event:
        if sheet.event in fs.HOLIDAYS:
            return table['holiday_family'], None
        entry = table['events'][sheet.event]
        return entry['family'], entry.get('word')
    if sheet.scene_words:
        word = sheet.scene_words[0]
        for name, family in table['families'].items():
            if word in family['members']:
                return name, word
    return None, None


def join_people(names: tuple[str, ...]) -> str:
    return names[0] if len(names) == 1 else f'{", ".join(names[:-1])} and {names[-1]}'


def render_table(sheet: fs.FactSheet, table: dict[str, Any]) -> str:
    family_name, word = choose_family(sheet, table)
    if family_name is None:
        return render_plain(sheet)
    family = table['families'][family_name]
    variant = variant_for(sheet)
    for v in (variant, *table['variant_fallback'][variant]):
        rows = family['word_rows'].get(word or '', {}).get(v) or family['rows'].get(v)
        if rows:
            break
    else:
        return render_plain(sheet)
    return str(rows[sheet.sheet_id % len(rows)]).format(
        place=sheet.place,
        person=sheet.named[0] if sheet.named else '',
        people=join_people(sheet.named) if sheet.named else '',
        month=fs.MONTHS[sheet.start.month - 1],
        year=sheet.start.year,
        event=sheet.event,
    )


def model_messages(sheet: fs.FactSheet) -> list[dict[str, str]]:
    named = f' (named: {join_people(sheet.named)})' if sheet.named else ''
    user = USER_PROMPT.format(
        dates=dates_text(sheet),
        place=sheet.place or 'none',
        people=sheet.people,
        named=named,
        event=sheet.event or 'none',
        scene_words=', '.join(sheet.scene_words) or 'none',
    )
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]


def load_model(path: str) -> Any:
    from llama_cpp import Llama

    return Llama(model_path=path, n_ctx=CONTEXT, n_threads=THREADS, seed=mem.SEED, verbose=False)


def ask_model(llm: Any, sheet: fs.FactSheet) -> tuple[str, float]:
    """One answer, no retry; returns the raw text and the seconds it took."""
    started = time.perf_counter()
    out = llm.create_chat_completion(
        messages=model_messages(sheet),
        response_format={'type': 'json_object', 'schema': SCHEMA},
        temperature=0.0,
        seed=mem.SEED,
        max_tokens=MAX_NEW_TOKENS,
    )
    return out['choices'][0]['message']['content'], time.perf_counter() - started


def peak_rss_mb() -> float:
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) / 1024
    raise RuntimeError('VmHWM not in /proc/self/status')


def build_sheets(memory_words_path: str | None, n: int = fs.N_SHEETS) -> list[fs.FactSheet]:
    vocab = mem.load_vocabulary()
    if memory_words_path:
        words = [tuple(w) for w in json.loads(Path(memory_words_path).read_text())]
    else:
        mem.extract_captions()
        coco = mem.load_coco()
        split = mem.build_split(coco.supers, coco.all_ids)
        truth = {
            p: mem.photo_truth(coco.captions[p], coco.categories.get(p, set()), vocab)
            for ph in split.memories
            for p in ph
        }
        words = mem.oracle_memory_words(split.memories, truth)
    return fs.make_sheets(words, vocab, mem.SEED, n)


def generate(args: argparse.Namespace) -> None:
    sheets = build_sheets(args.memory_words, args.n)
    table = load_templates()
    llm = load_model(args.model_path) if args.way == 'model' else None
    with Path(args.out).open('w') as out:
        for sheet in sheets:
            row: dict[str, Any] = {
                'sheet': sheet.sheet_id,
                'way': args.name or args.way,
                'raw': None,
            }
            if llm is not None:
                row['raw'], row['seconds'] = ask_model(llm, sheet)
                parsed = score.parse_answer(row['raw'])
                row['title'], row['subtitle'] = parsed if parsed else (None, None)
            else:
                plain = args.way == 'plain'
                row['title'] = render_plain(sheet) if plain else render_table(sheet, table)
                row['subtitle'] = template_subtitle(sheet)
            out.write(json.dumps(row) + '\n')


def timing(args: argparse.Namespace) -> None:
    """3 repetitions over the first ``--sheets`` sheets; one title at a time, 4 threads."""
    sheets = build_sheets(args.memory_words)[: args.sheets]
    llm = load_model(args.model_path)
    ask_model(llm, sheets[0])  # warm-up, not recorded
    reps: list[dict[str, Any]] = []
    for rep in range(args.reps):
        load = os.getloadavg()[0]
        secs = [ask_model(llm, s)[1] for s in sheets]
        reps.append({'rep': rep, 'load_1min': load, 'seconds': secs})
    every = [s for r in reps for s in r['seconds']]
    summary = score.timing_summary(every, 2.0, peak_rss_mb(), 1024.0)
    Path(args.out).write_text(
        json.dumps({'model': args.name, 'reps': reps, 'summary': summary}, indent=1)
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--way', choices=['plain', 'table', 'model'], required=True)
    g.add_argument('--model-path')
    g.add_argument('--name')
    g.add_argument('--out', required=True)
    g.add_argument('--n', type=int, default=fs.N_SHEETS)
    g.add_argument('--memory-words')
    g.set_defaults(func=generate)
    t = sub.add_parser('timing')
    t.add_argument('--model-path', required=True)
    t.add_argument('--name', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--sheets', type=int, default=30)
    t.add_argument('--reps', type=int, default=3)
    t.add_argument('--memory-words')
    t.set_defaults(func=timing)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
