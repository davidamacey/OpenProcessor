"""Run R8 data: the scene vocabulary, the 60 test memories, the tuning set and the truth.

Folders come from the environment (``R8_COCO_IMAGES``, ``R8_ANNOTATIONS_DIR``, ``R8_RUN_DIR``); the
run's command sets them. Truth (a scene word is true for a photo): the word or a synonym is in one of the photo's five COCO
captions, or one of the word's proof categories is among the photo's COCO annotations.
"""

from __future__ import annotations

import json
import os
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path


SEED = 20261117
N_MEMORIES = 60
PHOTOS_PER_MEMORY = 3
N_TUNING = 1000

HERE = Path(__file__).resolve().parent
VOCABULARY_PATH = HERE / 'scene_vocabulary.json'

# The data and run folders are given by the environment, so no host path is written in this tree:
#   R8_COCO_IMAGES       the folder of the 5,000 val2017 pictures
#   R8_ANNOTATIONS_DIR   the folder that holds annotations_trainval2017.zip and its extracted copy
#   R8_RUN_DIR           where the run writes (captions extract, vectors, descriptions, titles)
_ENV_HELP = 'set {name} (see the module docstring of scripts/research/title_model/memories.py)'


def env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(_ENV_HELP.format(name=name))
    return Path(value)


def run_dir() -> Path:
    return env_path('R8_RUN_DIR')


def annotations_zip() -> Path:
    return env_path('R8_ANNOTATIONS_DIR') / 'annotations_trainval2017.zip'


def instances_path() -> Path:
    return (
        env_path('R8_ANNOTATIONS_DIR')
        / 'annotations_trainval2017'
        / 'annotations'
        / 'instances_val2017.json'
    )


def captions_path() -> Path:
    return run_dir() / 'data' / 'captions_val2017.json'


_NON_WORD = re.compile(r'[^a-z0-9]+')


@dataclass(frozen=True)
class SceneWord:
    word: str
    synonyms: tuple[str, ...]
    phrases: tuple[str, ...]
    proof: tuple[str, ...]

    @property
    def forms(self) -> tuple[str, ...]:
        """The word and its synonyms: every form that finds the word in a sentence."""
        return (self.word, *self.synonyms)


@dataclass(frozen=True)
class Split:
    memories: list[list[int]]
    tuning: list[int]


def load_vocabulary(path: Path = VOCABULARY_PATH) -> list[SceneWord]:
    rows = json.loads(path.read_text())['words']
    return [
        SceneWord(r['word'], tuple(r['synonyms']), tuple(r['phrases']), tuple(r['proof']))
        for r in rows
    ]


def normalize(text: str) -> str:
    """Lower case, every run of non-alphanumerics a single space, no edge spaces."""
    return _NON_WORD.sub(' ', text.lower()).strip()


def find_words(text: str, vocabulary: list[SceneWord]) -> set[str]:
    """Scene words whose word or synonym is in ``text`` as whole words ("beachfront" is not "beach")."""
    padded = f' {normalize(text)} '
    return {w.word for w in vocabulary if any(f' {normalize(f)} ' in padded for f in w.forms)}


def photo_truth(captions: list[str], categories: set[str], vocabulary: list[SceneWord]) -> set[str]:
    found = find_words(' . '.join(captions), vocabulary)
    found |= {w.word for w in vocabulary if categories & set(w.proof)}
    return found


def extract_captions() -> Path:
    """Extract ``captions_val2017.json`` from the annotations archive once."""
    zip_path, out_path = annotations_zip(), captions_path()
    if not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            out_path.write_bytes(zf.read('annotations/captions_val2017.json'))
    return out_path


@dataclass(frozen=True)
class CocoIndex:
    """Per photo: category names, supercategory names and the five captions."""

    categories: dict[int, set[str]]
    supers: dict[int, set[str]]
    captions: dict[int, list[str]]
    all_ids: list[int]


def load_coco() -> CocoIndex:
    inst = json.loads(instances_path().read_text())
    cat = {c['id']: (c['name'], c['supercategory']) for c in inst['categories']}
    categories: dict[int, set[str]] = {}
    supers: dict[int, set[str]] = {}
    for a in inst['annotations']:
        name, sup = cat[a['category_id']]
        categories.setdefault(a['image_id'], set()).add(name)
        supers.setdefault(a['image_id'], set()).add(sup)
    captions: dict[int, list[str]] = {}
    for a in json.loads(captions_path().read_text())['annotations']:
        captions.setdefault(a['image_id'], []).append(a['caption'])
    ids = sorted(int(i['id']) for i in inst['images'])
    return CocoIndex(categories, supers, captions, ids)


def build_split(
    supers: dict[int, set[str]],
    all_ids: list[int],
    seed: int = SEED,
    n_memories: int = N_MEMORIES,
    n_tuning: int = N_TUNING,
) -> Split:
    """60 memories of 3 photos that share a supercategory, then 1,000 other photos for tuning.

    Supercategories are dealt round-robin from a seeded shuffle, so every one has the same number
    of memories (+/- 1); no photo is used twice anywhere.
    """
    rng = random.Random(seed)
    names = sorted({s for ss in supers.values() for s in ss})
    rng.shuffle(names)
    pools = {n: sorted(i for i, ss in supers.items() if n in ss) for n in names}
    used: set[int] = set()
    memories: list[list[int]] = []
    for i in range(n_memories):
        pool = [p for p in pools[names[i % len(names)]] if p not in used]
        picked = sorted(rng.sample(pool, PHOTOS_PER_MEMORY))
        used.update(picked)
        memories.append(picked)
    rest = [i for i in all_ids if i not in used]
    return Split(memories, sorted(rng.sample(rest, n_tuning)))


def image_path(photo_id: int) -> Path:
    return env_path('R8_COCO_IMAGES') / f'{photo_id:012d}.jpg'


def oracle_memory_words(
    memories: list[list[int]], truth: dict[int, set[str]], limit: int = 3
) -> list[tuple[str, ...]]:
    """Per memory the words true for at least 2 of its 3 photos, most photos first, at most ``limit``.

    The fact sheets of the title run use these (the words a perfect scene-words step would keep), so
    title quality is measured on its own and one title run serves every scene-words decision.
    """
    out = []
    for photos in memories:
        counts: dict[str, int] = {}
        for p in photos:
            for w in truth[p]:
                counts[w] = counts.get(w, 0) + 1
        kept = sorted((w for w, c in counts.items() if c >= 2), key=lambda w: (-counts[w], w))
        out.append(tuple(kept[:limit]))
    return out
