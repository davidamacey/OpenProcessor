"""Run R8 arm A: scene words from the embeddings the product already computes.

    python -m scripts.research.title_model.scene_words_embed embed --encoder mobileclip2_s2 \\
        --api-url http://127.0.0.1:4603 --out vectors_mobileclip2_s2.npz
    python -m scripts.research.title_model.scene_words_embed embed --encoder pe_core_l14_336 \\
        --triton-url 127.0.0.1:4600 --out vectors_pe_core_l14_336.npz
    python -m scripts.research.title_model.scene_words_embed score --vectors V.npz --out result.json
    python -m scripts.research.title_model.scene_words_embed timing --vectors V.npz --out timing.json
    python -m scripts.research.title_model.scene_words_embed dryrun --out dryrun.json

MobileCLIP2-S2 vectors come from ``POST /embed/image`` and ``POST /embed/text`` of the product API. The
``/embed/*`` routes serve MobileCLIP only, so PE-Core-L14-336 is read from the product's Triton models
``pe_image_encoder`` and ``pe_text_encoder`` with the product's whole-frame preprocessing
(``src/utils/pe_preprocess.py``) and the PE tokenizer of the pinned ``perception_models``.
Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.title_model import memories as mem, score


HTTP_TIMEOUT_S = 60.0
PE_TEXT_CONTEXT = 32
PRECISION = 0.90
ENCODERS = {
    'mobileclip2_s2': ('mobileclip2_s2_image_encoder', 'mobileclip2_s2_text_encoder'),
    'pe_core_l14_336': ('pe_image_encoder', 'pe_text_encoder'),
}


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def phrase_table(vocabulary: list[mem.SceneWord]) -> tuple[list[str], np.ndarray]:
    """All phrases in word order, and the start offset of each word's phrases (for segment max)."""
    phrases: list[str] = []
    starts = []
    for w in vocabulary:
        starts.append(len(phrases))
        phrases.extend(w.phrases)
    return phrases, np.asarray(starts)


def word_scores(image_vecs: np.ndarray, text_vecs: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """``(photos, words)``: each word's score is the highest cosine of its phrases.

    Both inputs are L2-normalised, so the matrix product is the cosine.
    """
    return np.maximum.reduceat(image_vecs @ text_vecs.T, starts, axis=1)


def fit_thresholds(
    scores: np.ndarray, truth: np.ndarray, vocabulary: list[mem.SceneWord]
) -> dict[str, float | None]:
    """Per word the lowest score with precision >= 0.90 on the tuning photos, or ``None`` (dropped)."""
    return {
        w.word: score.pick_threshold(scores[:, j].tolist(), truth[:, j].tolist(), PRECISION)
        for j, w in enumerate(vocabulary)
    }


def photo_findings(
    row: np.ndarray, thresholds: dict[str, float | None], vocabulary: list[mem.SceneWord]
) -> dict[str, float]:
    """Words found in one photo: score at or above the word's threshold; value = score - threshold."""
    out = {}
    for j, w in enumerate(vocabulary):
        t = thresholds[w.word]
        if t is not None and row[j] >= t:
            out[w.word] = float(row[j] - t)
    return out


class MobileClipClient:
    """MobileCLIP2-S2 through the product API routes ``/embed/image`` and ``/embed/text``."""

    def __init__(self, api_url: str) -> None:
        import requests

        self.api, self.http = api_url.rstrip('/'), requests.Session()

    def image(self, path: Path) -> np.ndarray:
        with path.open('rb') as f:
            r = self.http.post(
                f'{self.api}/embed/image',
                files={'image': (path.name, f, 'image/jpeg')},
                timeout=HTTP_TIMEOUT_S,
            )
        r.raise_for_status()
        return np.asarray(r.json()['embedding'], dtype=np.float32)

    def text(self, phrase: str) -> np.ndarray:
        r = self.http.post(f'{self.api}/embed/text', json={'text': phrase}, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
        return np.asarray(r.json()['embedding'], dtype=np.float32)

    def versions(self) -> dict[str, str]:
        return {'route': '/embed/image,/embed/text'}


class PeClient:
    """PE-Core-L14-336 through Triton's ``pe_image_encoder`` and ``pe_text_encoder``."""

    def __init__(self, triton_url: str) -> None:
        import tritonclient.http as httpclient
        from core.vision_encoder.tokenizer import SimpleTokenizer

        self.mod = httpclient
        self.client = httpclient.InferenceServerClient(
            triton_url, connection_timeout=HTTP_TIMEOUT_S, network_timeout=HTTP_TIMEOUT_S
        )
        self.tokenizer = SimpleTokenizer(context_length=PE_TEXT_CONTEXT)

    def image(self, path: Path) -> np.ndarray:
        from src.utils.pe_preprocess import whole_frame_chw

        chw = whole_frame_chw(str(path))
        if chw is None:
            raise RuntimeError(f'cannot decode {path.name}')
        inp = self.mod.InferInput('images', [1, 3, 336, 336], 'UINT8')
        inp.set_data_from_numpy(chw[None])
        res = self.client.infer(
            'pe_image_encoder', [inp], outputs=[self.mod.InferRequestedOutput('image_embeddings')]
        )
        return np.asarray(res.as_numpy('image_embeddings')[0], dtype=np.float32)

    def text(self, phrase: str) -> np.ndarray:
        tokens = np.asarray(self.tokenizer([phrase]), dtype=np.int64)
        inp = self.mod.InferInput('text_tokens', list(tokens.shape), 'INT64')
        inp.set_data_from_numpy(tokens)
        res = self.client.infer(
            'pe_text_encoder', [inp], outputs=[self.mod.InferRequestedOutput('text_embeddings')]
        )
        return np.asarray(res.as_numpy('text_embeddings')[0], dtype=np.float32)

    def versions(self) -> dict[str, str]:
        return {
            name: ','.join(self.client.get_model_metadata(name).get('versions', ['unknown']))
            for name in ENCODERS['pe_core_l14_336']
        }


def photo_ids() -> tuple[list[int], list[int], mem.CocoIndex, list[list[int]]]:
    mem.extract_captions()
    coco = mem.load_coco()
    split = mem.build_split(coco.supers, coco.all_ids)
    memory_ids = [p for photos in split.memories for p in photos]
    return split.tuning, memory_ids, coco, split.memories


def embed(args: argparse.Namespace) -> None:
    client: Any = (
        MobileClipClient(args.api_url)
        if args.encoder == 'mobileclip2_s2'
        else PeClient(args.triton_url)
    )
    vocab = mem.load_vocabulary()
    phrases, _ = phrase_table(vocab)
    tuning, memory_ids, _, _ = photo_ids()
    ids = tuning + memory_ids
    images = np.stack([client.image(mem.image_path(p)) for p in ids])
    texts = np.stack([client.text(t) for t in phrases])
    np.savez(
        args.out,
        ids=np.asarray(ids),
        n_tuning=len(tuning),
        phrases=np.asarray(phrases),
        image=images,
        text=texts,
        encoder=args.encoder,
        versions=json.dumps(client.versions()),
    )


def evaluate(
    data: dict[str, Any],
    vocab: list[mem.SceneWord],
    tuning: list[int],
    memories: list[list[int]],
    truth: dict[int, set[str]],
) -> dict[str, Any]:
    """Fit thresholds on the tuning photos, keep words for the test memories, score them."""
    ids = list(data['ids'])
    row = {p: i for i, p in enumerate(ids)}
    _, starts = phrase_table(vocab)
    scores = word_scores(l2_normalize(data['image']), l2_normalize(data['text']), starts)
    words = [w.word for w in vocab]
    t_rows = [row[p] for p in tuning]
    tr = np.array([[w in truth[p] for w in words] for p in tuning])
    thresholds = fit_thresholds(scores[t_rows], tr, vocab)
    kept = [
        score.keep_words(
            [photo_findings(scores[row[p]], thresholds, vocab) for p in photos], rank='margin'
        )
        for photos in memories
    ]
    truths = [[truth[p] for p in photos] for photos in memories]
    return {
        'encoder': str(data['encoder']),
        'versions': json.loads(str(data['versions'])),
        'thresholds': thresholds,
        'dropped_words': sorted(w for w, t in thresholds.items() if t is None),
        'kept': kept,
        'summary': score.scene_summary(kept, truths),
    }


def _truth(coco: mem.CocoIndex, vocab: list[mem.SceneWord], ids: list[int]) -> dict[int, set[str]]:
    return {p: mem.photo_truth(coco.captions[p], coco.categories.get(p, set()), vocab) for p in ids}


def score_cmd(args: argparse.Namespace) -> None:
    data = dict(np.load(args.vectors))
    vocab = mem.load_vocabulary()
    tuning, memory_ids, coco, memories = photo_ids()
    truth = _truth(coco, vocab, tuning + memory_ids)
    Path(args.out).write_text(json.dumps(evaluate(data, vocab, tuning, memories, truth), indent=1))


def timing(args: argparse.Namespace) -> None:
    """Time the cosine of one photo against all phrases (embeddings given), 3 repetitions."""
    data = dict(np.load(args.vectors))
    vocab = mem.load_vocabulary()
    _, starts = phrase_table(vocab)
    text = l2_normalize(data['text'])
    images = l2_normalize(data['image'][int(data['n_tuning']) :])
    reps: list[dict[str, Any]] = []
    for rep in range(args.reps):
        load = os.getloadavg()[0]
        secs: list[float] = []
        for vec in images:
            started = time.perf_counter()
            word_scores(vec[None], text, starts)
            secs.append(time.perf_counter() - started)
        reps.append({'rep': rep, 'load_1min': load, 'seconds': secs})
    every = [s for r in reps for s in r['seconds']]
    summary = score.timing_summary(every, 0.001)
    Path(args.out).write_text(
        json.dumps({'encoder': str(data['encoder']), 'reps': reps, 'summary': summary}, indent=1)
    )


def dryrun(args: argparse.Namespace) -> None:
    """End to end on made-up vectors: 5 memories and 100 tuning photos, no service, no model."""
    vocab = mem.load_vocabulary()
    tuning, _, coco, memories = photo_ids()
    tuning, memories = tuning[:100], memories[:5]
    ids = tuning + [p for photos in memories for p in photos]
    phrases, _ = phrase_table(vocab)
    rng = np.random.default_rng(mem.SEED)
    data = {
        'ids': np.asarray(ids),
        'n_tuning': len(tuning),
        'encoder': 'made_up',
        'versions': '{}',
        'image': rng.normal(size=(len(ids), 512)),
        'text': rng.normal(size=(len(phrases), 512)),
    }
    truth = _truth(coco, vocab, ids)
    Path(args.out).write_text(json.dumps(evaluate(data, vocab, tuning, memories, truth), indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('embed')
    e.add_argument('--encoder', choices=sorted(ENCODERS), required=True)
    e.add_argument('--api-url')
    e.add_argument('--triton-url')
    e.add_argument('--out', required=True)
    e.set_defaults(func=embed)
    s = sub.add_parser('score')
    s.add_argument('--vectors', required=True)
    s.add_argument('--out', required=True)
    s.set_defaults(func=score_cmd)
    t = sub.add_parser('timing')
    t.add_argument('--vectors', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--reps', type=int, default=3)
    t.set_defaults(func=timing)
    d = sub.add_parser('dryrun')
    d.add_argument('--out', required=True)
    d.set_defaults(func=dryrun)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
