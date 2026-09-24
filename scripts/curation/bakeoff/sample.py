"""Cluster-stratified representative sampler for the bake-off datasets.

To compare models across datasets of very different sizes (and with known
near-duplicate contamination, e.g. Roboflow), we draw a *representative*
fixed-size sample from each dataset rather than scoring whole test splits of
unequal size. Sampling is embedding-cluster-stratified using the SAME encoder
our own model's data pipeline uses --- PE-Core-L14-336 served on Triton --- so
the sample-selection method is identical for every dataset and cannot favor
ours. Within each dataset we cluster the test frames by PE embedding
(agglomerative, cosine) and draw proportionally from every cluster, which both
covers the appearance space evenly and collapses near-duplicate clusters to a
representative few.

Output is a standard ``images/test`` + ``labels/test`` tree (symlinked) plus a
``SAMPLE.json`` manifest, ready to freeze with :mod:`freeze` and score with the
normal harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


_PE_INPUT_SIZE = 336
# Must equal src/services/detection/pe_preprocess.PE_MEAN/PE_STD (PE-Core's own
# 0.5/0.5 normalization); tests/curation/test_pe_preprocess.py pins the two together.
_PE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
_PE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
_PE_MODEL = 'pe_image_encoder'
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def _preprocess_for_pe(img: Image.Image, target: int = _PE_INPUT_SIZE) -> np.ndarray:
    """Resize-by-shorter-edge + center-crop to PE-Core 336 PE-normalized CHW.

    Identical convention to the production PE image path so embeddings match.
    """
    rgb = img.convert('RGB')
    w, h = rgb.size
    if w == 0 or h == 0:
        return np.zeros((3, target, target), dtype=np.float32)
    scale = target / min(w, h)
    new_w = max(target, round(w * scale))
    new_h = max(target, round(h * scale))
    resized = rgb.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - target) // 2
    upper = (new_h - target) // 2
    centered = resized.crop((left, upper, left + target, upper + target))
    arr = np.asarray(centered, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
    return ((chw - _PE_MEAN) / _PE_STD).astype(np.float32, copy=False)


def _list_images(src: Path, split: str) -> list[Path]:
    img_dir = src / 'images' / split
    if not img_dir.is_dir():
        raise SystemExit(f'no images/{split} under {src}')
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)


def _embed(images: list[Path], triton_url: str, batch: int = 32) -> np.ndarray:
    """Embed every image with PE-Core on Triton; returns (N, 1024) L2-norm."""
    from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

    client = InferenceServerClient(url=triton_url)
    out: list[np.ndarray] = []
    for i in range(0, len(images), batch):
        chunk = images[i : i + batch]
        tensors = np.stack([_preprocess_for_pe(Image.open(p)) for p in chunk]).astype(np.float32)
        inp = InferInput('images', list(tensors.shape), 'FP32')
        inp.set_data_from_numpy(np.ascontiguousarray(tensors))
        res = client.infer(_PE_MODEL, [inp], outputs=[InferRequestedOutput('image_embeddings')])
        out.append(np.asarray(res.as_numpy('image_embeddings'), dtype=np.float32))
    emb = np.vstack(out)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norms == 0, 1.0, norms)


def _cluster(emb: np.ndarray, distance_threshold: float) -> np.ndarray:
    """Agglomerative clustering (cosine, average linkage) -> labels.

    Same cosine/threshold family as the production residual clustering.
    """
    from sklearn.cluster import AgglomerativeClustering

    if len(emb) == 1:
        return np.zeros(1, dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average',
    )
    return model.fit_predict(emb)


def _stratified_sample(labels: np.ndarray, n: int, seed: int) -> list[int]:
    """Pick ~n indices proportionally across clusters (>=1 per cluster)."""
    rng = np.random.default_rng(seed)
    clusters = {int(c): np.where(labels == c)[0] for c in np.unique(labels)}
    total = len(labels)
    if n >= total:
        return list(range(total))
    picked: list[int] = []
    for members in clusters.values():
        take = max(1, round(n * len(members) / total))
        take = min(take, len(members))
        picked.extend(int(i) for i in rng.choice(members, size=take, replace=False))
    # Trim/pad to exactly n deterministically.
    picked = sorted(set(picked))
    if len(picked) > n:
        picked = list(rng.choice(picked, size=n, replace=False))
    return sorted(picked)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', required=True, help='dataset root with images/<split>+labels/<split>')
    ap.add_argument('--out', required=True, help='output root for the sampled dataset')
    ap.add_argument('--split', default='test')
    ap.add_argument('--n', type=int, default=200, help='target sample size')
    ap.add_argument('--distance-threshold', type=float, default=0.25)
    ap.add_argument('--triton-url', default='localhost:4601')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    images = _list_images(src, args.split)
    print(f'{len(images)} images in {src.name}/{args.split}; embedding via PE on Triton...')
    emb = _embed(images, args.triton_url)
    labels = _cluster(emb, args.distance_threshold)
    n_clusters = len(np.unique(labels))
    idx = _stratified_sample(labels, args.n, args.seed)
    print(f'{n_clusters} clusters -> sampled {len(idx)} frames')

    img_out = out / 'images' / args.split
    lbl_out = out / 'labels' / args.split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for i in idx:
        img = images[i]
        (img_out / img.name).unlink(missing_ok=True)
        (img_out / img.name).symlink_to(img.resolve())
        lbl_src = src / 'labels' / args.split / f'{img.stem}.txt'
        text = lbl_src.read_text(encoding='utf-8') if lbl_src.is_file() else ''
        (lbl_out / f'{img.stem}.txt').write_text(text)

    manifest = {
        'source': str(src),
        'split': args.split,
        'encoder': _PE_MODEL,
        'n_source': len(images),
        'n_clusters': n_clusters,
        'n_sampled': len(idx),
        'distance_threshold': args.distance_threshold,
        'seed': args.seed,
    }
    (out / 'SAMPLE.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'wrote sample -> {out} ({len(idx)} frames, {n_clusters} clusters)')


if __name__ == '__main__':
    main()
