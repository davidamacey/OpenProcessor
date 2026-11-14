# Face grouping evaluation

Scores how faces are grouped into people on public sets with identity labels (LFW, CALFW, CPLFW), on the
embeddings the stack serves (SCRFD + ArcFace `w600k_r50`). Images, crops and vectors are never committed;
scripts, manifests (hashes) and aggregate scores are.

Dependencies: `scripts/research/requirements.txt` (`finch-clust`, `chinese-whispers`); the rest is in
`requirements-test.txt`. Run Python that reads downloaded data with `-I`.

1. Copy each set read-only to scratch (the `data` link of its catalog data card): LFW `images/`,
   `labels.jsonl`, `manifest.json` and `source/pairs.txt`; CALFW and CPLFW `aligned images/`, `labels.jsonl`,
   `manifest.json` and the pairs file. Each `labels.jsonl` row is `{"identity", "path"}`; the identity is the
   directory name (LFW) or the file-name prefix before the trailing index (CALFW, CPLFW).
2. Embeddings (needs the stack; one GPU run, hold the host lock):
   `python -I scripts/research/face_grouping/extract_embeddings.py <dir>/<set> <dir>/<set>.npz`
   sends every image to `POST /faces/recognize` with 8 clients and stores every detected face (box, landmarks,
   score, route quality, embedding).
3. Scores (CPU only): `python -I scripts/research/face_grouping/run_eval.py <dir> <out_dir>` writes
   `results.json` and `results.md`.

Modules: `face_table` (main face per image, pairs files), `verification` (10-fold accuracy, TAR at FAR),
`scores` (identity report; pair and BCubed come from `scripts/bench/cluster_eval_metrics.py`), `grouping`
(single linkage, Immich-style core points, average and complete linkage, mutual kNN with a cluster merge, FINCH,
Chinese Whispers), `quality` (signals, reject curve, quality gate), `incremental`, `sweep`, `evaluate`, `report`.

Immich rule (`handleRecognizeFaces`): a face with at least `min_faces` hits (itself included) within
`max_distance` may found a group; it joins the group of its first hit that has one, else its nearest grouped
neighbour within `max_distance`; a face with no group is deferred once; only a core face creates a group.
The "nearest grouped neighbour" search is limited to the 30 exact neighbours the graph stores.

`python -I` ignores `PYTHONPATH`: if `finch` or `chinese_whispers` live outside the venv, append their directory
with `sys.path.append(...)` and `runpy.run_path('scripts/research/face_grouping/run_eval.py', run_name='__main__')`.
The scoring takes about an hour on 8 niced threads (pooled average and complete linkage dominate).
