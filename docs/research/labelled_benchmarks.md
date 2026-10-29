# Labelled benchmarks for embedding and clustering evaluation

Snapshot of 2026-10-29, written on branch `research/labelled-benchmarks` from `origin/main` at
`0e2d3fe70`. Host: the dev workstation, CPU and network only; no GPU used, no model run. Every
external claim has a URL; a figure I could not open at the source is labelled as such. No number
here is a measurement of OpenProcessor.

## Why

The Commons set C result for the embedding and clustering work (the "WB" run) rests on labels taken
from Wikimedia category names. [cluster_proxy_datasets.md](cluster_proxy_datasets.md) already says
the label error rate there is unknown (the 10 % review sample was never marked), and a label check
was inconclusive. A ranking of embedding models or clustering methods that rests on those labels is
weak evidence. This note lists benchmarks whose labels were produced and checked by the dataset
authors, with a published split, metric and protocol, so a result can be compared with the
literature and the pipeline can be sanity-checked before any model decision.

Decision status: the embedding choice (#347, #365) is **pending a re-run on the sets below**.

## Summary of what is usable

| Kind | First benchmark | Local status |
|---|---|---|
| Vehicle make/model (our WB analogue) | CARS196 (Stanford Cars), 196 make/model/year classes | downloaded, indexed |
| Vehicle make/model, scale and noise | VMMRdb, 9,169 classes, 285,086 images in the zip | downloaded, indexed |
| Generic fine-grained | CUB-200-2011 | downloaded, indexed |
| Whole images / products | Stanford Online Products (SOP) | downloaded, indexed |
| Faces | LFW (already local), CALFW, CPLFW; IJB-B/C clustering needs a request | CALFW, CPLFW downloaded |
| Vehicle re-identification, CompCars, BoxCars, In-shop, IJB, MS1M, VGGFace2 | access by request, signed form, withdrawn or host dead | skipped, reasons below |

## 1. Vehicle benchmarks

| Set | What | Access and licence (checked 2026-10-29) | Status |
|---|---|---|---|
| CARS196 / Stanford Cars | 16,185 images, 196 classes (make, model, body, year); 8,144 train / 8,041 test images ([paper page](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), now unreachable from here: connection refused) | Original host dead. Community mirror [tanganke/stanford_cars](https://huggingface.co/datasets/tanganke/stanford_cars): parquet, train 8,144 and test 8,041 rows, the 196 class names match the devkit order, no licence on the card. Other mirrors ([lance-format](https://huggingface.co/datasets/lance-format/stanford-cars-lance), [mteb](https://huggingface.co/datasets/mteb/StanfordCarsZeroShot), Kaggle) say "other", "unknown" or nothing; lance-format states the original terms are non-commercial research. Treat as non-commercial research: local evaluation only, never redistribute | downloaded |
| VMMRdb | 9,170 classes (make, model, year), 291,752 images from classified-ad photos, 1950 to 2016 ([CVPR-W 2017](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w9/papers/Tafazzoli_A_Large_and_CVPR_2017_paper.pdf)) | [Repo](https://github.com/faezetta/VMMRdb) carries an MIT licence (LICENSE file read); data is a 12.35 GB Dropbox zip that answered. This corrects the earlier "no licence found, host did not answer" in the proxy note. Labels come from ad titles: noisy and long-tailed, US-centric | downloaded, indexed |
| CompCars | 163 makes, 1,716 models, 136,726 web images with boxes and viewpoint, plus 50,000 surveillance images ([page](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)) | Non-commercial research only, no redistribution; password-protected archives with the password issued on a signed release form | skipped: needs the signed agreement. `fetch_vehicle_proxy.py compcars --root` indexes a licensed copy |
| VeRi-776 | over 50,000 images of 776 vehicles, 20 cameras ([page](https://vehiclereid.github.io/VeRi/)) | Email request (name, affiliation), non-commercial, not shared with third parties | skipped: request-only. Re-identification, not make/model |
| VeRi-Wild | 416,314 images, 40,671 identities, 174 cameras ([repo](https://github.com/PKU-IMRE/VERI-Wild)) | Email request, non-commercial, no public redistribution | skipped: request-only |
| VehicleID | 221,763 images, 26,267 vehicles (survey figure, [arXiv 2102.09744](https://arxiv.org/pdf/2102.09744)) | No public download process found | skipped: no access path found |
| BoxCars116k | 116k images with fine-grained labels, traffic cameras ([repo](https://github.com/JakubSochor/BoxCars)) | Zip on `medusa.fit.vutbr.cz` and a Google Drive backup; repo states "research only", no dataset licence | skipped: the host did not resolve from this machine; no licence stated |

Re-identification sets (VeRi, VehicleID, VeRi-Wild) measure identity of one physical vehicle across
cameras, which is a different question from make/model clustering; they are the right benchmark
only for an instance-level (same car) feature. All three are request-only, so none is fetched.

## 2. Deep-metric-learning and clustering benchmarks

### Protocols (as papers report them)

| Set | Train / test | Metric protocol | Source |
|---|---|---|---|
| CUB-200-2011 | 200 bird species, 11,788 images; classes 1 to 100 train (5,864 images), 101 to 200 test (5,924). Zero-shot: no class overlap | Recall@K, K in 1,2,4,8,16,32: a query scores 1 if any of its K nearest neighbours (query itself excluded) has the same class; NMI of k-means (K = number of test classes) on the test embeddings; some papers add F1 | [Hierarchical Triplet Loss](https://arxiv.org/pdf/1810.06951) |
| CARS196 | classes 1 to 98 train, 99 to 196 test (8,131 test images) | same as CUB | [arXiv 1912.08275](https://arxiv.org/pdf/1912.08275) and [2003.02546](https://arxiv.org/pdf/2003.02546) |
| Stanford Online Products | 120,053 images, 22,634 products, 12 coarse categories; first 11,318 products train (59,551 images), remaining 11,316 test (60,502). Query set = gallery = test set | Recall@K, K in 1,10,100,1000; NMI as above | [Oh Song et al. 2016](https://arxiv.org/pdf/1511.06452); split numbers per [ChainerCV loader](https://github.com/chainer/chainercv/blob/master/chainercv/datasets/online_products/online_products_dataset.py) |
| In-shop Clothes | 3,997 classes train (25,882 images), 3,985 test (28,760); query 14,218, gallery 12,612 | Recall@K, K in 1,10,20,30,40,50 (some papers 1,2,4,8,16,32); a hit if any of the top K gallery images matches | [Hierarchical Triplet Loss](https://arxiv.org/pdf/1810.06951) |

Pretrained, frozen encoders (our case) need no training, so the train classes are optional: use the
**test classes only** (for CARS196 and CUB the 98 or 100 held-out classes, for SOP the test half)
to stay comparable with the table below, and optionally report the full set as a second row.

Clustering metrics: NMI (arithmetic normalisation, as in scikit-learn), pairwise F-score, BCubed
F-score; for k-means-based DML numbers K equals the class count.

In-shop (DeepFashion) is non-commercial research only, behind a Google Drive download after a
[licence agreement](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html); skipped. It adds
nothing for vehicles or faces.

### Published numbers, and how far they transfer

Honest statement: I did **not** find a paper that tabulates frozen CLIP, SigLIP and DINOv2
Recall@1 and NMI on CUB, CARS196 and SOP under the DML protocol. What exists:

1. **NMI, frozen embeddings, generic clustering** (MIEB, [arXiv 2504.10471](https://arxiv.org/abs/2504.10471),
   Table 9, NMI x 100, as read from the HTML version of the paper). Not the fine-grained sets, but
   a published cross-check for the clustering code path on frozen CLIP, SigLIP and DINOv2:

   | Model | CIFAR10 | CIFAR100 | ImageNet10 | ImageNetDog15 | TinyImageNet |
   |---|---|---|---|---|---|
   | dinov2-large | 79.90 | 79.93 | 92.23 | 86.20 | 77.22 |
   | dinov2-base | 82.62 | 77.20 | 93.93 | 85.67 | 74.31 |
   | clip-vit-large-patch14 (OpenAI) | 80.87 | 64.54 | 94.00 | 72.83 | 69.82 |
   | clip-vit-base-patch16 (OpenAI) | 69.25 | 59.35 | 92.58 | 63.25 | 62.90 |
   | siglip-so400m-patch14-384 | 83.79 | 76.67 | 98.19 | 83.57 | 68.31 |
   | siglip-base-patch16-224 | 76.11 | 67.01 | 92.61 | 78.18 | 58.58 |

   MIEB also defines `CUB200I2IRetrieval`, `StanfordCarsI2IRetrieval` and `SOPI2IRetrieval`
   (Recall@1) in its task list, so per-model numbers on exactly these sets exist in the MIEB
   leaderboard; I could not open that table, so the values are **not** quoted here.
2. **Linear-probe accuracy on Cars** (MIEB fine-grained table, accuracy): dinov2-giant 83.18,
   dinov2-base 78.53, clip-vit-large-patch14 87.43, clip-vit-base-patch16 80.58,
   siglip-so400m-patch14-384 95.75. Classification, not retrieval; shows the ordering SigLIP >
   CLIP > DINOv2 on cars, the reverse of the generic-clustering table above.
3. **Fine-tuned ViT-S/16, 384-d, Recall@1** ([Hyp-ViT, arXiv 2203.10833](https://arxiv.org/pdf/2203.10833),
   as summarised from the paper): Sph-DINO CUB 78.7 / Cars 86.6 / SOP 82.2 / In-shop 90.1;
   Hyp-DINO 80.9 / 89.2 / 85.1 / 92.4; Hyp-ViT (ImageNet-21k) 85.6 / 86.5 / 85.9 / 92.5. These are
   trained on the train classes, so they are an upper reference for a frozen encoder, not a target.
4. **Do not use** the FAISS-index repositories such as
   [pawlo2013/StanfordOnlineProducts_dino3s](https://huggingface.co/datasets/pawlo2013/StanfordOnlineProducts_dino3s)
   (frozen DINOv3-S, reported R@1 64.87) or `Cars196_dino3s` (R@1 96.77): the Cars index holds
   109,679 embeddings and the SOP index 171,079, more than the datasets have, so the protocol is
   not the standard one and the numbers are not comparable.

Pipeline sanity check, in order: (a) reproduce the MIEB NMI rows for one DINOv2 and one CLIP model
on ImageNet10 or CIFAR10 within a few points (needs the ImageNet subset or CIFAR; CIFAR is small
and 32 px, so it only checks the clustering code, not crops); (b) on CARS196 test classes compute
Recall@1/2/4/8 and k-means NMI for the encoders and confirm the ordering is stable across the
98 test classes and that numbers sit below the fine-tuned reference in item 3; (c) only then run
the repo's own cluster pass and score it against the same labels. Any frozen number computed here
is a new measurement and must say so; it is not a reproduction of a published one.

## 3. Face benchmarks and clustering protocols

| Set | What | Licence / status | Local |
|---|---|---|---|
| LFW | 13,233 images, 5,749 identities; 6,000-pair verification, 10 folds ([site](https://vis-www.cs.umass.edu/lfw/), unreachable here, figshare copy used) | No licence; web news photos | already at `clustering/lfw` |
| CALFW | LFW identities, 3,000 positive pairs with age gaps plus 3,000 negative; lists updated 2018-09-19 ([page](http://whdeng.cn/CALFW/index.html)) | No licence stated; Google Drive and Baidu | downloaded: 12,174 aligned images, 4,025 identities |
| CPLFW | same with pose gaps ([page](http://www.whdeng.cn/CPLFW/index.html)) | no licence stated | downloaded: 11,652 aligned images, 3,933 identities |
| IJB-B / IJB-C | NIST unconstrained video and still faces | access through a NIST request and agreement ([request page](https://nigos.nist.gov/datasets/ijbc/request)) | skipped: request-only |
| MS-Celeb-1M | 5.8M images, 86K identities | Withdrawn by Microsoft in June 2019 ([CACM](https://cacm.acm.org/news/237357-microsoft-quietly-deletes-largest-public-face-recognition-dataset/fulltext)); copies circulate | skipped: withdrawn, no licence |
| CASIA-WebFace | 494,414 images, 10,575 identities, scraped from IMDb | Research-only; mirrors on Hugging Face and Kaggle | skipped: no clear licence, not needed for the protocols below |
| VGGFace2 | 3.3M images | The Oxford page says [the download links are no longer available](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/); mirrors exist with `unknown` or `cc-by-nc-4.0` | skipped: official distribution ended |

CALFW and CPLFW are **verification** sets (pairs). They contain the identity labels needed for
clustering (the identity is the file-name stem), and they give harder age and pose variants of LFW
identities, but there is no published clustering protocol on them. Use them as extra labelled
face pools and use the pair files for the standard verification accuracy as the sanity check
of the face embedding.

### Face clustering protocol in the literature

- **MS1M protocol** (Yang et al., L-GCN, then [STAR-FC](https://openaccess.thecvf.com/content/CVPR2021/papers/Shen_Structure-Aware_Face_Clustering_on_a_Large-Scale_Graph_With_107_Nodes_CVPR_2021_paper.pdf),
  [Ada-NETS](https://arxiv.org/pdf/2202.03800)): MS-Celeb-1M is split by identity into 10 parts of
  about 0.58 M images; part 0 trains, parts 1 to 9 test, evaluated on 1, 3, 5, 7 and 9 parts
  (584 K to 5.21 M images). Metrics: pairwise F-score (Fp) and BCubed F-score (Fb); NMI is also
  reported by the code base. Pairwise F at 584 K / 1.74 M / 2.89 M / 4.05 M / 5.21 M, as quoted in
  the Ada-NETS table: L-GCN 78.68 / 75.83 / 74.29 / 73.70 / 72.99, STAR-FC 91.97 / 88.28 / 86.17 /
  84.70 / 83.46, Ada-NETS 92.79 / 89.33 / 87.50 / 85.40 / 83.99. The Fb columns were garbled in my
  extraction and are not quoted. The data is MS-Celeb-1M, withdrawn, so this protocol cannot be
  run as published.
- **IJB-B protocol**: three subsets of 512, 1,024 and 1,845 identities (18,171, 36,575 and 68,195
  images); training on a random CASIA subset (5K identities, 200K images). The FaceT paper warns
  the CASIA subsampling is not aligned across papers
  ([arXiv 2104.11502](https://arxiv.org/pdf/2104.11502)). L-GCN is quoted at 83.3 / 83.3 / 81.4
  (F512, F1024, F1845) in the [pairwise-classification paper](https://arxiv.org/pdf/2205.13117);
  treat as unverified.
- Metric code: [yl-1993/learn-to-cluster](https://github.com/yl-1993/learn-to-cluster) (MIT)
  implements pairwise, BCubed and NMI; its data is on Google Drive without a stated licence.

Because the published face-clustering data is either withdrawn or request-only, the runnable
face check here is: embed LFW/CALFW/CPLFW, run the verification protocol (10-fold accuracy on the
supplied pairs) as the embedding sanity check, then cluster identities with at least 2 images and
report pairwise F, BCubed F and NMI against the identity labels. This is our own protocol on
public identities, not a reproduction of the MS1M table.

## 4. Vector databases that come with labels

Searched; honest outcome:

- **ann-benchmarks** ([repo](https://github.com/erikbern/ann-benchmarks)): SIFT, GIST, GloVe,
  Fashion-MNIST, MNIST, NYTimes as HDF5 with train, test and top-100 exact neighbours. Ground truth
  is nearest neighbours, not class labels. Useful for index speed and recall, useless for
  clustering quality.
- **VectorDBBench** ([repo](https://github.com/zilliztech/vectordbbench)): Cohere/OpenAI text
  embeddings, SIFT, GIST, filtered-search workloads; neighbour ground truth. Same limitation.
- **Big-ANN NeurIPS'23 filtered track** ([page](https://big-ann-benchmarks.com/neurips23.html)):
  10M YFCC images as 192-d CLIP vectors with tag sets (200,386-word vocabulary), CC BY 4.0. These
  are free-text tags used as filters, not curated class labels.
- **BigVectorBench** ([repo](https://github.com/BenchCouncil/BigVectorBench)): ag_news text vectors
  with a class id used as a filter; a Wikipedia image set with text. Labels serve filtering, recall
  is still scored against exact neighbours.
- A [Frontiers paper](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1819991/full)
  filters CIFAR-10 and STL-10 embeddings by class label inside FAISS, Milvus, Qdrant and pgvector;
  recall is again against exact search.
- **Hugging Face index repos with labels**: `pawlo2013/*_dino3*` repositories ship a
  `faiss_index.bin` plus a CSV mapping rows to file, split and label. This is the only
  "vector index with labels" artifact I found, and its protocol is non-standard (see above).
  [Marqo/marqo-GS-10M](https://huggingface.co/datasets/Marqo/marqo-GS-10M) is a labelled
  (graded-relevance) shopping dataset, not an index; I did not verify its licence and did not fetch it.

Conclusion: no established benchmark scores a vector database against curated class labels.
The practical route is to embed a labelled dataset from section 2 with our own encoder, and score
label-aware precision@k (the fraction of the K neighbours with the same class), which is exactly the
Recall@K / precision protocol of the DML papers. If a specific product was meant by "a vector
database with labels", I could not identify which; the owner should name it.

## 5. Recommended minimal suite

| Cluster kind | Set | Size | Licence | Download | Split and metrics | Reproduce first |
|---|---|---|---|---|---|---|
| Vehicle make/model | **CARS196** | 16,185 images, 196 classes, 0.8 GB parquet | non-commercial research (mirror: none stated) | done | the 98 test classes (8,131 images); Recall@1/2/4/8, k-means NMI with K=98; also run our cluster pass and score ARI, NMI, BCubed against the class labels, and against make only | Hyp-ViT/Hyp-DINO table as an upper reference; MIEB ordering SigLIP > CLIP > DINOv2 for cars |
| Vehicle, scale | VMMRdb | 285,086 images in the zip (published 291,752), 9,169 classes (published 9,170) | MIT (repo); photos from ads | downloaded, indexed | classes with at least 40 images, sampled 40 per class like set C; same metrics. Noisy labels: report with that caveat | none published for frozen encoders |
| Generic objects (fine-grained) | **CUB-200-2011** | 11,788 images, 200 classes, 1.2 GB | CC-BY record (archive has no licence text) | downloaded, indexed | classes 101 to 200; Recall@K, NMI | Hyp-ViT/Hyp-DINO row |
| Whole images / products | **SOP** | 120,053 images, 22,634 products, 3.8 GB | MIT | done | test half (60,502 images, 11,316 products), Recall@1,10,100,1000 | Hyp-ViT/Hyp-DINO row; MIEB `SOPI2IRetrieval` |
| Whole images, coarse | ImageNet10 / ImageNetDog15 / CIFAR | from ImageNet (research terms) | not fetched | not fetched | NMI | MIEB Table 9 |
| Faces | LFW, CALFW, CPLFW | 13,233 / 12,174 / 11,652 images | none stated, local only | done | verification accuracy on the pairs, then identity clustering | no published reference found for our encoders; compare against the LFW pairs result of the same model |

**Recommended first benchmark to re-run WB on: CARS196, test classes.** It is the direct analogue
of set C (make, model and year for passenger cars), has labels written by the dataset authors,
is small enough for a 10-minute run, has a fixed zero-shot protocol, and has a ready fine-tuned
upper reference. VMMRdb is the follow-up for scale. If a result on CARS196 and one on SOP agree
on the embedding ranking, that is the evidence the decision (#347, #365) needs.

## Status of downloads

Directory root: `<datasets root>/clustering/<name>/`; each has `manifest.json` (SHA-256 of every
file), `labels.jsonl` and the original archive or parquet kept in place. Nothing is committed to
the repository.

| Set | Directory | Images and classes | On disk | Check done |
|---|---|---|---|---|
| CARS196 | `stanford_cars` | 16,185 images (train 8,144, test 8,041), 196 classes; DML test classes 8,131 images | 3.8 GB (parquet kept, images extracted) | counts equal the published 16,185 / 8,144 / 8,041 |
| Stanford Online Products | `stanford_online_products` | 120,053 images; train 59,551 images in 11,318 products, test 60,502 in 11,316 | 6.1 GB | counts equal the published protocol exactly |
| CUB-200-2011 | `cub200_2011` | 11,788 images, 200 classes; DML train 5,864, test 5,924 | 2.3 GB (archive kept) | archive MD5 matches the CaltechDATA value |
| VMMRdb | `vmmrdb` | 285,086 images in 9,169 class folders (make_model_year); 3,704 classes have at least 15 images, 1,915 at least 40 | 24 GB (zip kept) | zip integrity test passed. The paper and README say 291,752 images and 9,170 classes; the zip I extracted has 285,086 and 9,169, so the published figures do not match the file (not investigated further) |
| CALFW | `calfw` | 12,174 aligned images, 4,025 identities, `pairs_CALFW.txt` | 668 MB | zip integrity test passed |
| CPLFW | `cplfw` | 11,652 aligned images, 3,933 identities, `pairs_CPLFW.txt` | 672 MB | zip integrity test passed |
| LFW | `lfw` | existing copy from the proxy note, untouched | | |

Manifests list every file with its byte count and SHA-256 (trust on first use, since no upstream
checksums exist except the CUB MD5). Labels files: `labels.jsonl` per directory (path, class id
and name or identity, original split, `dml_split` where the protocol defines one). CARS196 and
CUB `dml_split` marks the zero-shot halves (classes 0 to 97 / 1 to 100 train).

## Skipped, with reasons

| Set | Reason |
|---|---|
| CompCars | Signed release form needed for the archive password; non-commercial, no redistribution |
| VeRi-776, VeRi-Wild | Request by email, promise not to share; non-commercial; re-identification rather than make/model |
| VehicleID | No public access process found |
| BoxCars116k | `medusa.fit.vutbr.cz` did not resolve; no dataset licence |
| In-shop Clothes (DeepFashion) | Non-commercial agreement and Google Drive request; not needed for vehicles or faces |
| IJB-B, IJB-C | NIST request and agreement |
| MS-Celeb-1M and its clustering parts | Withdrawn by Microsoft; no licence |
| CASIA-WebFace | Research-only, no clear licence, mirrors only; not needed |
| VGGFace2 | Official download ended; mirrors with unclear or non-commercial terms |
| ImageNet subsets, CIFAR | Not fetched: ImageNet needs a terms-of-access login, CIFAR is 32 px so says little about crops; only used as published reference rows |
| Marqo-GS-10M | Not a class-label benchmark; licence not verified |

## Reproduce

Downloads use `curl -L --limit-rate 30M` under `nice -n 10`; the tools to unpack and write the
manifests were scratch scripts and are not part of the repository. Source URLs:

```bash
# CARS196 (parquet mirror)
https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main/data/{train,test}-0000{0,1}-of-00002.parquet
# SOP (parquet of the original archive)
https://huggingface.co/datasets/nyris/stanford-online-products-v1/resolve/main/data/{train,test}-0000{0..3}-of-00004.parquet
# CUB-200-2011 (MD5 97eceeb196236b17998738112f37df78)
https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
# VMMRdb
https://www.dropbox.com/scl/fi/3r3a0oz71nlvhqhu3tocf/VMMRdb.zip?rlkey=e5bb62ae88x2pzxnsggobxpe0&dl=1
# CALFW / CPLFW (official Google Drive ids from the authors' pages)
https://drive.usercontent.google.com/download?id=1_cYgy7VFCy6JqkR8EvOxCVS02jHN1ozm&export=download&confirm=t
https://drive.usercontent.google.com/download?id=1aInOZtuvKkiV-Gtitcv1-daZshL-8PAE&export=download&confirm=t
```

Open items: mark the Commons set C review sample so its error rate is known; run the CARS196 and
SOP encoder comparison (needs the encoders, so GPU or a long CPU run) and file the numbers in a
dated log; decide whether to ask for CompCars and VeRi-776 access.
