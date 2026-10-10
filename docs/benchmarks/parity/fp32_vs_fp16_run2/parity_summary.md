# Parity: fp16-c1 against fp32-c1 (PASS)

Profile `precision`: A change that moves numerics of one stage by rounding only (engine precision, input dtype): matched detections and vectors must stay equivalent.

| | reference | candidate |
|---|---|---|
| tag | fp32-c1 | fp16-c1 |
| engine | fp32 | fp16 |
| git sha | 3b371469c820 | 3b371469c820 |
| manifest sha256 | edde8ae3d8cecfb2 | edde8ae3d8cecfb2 |

| Gate | Metric | Observed | Rule | Status |
|---|---|---:|---|---|
| P-box | `boxes.images_identical_share` | 0.99343 | >= 0.98 | PASS |
| P-box-confident | `boxes.unmatched_confident` | 0 | <= 0 | PASS |
| P-item | `boxes.item_count_class_identical_share` | 0.99949 | >= 0.99 | PASS |
| P-pe-p01 | `pe_crop.p01` | 0.99900 | >= 0.995 | PASS |
| P-pe-median | `pe_crop.median` | 0.99990 | >= 0.999 | PASS |
| P-pew-p01 | `pe_frame.p01` | 0.99749 | >= 0.995 | PASS |
| P-pew-median | `pe_frame.median` | 0.99981 | >= 0.999 | PASS |
| P-knn | `knn.overlap_at_k` | 0.97060 | >= 0.95 | PASS |
| P-dup | `knn.near_dup_agreement` | 1.00000 | >= 0.995 | PASS |
| P-cluster-assign | `cluster.assignment_agreement` | 0.99773 | >= 0.98 | PASS |
| P-cluster-purity | `cluster.purity_drop_points` | -0.45348 | <= 0.5 | PASS |
| P-label-vote | `knn.label_vote_agreement` | 0.98200 | >= 0.97 | PASS |
| P-knn-precision | `knn.precision_drop_points` | -0.06545 | <= 0.5 | PASS |
| P-label | `labels.class_agreement` | 1.00000 | >= 0.99 | PASS |
| P-vlm | `labels.vlm_agreement` | n/a | >= 0.97 | SKIP |
| P-clip | `clip.p01` | 1.00000 | >= 0.995 | PASS |
| P-face | `face.p01` | 1.00000 | >= 0.995 | PASS |
| P-ocr | `ocr.exact_share` | 1.00000 | >= 0.98 | PASS |

Recorded, not gated:

| Metric | Value |
|---|---:|
| `boxes.crop_rect_identical_share` | 0.99226 |
| `pe_crop.identical_share` | 0.00000 |
| `pe_frame.identical_share` | 0.00000 |
| `cluster.ari` | 0.33167 |
| `cluster.nmi` | 0.77012 |
| `cluster.objective_rel_diff` | -0.00590 |
| `cluster.purity_ref` | 0.62017 |
| `cluster.similarity_abs_delta_mean` | 0.02223 |
| `cluster.similarity_abs_delta_p99` | 0.14670 |
| `cluster.core_agreement` | 0.96005 |
| `knn.precision_ref` | 0.58495 |
| `knn.precision_cand` | 0.58560 |
| `cluster.purity_cand` | 0.62471 |
| `knn.top1_recall_at_k` | 1.00000 |
| `pe_crop.min` | 0.92573 |
| `pe_crop.mean` | 0.99978 |
| `pe_frame.min` | 0.99145 |
| `pe_frame.mean` | 0.99966 |
| `boxes.iou_min` | 0.78329 |
| `boxes.score_delta_max` | 0.00000 |
| `ocr.char_error_rate` | 0.00000 |

Items: 12791 reference, 12790 candidate, 12790 matched.
