# Parity: det-u8-repeat against det-u8 (FAIL)

Profile `precision`: A change that moves numerics of one stage by rounding only (engine precision, input dtype): matched detections and vectors must stay equivalent.

| | reference | candidate |
|---|---|---|
| tag | det-u8 | det-u8-repeat |
| engine | fp16 | fp16 |
| git sha | 57dfb3cfe2bc | 57dfb3cfe2bc |
| manifest sha256 | edde8ae3d8cecfb2 | edde8ae3d8cecfb2 |

| Gate | Metric | Observed | Rule | Status |
|---|---|---:|---|---|
| P-box | `boxes.images_identical_share` | 0.99191 | >= 0.98 | PASS |
| P-box-confident | `boxes.unmatched_confident` | 0 | <= 0 | PASS |
| P-item | `boxes.item_count_class_identical_share` | 0.99949 | >= 0.99 | PASS |
| P-pe-p01 | `pe_crop.p01` | 0.99994 | >= 0.995 | PASS |
| P-pe-median | `pe_crop.median` | 1.00000 | >= 0.999 | PASS |
| P-pew-p01 | `pe_frame.p01` | 0.99996 | >= 0.995 | PASS |
| P-pew-median | `pe_frame.median` | 1.00000 | >= 0.999 | PASS |
| P-knn | `knn.overlap_at_k` | 0.99420 | >= 0.95 | PASS |
| P-dup | `knn.near_dup_agreement` | 1.00000 | >= 0.995 | PASS |
| P-cluster-assign | `cluster.assignment_agreement` | 0.99945 | >= 0.98 | PASS |
| P-cluster-purity | `cluster.purity_drop_points` | 0.90682 | <= 0.5 | FAIL |
| P-label-vote | `knn.label_vote_agreement` | 1.00000 | >= 0.97 | PASS |
| P-knn-precision | `knn.precision_drop_points` | 0.01318 | <= 0.5 | PASS |
| P-label | `labels.class_agreement` | 1.00000 | >= 0.99 | PASS |
| P-vlm | `labels.vlm_agreement` | n/a | >= 0.97 | SKIP |
| P-clip | `clip.p01` | 1.00000 | >= 0.995 | PASS |
| P-face | `face.p01` | 1.00000 | >= 0.995 | PASS |
| P-ocr | `ocr.exact_share` | 1.00000 | >= 0.98 | PASS |

Recorded, not gated:

| Metric | Value |
|---|---:|
| `boxes.crop_rect_identical_share` | 0.99101 |
| `pe_crop.identical_share` | 0.07559 |
| `pe_frame.identical_share` | 0.06750 |
| `cluster.ari` | 0.34647 |
| `cluster.nmi` | 0.77528 |
| `cluster.objective_rel_diff` | 0.00264 |
| `cluster.purity_ref` | 0.62602 |
| `cluster.similarity_abs_delta_mean` | 0.02136 |
| `cluster.similarity_abs_delta_p99` | 0.13890 |
| `cluster.core_agreement` | 0.96083 |
| `knn.precision_ref` | 0.60040 |
| `knn.precision_cand` | 0.60026 |
| `cluster.purity_cand` | 0.61695 |
| `knn.top1_recall_at_k` | 0.99900 |
| `pe_crop.min` | 0.90895 |
| `pe_crop.mean` | 0.99990 |
| `pe_frame.min` | 0.99976 |
| `pe_frame.mean` | 1.00000 |
| `boxes.iou_min` | 0.71410 |
| `boxes.score_delta_max` | 0.17700 |
| `ocr.char_error_rate` | 0.00000 |

Items: 12792 reference, 12793 candidate, 12792 matched.
