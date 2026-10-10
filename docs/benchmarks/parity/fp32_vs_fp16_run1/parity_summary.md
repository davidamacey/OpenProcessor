# Parity: fp16 against fp32 (PASS)

Profile `precision`: A change that moves numerics of one stage by rounding only (engine precision, input dtype): matched detections and vectors must stay equivalent.

| | reference | candidate |
|---|---|---|
| tag | fp32 | fp16 |
| engine | fp32 | fp16 |
| git sha | 0567748f7578 | 0567748f7578 |
| manifest sha256 | edde8ae3d8cecfb2 | edde8ae3d8cecfb2 |

| Gate | Metric | Observed | Rule | Status |
|---|---|---:|---|---|
| P-box | `boxes.images_identical_share` | 0.99343 | >= 0.98 | PASS |
| P-box-confident | `boxes.unmatched_confident` | 0 | <= 0 | PASS |
| P-item | `boxes.item_count_class_identical_share` | 0.99949 | >= 0.99 | PASS |
| P-pe-p01 | `pe_crop.p01` | 0.99840 | >= 0.995 | PASS |
| P-pe-median | `pe_crop.median` | 0.99990 | >= 0.999 | PASS |
| P-pew-p01 | `pe_frame.p01` | 0.99751 | >= 0.995 | PASS |
| P-pew-median | `pe_frame.median` | 0.99981 | >= 0.999 | PASS |
| P-knn | `knn.overlap_at_k` | 0.96810 | >= 0.95 | PASS |
| P-dup | `knn.near_dup_agreement` | 1.00000 | >= 0.995 | PASS |
| P-cluster-assign | `cluster.assignment_agreement` | 0.99773 | >= 0.98 | PASS |
| P-cluster-purity | `cluster.purity_drop_points` | -0.71155 | <= 0.5 | PASS |
| P-label-vote | `knn.label_vote_agreement` | 0.97900 | >= 0.97 | PASS |
| P-knn-precision | `knn.precision_drop_points` | 0.03822 | <= 0.5 | PASS |
| P-label | `labels.class_agreement` | 1.00000 | >= 0.99 | PASS |
| P-vlm | `labels.vlm_agreement` | n/a | >= 0.97 | SKIP |
| P-clip | `clip.p01` | 1.00000 | >= 0.995 | PASS |
| P-face | `face.p01` | 1.00000 | >= 0.995 | PASS |
| P-ocr | `ocr.exact_share` | 1.00000 | >= 0.98 | PASS |

Recorded, not gated:

| Metric | Value |
|---|---:|
| `boxes.crop_rect_identical_share` | 0.98991 |
| `pe_crop.identical_share` | 0.00000 |
| `pe_frame.identical_share` | 0.00000 |
| `cluster.ari` | 0.34088 |
| `cluster.nmi` | 0.77644 |
| `cluster.objective_rel_diff` | -0.00159 |
| `cluster.purity_ref` | 0.61623 |
| `cluster.similarity_abs_delta_mean` | 0.02154 |
| `cluster.similarity_abs_delta_p99` | 0.14104 |
| `cluster.core_agreement` | 0.96192 |
| `knn.precision_ref` | 0.57427 |
| `knn.precision_cand` | 0.57389 |
| `cluster.purity_cand` | 0.62335 |
| `knn.top1_recall_at_k` | 1.00000 |
| `pe_crop.min` | 0.92198 |
| `pe_crop.mean` | 0.99975 |
| `pe_frame.min` | 0.99145 |
| `pe_frame.mean` | 0.99966 |
| `boxes.iou_min` | 0.78329 |
| `boxes.score_delta_max` | 0.00000 |
| `ocr.char_error_rate` | 0.00000 |

Items: 12789 reference, 12790 candidate, 12789 matched.
