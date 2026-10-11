# Parity: det-u8 against fp16 (FAIL)

Profile `precision`: A change that moves numerics of one stage by rounding only (engine precision, input dtype): matched detections and vectors must stay equivalent.

| | reference | candidate |
|---|---|---|
| tag | fp16 | det-u8 |
| engine | fp16 | fp16 |
| git sha | 0567748f7578 | 57dfb3cfe2bc |
| manifest sha256 | edde8ae3d8cecfb2 | edde8ae3d8cecfb2 |

| Gate | Metric | Observed | Rule | Status |
|---|---|---:|---|---|
| P-box | `boxes.images_identical_share` | 0.93478 | >= 0.98 | FAIL |
| P-box-confident | `boxes.unmatched_confident` | 2 | <= 0 | FAIL |
| P-item | `boxes.item_count_class_identical_share` | 0.96967 | >= 0.99 | FAIL |
| P-pe-p01 | `pe_crop.p01` | 0.97794 | >= 0.995 | FAIL |
| P-pe-median | `pe_crop.median` | 0.99995 | >= 0.999 | PASS |
| P-pew-p01 | `pe_frame.p01` | 0.99926 | >= 0.995 | PASS |
| P-pew-median | `pe_frame.median` | 0.99993 | >= 0.999 | PASS |
| P-knn | `knn.overlap_at_k` | 0.94690 | >= 0.95 | FAIL |
| P-dup | `knn.near_dup_agreement` | 0.99981 | >= 0.995 | PASS |
| P-cluster-assign | `cluster.assignment_agreement` | 0.99248 | >= 0.98 | PASS |
| P-cluster-purity | `cluster.purity_drop_points` | -0.28218 | <= 0.5 | PASS |
| P-label-vote | `knn.label_vote_agreement` | 0.97300 | >= 0.97 | PASS |
| P-knn-precision | `knn.precision_drop_points` | -0.23317 | <= 0.5 | PASS |
| P-label | `labels.class_agreement` | 1.00000 | >= 0.99 | PASS |
| P-vlm | `labels.vlm_agreement` | n/a | >= 0.97 | SKIP |
| P-clip | `clip.p01` | 1.00000 | >= 0.995 | PASS |
| P-face | `face.p01` | 1.00000 | >= 0.995 | PASS |
| P-ocr | `ocr.exact_share` | 1.00000 | >= 0.98 | PASS |

Recorded, not gated:

| Metric | Value |
|---|---:|
| `boxes.crop_rect_identical_share` | 0.86087 |
| `pe_crop.identical_share` | 0.00000 |
| `pe_frame.identical_share` | 0.00000 |
| `cluster.ari` | 0.33515 |
| `cluster.nmi` | 0.77333 |
| `cluster.objective_rel_diff` | -0.00004 |
| `cluster.purity_ref` | 0.62377 |
| `cluster.similarity_abs_delta_mean` | 0.02240 |
| `cluster.similarity_abs_delta_p99` | 0.14530 |
| `cluster.core_agreement` | 0.96340 |
| `knn.precision_ref` | 0.58841 |
| `knn.precision_cand` | 0.59074 |
| `cluster.purity_cand` | 0.62659 |
| `knn.top1_recall_at_k` | 0.99700 |
| `pe_crop.min` | 0.80584 |
| `pe_crop.mean` | 0.99871 |
| `pe_frame.min` | 0.99740 |
| `pe_frame.mean` | 0.99989 |
| `boxes.iou_min` | 0.71228 |
| `boxes.score_delta_max` | 0.18018 |
| `ocr.char_error_rate` | 0.00000 |

Items: 12790 reference, 12792 candidate, 12758 matched.
