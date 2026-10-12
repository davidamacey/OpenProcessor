# Parity: cand-rope against ref-u8fp16 (PASS)

Profile `precision`: A change that moves numerics of one stage by rounding only (engine precision, input dtype): matched detections and vectors must stay equivalent.

| | reference | candidate |
|---|---|---|
| tag | ref-u8fp16 | cand-rope |
| engine | fp16 | fp16 |
| git sha | 9ae52c0949fb | 54e7c1a1745b |
| manifest sha256 | edde8ae3d8cecfb2 | edde8ae3d8cecfb2 |

| Gate | Metric | Observed | Rule | Status |
|---|---|---:|---|---|
| P-box | `boxes.images_identical_share` | 0.99444 | >= 0.98 | PASS |
| P-box-confident | `boxes.unmatched_confident` | 0 | <= 0 | PASS |
| P-item | `boxes.item_count_class_identical_share` | 1.00000 | >= 0.99 | PASS |
| P-pe-p01 | `pe_crop.p01` | 0.99976 | >= 0.995 | PASS |
| P-pe-median | `pe_crop.median` | 0.99996 | >= 0.999 | PASS |
| P-pew-p01 | `pe_frame.p01` | 0.99945 | >= 0.995 | PASS |
| P-pew-median | `pe_frame.median` | 0.99994 | >= 0.999 | PASS |
| P-knn | `knn.overlap_at_k` | 0.98260 | >= 0.95 | PASS |
| P-dup | `knn.near_dup_agreement` | 1.00000 | >= 0.995 | PASS |
| P-cluster-assign | `cluster.assignment_agreement` | 0.99867 | >= 0.98 | PASS |
| P-cluster-purity | `cluster.purity_drop_points` | 0.43777 | <= 0.5 | PASS |
| P-label-vote | `knn.label_vote_agreement` | 0.99400 | >= 0.97 | PASS |
| P-knn-precision | `knn.precision_drop_points` | -0.11858 | <= 0.5 | PASS |
| P-label | `labels.class_agreement` | 1.00000 | >= 0.99 | PASS |
| P-vlm | `labels.vlm_agreement` | n/a | >= 0.97 | SKIP |
| P-clip | `clip.p01` | 1.00000 | >= 0.995 | PASS |
| P-face | `face.p01` | 1.00000 | >= 0.995 | PASS |
| P-ocr | `ocr.exact_share` | 1.00000 | >= 0.98 | PASS |

Recorded, not gated:

| Metric | Value |
|---|---:|
| `boxes.crop_rect_identical_share` | 0.99203 |
| `pe_crop.identical_share` | 0.00000 |
| `pe_frame.identical_share` | 0.00000 |
| `cluster.ari` | 0.34191 |
| `cluster.nmi` | 0.77679 |
| `cluster.objective_rel_diff` | 0.00194 |
| `cluster.purity_ref` | 0.62086 |
| `cluster.similarity_abs_delta_mean` | 0.02192 |
| `cluster.similarity_abs_delta_p99` | 0.14244 |
| `cluster.core_agreement` | 0.95833 |
| `knn.precision_ref` | 0.59947 |
| `knn.precision_cand` | 0.60066 |
| `cluster.purity_cand` | 0.61648 |
| `knn.top1_recall_at_k` | 1.00000 |
| `pe_crop.min` | 0.93585 |
| `pe_crop.mean` | 0.99989 |
| `pe_frame.min` | 0.99704 |
| `pe_frame.mean` | 0.99991 |
| `boxes.iou_min` | 0.78880 |
| `boxes.score_delta_max` | 0.01904 |
| `ocr.char_error_rate` | 0.00000 |

Items: 12792 reference, 12792 candidate, 12792 matched.
