# Parity: ref-ship against ref-u8fp16 (FAIL)

Profile `precision`: A change that moves numerics of one stage by rounding only (engine precision, input dtype): matched detections and vectors must stay equivalent.

| | reference | candidate |
|---|---|---|
| tag | ref-u8fp16 | ref-ship |
| engine | fp16 | fp16 |
| git sha | 9ae52c0949fb | ed3388671e81 |
| manifest sha256 | edde8ae3d8cecfb2 | edde8ae3d8cecfb2 |

| Gate | Metric | Observed | Rule | Status |
|---|---|---:|---|---|
| P-box | `boxes.images_identical_share` | 0.99444 | >= 0.98 | PASS |
| P-box-confident | `boxes.unmatched_confident` | 0 | <= 0 | PASS |
| P-item | `boxes.item_count_class_identical_share` | 0.99899 | >= 0.99 | PASS |
| P-pe-p01 | `pe_crop.p01` | 0.99978 | >= 0.995 | PASS |
| P-pe-median | `pe_crop.median` | 0.99996 | >= 0.999 | PASS |
| P-pew-p01 | `pe_frame.p01` | 0.65058 | >= 0.995 | FAIL |
| P-pew-median | `pe_frame.median` | 0.81585 | >= 0.999 | FAIL |
| P-knn | `knn.overlap_at_k` | 0.98180 | >= 0.95 | PASS |
| P-dup | `knn.near_dup_agreement` | 1.00000 | >= 0.995 | PASS |
| P-cluster-assign | `cluster.assignment_agreement` | 0.99891 | >= 0.98 | PASS |
| P-cluster-purity | `cluster.purity_drop_points` | -0.35181 | <= 0.5 | PASS |
| P-label-vote | `knn.label_vote_agreement` | 0.99300 | >= 0.97 | PASS |
| P-knn-precision | `knn.precision_drop_points` | -0.09174 | <= 0.5 | PASS |
| P-label | `labels.class_agreement` | 1.00000 | >= 0.99 | PASS |
| P-vlm | `labels.vlm_agreement` | n/a | >= 0.97 | SKIP |
| P-clip | `clip.p01` | n/a | >= 0.995 | SKIP |
| P-face | `face.p01` | n/a | >= 0.995 | SKIP |
| P-ocr | `ocr.exact_share` | n/a | >= 0.98 | SKIP |

Recorded, not gated:

| Metric | Value |
|---|---:|
| `boxes.crop_rect_identical_share` | 0.99234 |
| `pe_crop.identical_share` | 0.00000 |
| `pe_frame.identical_share` | 0.00000 |
| `cluster.ari` | 0.35429 |
| `cluster.nmi` | 0.77692 |
| `cluster.objective_rel_diff` | -0.00017 |
| `cluster.purity_ref` | 0.62091 |
| `cluster.similarity_abs_delta_mean` | 0.02176 |
| `cluster.similarity_abs_delta_p99` | 0.14677 |
| `cluster.core_agreement` | 0.95974 |
| `knn.precision_ref` | 0.61009 |
| `knn.precision_cand` | 0.61101 |
| `cluster.purity_cand` | 0.62442 |
| `knn.top1_recall_at_k` | 0.99900 |
| `pe_crop.min` | 0.93580 |
| `pe_crop.mean` | 0.99989 |
| `pe_frame.min` | 0.55685 |
| `pe_frame.mean` | 0.80735 |
| `boxes.iou_min` | 0.78880 |
| `boxes.score_delta_max` | 0.01904 |
| `ocr.char_error_rate` | n/a |

Items: 12792 reference, 12792 candidate, 12791 matched.
