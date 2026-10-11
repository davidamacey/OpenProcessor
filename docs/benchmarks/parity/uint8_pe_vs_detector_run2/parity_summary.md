# Parity: new-u8 against det-u8-repeat (PASS)

Profile `precision`: A change that moves numerics of one stage by rounding only (engine precision, input dtype): matched detections and vectors must stay equivalent.

| | reference | candidate |
|---|---|---|
| tag | det-u8-repeat | new-u8 |
| engine | fp16 | fp16 |
| git sha | 57dfb3cfe2bc | 27e4570572a8 |
| manifest sha256 | edde8ae3d8cecfb2 | edde8ae3d8cecfb2 |

| Gate | Metric | Observed | Rule | Status |
|---|---|---:|---|---|
| P-box | `boxes.images_identical_share` | 0.99444 | >= 0.98 | PASS |
| P-box-confident | `boxes.unmatched_confident` | 0 | <= 0 | PASS |
| P-item | `boxes.item_count_class_identical_share` | 0.99949 | >= 0.99 | PASS |
| P-pe-p01 | `pe_crop.p01` | 0.99994 | >= 0.995 | PASS |
| P-pe-median | `pe_crop.median` | 1.00000 | >= 0.999 | PASS |
| P-pew-p01 | `pe_frame.p01` | 0.99996 | >= 0.995 | PASS |
| P-pew-median | `pe_frame.median` | 1.00000 | >= 0.999 | PASS |
| P-knn | `knn.overlap_at_k` | 0.99500 | >= 0.95 | PASS |
| P-dup | `knn.near_dup_agreement` | 1.00000 | >= 0.995 | PASS |
| P-cluster-assign | `cluster.assignment_agreement` | 0.99945 | >= 0.98 | PASS |
| P-cluster-purity | `cluster.purity_drop_points` | 0.28924 | <= 0.5 | PASS |
| P-label-vote | `knn.label_vote_agreement` | 0.99800 | >= 0.97 | PASS |
| P-knn-precision | `knn.precision_drop_points` | 0.06579 | <= 0.5 | PASS |
| P-label | `labels.class_agreement` | 1.00000 | >= 0.99 | PASS |
| P-vlm | `labels.vlm_agreement` | n/a | >= 0.97 | SKIP |
| P-clip | `clip.p01` | 1.00000 | >= 0.995 | PASS |
| P-face | `face.p01` | 1.00000 | >= 0.995 | PASS |
| P-ocr | `ocr.exact_share` | 1.00000 | >= 0.98 | PASS |

Recorded, not gated:

| Metric | Value |
|---|---:|
| `boxes.crop_rect_identical_share` | 0.99179 |
| `pe_crop.identical_share` | 0.07067 |
| `pe_frame.identical_share` | 0.06450 |
| `cluster.ari` | 0.34222 |
| `cluster.nmi` | 0.77464 |
| `cluster.objective_rel_diff` | -0.00118 |
| `cluster.purity_ref` | 0.61695 |
| `cluster.similarity_abs_delta_mean` | 0.02141 |
| `cluster.similarity_abs_delta_p99` | 0.13960 |
| `cluster.core_agreement` | 0.95865 |
| `knn.precision_ref` | 0.60013 |
| `knn.precision_cand` | 0.59947 |
| `cluster.purity_cand` | 0.61406 |
| `knn.top1_recall_at_k` | 1.00000 |
| `pe_crop.min` | 0.94803 |
| `pe_crop.mean` | 0.99992 |
| `pe_frame.min` | 0.99975 |
| `pe_frame.mean` | 1.00000 |
| `boxes.iou_min` | 0.78710 |
| `boxes.score_delta_max` | 0.00000 |
| `ocr.char_error_rate` | 0.00000 |

Items: 12793 reference, 12792 candidate, 12792 matched.
