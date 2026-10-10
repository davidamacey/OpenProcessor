# Parity: fp32-repeat against fp32 (FAIL)

Profile `precision`: A change that moves numerics of one stage by rounding only (engine precision, input dtype): matched detections and vectors must stay equivalent.

| | reference | candidate |
|---|---|---|
| tag | fp32 | fp32-repeat |
| engine | fp32 | fp32 |
| git sha | 0567748f7578 | 3b371469c820 |
| manifest sha256 | edde8ae3d8cecfb2 | edde8ae3d8cecfb2 |

| Gate | Metric | Observed | Rule | Status |
|---|---|---:|---|---|
| P-box | `boxes.images_identical_share` | 0.99242 | >= 0.98 | PASS |
| P-box-confident | `boxes.unmatched_confident` | 1 | <= 0 | FAIL |
| P-item | `boxes.item_count_class_identical_share` | 0.99899 | >= 0.99 | PASS |
| P-pe-p01 | `pe_crop.p01` | 1.00000 | >= 0.995 | PASS |
| P-pe-median | `pe_crop.median` | 1.00000 | >= 0.999 | PASS |
| P-pew-p01 | `pe_frame.p01` | 1.00000 | >= 0.995 | PASS |
| P-pew-median | `pe_frame.median` | 1.00000 | >= 0.999 | PASS |
| P-knn | `knn.overlap_at_k` | 0.99780 | >= 0.95 | PASS |
| P-dup | `knn.near_dup_agreement` | 1.00000 | >= 0.995 | PASS |
| P-cluster-assign | `cluster.assignment_agreement` | 0.99961 | >= 0.98 | PASS |
| P-cluster-purity | `cluster.purity_drop_points` | -0.39878 | <= 0.5 | PASS |
| P-label-vote | `knn.label_vote_agreement` | 0.99900 | >= 0.97 | PASS |
| P-knn-precision | `knn.precision_drop_points` | 0.01274 | <= 0.5 | PASS |
| P-label | `labels.class_agreement` | 1.00000 | >= 0.99 | PASS |
| P-vlm | `labels.vlm_agreement` | n/a | >= 0.97 | SKIP |
| P-clip | `clip.p01` | 1.00000 | >= 0.995 | PASS |
| P-face | `face.p01` | 1.00000 | >= 0.995 | PASS |
| P-ocr | `ocr.exact_share` | 1.00000 | >= 0.98 | PASS |

Recorded, not gated:

| Metric | Value |
|---|---:|
| `boxes.crop_rect_identical_share` | 0.99171 |
| `pe_crop.identical_share` | 0.98225 |
| `pe_frame.identical_share` | 0.98900 |
| `cluster.ari` | 0.35073 |
| `cluster.nmi` | 0.77911 |
| `cluster.objective_rel_diff` | 0.00236 |
| `cluster.purity_ref` | 0.61623 |
| `cluster.similarity_abs_delta_mean` | 0.02128 |
| `cluster.similarity_abs_delta_p99` | 0.14889 |
| `cluster.core_agreement` | 0.96161 |
| `knn.precision_ref` | 0.57427 |
| `knn.precision_cand` | 0.57414 |
| `cluster.purity_cand` | 0.62022 |
| `knn.top1_recall_at_k` | 1.00000 |
| `pe_crop.min` | 0.95163 |
| `pe_crop.mean` | 0.99991 |
| `pe_frame.min` | 1.00000 |
| `pe_frame.mean` | 1.00000 |
| `boxes.iou_min` | 0.77363 |
| `boxes.score_delta_max` | 0.08350 |
| `ocr.char_error_rate` | 0.00000 |

Items: 12789 reference, 12791 candidate, 12789 matched.
