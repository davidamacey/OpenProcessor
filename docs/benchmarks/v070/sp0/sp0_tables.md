### set A (COCO 1,946)

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 1846 |
| images/s median (min to max) | 13.14 (13.12 to 13.47) |
| crops per image | 6.42 |
| GPU util mean / p50 % | 45 / 30 |
| Triton cores (docker stats) | 0.23 |
| API CPU-s per image | 0.0778 |
| 1-minute load at start (max) | 27.0 |
| noisy repetitions | 1, 2 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| embed | 434.28 | 54.5% |
| opensearch_write | 335.34 | 42.1% |
| whole_frame_preprocess | 5.18 | 0.7% |
| pe_preprocess | 4.74 | 0.6% |
| decode | 4.67 | 0.6% |
| jpeg_encode | 3.32 | 0.4% |
| blur | 2.64 | 0.3% |
| letterbox | 2.43 | 0.3% |
| triton_request | 1.10 | 0.1% |
| bulk_serialize | 0.79 | 0.1% |
| crop | 0.54 | 0.1% |
| file_read | 0.53 | 0.1% |
| doc_build | 0.42 | 0.1% |
| hash | 0.12 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| gpu_decode | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| pe_image_encoder | 16.79 | 17.96 |
| yolov11_small_trt_end2end | 0.03 | 0.62 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 0.00 | 0.0% |
| crop_copy | 0.00 | 0.0% |
| crop_jpeg | 0.00 | 0.0% |
| decode | 0.00 | 0.0% |
| detect | 0.00 | 0.0% |
| embed_crops | 0.00 | 0.0% |
| embed_frame | 0.00 | 0.0% |
| embed_pe | 0.00 | 0.0% |
| letterbox | 0.00 | 0.0% |
| queue | 0.00 | 0.0% |

### set B (500 photos)

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 400 |
| images/s median (min to max) | 9.88 (9.63 to 9.96) |
| crops per image | 1.65 |
| GPU util mean / p50 % | 15 / 0 |
| Triton cores (docker stats) | 1.90 |
| API CPU-s per image | 0.2618 |
| 1-minute load at start (max) | 10.2 |
| noisy repetitions | none |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 929.88 | 70.0% |
| opensearch_write | 276.37 | 20.8% |
| decode | 70.90 | 5.3% |
| blur | 12.15 | 0.9% |
| whole_frame_preprocess | 12.02 | 0.9% |
| letterbox | 11.63 | 0.9% |
| embed | 6.70 | 0.5% |
| jpeg_encode | 3.71 | 0.3% |
| file_read | 3.54 | 0.3% |
| crop | 0.89 | 0.1% |
| pe_preprocess | 0.67 | 0.1% |
| bulk_serialize | 0.35 | 0.0% |
| doc_build | 0.21 | 0.0% |
| hash | 0.16 | 0.0% |
| triton_request | 0.08 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 0.34 | 6.06 |
| pe_image_encoder | 4.83 | 11.85 |
| yolov11_small_trt_end2end | 16.43 | 12.02 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 4.61 | 1.9% |
| crop_copy | 6.48 | 2.6% |
| crop_jpeg | 29.69 | 12.0% |
| decode | 86.84 | 39.5% |
| detect | 39.70 | 15.5% |
| embed_crops | 4.86 | 2.0% |
| embed_frame | 5.20 | 2.1% |
| embed_pe | 56.37 | 22.3% |
| letterbox | 5.38 | 2.2% |
| queue | 0.35 | 0.2% |

### set P (3,000 photos)

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 2900 |
| images/s median (min to max) | 14.80 (11.92 to 16.47) |
| crops per image | 2.41 |
| GPU util mean / p50 % | 35 / 0 |
| Triton cores (docker stats) | 2.53 |
| API CPU-s per image | 0.0779 |
| 1-minute load at start (max) | 20.2 |
| noisy repetitions | 2, 3 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 818.04 | 72.1% |
| opensearch_write | 312.32 | 27.5% |
| file_read | 2.35 | 0.2% |
| bulk_serialize | 0.44 | 0.0% |
| decode | 0.39 | 0.0% |
| doc_build | 0.25 | 0.0% |
| blur | 0.25 | 0.0% |
| whole_frame_preprocess | 0.21 | 0.0% |
| letterbox | 0.20 | 0.0% |
| embed | 0.19 | 0.0% |
| hash | 0.15 | 0.0% |
| jpeg_encode | 0.05 | 0.0% |
| crop | 0.05 | 0.0% |
| pe_preprocess | 0.02 | 0.0% |
| triton_request | 0.00 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 3.10 | 8.51 |
| pe_image_encoder | 7.39 | 16.98 |
| yolov11_small_trt_end2end | 17.35 | 17.62 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 10.63 | 2.2% |
| crop_copy | 28.81 | 5.8% |
| crop_jpeg | 70.58 | 15.8% |
| decode | 89.91 | 19.3% |
| detect | 52.05 | 11.5% |
| embed_crops | 53.33 | 12.0% |
| embed_frame | 6.82 | 1.4% |
| embed_pe | 126.24 | 27.6% |
| letterbox | 21.45 | 4.5% |
| queue | 0.80 | 0.2% |

