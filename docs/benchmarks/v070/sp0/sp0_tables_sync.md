### set B, STAGE_SYNC on

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 400 |
| images/s median (min to max) | 10.61 (9.76 to 10.86) |
| crops per image | 1.65 |
| GPU util mean / p50 % | 21 / 0 |
| Triton cores (docker stats) | 1.83 |
| API CPU-s per image | 0.2515 |
| 1-minute load at start (max) | 17.3 |
| noisy repetitions | 3 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 817.60 | 68.9% |
| opensearch_write | 249.89 | 21.0% |
| decode | 70.82 | 6.0% |
| letterbox | 12.67 | 1.1% |
| blur | 11.82 | 1.0% |
| whole_frame_preprocess | 11.81 | 1.0% |
| embed | 6.17 | 0.5% |
| file_read | 3.52 | 0.3% |
| crop | 0.87 | 0.1% |
| pe_preprocess | 0.61 | 0.1% |
| jpeg_encode | 0.60 | 0.1% |
| bulk_serialize | 0.36 | 0.0% |
| doc_build | 0.21 | 0.0% |
| hash | 0.15 | 0.0% |
| triton_request | 0.09 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 3.61 | 9.91 |
| pe_image_encoder | 4.84 | 11.28 |
| yolov11_small_trt_end2end | 17.31 | 11.89 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 4.97 | 1.8% |
| crop_copy | 9.95 | 3.8% |
| crop_jpeg | 29.42 | 10.5% |
| decode | 87.90 | 37.6% |
| detect | 41.92 | 15.8% |
| embed_crops | 7.75 | 2.8% |
| embed_frame | 7.17 | 2.6% |
| embed_pe | 58.13 | 20.7% |
| letterbox | 10.36 | 3.8% |
| queue | 0.51 | 0.2% |

### set P, STAGE_SYNC on

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 2900 |
| images/s median (min to max) | 15.47 (14.82 to 16.15) |
| crops per image | 2.41 |
| GPU util mean / p50 % | 35 / 0 |
| Triton cores (docker stats) | 2.92 |
| API CPU-s per image | 0.0787 |
| 1-minute load at start (max) | 17.9 |
| noisy repetitions | 1, 2 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 823.68 | 74.4% |
| opensearch_write | 278.21 | 25.1% |
| file_read | 2.39 | 0.2% |
| decode | 0.61 | 0.1% |
| bulk_serialize | 0.46 | 0.0% |
| blur | 0.45 | 0.0% |
| letterbox | 0.30 | 0.0% |
| whole_frame_preprocess | 0.27 | 0.0% |
| doc_build | 0.26 | 0.0% |
| embed | 0.24 | 0.0% |
| hash | 0.16 | 0.0% |
| crop | 0.08 | 0.0% |
| pe_preprocess | 0.02 | 0.0% |
| jpeg_encode | 0.02 | 0.0% |
| triton_request | 0.00 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 0.62 | 7.79 |
| pe_image_encoder | 7.35 | 17.14 |
| yolov11_small_trt_end2end | 17.41 | 16.91 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 11.25 | 2.4% |
| crop_copy | 28.92 | 6.1% |
| crop_jpeg | 74.11 | 15.5% |
| decode | 91.75 | 19.2% |
| detect | 50.68 | 10.6% |
| embed_crops | 61.40 | 12.9% |
| embed_frame | 9.06 | 1.9% |
| embed_pe | 125.32 | 26.1% |
| letterbox | 24.19 | 5.1% |
| queue | 0.71 | 0.1% |

### set B, counters removed

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 400 |
| images/s median (min to max) | 10.10 (10.06 to 10.59) |
| crops per image | 1.65 |
| GPU util mean / p50 % | 17 / 0 |
| Triton cores (docker stats) | 1.15 |
| API CPU-s per image | 0.2723 |
| 1-minute load at start (max) | 21.4 |
| noisy repetitions | 2, 3 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 870.99 | 69.6% |
| opensearch_write | 258.26 | 20.6% |
| decode | 72.18 | 5.8% |
| letterbox | 13.00 | 1.0% |
| blur | 11.84 | 0.9% |
| whole_frame_preprocess | 11.71 | 0.9% |
| embed | 6.55 | 0.5% |
| file_read | 3.53 | 0.3% |
| crop | 0.95 | 0.1% |
| pe_preprocess | 0.66 | 0.1% |
| jpeg_encode | 0.65 | 0.1% |
| bulk_serialize | 0.35 | 0.0% |
| doc_build | 0.21 | 0.0% |
| hash | 0.14 | 0.0% |
| triton_request | 0.08 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 0.44 | 7.74 |
| pe_image_encoder | 5.14 | 10.69 |
| yolov11_small_trt_end2end | 17.46 | 10.89 |

