### set P, 8 batches, headroom (8 unused models unloaded)

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 2900 |
| images/s median (min to max) | 48.46 (47.25 to 48.69) |
| crops per image | 2.41 |
| GPU util mean / p50 % | 97 / 100 |
| Triton cores (docker stats) | 7.89 |
| API CPU-s per image | 0.0624 |
| 1-minute load at start (max) | 15.2 |
| noisy repetitions | none |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 3159.07 | 91.7% |
| opensearch_write | 278.47 | 8.1% |
| file_read | 4.04 | 0.1% |
| embed | 2.63 | 0.1% |
| bulk_serialize | 0.43 | 0.0% |
| doc_build | 0.25 | 0.0% |
| hash | 0.15 | 0.0% |
| decode | 0.13 | 0.0% |
| letterbox | 0.07 | 0.0% |
| whole_frame_preprocess | 0.07 | 0.0% |
| blur | 0.06 | 0.0% |
| pe_preprocess | 0.01 | 0.0% |
| jpeg_encode | 0.01 | 0.0% |
| crop | 0.01 | 0.0% |
| triton_request | 0.00 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 7.08 | 5.96 |
| pe_image_encoder | 122.16 | 47.41 |
| yolov11_small_trt_end2end | 42.89 | 113.51 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 19.85 | 0.7% |
| crop_copy | 34.97 | 1.2% |
| crop_jpeg | 73.79 | 2.6% |
| decode | 89.88 | 3.2% |
| detect | 243.41 | 8.6% |
| embed_crops | 89.28 | 3.2% |
| embed_frame | 10.34 | 0.4% |
| embed_pe | 744.26 | 26.3% |
| letterbox | 21.89 | 0.8% |
| queue | 1510.94 | 53.1% |

### set P, 4 batches, headroom

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 2900 |
| images/s median (min to max) | 38.55 (36.01 to 39.08) |
| crops per image | 2.41 |
| GPU util mean / p50 % | 82 / 100 |
| Triton cores (docker stats) | 4.93 |
| API CPU-s per image | 0.0567 |
| 1-minute load at start (max) | 102.9 |
| noisy repetitions | 2, 3 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 1494.47 | 81.1% |
| opensearch_write | 342.15 | 18.6% |
| file_read | 3.40 | 0.2% |
| embed | 0.75 | 0.0% |
| bulk_serialize | 0.38 | 0.0% |
| doc_build | 0.24 | 0.0% |
| hash | 0.12 | 0.0% |
| decode | 0.12 | 0.0% |
| letterbox | 0.07 | 0.0% |
| whole_frame_preprocess | 0.06 | 0.0% |
| blur | 0.05 | 0.0% |
| crop | 0.01 | 0.0% |
| jpeg_encode | 0.01 | 0.0% |
| pe_preprocess | 0.01 | 0.0% |
| triton_request | 0.00 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 57.12 | 10.35 |
| pe_image_encoder | 46.47 | 31.09 |
| yolov11_small_trt_end2end | 20.65 | 47.07 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 28.38 | 2.3% |
| crop_copy | 64.16 | 5.6% |
| crop_jpeg | 75.47 | 6.2% |
| decode | 105.36 | 8.9% |
| detect | 118.08 | 10.1% |
| embed_crops | 204.78 | 17.5% |
| embed_frame | 28.92 | 2.6% |
| embed_pe | 365.49 | 30.7% |
| letterbox | 50.86 | 4.4% |
| queue | 144.29 | 12.4% |

### set P, 8 batches, all default models

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 2900 |
| images/s median (min to max) | 40.41 (29.84 to 43.02) |
| crops per image | 2.41 |
| GPU util mean / p50 % | 90 / 100 |
| Triton cores (docker stats) | 6.94 |
| API CPU-s per image | 0.0945 |
| 1-minute load at start (max) | 34.4 |
| noisy repetitions | 1, 2, 3 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 3138.37 | 89.9% |
| opensearch_write | 309.27 | 8.9% |
| embed | 13.11 | 0.4% |
| decode | 8.84 | 0.3% |
| blur | 6.35 | 0.2% |
| letterbox | 4.48 | 0.1% |
| whole_frame_preprocess | 4.46 | 0.1% |
| file_read | 4.06 | 0.1% |
| crop | 0.84 | 0.0% |
| bulk_serialize | 0.44 | 0.0% |
| jpeg_encode | 0.43 | 0.0% |
| pe_preprocess | 0.30 | 0.0% |
| doc_build | 0.28 | 0.0% |
| hash | 0.16 | 0.0% |
| triton_request | 0.04 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 111.52 | 7.84 |
| pe_image_encoder | 20.19 | 22.11 |
| yolov11_small_trt_end2end | 18.75 | 35.55 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 90.71 | 3.1% |
| crop_copy | 152.73 | 5.8% |
| crop_jpeg | 74.72 | 2.8% |
| decode | 134.97 | 5.3% |
| detect | 84.37 | 3.1% |
| embed_crops | 492.15 | 18.5% |
| embed_frame | 95.60 | 3.6% |
| embed_pe | 200.46 | 7.6% |
| letterbox | 142.90 | 5.4% |
| queue | 1235.99 | 44.3% |

### set P, 4 batches, all default models

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 2900 |
| images/s median (min to max) | 30.66 (22.52 to 32.08) |
| crops per image | 2.41 |
| GPU util mean / p50 % | 66 / 87 |
| Triton cores (docker stats) | 5.23 |
| API CPU-s per image | 0.1259 |
| 1-minute load at start (max) | 31.9 |
| noisy repetitions | 1, 2, 3 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 1828.17 | 82.6% |
| opensearch_write | 313.16 | 14.1% |
| decode | 17.82 | 0.8% |
| embed | 14.46 | 0.7% |
| blur | 14.04 | 0.6% |
| letterbox | 9.37 | 0.4% |
| whole_frame_preprocess | 9.10 | 0.4% |
| file_read | 3.00 | 0.1% |
| crop | 1.57 | 0.1% |
| jpeg_encode | 0.84 | 0.0% |
| pe_preprocess | 0.78 | 0.0% |
| bulk_serialize | 0.44 | 0.0% |
| doc_build | 0.28 | 0.0% |
| hash | 0.15 | 0.0% |
| triton_request | 0.07 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 70.30 | 8.90 |
| pe_image_encoder | 21.32 | 24.35 |
| yolov11_small_trt_end2end | 16.59 | 27.87 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 50.86 | 3.0% |
| crop_copy | 133.25 | 9.1% |
| crop_jpeg | 75.63 | 4.6% |
| decode | 136.38 | 9.8% |
| detect | 75.01 | 5.1% |
| embed_crops | 330.18 | 21.3% |
| embed_frame | 71.45 | 4.8% |
| embed_pe | 229.76 | 15.5% |
| letterbox | 146.46 | 10.2% |
| queue | 268.85 | 18.3% |

### set P, 4 batches, headroom, control

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 2900 |
| images/s median (min to max) | 41.63 (40.74 to 41.94) |
| crops per image | 2.41 |
| GPU util mean / p50 % | 87 / 100 |
| Triton cores (docker stats) | 7.10 |
| API CPU-s per image | 0.0639 |
| 1-minute load at start (max) | 11.5 |
| noisy repetitions | none |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 1346.52 | 81.2% |
| opensearch_write | 305.75 | 18.4% |
| file_read | 3.47 | 0.2% |
| embed | 0.53 | 0.0% |
| bulk_serialize | 0.46 | 0.0% |
| doc_build | 0.28 | 0.0% |
| hash | 0.16 | 0.0% |
| decode | 0.13 | 0.0% |
| whole_frame_preprocess | 0.06 | 0.0% |
| letterbox | 0.06 | 0.0% |
| blur | 0.05 | 0.0% |
| jpeg_encode | 0.01 | 0.0% |
| crop | 0.01 | 0.0% |
| pe_preprocess | 0.01 | 0.0% |
| triton_request | 0.00 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 12.00 | 7.73 |
| pe_image_encoder | 46.32 | 32.57 |
| yolov11_small_trt_end2end | 19.69 | 47.80 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 23.86 | 2.2% |
| crop_copy | 56.72 | 5.3% |
| crop_jpeg | 74.72 | 6.9% |
| decode | 104.47 | 9.5% |
| detect | 119.92 | 11.6% |
| embed_crops | 172.89 | 16.1% |
| embed_frame | 23.39 | 2.2% |
| embed_pe | 373.07 | 34.7% |
| letterbox | 40.91 | 3.9% |
| queue | 86.06 | 8.0% |

### set P, 4 batches, headroom, MAX_IN_FLIGHT 16

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 2900 |
| images/s median (min to max) | 45.12 (44.66 to 45.75) |
| crops per image | 2.41 |
| GPU util mean / p50 % | 93 / 100 |
| Triton cores (docker stats) | 6.60 |
| API CPU-s per image | 0.0693 |
| 1-minute load at start (max) | 38.1 |
| noisy repetitions | 1, 2, 3 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 1314.63 | 82.4% |
| opensearch_write | 275.65 | 17.3% |
| file_read | 2.95 | 0.2% |
| embed | 1.12 | 0.1% |
| bulk_serialize | 0.43 | 0.0% |
| doc_build | 0.28 | 0.0% |
| hash | 0.14 | 0.0% |
| decode | 0.14 | 0.0% |
| letterbox | 0.07 | 0.0% |
| whole_frame_preprocess | 0.06 | 0.0% |
| blur | 0.06 | 0.0% |
| crop | 0.01 | 0.0% |
| pe_preprocess | 0.01 | 0.0% |
| jpeg_encode | 0.01 | 0.0% |
| triton_request | 0.00 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 379.27 | 34.20 |
| pe_image_encoder | 23.44 | 33.09 |
| yolov11_small_trt_end2end | 17.23 | 62.38 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 13.65 | 1.9% |
| crop_copy | 27.31 | 3.9% |
| crop_jpeg | 71.25 | 10.0% |
| decode | 81.30 | 11.4% |
| detect | 106.89 | 15.0% |
| embed_crops | 66.18 | 9.4% |
| embed_frame | 6.23 | 0.9% |
| embed_pe | 295.21 | 41.9% |
| letterbox | 11.96 | 1.7% |
| queue | 28.69 | 4.0% |

### set P, 4 batches, headroom, WORKERS 16

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 2900 |
| images/s median (min to max) | 46.76 (46.05 to 46.88) |
| crops per image | 2.41 |
| GPU util mean / p50 % | 94 / 100 |
| Triton cores (docker stats) | 7.37 |
| API CPU-s per image | 0.0781 |
| 1-minute load at start (max) | 17.0 |
| noisy repetitions | 3 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 1232.47 | 82.1% |
| opensearch_write | 263.82 | 17.6% |
| file_read | 3.09 | 0.2% |
| embed | 1.19 | 0.1% |
| bulk_serialize | 0.47 | 0.0% |
| doc_build | 0.28 | 0.0% |
| hash | 0.16 | 0.0% |
| decode | 0.13 | 0.0% |
| whole_frame_preprocess | 0.07 | 0.0% |
| letterbox | 0.07 | 0.0% |
| blur | 0.06 | 0.0% |
| jpeg_encode | 0.01 | 0.0% |
| pe_preprocess | 0.01 | 0.0% |
| crop | 0.01 | 0.0% |
| triton_request | 0.00 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 9.04 | 6.49 |
| pe_image_encoder | 25.44 | 34.41 |
| yolov11_small_trt_end2end | 16.97 | 68.16 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 12.91 | 1.3% |
| crop_copy | 23.81 | 2.4% |
| crop_jpeg | 74.12 | 7.5% |
| decode | 82.20 | 8.4% |
| detect | 109.35 | 11.0% |
| embed_crops | 51.19 | 5.2% |
| embed_frame | 4.49 | 0.5% |
| embed_pe | 305.12 | 30.7% |
| letterbox | 5.46 | 0.6% |
| queue | 324.88 | 32.6% |

