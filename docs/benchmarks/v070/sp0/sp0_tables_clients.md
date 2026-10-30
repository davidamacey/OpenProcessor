### set B, 4 clients

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 400 |
| images/s median (min to max) | 26.64 (25.21 to 29.00) |
| crops per image | 1.65 |
| GPU util mean / p50 % | 49 / 43 |
| Triton cores (docker stats) | 2.54 |
| API CPU-s per image | 0.3034 |
| 1-minute load at start (max) | 21.0 |
| noisy repetitions | 2, 3 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 1149.95 | 70.4% |
| opensearch_write | 269.16 | 16.5% |
| decode | 105.36 | 6.5% |
| blur | 32.89 | 2.0% |
| letterbox | 24.21 | 1.5% |
| whole_frame_preprocess | 23.39 | 1.4% |
| embed | 16.82 | 1.0% |
| file_read | 7.25 | 0.4% |
| crop | 1.69 | 0.1% |
| pe_preprocess | 0.92 | 0.1% |
| jpeg_encode | 0.79 | 0.0% |
| bulk_serialize | 0.34 | 0.0% |
| triton_request | 0.23 | 0.0% |
| doc_build | 0.21 | 0.0% |
| hash | 0.16 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 44.31 | 17.42 |
| pe_image_encoder | 7.09 | 14.22 |
| yolov11_small_trt_end2end | 14.97 | 14.15 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 20.44 | 3.6% |
| crop_copy | 24.65 | 4.4% |
| crop_jpeg | 31.11 | 5.1% |
| decode | 121.19 | 26.2% |
| detect | 50.40 | 8.9% |
| embed_crops | 29.38 | 5.0% |
| embed_frame | 45.67 | 7.8% |
| embed_pe | 79.20 | 15.4% |
| letterbox | 97.63 | 17.9% |
| queue | 21.84 | 4.3% |

### set B, 4 clients, second run

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 400 |
| images/s median (min to max) | 28.53 (26.23 to 28.84) |
| crops per image | 1.65 |
| GPU util mean / p50 % | 46 / 44 |
| Triton cores (docker stats) | 1.62 |
| API CPU-s per image | 0.2594 |
| 1-minute load at start (max) | 59.4 |
| noisy repetitions | 1, 2, 3 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 1228.43 | 73.6% |
| opensearch_write | 288.03 | 17.3% |
| decode | 85.63 | 5.1% |
| letterbox | 15.44 | 0.9% |
| whole_frame_preprocess | 14.56 | 0.9% |
| blur | 14.18 | 0.8% |
| embed | 9.39 | 0.6% |
| file_read | 6.91 | 0.4% |
| jpeg_encode | 4.45 | 0.3% |
| crop | 1.23 | 0.1% |
| pe_preprocess | 0.66 | 0.0% |
| bulk_serialize | 0.28 | 0.0% |
| doc_build | 0.19 | 0.0% |
| hash | 0.15 | 0.0% |
| triton_request | 0.10 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 61.99 | 10.45 |
| pe_image_encoder | 11.41 | 17.37 |
| yolov11_small_trt_end2end | 15.15 | 14.08 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 28.99 | 4.8% |
| crop_copy | 32.93 | 5.5% |
| crop_jpeg | 30.76 | 4.8% |
| decode | 114.60 | 19.7% |
| detect | 61.03 | 9.6% |
| embed_crops | 64.50 | 9.1% |
| embed_frame | 82.80 | 11.7% |
| embed_pe | 126.56 | 16.4% |
| letterbox | 103.51 | 17.0% |
| queue | 10.20 | 2.0% |

### set B, 4 clients, counters removed

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 400 |
| images/s median (min to max) | 29.21 (25.60 to 30.33) |
| crops per image | 1.65 |
| GPU util mean / p50 % | 43 / 28 |
| Triton cores (docker stats) | 2.88 |
| API CPU-s per image | 0.2493 |
| 1-minute load at start (max) | 25.4 |
| noisy repetitions | 1, 2 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 1183.44 | 72.0% |
| opensearch_write | 311.77 | 19.0% |
| decode | 82.06 | 5.0% |
| letterbox | 15.41 | 0.9% |
| whole_frame_preprocess | 13.88 | 0.8% |
| blur | 13.38 | 0.8% |
| embed | 9.05 | 0.6% |
| file_read | 8.31 | 0.5% |
| jpeg_encode | 4.18 | 0.3% |
| crop | 1.19 | 0.1% |
| pe_preprocess | 0.73 | 0.0% |
| bulk_serialize | 0.32 | 0.0% |
| doc_build | 0.21 | 0.0% |
| hash | 0.16 | 0.0% |
| triton_request | 0.10 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 33.62 | 9.16 |
| pe_image_encoder | 11.33 | 17.41 |
| yolov11_small_trt_end2end | 15.68 | 19.03 |

### set B, 4 clients, counters (interleaved)

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 400 |
| images/s median (min to max) | 29.95 (26.50 to 31.75) |
| crops per image | 1.65 |
| GPU util mean / p50 % | 53 / 58 |
| Triton cores (docker stats) | 2.86 |
| API CPU-s per image | 0.2370 |
| 1-minute load at start (max) | 11.1 |
| noisy repetitions | none |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 1121.70 | 72.4% |
| opensearch_write | 288.95 | 18.6% |
| decode | 76.47 | 4.9% |
| letterbox | 13.37 | 0.9% |
| whole_frame_preprocess | 13.08 | 0.8% |
| blur | 12.43 | 0.8% |
| embed | 10.89 | 0.7% |
| file_read | 7.25 | 0.5% |
| jpeg_encode | 3.75 | 0.2% |
| crop | 0.98 | 0.1% |
| pe_preprocess | 0.64 | 0.0% |
| bulk_serialize | 0.31 | 0.0% |
| doc_build | 0.20 | 0.0% |
| hash | 0.15 | 0.0% |
| triton_request | 0.09 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 15.43 | 7.85 |
| pe_image_encoder | 9.23 | 18.04 |
| yolov11_small_trt_end2end | 15.99 | 18.68 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 23.51 | 4.4% |
| crop_copy | 29.82 | 5.2% |
| crop_jpeg | 30.44 | 5.5% |
| decode | 107.70 | 23.0% |
| detect | 60.12 | 10.9% |
| embed_crops | 35.39 | 6.9% |
| embed_frame | 43.69 | 8.6% |
| embed_pe | 103.50 | 20.3% |
| letterbox | 68.22 | 12.3% |
| queue | 5.24 | 0.8% |

### set P, 4 clients

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 2900 |
| images/s median (min to max) | 24.98 (23.50 to 33.08) |
| crops per image | 2.41 |
| GPU util mean / p50 % | 58 / 60 |
| Triton cores (docker stats) | 4.36 |
| API CPU-s per image | 0.2004 |
| 1-minute load at start (max) | 26.4 |
| noisy repetitions | 1 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 1757.81 | 77.4% |
| opensearch_write | 332.62 | 14.7% |
| decode | 49.49 | 2.2% |
| blur | 37.03 | 1.6% |
| embed | 29.58 | 1.3% |
| whole_frame_preprocess | 25.37 | 1.1% |
| letterbox | 23.96 | 1.1% |
| crop | 4.45 | 0.2% |
| file_read | 3.41 | 0.2% |
| jpeg_encode | 2.86 | 0.1% |
| pe_preprocess | 2.10 | 0.1% |
| bulk_serialize | 0.41 | 0.0% |
| doc_build | 0.26 | 0.0% |
| triton_request | 0.15 | 0.0% |
| hash | 0.15 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 74.94 | 12.87 |
| pe_image_encoder | 17.37 | 21.80 |
| yolov11_small_trt_end2end | 15.35 | 16.11 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 49.59 | 3.0% |
| crop_copy | 186.47 | 10.8% |
| crop_jpeg | 81.01 | 4.8% |
| decode | 159.71 | 10.8% |
| detect | 68.26 | 4.2% |
| embed_crops | 325.16 | 21.3% |
| embed_frame | 96.30 | 5.7% |
| embed_pe | 203.21 | 11.3% |
| letterbox | 183.49 | 11.2% |
| queue | 180.82 | 14.0% |

### set P, 8 clients

| measure | value |
|---|---|
| repetitions | 3 |
| images per repetition | 2900 |
| images/s median (min to max) | 20.13 (19.20 to 21.91) |
| crops per image | 2.41 |
| GPU util mean / p50 % | 39 / 19 |
| Triton cores (docker stats) | 2.87 |
| API CPU-s per image | 0.7007 |
| 1-minute load at start (max) | 23.0 |
| noisy repetitions | 2, 3 |

#### API stages (op_pipeline_stage_seconds, sums over concurrent work)

| stage | ms / image | share |
|---|---|---|
| gpu_decode | 2987.91 | 67.0% |
| opensearch_write | 398.22 | 8.9% |
| embed | 349.04 | 7.8% |
| decode | 278.05 | 6.2% |
| blur | 170.02 | 3.8% |
| letterbox | 125.94 | 2.8% |
| whole_frame_preprocess | 105.41 | 2.4% |
| crop | 16.46 | 0.4% |
| jpeg_encode | 10.63 | 0.2% |
| pe_preprocess | 8.31 | 0.2% |
| file_read | 4.73 | 0.1% |
| triton_request | 0.73 | 0.0% |
| bulk_serialize | 0.46 | 0.0% |
| doc_build | 0.33 | 0.0% |
| hash | 0.15 | 0.0% |
| admission_wait | 0.00 | 0.0% |
| persist | 0.00 | 0.0% |

#### Triton models

| model | queue ms / request | compute ms / request |
|---|---|---|
| gpu_image_prep | 1079.75 | 33.77 |
| pe_image_encoder | 38.80 | 35.75 |
| yolov11_small_trt_end2end | 7.30 | 10.85 |

#### gpu_image_prep stages (host wall, sums over concurrent requests)

| stage | ms / image | share |
|---|---|---|
| blur | 122.31 | 2.5% |
| crop_copy | 382.22 | 9.1% |
| crop_jpeg | 94.65 | 2.0% |
| decode | 469.53 | 30.5% |
| detect | 111.45 | 2.6% |
| embed_crops | 683.04 | 14.2% |
| embed_frame | 227.71 | 4.9% |
| embed_pe | 196.07 | 4.3% |
| letterbox | 475.21 | 14.3% |
| queue | 328.65 | 18.6% |

