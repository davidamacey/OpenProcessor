# Third-Party Code Attribution

This document provides attribution for third-party code and reference architectures used in this project.

---

## Ultralytics YOLO - End2End ONNX Export

This project uses modified code from the **levipereira/ultralytics** fork to enable GPU-accelerated end-to-end YOLO inference with TensorRT EfficientNMS plugin integration.

### Repository Information
- **Fork Repository:** https://github.com/levipereira/ultralytics
- **Original Repository:** https://github.com/ultralytics/ultralytics (official)
- **Fork Version:** 8.3.18 (October 20, 2024)
- **License:** AGPL-3.0 (same as official ultralytics)
- **Fork Author:** Levi Pereira (@levipereira)

### Code Used

**Approximately 600 lines of custom code** from the fork, specifically:

1. **`export_onnx_trt()` method** (~365 lines)
   - Location: `ultralytics/engine/exporter.py` (lines 460-592 in fork)
   - Purpose: Adds TensorRT EfficientNMS plugin integration to ONNX export graph
   - Enables GPU-accelerated Non-Maximum Suppression (NMS) embedded in model

2. **TensorRT Custom Operators** (~280 lines)
   - `TRT_EfficientNMS` class (torch.autograd.Function)
   - `TRT_EfficientNMS_85` variant (80 classes + 5 additional outputs)
   - `TRT_EfficientNMSX` variant (extended functionality)
   - Location: Lines 1355-1647 in fork
   - Purpose: PyTorch operators that map to TensorRT's EfficientNMS plugin

3. **`End2End_TRT` wrapper class**
   - Wraps YOLO model with NMS layer for end-to-end inference
   - Enables single-pass GPU inference without CPU post-processing

### What This Enables

**TensorRT + GPU NMS:**
- Embeds Non-Maximum Suppression directly into TensorRT engine
- Eliminates CPU post-processing bottleneck
- Achieves **2-5x speedup** by avoiding CPU↔GPU memory transfers for NMS

### Fork Status

As of November 2025:
- Fork is **210 versions behind** official ultralytics (8.3.18 vs 8.3.228)
- Fork provides critical functionality not available in official repository
- Custom operators stable and production-tested by fork maintainer

### Usage in This Project

The fork's end2end export functionality is used to generate:
- `models/yolov11_small_trt_end2end/` - YOLO11 object detection with GPU NMS

---

## NVIDIA TensorRT EfficientNMS Plugin

The end2end models use NVIDIA's **TensorRT EfficientNMS plugin** for GPU-accelerated post-processing.

### Information
- **Provider:** NVIDIA Corporation
- **Documentation:** https://docs.nvidia.com/deeplearning/tensorrt/
- **Plugin:** EfficientNMS_TRT
- **Purpose:** GPU-accelerated Non-Maximum Suppression
- **License:** NVIDIA Deep Learning Software License

### Integration
- Embedded via levipereira/ultralytics fork (see above)
- Compiled into TensorRT engine at model build time
- Executes entirely on GPU, eliminating CPU bottleneck

---

## Apple MobileCLIP

MobileCLIP2-S2 is used for generating image and text embeddings for visual search.

### Repository Information
- **Repository:** https://github.com/apple/ml-mobileclip
- **Paper:** "MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training"
- **Model:** MobileCLIP2-S2 (35.7M parameters, 77.2% ImageNet accuracy)
- **License:** Apple Sample Code License

### Usage in This Project
- Image encoder exported to TensorRT for GPU-accelerated embedding generation
- Text encoder exported to TensorRT for text query embedding
- Reference implementation cloned to `reference_repos/ml-mobileclip/` during setup
- 512-dimensional L2-normalized embeddings for similarity search

### Integration
The MobileCLIP models are exported to TensorRT via the export scripts. The repository applies patches to OpenCLIP for MobileCLIP2 support.

---

## OpenSearch

OpenSearch provides k-NN vector similarity search for all embedding types.

### Information
- **Provider:** OpenSearch Project (AWS-backed)
- **Documentation:** https://opensearch.org/docs/latest/
- **Version:** 3.3.1
- **License:** Apache License 2.0

### Features Used
- k-NN plugin with HNSW algorithm for approximate nearest neighbor search
- Cosine similarity for embedding comparisons
- Nested document queries for per-object search
- Bulk ingestion API for batch indexing

---

## InsightFace SCRFD (Face Detection with Landmarks)

The SCRFD face detection model and post-processing pipeline are based on InsightFace's official implementation.

### Repository Information
- **Repository:** https://github.com/deepinsight/insightface
- **Paper:** "Sample and Computation Redistribution for Efficient Face Detection" (ICLR 2022)
- **Model:** SCRFD-10G with Batch Normalization and Keypoints (scrfd_10g_bnkps)
- **License:** MIT License (InsightFace)
- **Authors:** Jia Guo, Jiankang Deng, Xiang An, Zongguang Yu

### Code Adapted
- **Post-processing pipeline** (`src/utils/scrfd_decode.py`): Anchor generation, `distance2bbox`, `distance2kps`, and NMS functions adapted from InsightFace's `insightface/model_zoo/scrfd.py` and `detection/scrfd/tools/scrfd.py`
- **Face alignment** (`src/utils/face_align.py`): Umeyama similarity transform and ArcFace reference template from `insightface/utils/face_align.py`
- **ArcFace model**: Pre-trained `w600k_r50` from InsightFace's buffalo_l model pack

### What This Enables
- 5-point facial landmark detection (left eye, right eye, nose, left mouth, right mouth)
- Umeyama similarity transform alignment for ArcFace (industry standard)
- 95.2% Easy / 93.9% Medium / 83.1% Hard on WiderFace benchmark

---

## Reference Architectures

The following repositories were used as **reference only** (no code directly copied):

### 1. levipereira/triton-server-yolo
- **URL:** https://github.com/levipereira/triton-server-yolo
- **Usage:** Reference architecture for deploying end2end YOLO models on Triton Inference Server
- **What We Learned:**
  - Ensemble model configuration patterns
  - Dynamic batching configuration for YOLO workloads
- **License:** Not specified in repository
- **Author:** Levi Pereira (@levipereira)

### 2. omarabid59/yolov8-triton
- **URL:** https://github.com/omarabid59/yolov8-triton
- **Usage:** Reference for Triton ensemble patterns and model repository structure
- **What We Learned:**
  - Triton model repository conventions
  - Ensemble preprocessing/inference/postprocessing patterns
- **License:** Not specified in repository

### 3. NVIDIA Triton Inference Server
- **URL:** https://github.com/triton-inference-server/server
- **Documentation:** https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Usage:** Official Triton documentation and examples
- **License:** BSD 3-Clause License

### 4. hiennguyen9874/triton-face-recognition
- **URL:** https://github.com/hiennguyen9874/triton-face-recognition
- **Usage:** Reference architecture for deploying face detection + recognition on Triton with dynamic batching
- **What We Learned:**
  - Triton config patterns for face detection with landmarks (end2end and raw outputs)
  - TensorRT export with dynamic batch shapes for face models
  - Face alignment (norm_crop) integration with Triton client pipelines
- **License:** Not specified in repository
- **Author:** Hien Nguyen (@hiennguyen9874)

### 5. SthPhoenix/InsightFace-REST
- **URL:** https://github.com/SthPhoenix/InsightFace-REST
- **Usage:** Reference for SCRFD TensorRT deployment with dynamic batching at scale
- **What We Learned:**
  - SCRFD model export strategies for TensorRT (handling batch-1 reshape limitation)
  - Performance benchmarks for SCRFD on various GPU hardware (820 FPS on RTX 4090)
  - Production face recognition pipeline architecture
- **License:** MIT License

---

## Acknowledgments

Special thanks to:

- **Levi Pereira (@levipereira)** - For the ultralytics fork with end2end TensorRT export and the triton-server-yolo reference architecture
- **Ultralytics Team** - For the YOLO models and official ultralytics library
- **NVIDIA Corporation** - For Triton Inference Server and TensorRT
- **Omar Abid (@omarabid59)** - For the yolov8-triton reference implementation
- **Apple Machine Learning Research** - For MobileCLIP efficient vision-language models
- **OpenSearch Project** - For the k-NN vector search engine
- **InsightFace Team (Jia Guo, Jiankang Deng et al.)** - For SCRFD face detection, ArcFace recognition, and face alignment algorithms
- **Hien Nguyen (@hiennguyen9874)** - For the triton-face-recognition reference implementation
- **OpenCLIP Contributors** - For the open-source CLIP implementation

---

## License Compliance

### This Project
This project's original code is licensed under **MIT License** (see [LICENSE](LICENSE)).

### Third-Party Components

| Component | License | Attribution Required |
|-----------|---------|---------------------|
| levipereira/ultralytics | AGPL-3.0 | ✓ Yes (this file) |
| Ultralytics YOLO | AGPL-3.0 | ✓ Yes (inherited) |
| NVIDIA Triton | BSD 3-Clause | ✓ Yes |
| NVIDIA TensorRT | NVIDIA DSLA | ✓ Yes |
| Apple MobileCLIP | Apple Sample Code | ✓ Yes (this file) |
| InsightFace (SCRFD, ArcFace) | MIT | ✓ Yes (this file) |
| OpenSearch | Apache 2.0 | ✓ Yes |
| OpenCLIP | MIT | ✓ Yes |

**Note:** The use of AGPL-3.0 licensed code (ultralytics fork) may impose obligations on derivative works. Consult the AGPL-3.0 license for details: https://www.gnu.org/licenses/agpl-3.0.en.html

---

## Contact and Questions

For questions about attribution or licensing:
1. Review the detailed analysis in [docs/Attribution/](docs/Attribution/)
2. Consult the original repository licenses linked above
3. For fork-specific questions, contact the fork maintainer: https://github.com/levipereira

---

**Last Updated:** January 2026
