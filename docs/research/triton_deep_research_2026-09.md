> Status: research snapshot dated 2026-09-26, written against the repository as it was then. Findings about the current code path may have been fixed since; the actionable plan derived from it is [`docs/design/triton_pipeline_optimization_plan.md`](../design/triton_pipeline_optimization_plan.md) (tracked in #40). Treat external references and case studies as the durable part.

# NVIDIA Triton Inference Server + OpenProcessor
## Deep Research, External Reference Implementations, Production Lessons, Performance Architecture, and Improvement Roadmap

**Research date:** September 26, 2026  
**Primary subject:** NVIDIA Triton Inference Server / NVIDIA Dynamo-Triton for high-throughput computer-vision inference  
**Repository under review:** https://github.com/davidamacey/OpenProcessor  
**Goal:** Determine how to turn OpenProcessor into a much higher-throughput, production-grade visual inference platform for object detection, face detection/recognition, image embeddings, CLIP-style models, Perception Encoder, OCR, and related image-processing workloads.

> **Important distinction:** OpenProcessor is the system being critiqued. It is included below only as the current implementation baseline. It is **not** treated as an independent reference implementation. The independent references are NVIDIA code/documentation, outside GitHub projects, GTC sessions, NVIDIA customer case studies, engineering blogs, public issue/forum discussions, and company production deployments.

---

# Table of Contents

1. Executive Summary
2. What Triton Is and Where It Stands in 2026
3. What Triton Is Good At
4. OpenProcessor Current Baseline
5. Current OpenProcessor Code-Path Audit
6. The Biggest Architectural Finding
7. Recommended Target Architecture
8. DALI and Server-Side Image Preprocessing
9. Decode Once, Fan Out to Many Models
10. Triton Dynamic Batching
11. Model Instances and Multi-Model GPU Scheduling
12. Ensembles vs BLS vs External Orchestration
13. gRPC, HTTP, Shared Memory, and Transport
14. Face Detection and Recognition Pipeline
15. MobileCLIP, PE-Core, and Embedding Serving
16. OCR as a Separate Workload Lane
17. OpenSearch and Persistence
18. FastAPI's Role
19. Directory and Large-Library Ingestion
20. TensorRT Optimization
21. Observability: Metrics, Tracing, and Profiling
22. Benchmark Methodology
23. Production Company Case Studies
24. Independent GitHub Repositories to Study
25. NVIDIA GTC, Training, and Video Resources
26. Community Issues and Failure Modes
27. DeepStream, CV-CUDA, nvImageCodec, and Video
28. Deployment Patterns: Single Server Through Kubernetes
29. Concrete OpenProcessor Implementation Roadmap
30. Proposed Pull Requests / Work Packages
31. Illustrative Code and Configuration
32. Performance Hypotheses to Test
33. What Not to Optimize First
34. Recommended Reading Order
35. Complete Resource Catalog
36. Appendix A — OpenProcessor Source Audit Links
37. Appendix B — Claims, Evidence, and Caveats

---

# 1. Executive Summary

The strongest conclusion from all of the research is:

> **Keep Triton. Change OpenProcessor so Triton owns much more of the image-processing dataplane.**

OpenProcessor already has a good foundation:

- TensorRT engines
- NVIDIA Triton
- gRPC
- dynamic batching
- YOLO detection
- SCRFD
- ArcFace
- MobileCLIP
- PE-Core
- OCR
- OpenSearch
- FastAPI
- Prometheus/Grafana
- batch APIs
- an asynchronous Triton client pool

The problem is not that OpenProcessor chose the wrong serving technology.

The problem is that the highest-volume execution path is still substantially **Python-centric and request-centric**.

The current architecture is approximately:

```text
image
  |
  v
FastAPI
  |
Python decode / resize / normalization
  |
Python orchestration
  |
sync Triton request
  |
wait
  |
another sync Triton request
  |
CPU postprocessing
  |
OpenSearch operations
```

The architecture demonstrated by the most relevant outside production systems is closer to:

```text
compressed image
      |
      v
bounded asynchronous ingest
      |
      v
Triton
  |
  +--> DALI/nvJPEG decode once
  |
  +--> preprocess branch --> YOLO TensorRT
  |
  +--> preprocess branch --> MobileCLIP TensorRT
  |
  +--> preprocess branch --> PE-Core TensorRT
  |
  +--> preprocess branch --> SCRFD TensorRT
                               |
                               v
                         GPU postprocess
                               |
                               v
                         face work queue
                               |
                               v
                         ArcFace batches

results
  |
  +--> OpenSearch bulk writer
  |
  +--> optional OCR/enrichment queue
```

## Highest-priority OpenProcessor findings

The following were re-verified against `main` on September 26, 2026.

### 1. `/ingest/batch` is still using a synchronous 8-thread path

OpenProcessor has an `AsyncTritonPool` configured for four channels and up to 64 concurrent requests, but the central bulk visual-search ingest path still acquires the synchronous Triton client and executes the per-image pipeline inside a `ThreadPoolExecutor` capped at eight inference workers.

Current source:

- `src/services/visual_search.py`
- https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/services/visual_search.py

This is one of the first paths that should be redesigned.

### 2. YOLO and MobileCLIP are artificially serialized

`infer_yolo_clip_cpu()`:

1. decodes with PIL
2. preprocesses YOLO on CPU
3. preprocesses MobileCLIP on CPU
4. runs YOLO through Triton
5. waits
6. runs MobileCLIP through Triton
7. waits

The two inference operations have no dependency on one another.

Current source:

- `src/clients/triton_client.py`
- https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/clients/triton_client.py

Even before DALI is introduced, those calls should at minimum be submitted concurrently.

### 3. OpenProcessor preprocesses images in Python before Triton

For YOLO and MobileCLIP the current path performs:

```text
JPEG bytes
  -> PIL/OpenCV
  -> NumPy
  -> resize/letterbox
  -> normalization
  -> transpose
  -> FP32 CHW
  -> gRPC
  -> Triton
```

This is precisely the class of bottleneck DALI was created to address.

A 640 × 640 × 3 FP32 tensor is about 4.9 MB before protocol overhead. A compressed JPEG can be dramatically smaller. Sending compressed media and processing it at the inference server can therefore reduce both CPU work and transport volume.

### 4. The face path bounces CPU -> GPU -> CPU -> GPU

Current code documents:

```text
FastAPI
  -> CPU image decode + resize
  -> SCRFD TensorRT
  -> CPU anchor decode
  -> CPU NMS
  -> CPU Umeyama affine alignment
  -> ArcFace TensorRT
```

Source:

- `src/clients/fast_face_client.py`
- https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/clients/fast_face_client.py

That is a good correctness-first implementation, but it is not the endpoint for maximum throughput.

### 5. Directory ingest is sequential at the batch level

Current code:

- first materializes the directory listing with `list(rglob(...))`
- reads the files for a batch
- calls `ingest_batch()`
- waits
- then reads the next batch

Source:

- `src/routers/ingest.py`
- https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/routers/ingest.py

That prevents steady-state overlap among storage reads, hashing, image decode, GPU inference, OpenSearch writes, and OCR enrichment.

### 6. The response middleware performs avoidable serialization work

For inference endpoints, OpenProcessor currently consumes the full response body, parses JSON with `orjson`, injects timing/request ID, serializes JSON again, and creates a replacement response.

For the high-throughput path, use headers such as:

```text
X-Process-Time
X-Request-ID
```

rather than reparsing large payloads.

### 7. Worker count and GPU concurrency are mixed together

Current `main.py` initializes:

```text
ThreadPoolExecutor(max_workers=64)
AsyncTritonPool(pool_size=4, max_concurrent=64)
```

for each FastAPI process.

The README separately suggests `--workers=64` for production.

A 64-process deployment could theoretically expose:

```text
64 x 64 = 4,096 executor worker slots
64 x 4  =   256 Triton client channels
```

Executor threads are lazy, but the design still risks using application-process concurrency to solve work that Triton's scheduler should solve.

### 8. The custom gRPC pool contains assumptions that need proof

`triton_pool.py` describes the multi-channel scheme as a “Fortune 100 pattern” and manually specifies keepalive, stream/window, and connection options.

I did **not** find independent evidence that the exact OpenProcessor recipe is a standard production Triton pattern.

Benchmark it against:

```text
1 normal aio client
4 clients
8 clients
current custom channel options
gRPC streaming
system shared memory
CUDA shared memory
```

### 9. Dynamic batching needs to be re-baselined

Current NVIDIA documentation explicitly says that `preferred_batch_size` should **not** be configured for most models.

NVIDIA recommends starting with ordinary dynamic batching and then adding queue delay only if the throughput increase is worth the extra latency.

Reference:

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html

### 10. Separate interactive and bulk personalities

Interactive:

```text
batch immediately
minimal queue delay
optimize p95/p99
```

Bulk:

```text
allow deeper queues
larger actual batches
optimize images/sec
```

It can be useful to expose the same engine through different Triton model aliases/configurations or separate worker lanes.

### 11. Use Model Analyzer for the workload mix, not one model at a time

OpenProcessor runs several models on the same GPU. Optimizing each in isolation can produce a globally bad deployment.

Current Model Analyzer supports single models, ensembles, BLS, and multi-model concurrent profiling.

Reference:

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_analyzer/README.html

### 12. NIO is the closest public production reference

NIO moved image preprocessing from client CPUs to server-side GPU processing using Triton, nvJPEG, and DALI and used BLS to orchestrate pipelines.

NVIDIA reports:

- up to 6× latency reduction in some core pipelines
- up to 5× overall throughput improvement

Reference:

https://developer.nvidia.com/blog/designing-an-optimal-ai-inference-pipeline-for-autonomous-driving/

### 13. Snap is a strong mixed-CV/Triton operations reference

Snap publicly describes:

- Triton as a universal serving layer
- ensembles
- TensorRT
- Model Analyzer
- OCR with custom logic
- Kubernetes
- Prometheus
- more than 1,000 T4/L4 GPUs in the cited deployment
- 3× OCR throughput in the cited workload

References:

https://developer.nvidia.com/blog/?p=82250  
https://www.nvidia.com/en-us/on-demand/session/gtc24-s61915/  
https://www.nvidia.com/en-us/on-demand/session/gtc24-s62137/

### 14. Oracle provides a useful FastAPI-vs-Triton case

Oracle OCI Vision publicly compared a custom serving implementation with Triton and reported approximately:

- 30–76% throughput improvement
- 30–51% latency reduction

depending on model and concurrency.

Reference:

https://blogs.oracle.com/ai-and-datascience/oci-ai-vision-nvidia-triton-inference-server

### 15. An independent repo measured Python preprocessing directly

The independent repository:

https://github.com/simon-bouchard/cv-inference-triton

reported approximately:

| Stage | Python preprocessing | C++ preprocessing |
|---|---:|---:|
| Preprocess p50 | ~22 ms | ~12.7 ms |
| Peak preprocess rate | ~87 img/s | ~114 img/s |
| Full TensorRT pipeline p50 | ~36.2 ms | ~25.0 ms |

The exact values are hardware-specific, but the lesson is strong:

> Measure preprocessing as a first-class workload.

---

# 2. What Triton Is and Where It Stands in 2026

NVIDIA Triton Inference Server is an open-source inference-serving system supporting multiple model backends and both CPU and GPU deployment.

Typical backends include:

- TensorRT
- ONNX Runtime
- PyTorch / LibTorch
- Python
- OpenVINO
- FIL
- custom C++ backends
- specialized backends

It exposes standardized inference APIs including HTTP/REST and gRPC.

## Current release

As of this research date, NVIDIA's 26.08 container corresponds to Triton 2.72.0 and includes a recent CUDA/TensorRT/DALI/nvImageCodec stack.

Release notes:

https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-08.html

## OpenProcessor currently uses 26.06

Current Dockerfile:

```text
FROM nvcr.io/nvidia/tritonserver:26.06-py3
```

Source:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/Dockerfile.triton

An upgrade to 26.08 is reasonable to test, but TensorRT plans should be rebuilt and regression-tested on the target stack.

## Dynamo-Triton naming

NVIDIA now presents Triton within the broader NVIDIA Dynamo product family.

That does **not** make it LLM-only.

Current product material:

https://www.nvidia.com/en-us/ai/dynamo-triton/  
https://developer.nvidia.com/triton-inference-server

For OpenProcessor's CV workload, the conventional Triton server remains highly relevant.

---

# 3. What Triton Is Good At

Triton's value is not merely “put a REST endpoint around a model.”

Capabilities include:

- dynamic batching
- concurrent model execution
- multiple model instances
- multi-GPU placement
- model versioning
- model repositories
- TensorRT execution
- ONNX execution
- Python and C++ custom backends
- server-side preprocessing/postprocessing
- model ensembles
- Business Logic Scripting
- rate limiting
- priorities
- response caching
- metrics
- tracing
- shared memory
- HTTP
- gRPC
- KServe-compatible protocol
- live model loading/unloading
- Kubernetes deployment
- Prometheus integration

OpenProcessor's real problem is:

```text
run several heterogeneous models
on shared GPU resources
with varying batch sizes
with high concurrency
while maintaining predictable latency
and avoiding repeated decode/preprocessing work
```

That is exactly the class of problem Triton targets.

---

# 4. OpenProcessor Current Baseline

Repository:

https://github.com/davidamacey/OpenProcessor

Current README version observed during research:

```text
0.3.0
```

Current major models include:

| Model | Purpose | Serving direction |
|---|---|---|
| YOLO11 | object detection | TensorRT |
| YOLO26 | object detection | TensorRT |
| SCRFD-10G | face detection + landmarks | TensorRT |
| ArcFace | face embeddings | TensorRT |
| MobileCLIP | image/text embeddings | TensorRT |
| PP-OCRv5 | OCR | TensorRT + orchestration |
| PE-Core-L14-336 | higher-quality image embedding | Triton path for image encoder |

Current README performance values are approximately:

| Operation | Published time | Published throughput |
|---|---:|---:|
| Object detection | 140–170 ms | ~6–7 RPS |
| Face detection | 100–150 ms | ~7–10 RPS |
| Face recognition | 105–130 ms | ~8–9 RPS |
| Image embedding / CLIP | 6–8 ms | ~120 RPS |
| Text embedding / CLIP | 5–17 ms | ~60–200 RPS |
| OCR | 170–350 ms | ~3–6 RPS |
| Full analyze | 280–430 ms | ~2–3 RPS |
| Single-image ingest | 750–950 ms | ~1–1.3 RPS |
| Batch ingest, 50 images | ~7.3 sec | ~6.8 img/s |

Source:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/README.md

The disparity suggests the whole ingest pipeline, not just the model kernel, is the main optimization target.

---

# 5. Current OpenProcessor Code-Path Audit

## `ingest_image()` already parallelizes some top-level work

Single-image ingestion launches the combined YOLO/CLIP path and face recognition in a small thread pool.

Good:

```text
image
  |
  +--> YOLO + CLIP branch
  |
  +--> SCRFD + ArcFace branch
```

But YOLO and CLIP remain serialized inside their branch.

## Bulk path still uses synchronous inference

Current `VisualSearchService.ingest_batch()` acquires synchronous clients and then uses a `ThreadPoolExecutor` with:

```text
inference_workers = min(8, len(images_bytes))
```

Source:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/services/visual_search.py

## Async Triton pool already exists

App startup creates:

```text
AsyncTritonPool(
    pool_size=4,
    max_concurrent=64
)
```

Source:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/main.py

The architectural groundwork already exists; the main ingest path just does not use it.

## Explicit batched methods already exist

`triton_client.py` contains batching helpers for MobileCLIP and ArcFace.

Source:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/clients/triton_client.py

## YOLO + MobileCLIP serialization

Current simplified flow:

```text
decode
preprocess YOLO
preprocess CLIP
infer YOLO
wait
infer MobileCLIP
wait
```

These inference calls are independent.

## Face pipeline

Current:

```text
CPU decode + resize
  -> SCRFD TensorRT
  -> CPU anchor decode + NMS
  -> CPU affine alignment
  -> ArcFace TensorRT
```

Source:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/clients/fast_face_client.py

## Directory traversal

Current directory ingest materializes `list(dir_path.rglob('*'))`, reads one batch, waits for ingestion, then proceeds.

Source:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/routers/ingest.py

## Response middleware

Current middleware reads and reserializes response bodies to inject timing/request data.

Source:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/main.py

## PE encoder

Current PE image encoder is already routed through the async Triton pool; text queries are CPU-side with LRU caching.

Source:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/clients/pe_encoder.py

---

# 6. The Biggest Architectural Finding

The central lesson from NIO, Snap, Oracle, NVIDIA DALI examples, and the independent CV repo is:

> **The next large speed gain is more likely to come from the data path than from squeezing another few percent from an already-TensorRT model.**

OpenProcessor has fast engines surrounded by a pipeline that:

- repeatedly decodes
- builds NumPy tensors
- uses synchronous calls
- waits at stage boundaries
- performs CPU postprocessing
- performs database work in the same workflow

Think of the system as a factory. The goal is to keep storage, CPU, decode, GPU, database, and OCR workers busy simultaneously.

---

# 7. Recommended Target Architecture

## Interactive/control plane

Keep FastAPI for:

- authentication
- jobs
- metadata
- query/search
- health
- model controls
- interactive inference

## Bulk dataplane

For high-volume local/S3/NAS processing:

```text
filesystem / NAS / S3 / MinIO
             |
             v
      ingest dispatcher
             |
       bounded queues
             |
             v
           Triton
             |
             v
       result queues
        /         \
       v           v
OpenSearch       OCR/enrichment
bulk writer      worker
```

At steady state:

```text
disk is reading       N+2
CPU is hashing        N+1
GPU is inferring      N
OpenSearch is writing N-1
OCR is enriching      N-2
```

Every queue should be bounded to provide backpressure.

---

# 8. DALI and Server-Side Image Preprocessing

NVIDIA DALI is one of the most important technologies for OpenProcessor.

It can run as a Triton backend and move decode/resize/normalize closer to the GPU server.

Instead of sending:

```text
decoded FP32 CHW tensor
```

send:

```text
compressed JPEG/PNG bytes
```

Potential operations:

- image decode
- resize
- crop
- normalization
- color conversion
- transpose/layout change
- padding

References:

https://developer.nvidia.com/blog/?p=30560  
https://developer.nvidia.com/blog/rapid-data-pre-processing-with-nvidia-dali/  
https://github.com/triton-inference-server/dali_backend  
https://github.com/triton-inference-server/dali_backend/blob/main/docs/tutorials/training_to_inference.md  
https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/nvidia-dali.md

Important nuance: DALI decode may be mixed CPU/GPU. The benefit is not “zero CPU”; it is eliminating repeated application-side PIL/OpenCV decode and tensor construction.

Do not blindly create many DALI instances; decoders and pipelines consume memory.

---

# 9. Decode Once, Fan Out to Many Models

Target:

```text
                       +--> YOLO transform --> YOLO
                       |
JPEG --> decode once --+--> CLIP transform --> MobileCLIP
                       |
                       +--> PE transform ----> PE-Core
                       |
                       +--> face transform --> SCRFD
                       |
                       +--> OCR transform ----> OCR
```

This reduces:

- repeated JPEG decompression
- repeated color conversion
- repeated allocations
- repeated runtime transitions
- repeated network transfer
- repeated CPU-GPU copies

NIO's public Triton/DALI architecture is the strongest precedent.

Reference:

https://developer.nvidia.com/blog/designing-an-optimal-ai-inference-pipeline-for-autonomous-driving/

---

# 10. Triton Dynamic Batching

Dynamic batching combines independent requests when possible.

Example:

```text
A: 1 image
B: 1 image
C: 1 image
D: 1 image
```

can become:

```text
batch of 4
```

Recommended starting configuration:

```protobuf
max_batch_size: 64

dynamic_batching {}
```

Then benchmark.

Source:

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html

NVIDIA explicitly says preferred batch sizes should not be used for most models.

Queue delay should only be added if throughput gains justify latency.

For interactive and bulk, maintain separate policies if necessary.

---

# 11. Model Instances and Multi-Model GPU Scheduling

Example:

```protobuf
instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]
```

Possible benefit:

- copy/compute overlap
- parallel small-model execution

Possible cost:

- extra memory
- contention
- cache pressure
- worse latency
- less room for other models

Use Model Analyzer:

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_analyzer/README.html

Profile the **whole model mix**, not just each model in isolation.

---

# 12. Ensembles vs BLS vs External Orchestration

## Triton Ensemble

Best for fixed DAGs:

```text
encoded image
  -> DALI
  -> model
  -> postprocess
```

Docs:

https://github.com/triton-inference-server/server/blob/main/docs/user_guide/ensemble_models.md

Tutorial:

https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_5-Model_Ensembles

## Business Logic Scripting

Best for conditional/data-dependent flows:

```text
run SCRFD
if faces:
    align
    call ArcFace
```

Docs:

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/bls.html

NIO used BLS in its public architecture.

## External async orchestrator

Best for:

- filesystem
- hashing
- database
- retries
- job state
- user policy
- workflow management

Recommended split:

```text
external worker -> controls job/storage/database
Triton -> owns preprocessing/model DAG/GPU-friendly postprocess
```

---

# 13. gRPC, HTTP, Shared Memory, and Transport

## gRPC vs HTTP

gRPC is a strong choice, but not always the bottleneck.

`cv-inference-triton` reported no meaningful difference in its tested workload because preprocessing/GPU work dominated.

https://github.com/simon-bouchard/cv-inference-triton

## Multiple gRPC channels

Benchmark:

```text
1
2
4
8
```

while watching throughput, p95/p99, CPU, network, and queue time.

## Shared memory

Triton supports:

- system shared memory
- CUDA shared memory

Docs:

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_shared_memory.html

Most useful when:

- client and server share a host
- tensors are large
- request rate is high
- serialization is measurable
- data is already in GPU memory

---

# 14. Face Detection and Recognition Pipeline

Current:

```text
JPEG
 -> OpenCV decode CPU
 -> SCRFD preprocess CPU
 -> SCRFD GPU
 -> raw tensors CPU
 -> anchor decode/NMS CPU
 -> affine alignment CPU
 -> ArcFace preprocess CPU
 -> ArcFace GPU
```

## Improvement 1 — batch faces globally

If 64 photos yield 137 faces, do:

```text
ArcFace batch 64
ArcFace batch 64
ArcFace batch 9
```

instead of small image-local batches.

## Improvement 2 — GPU SCRFD postprocessing

Possible path:

1. TensorRT where practical
2. Triton Python backend with GPU tensors/DLPack
3. C++/CUDA backend if needed

## Improvement 3 — GPU alignment

Possible tools:

- DALI
- CV-CUDA
- CUDA
- OpenCV CUDA
- custom backend

## External face repos

https://github.com/hiennguyen9874/triton-face-recognition  
https://github.com/yiqisoft/Face-Recognition-with-Triton-Inference-Server

---

# 15. MobileCLIP, PE-Core, and Embedding Serving

OpenProcessor's MobileCLIP image encoder is already much faster than the complete ingest path.

This suggests optimization should focus on:

- decode
- preprocessing
- batching
- transport
- persistence

before changing the model solely for speed.

Current PE-Core path:

- image encoder via Triton async pool
- 1024-dimensional image embeddings
- CPU text encoder
- LRU text cache
- crop batching
- L2 normalization

Source:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/clients/pe_encoder.py

Do not mix MobileCLIP and PE vectors in one index.

Use model/version-specific indices, for example:

```text
images_mobileclip_s2_v1
images_pe_core_l14_336_v1
faces_arcface_r50_v1
```

Apple MobileCLIP repository:

https://github.com/apple/ml-mobileclip

---

# 16. OCR as a Separate Workload Lane

OCR is much slower than MobileCLIP in OpenProcessor's published numbers.

For bulk ingest, consider:

```text
image
  -> YOLO / embedding / face
  -> primary index complete

         |
         v

      OCR queue
         |
         v
      OCR models
         |
         v
   patch search document
```

Use synchronous OCR only when the caller explicitly needs it immediately.

Snap's public OCR session reports ~3× OCR throughput for its workload after adopting Triton-oriented serving.

https://www.nvidia.com/en-us/on-demand/session/gtc24-s62137/

---

# 17. OpenSearch and Persistence

Do not let OpenSearch latency directly idle the GPU.

Bad:

```text
infer -> wait -> index -> wait -> infer
```

Better:

```text
GPU inference
   |
bounded result queue
   |
bulk writer(s)
```

Benchmark bulk sizes such as:

```text
128
256
512
1000
```

based on real document/vector size.

If OpenSearch slows down, queue backpressure should eventually throttle inference without unbounded memory growth.

Measure OpenSearch separately with pregenerated documents.

---

# 18. FastAPI's Role

Keep FastAPI for:

- REST
- auth
- search
- metadata
- job submission
- status
- management
- interactive inference

Do not make FastAPI the high-volume GPU scheduler.

A local million-image job should look more like:

```text
FastAPI creates job
worker feeds Triton directly
FastAPI reports status
```

Benchmark lower API process counts rather than assuming 64 workers are useful on a one-GPU deployment.

---

# 19. Directory and Large-Library Ingestion

Current:

```text
list entire directory
 -> read batch
 -> ingest batch
 -> wait
 -> next batch
```

Target:

```text
directory iterator
   |
file-path queue
   |
reader workers
   |
encoded-byte queue
   |
hash/dedupe
   |
inference dispatcher
   |
result queue
   +--> OpenSearch bulk
   +--> OCR
```

Persist progress:

```text
processed
failed
duplicate
indexed
checkpoint
```

Tune readers differently for:

- NVMe
- NAS/NFS/SMB
- S3/MinIO

Queue limits should consider bytes, not just item counts.

---

# 20. TensorRT Optimization

Benchmark the engine first with `trtexec`.

Hierarchy:

```text
trtexec
 -> Triton model
 -> preprocessing
 -> ensemble
 -> FastAPI endpoint
 -> folder ingest
```

## FP16

Reasonable default, but quality-test every model.

## INT8

Potential throughput improvement, but validate:

- YOLO mAP
- ArcFace ROC/TAR/FAR
- embedding Recall@K/MRR/nDCG
- OCR error metrics

## CLIP precision caution

Public issue:

https://github.com/triton-inference-server/server/issues/4105

A user observed a CLIP embedding discrepancy after TensorRT FP16 conversion.

Lesson:

> Embedding models require semantic regression tests, not merely shape checks.

## TensorRT references

https://developer.nvidia.com/tensorrt  
https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html  
https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html  
https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html  
https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html  
https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-nvidia-tensorrt-deployment/

CUDA Graphs and lower-level optimizations belong after higher-level serialization/data-path issues are fixed.

---

# 21. Observability: Metrics, Tracing, and Profiling

Important Triton metrics include:

```text
nv_inference_request_duration_us
nv_inference_queue_duration_us
nv_inference_compute_input_duration_us
nv_inference_compute_infer_duration_us
nv_inference_compute_output_duration_us
```

Docs:

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html

Interpretation:

- **queue high** -> capacity/instances/batching/contention
- **input high** -> copies/tensor size/preprocessing/transport
- **infer high** -> engine/model/GPU
- **output high** -> large outputs/copies/postprocessing

Tracing:

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/trace.html

Use Nsight Systems when aggregate metrics do not explain idle GPU gaps.

---

# 22. Benchmark Methodology

## Layer 1 — TensorRT engine

Use `trtexec`.

## Layer 2 — Triton model

Use Perf Analyzer:

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/README.html

## Layer 3 — preprocessing

Compare:

- PIL
- OpenCV
- C++
- DALI

## Layer 4 — ensemble

Measure compressed JPEG -> preprocess -> model -> output.

## Layer 5 — FastAPI endpoint

Include multipart/middleware/serialization.

## Layer 6 — full ingest

Include storage/hash/dedupe/inference/search/OCR.

Record:

| Metric | Unit |
|---|---|
| request throughput | req/s |
| image throughput | img/s |
| face throughput | faces/s |
| p50/p95/p99 | ms |
| GPU utilization | % |
| GPU memory | MiB |
| GPU power | W |
| CPU utilization | % |
| disk throughput | MB/s |
| network throughput | MB/s |
| Triton queue/input/infer/output | µs |
| batch-size distribution | count |
| OpenSearch throughput | docs/s |

Use varied images, not one repeated JPEG.

Sweep concurrency:

```text
1, 2, 4, 8, 16, 32, 64, 128
```

Sweep batch sizes within engine/profile limits.

NVIDIA optimization guide:

https://github.com/triton-inference-server/server/blob/main/docs/user_guide/optimization.md

---

# 23. Production Company Case Studies

Performance results below are vendor/company-reported and workload-specific. They are evidence, not promises.

## NIO

Closest architecture match to OpenProcessor.

Reported technologies:

- Triton
- DALI
- nvJPEG
- BLS
- Kubernetes
- Argo
- Istio
- Prometheus
- Grafana

Reported results:

- up to 6× latency reduction in some core pipelines
- up to 5× overall throughput improvement

Source:

https://developer.nvidia.com/blog/designing-an-optimal-ai-inference-pipeline-for-autonomous-driving/

## Snap — universal model serving

Publicly describes:

- standardized Triton serving
- ensembles
- ONNX
- TensorRT
- Python/PyTorch preprocessing in some paths
- Model Analyzer
- Kubernetes
- Prometheus
- >1,000 T4/L4 GPUs in the cited deployment

Source:

https://developer.nvidia.com/blog/?p=82250

GTC:

https://www.nvidia.com/en-us/on-demand/session/gtc24-s61915/

## Snap — OCR

Reported ~3× OCR inference throughput in the cited workload.

https://www.nvidia.com/en-us/on-demand/session/gtc24-s62137/

## Oracle OCI Vision

Published FastAPI-like serving vs Triton examples show roughly:

- 30–76% throughput improvement
- 30–51% latency reduction

depending on model/concurrency.

Source:

https://blogs.oracle.com/ai-and-datascience/oci-ai-vision-nvidia-triton-inference-server

Additional Oracle material:

https://blogs.oracle.com/ai-and-datascience/oci-nvidia-triton-inference-server  
https://blogs.oracle.com/ai-and-datascience/ml-models-triton-inference-server-oke

## Volkswagen Computer Vision Workbench

Used Triton as a model-serving layer to abstract multiple frameworks/model types inside an internal CV ecosystem.

NVIDIA summary:

https://developer.nvidia.com/blog/?p=30016

Triton GTC playlist:

https://www.nvidia.com/en-us/on-demand/playlist/playList-fb60ec5d-c184-4416-9d5c-5b5eb236a286/

## USPS

NVIDIA states USPS uses Triton in a microservices architecture for package analytics across 192 distribution centers.

Source:

https://developer.nvidia.com/blog/?p=30016

## Microsoft Bing

NVIDIA reported a 7× throughput-per-GPU result in a migration involving Triton, A100/MIG, and model/system changes.

Do not attribute that multiplier to Triton alone.

Source:

https://blogs.nvidia.com/blog/microsoft-bing-triton/

## Tencent

NVIDIA reports a centralized Triton ML platform handling around 1.5 million queries/day in the cited deployment.

Source:

https://developer.nvidia.com/blog/solving-ai-inference-challenges-with-nvidia-triton/

## Yahoo Japan

NVIDIA reports Triton usage in a centralized ML platform including image-similarity/location search.

Source:

https://developer.nvidia.com/blog/solving-ai-inference-challenges-with-nvidia-triton/

## Airtel

NVIDIA reports approximately 2× GPU throughput for a cited ASR-serving migration to Triton.

Source:

https://developer.nvidia.com/blog/solving-ai-inference-challenges-with-nvidia-triton/

## Wealthsimple

Used Triton on CPUs for standardized fraud/fintech serving.

Source:

https://developer.nvidia.com/blog/solving-ai-inference-challenges-with-nvidia-triton/

## Alibaba Intelligent Connectivity

NVIDIA describes Triton in a streaming TTS architecture.

Source:

https://developer.nvidia.com/blog/solving-ai-inference-challenges-with-nvidia-triton/

## GE Healthcare

NVIDIA describes Triton in GE Healthcare AI deployment environments supporting heterogeneous model frameworks.

Source:

https://developer.nvidia.com/blog/solving-ai-inference-challenges-with-nvidia-triton/

GE platform background:

https://www.gehealthcare.com/en-us/about/newsroom/press-releases/ge-healthcare-digital-health-platform-to-help-providers-accelerate-digital-transformation

## Sansera / Aixia industrial visual inspection

NVIDIA describes:

- Triton
- Jetson edge
- A10 data-center GPUs
- multiple sequential visual models
- consolidated pre/postprocessing

Source:

https://developer.nvidia.com/blog/implementing-industrial-inference-pipelines-for-smart-manufacturing/

## Criteo

GTC 2026 session discusses:

- >500 billion requests/day
- ~20 ms latency budget
- dozens of models
- client-side batching
- tensor fusion
- dynamic-batcher bottlenecks
- serving-layer bottlenecks
- 360K inferences/sec/GPU in the session title

Not a CV workload, but useful for extreme serving lessons.

https://www.nvidia.com/en-us/on-demand/session/gtc26-s82431/

## Other NVIDIA-listed users

NVIDIA has publicly cited users such as:

- Amazon
- Microsoft
- Oracle Cloud
- American Express
- Snap
- DocuSign
- NIO
- GE Healthcare
- Wealthsimple
- USPS
- Yahoo Japan
- Tencent
- Airtel

Customer index:

https://www.nvidia.com/en-us/deep-learning-ai/solutions/inference-platform.md/

---

# 24. Independent GitHub Repositories to Study

## simon-bouchard/cv-inference-triton

https://github.com/simon-bouchard/cv-inference-triton

Strongest independent benchmark-oriented CV reference found.

Useful for:

- raw JPEG pipeline
- Python vs C++ preprocessing
- TensorRT
- Triton
- Perf Analyzer methodology
- proving when batching/protocol changes do or do not matter

## NVIDIA DALI Triton Backend

https://github.com/triton-inference-server/dali_backend

Study:

- `external_source`
- encoded-image input
- decoder
- normalization
- batching
- autoserialization
- model config

## DALI training-to-inference

https://github.com/triton-inference-server/dali_backend/blob/main/docs/tutorials/training_to_inference.md

## Official Triton tutorials

https://github.com/triton-inference-server/tutorials

## Triton client repo

https://github.com/triton-inference-server/client

## Official custom backend repo

https://github.com/triton-inference-server/backend

Recommended backend example:

https://github.com/triton-inference-server/backend/blob/main/examples/backends/recommended/src/recommended.cc

## Python backend

https://github.com/triton-inference-server/python_backend

Preprocessing example:

https://github.com/triton-inference-server/python_backend/blob/main/examples/preprocessing/README.md

## Ultralytics Triton + DALI

https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/nvidia-dali.md

Very close to:

```text
raw JPEG -> DALI -> TensorRT YOLO -> Triton ensemble
```

## levipereira/triton-server-yolo

https://github.com/levipereira/triton-server-yolo

Useful for:

- YOLO TensorRT
- EfficientNMS-style flows
- dynamic shapes
- FP16/INT8
- batch/instance configuration

## hiennguyen9874/triton-face-recognition

https://github.com/hiennguyen9874/triton-face-recognition

Useful for:

- detector + ArcFace
- model-repository layout
- TensorRT export
- dynamic batching

## yiqisoft/Face-Recognition-with-Triton-Inference-Server

https://github.com/yiqisoft/Face-Recognition-with-Triton-Inference-Server

Uses RetinaFace and ArcFace.

## ybai789/yolov8-triton-tensorrt

https://github.com/ybai789/yolov8-triton-tensorrt

Community example pushing YOLO inference/postprocessing closer to Triton.

## joaquincabezas/clip_is_awesome

https://github.com/joaquincabezas/clip_is_awesome

Community CLIP optimization/serving reference.

## Apple MobileCLIP

https://github.com/apple/ml-mobileclip

Model source/reference, useful for future embedding upgrades.

---

# 25. NVIDIA GTC, Training, and Video Resources

## Triton On-Demand playlist

https://www.nvidia.com/en-us/on-demand/playlist/playList-fb60ec5d-c184-4416-9d5c-5b5eb236a286/

Includes material on:

- Triton at scale
- MIG/Kubernetes
- Volkswagen CV
- video analytics
- USPS
- deployment patterns

## High-Performance Inferencing at Scale Using Triton

GTC 2020:

https://developer.nvidia.com/gtc/2020/video/s22418-vid

Topics:

- batching
- scheduling
- model pipelines
- GPU-memory intermediates
- system/CUDA shared memory
- end-to-end performance

## Snap — Universal Model Serving

https://www.nvidia.com/en-us/on-demand/session/gtc24-s61915/

## Snap — OCR

https://www.nvidia.com/en-us/on-demand/session/gtc24-s62137/

## Model Analyzer video

https://www.youtube.com/watch?v=UU9Rh00yZMY

## DeepStream inference options with Triton/TensorRT

https://www.youtube.com/watch?v=eM4nKWy6anA

## Volkswagen Computer Vision Zoo

Available through:

https://www.nvidia.com/en-us/on-demand/playlist/playList-fb60ec5d-c184-4416-9d5c-5b5eb236a286/

## Boost Video Analytics Throughput by 6x Using Triton + DeepStream

Also in NVIDIA's Triton on-demand material.

Use the techniques, not the 6× number, as the transferable lesson.

## Triton at scale with MIG + Kubernetes

https://developer.nvidia.com/blog/deploying-nvidia-triton-at-scale-with-mig-and-kubernetes/

## Criteo — 360K Inferences/s/GPU

https://www.nvidia.com/en-us/on-demand/session/gtc26-s82431/

## Vision AI Demystified

https://www.nvidia.com/en-us/on-demand/session/gtc24-s62607/

Covers Triton, DALI, CV-CUDA, DeepStream, Video Codec SDK, TAO, and Metropolis.

## Medical imaging / MONAI + Triton

https://www.nvidia.com/en-us/on-demand/session/gtc25-s73347/

---

# 26. Community Issues and Failure Modes

## DALI + larger batch can perform worse

https://github.com/triton-inference-server/dali_backend/issues/178

Lesson: Perf Analyzer and real-app results both matter.

## Ensemble branches may not yield expected parallelism

https://github.com/triton-inference-server/server/issues/6982

Lesson: a parallel-looking DAG still needs profiling.

## Triton can be slower than direct execution when configured poorly

https://github.com/triton-inference-server/server/issues/5229

Lesson: serving overhead must be offset by batching/concurrency/operational benefits.

## Ensemble/preprocessing overhead can dominate

https://github.com/triton-inference-server/server/issues/3245

Lesson: “inside Triton” is not automatically free.

## Model instance concurrency discussion

https://github.com/triton-inference-server/server/issues/8671

## CLIP FP16 discrepancy

https://github.com/triton-inference-server/server/issues/4105

---

# 27. DeepStream, CV-CUDA, nvImageCodec, and Video

## DALI

Best for:

- batched image preprocessing
- decode
- resize
- normalization
- server-side image data pipelines

## CV-CUDA

Potential fit for:

- custom GPU CV operations
- resize
- crop
- warp
- color conversion
- normalization
- face alignment

## nvImageCodec

Modern image-codec library included in current Triton stack.

Potential role:

- high-throughput image decode
- custom backend
- cases where DALI abstraction is too restrictive

## DeepStream

Best for:

- live camera streams
- GStreamer
- NVDEC
- cross-stream batching
- tracking
- inference
- message brokers

For still-image folders, DALI/Triton is the more direct fit.

For many live H.264/H.265 streams, DeepStream becomes more compelling.

DeepStream + Triton:

https://developer.nvidia.com/blog/building-iva-apps-using-deepstream-5-0-updated-for-ga/

DeepStream 7:

https://developer.nvidia.com/blog/nvidia-deepstream-7-0-milestone-release-for-next-gen-vision-ai-development/

---

# 28. Deployment Patterns: Single Server Through Kubernetes

## Stage 1 — single GPU server

Best optimization environment:

```text
FastAPI
Triton
OpenSearch
Prometheus
Grafana
one GPU
```

Do not add Kubernetes before understanding per-GPU behavior.

## Stage 2 — multi-GPU server

Compare:

- one Triton using multiple GPUs via `instance_group`
- one Triton process/container per GPU

## Stage 3 — Kubernetes/cloud

Potential stack:

```text
load balancer
  -> Triton pods
  -> GPU nodes
```

Useful components:

- NVIDIA GPU Operator
- DCGM exporter
- Prometheus
- KEDA/HPA
- Karpenter
- queue-based worker scaling

Scale on more than GPU utilization:

- queue depth
- queue duration
- request rate
- job backlog
- p95
- GPU utilization

MIG reference:

https://developer.nvidia.com/blog/deploying-nvidia-triton-at-scale-with-mig-and-kubernetes/

MIG is not an RTX A6000 feature; it is relevant to supported data-center GPU families.

---

# 29. Concrete OpenProcessor Implementation Roadmap

## Phase 0 — freeze a reproducible baseline

Record:

- corpus
- GPU
- driver
- Triton container
- engines
- model versions
- OpenSearch state

Capture:

```text
trtexec
perf_analyzer
FastAPI endpoint
folder ingest
```

## Phase 1 — instrument everything

Track:

- request p50/p95/p99
- queue duration
- input/infer/output duration
- batch distribution
- GPU util/memory
- CPU
- disk
- network
- OpenSearch latency
- queue depth

## Phase 2 — fix existing async path

Replace:

```text
ThreadPoolExecutor(8)
sync Triton calls
```

with bounded async submission.

Benchmark concurrency:

```text
1 2 4 8 16 32 64 128
```

## Phase 3 — parallelize independent models

Change:

```text
YOLO -> wait -> CLIP
```

to:

```text
YOLO --\
        +--> gather
CLIP --/
```

Benchmark GPU traces.

## Phase 4 — pipeline directory ingest

Overlap:

```text
scan
read
hash
infer
format
index
OCR
```

## Phase 5 — DALI YOLO proof of concept

Compare:

```text
JPEG -> current CPU preprocess -> YOLO TRT
```

against:

```text
JPEG -> DALI -> YOLO TRT
```

Measure client CPU, transport bytes, latency, throughput, GPU utilization, and correctness.

## Phase 6 — DALI fan-out

Add branches for:

- YOLO
- MobileCLIP
- PE
- SCRFD

Maintain preprocessing parity.

## Phase 7 — global face batching

Measure ArcFace faces/sec and actual batch-size distribution.

## Phase 8 — GPU face postprocessing

Try:

1. Python backend + DLPack
2. CV-CUDA/DALI where suitable
3. C++/CUDA backend if still needed

## Phase 9 — decouple OCR and persistence

Add bounded queues and bulk writers.

## Phase 10 — Model Analyzer

Tune:

- instance counts
- batch size
- dynamic batching
- queue delay
- multi-model competition

## Phase 11 — transport optimization

Compare:

- aio gRPC
- custom pool
- streaming
- system SHM
- CUDA SHM

## Phase 12 — precision/model optimization

Then investigate:

- INT8
- TensorRT profiles
- CUDA Graphs
- alternate model sizes
- MobileCLIP generation
- PE model selection

Always pair speed tests with quality tests.

## Phase 13 — current Triton upgrade

Test OpenProcessor's 26.06 base against current 26.08 with rebuilt/revalidated TensorRT engines.

---

# 30. Proposed Pull Requests / Work Packages

1. **PR — benchmark baseline**  
   Add reproducible benchmark corpus and machine-readable output.

2. **PR — async batch dispatcher**  
   Feature flag:
   ```text
   (proposed) an ingest-engine switch: sync or async
   ```

3. **PR — concurrent YOLO + CLIP**  
   Pure scheduling change.

4. **PR — streaming directory scanner**  
   Add bounded scan/read/infer/result queues.

5. **PR — OpenSearch bulk writer**  
   Decouple persistence from inference.

6. **PR — OCR enrichment queue**  
   Add sync/async OCR modes.

7. **PR — DALI YOLO model**  
   First server-side image preprocessing proof.

8. **PR — DALI multi-output preprocessing**  
   Add YOLO/MobileCLIP/PE/SCRFD branches.

9. **PR — visual ensemble/BLS**  
   Expose one visual-core Triton API.

10. **PR — face aggregation queue**  
    Batch ArcFace globally.

11. **PR — GPU face postprocess experiment**  
    Python backend/DLPack first.

12. **PR — worker simplification**  
    Remove unproven excessive process/thread/channel concurrency.

13. **PR — Model Analyzer tooling**  
    Repeatable multi-model profiling.

14. **PR — same-host shared-memory experiment**

15. **PR — Triton 26.08 upgrade and engine regression**

---

# 31. Illustrative Code and Configuration

These are simplified architecture examples.

## Async fan-out

```python
async def infer_visual_core(image_bytes):
    yolo_task = asyncio.create_task(infer_yolo(image_bytes))
    clip_task = asyncio.create_task(infer_mobileclip(image_bytes))
    face_task = asyncio.create_task(infer_faces(image_bytes))

    yolo, clip, faces = await asyncio.gather(
        yolo_task,
        clip_task,
        face_task,
    )

    return {
        "detections": yolo,
        "embedding": clip,
        "faces": faces,
    }
```

## Bounded ingestion

```python
async def reader(paths, output_queue):
    for path in paths:
        data = await read_image(path)
        await output_queue.put((path, data))

async def inference_worker(input_queue, result_queue):
    while True:
        item = await input_queue.get()
        if item is STOP:
            break

        path, encoded = item
        result = await triton_infer(encoded)
        await result_queue.put((path, result))
        input_queue.task_done()

async def index_writer(result_queue):
    buffer = []

    while True:
        item = await result_queue.get()

        if item is STOP:
            break

        buffer.append(item)

        if len(buffer) >= BULK_SIZE:
            await opensearch_bulk(buffer)
            buffer.clear()

        result_queue.task_done()
```

Production code should add retries, cancellation, byte-based queue limits, checkpoints, metrics, failure routing, and graceful shutdown.

## Dynamic batching baseline

```protobuf
name: "mobileclip_image"
platform: "tensorrt_plan"
max_batch_size: 64

dynamic_batching {}

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
```

Then benchmark before adding preferred batches or long queue delays.

## DALI conceptual preprocessing

```python
@pipeline_def(
    batch_size=64,
    num_threads=4,
    device_id=0,
)
def visual_preprocess():
    encoded = fn.external_source(
        name="ENCODED_IMAGE",
        device="cpu",
        dtype=types.UINT8,
        ndim=1,
    )

    rgb = fn.decoders.image(
        encoded,
        device="mixed",
        output_type=types.RGB,
    )

    yolo = make_yolo_tensor(rgb)
    clip = make_mobileclip_tensor(rgb)
    pe = make_pe_tensor(rgb)
    face = make_scrfd_tensor(rgb)

    return yolo, clip, pe, face
```

## Ensemble concept

```protobuf
name: "visual_core"
platform: "ensemble"

input [
  {
    name: "ENCODED_IMAGE"
    data_type: TYPE_UINT8
    dims: [ -1 ]
  }
]

output [
  {
    name: "DETECTIONS"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  },
  {
    name: "IMAGE_EMBEDDING"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "dali_visual_preprocess"
      model_version: -1
      input_map {
        key: "ENCODED_IMAGE"
        value: "ENCODED_IMAGE"
      }
      output_map {
        key: "YOLO_INPUT"
        value: "YOLO_INPUT_TENSOR"
      }
      output_map {
        key: "CLIP_INPUT"
        value: "CLIP_INPUT_TENSOR"
      }
    },
    {
      model_name: "yolo_trt"
      model_version: -1
      input_map {
        key: "images"
        value: "YOLO_INPUT_TENSOR"
      }
      output_map {
        key: "detections"
        value: "DETECTIONS"
      }
    },
    {
      model_name: "mobileclip_trt"
      model_version: -1
      input_map {
        key: "images"
        value: "CLIP_INPUT_TENSOR"
      }
      output_map {
        key: "image_embeddings"
        value: "IMAGE_EMBEDDING"
      }
    }
  ]
}
```

## Face work queue

```python
@dataclass
class FaceWork:
    image_id: str
    face_index: int
    aligned_chw: np.ndarray

pending_faces = []

for image in detector_results:
    for i, aligned in enumerate(image.aligned_faces):
        pending_faces.append(
            FaceWork(
                image_id=image.id,
                face_index=i,
                aligned_chw=aligned,
            )
        )

for chunk in chunks(pending_faces, 128):
    batch = np.stack([x.aligned_chw for x in chunk])
    embeddings = await arcface(batch)

    for work, embedding in zip(chunk, embeddings):
        attach_embedding(
            work.image_id,
            work.face_index,
            embedding,
        )
```

## Perf Analyzer

```bash
perf_analyzer \
  -m mobileclip2_s2_image_encoder \
  -u localhost:8001 \
  -i grpc \
  --concurrency-range 1:128:4
```

```bash
perf_analyzer \
  -m yolo11_trt \
  -u localhost:8001 \
  -i grpc \
  --request-rate-range 10:500:10
```

Docs:

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/README.html

## `trtexec`

```bash
trtexec \
  --loadEngine=model.plan \
  --shapes=images:16x3x640x640 \
  --warmUp=3000 \
  --duration=30
```

Run across supported optimization profiles.

---

# 32. Performance Hypotheses to Test

## A — async bulk dispatcher improves throughput

Why plausible: current bulk path is explicitly capped at eight synchronous threads.

Prove by changing only dispatch strategy.

## B — DALI reduces CPU load and increases images/sec

Why plausible: Python preprocessing is substantial, and external code plus NIO's case study show preprocessing can dominate.

Prove with YOLO-only A/B.

## C — decode-once fan-out is more valuable than micro-optimizing one engine

Why plausible: the same image feeds many models.

Measure CPU, bytes moved, and images/sec.

## D — global ArcFace batching improves face throughput

Measure face-batch histogram before and after.

## E — 64 FastAPI workers hurt a single-GPU deployment

Sweep:

```text
1, 2, 4, 8, 16, 32, 64
```

## F — custom four-channel gRPC pool becomes unnecessary

Compare after higher-level pipeline fixes.

## G — OpenSearch or OCR becomes the next bottleneck

Observe queue growth after inference speed improves.

---

# 33. What Not to Optimize First

Do not start with:

- exotic CUDA Graph tuning
- dozens of Triton instances
- custom protocol work
- Kubernetes
- MIG
- rewriting every preprocessing op in CUDA C++
- extreme Uvicorn worker counts
- blindly increasing batches
- blindly converting everything to INT8

before answering:

```text
Where is the current wall time?
```

The fastest engineering sequence is:

```text
measure
remove serialization
pipeline stages
remove repeated preprocessing
batch the real work units
then tune
```

---

# 34. Recommended Reading Order

1. **NIO production architecture**  
   https://developer.nvidia.com/blog/designing-an-optimal-ai-inference-pipeline-for-autonomous-driving/

2. **Independent `cv-inference-triton` repo**  
   https://github.com/simon-bouchard/cv-inference-triton

3. **DALI Triton backend**  
   https://github.com/triton-inference-server/dali_backend

4. **DALI training-to-inference tutorial**  
   https://github.com/triton-inference-server/dali_backend/blob/main/docs/tutorials/training_to_inference.md

5. **Ultralytics DALI + Triton guide**  
   https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/nvidia-dali.md

6. **Snap universal serving GTC**  
   https://www.nvidia.com/en-us/on-demand/session/gtc24-s61915/

7. **Snap OCR GTC**  
   https://www.nvidia.com/en-us/on-demand/session/gtc24-s62137/

8. **Dynamic batching docs**  
   https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html

9. **Model Analyzer**  
   https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_analyzer/README.html

10. **Oracle OCI Vision comparison**  
    https://blogs.oracle.com/ai-and-datascience/oci-ai-vision-nvidia-triton-inference-server

11. **High-performance Triton GTC session**  
    https://developer.nvidia.com/gtc/2020/video/s22418-vid

12. **Triton On-Demand playlist**  
    https://www.nvidia.com/en-us/on-demand/playlist/playList-fb60ec5d-c184-4416-9d5c-5b5eb236a286/

---

# 35. Complete Resource Catalog

## Core Triton documentation

- Main docs: https://docs.nvidia.com/deeplearning/triton-inference-server/
- 26.08 release notes: https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-08.html
- Server GitHub: https://github.com/triton-inference-server/server
- Quickstart: https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md
- Tutorials: https://github.com/triton-inference-server/tutorials
- Dynamic batching: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html
- Model execution: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_execution.html
- Optimization guide: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/optimization.md
- Model Analyzer: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_analyzer/README.html
- Perf Analyzer: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/README.html
- Metrics: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html
- Trace: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/trace.html
- Shared memory: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_shared_memory.html
- Rate limiter: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/rate_limiter.html
- Response cache: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/response_cache.html
- Model repository: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html
- Model management: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html
- Model config: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html
- Ensemble docs: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/ensemble_models.md
- Ensemble tutorial: https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_5-Model_Ensembles
- BLS: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/bls.html
- Python backend docs: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html
- Python backend GitHub: https://github.com/triton-inference-server/python_backend
- C++ backend GitHub: https://github.com/triton-inference-server/backend
- Client GitHub: https://github.com/triton-inference-server/client

## DALI

- DALI Triton backend: https://github.com/triton-inference-server/dali_backend
- Training-to-inference: https://github.com/triton-inference-server/dali_backend/blob/main/docs/tutorials/training_to_inference.md
- DALI article: https://developer.nvidia.com/blog/rapid-data-pre-processing-with-nvidia-dali/
- Triton + DALI article: https://developer.nvidia.com/blog/?p=30560
- Ultralytics DALI guide: https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/nvidia-dali.md

## TensorRT

- TensorRT: https://developer.nvidia.com/tensorrt
- Getting started: https://developer.nvidia.com/tensorrt-getting-started
- Performance optimization: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html
- Best practices: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html
- Accuracy: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html
- Quantization: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html
- Timing cache/deployment: https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-nvidia-tensorrt-deployment/
- Triton + TensorRT optimization: https://developer.nvidia.com/blog/?p=50553

## NVIDIA architecture/company articles

- NIO: https://developer.nvidia.com/blog/designing-an-optimal-ai-inference-pipeline-for-autonomous-driving/
- Production inference overview: https://developer.nvidia.com/blog/solving-ai-inference-challenges-with-nvidia-triton/
- Simplifying inference in production: https://developer.nvidia.com/blog/?p=30016
- Snap: https://developer.nvidia.com/blog/?p=82250
- Smart manufacturing: https://developer.nvidia.com/blog/implementing-industrial-inference-pipelines-for-smart-manufacturing/
- MIG + Kubernetes: https://developer.nvidia.com/blog/deploying-nvidia-triton-at-scale-with-mig-and-kubernetes/
- GKE: https://developer.nvidia.com/blog/one-click-deployment-of-triton-inference-server-to-simplify-ai-inference-on-google-kubernetes-engine-gke/
- Metaflow + Triton: https://developer.nvidia.com/blog/develop-ml-ai-with-metaflow-deploy-with-triton-inference-server/
- GPU preprocessing / ensemble: https://developer.nvidia.com/blog/?p=61372
- Historic overview: https://developer.nvidia.com/blog/nvidia-triton-inference-server-boosts-deep-learning-inference/

## Company material

- Oracle Vision: https://blogs.oracle.com/ai-and-datascience/oci-ai-vision-nvidia-triton-inference-server
- Oracle Triton: https://blogs.oracle.com/ai-and-datascience/oci-nvidia-triton-inference-server
- Oracle OKE: https://blogs.oracle.com/ai-and-datascience/ml-models-triton-inference-server-oke
- Microsoft Bing: https://blogs.nvidia.com/blog/microsoft-bing-triton/
- GE Healthcare Edison: https://www.gehealthcare.com/en-us/about/newsroom/press-releases/ge-healthcare-digital-health-platform-to-help-providers-accelerate-digital-transformation
- NVIDIA inference customers: https://www.nvidia.com/en-us/deep-learning-ai/solutions/inference-platform.md/

## GTC / videos

- Triton On-Demand playlist: https://www.nvidia.com/en-us/on-demand/playlist/playList-fb60ec5d-c184-4416-9d5c-5b5eb236a286/
- GTC 2020 high-performance Triton: https://developer.nvidia.com/gtc/2020/video/s22418-vid
- Snap universal serving: https://www.nvidia.com/en-us/on-demand/session/gtc24-s61915/
- Snap OCR: https://www.nvidia.com/en-us/on-demand/session/gtc24-s62137/
- Model Analyzer video: https://www.youtube.com/watch?v=UU9Rh00yZMY
- DeepStream inference options: https://www.youtube.com/watch?v=eM4nKWy6anA
- Vision AI Demystified: https://www.nvidia.com/en-us/on-demand/session/gtc24-s62607/
- Criteo: https://www.nvidia.com/en-us/on-demand/session/gtc26-s82431/
- MONAI/Triton: https://www.nvidia.com/en-us/on-demand/session/gtc25-s73347/

## Independent/community repos

- cv-inference-triton: https://github.com/simon-bouchard/cv-inference-triton
- triton-server-yolo: https://github.com/levipereira/triton-server-yolo
- triton-face-recognition: https://github.com/hiennguyen9874/triton-face-recognition
- Face Recognition with Triton: https://github.com/yiqisoft/Face-Recognition-with-Triton-Inference-Server
- YOLOv8 Triton TensorRT: https://github.com/ybai789/yolov8-triton-tensorrt
- CLIP community example: https://github.com/joaquincabezas/clip_is_awesome
- MobileCLIP: https://github.com/apple/ml-mobileclip

## Community issues/discussions

- DALI batching: https://github.com/triton-inference-server/dali_backend/issues/178
- Ensemble concurrency: https://github.com/triton-inference-server/server/issues/6982
- Triton/direct performance: https://github.com/triton-inference-server/server/issues/5229
- Ensemble overhead: https://github.com/triton-inference-server/server/issues/3245
- Model instance concurrency: https://github.com/triton-inference-server/server/issues/8671
- CLIP FP16 discrepancy: https://github.com/triton-inference-server/server/issues/4105

---

# Appendix A — OpenProcessor Source Audit Links

Repository:

https://github.com/davidamacey/OpenProcessor

README:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/README.md

Triton Dockerfile:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/Dockerfile.triton

Visual search / ingest service:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/services/visual_search.py

Triton synchronous client:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/clients/triton_client.py

Async Triton pool:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/clients/triton_pool.py

Face client:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/clients/fast_face_client.py

Directory ingest router:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/routers/ingest.py

Main application:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/main.py

PE encoder:

https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/src/clients/pe_encoder.py

---

# Appendix B — Claims, Evidence, and Caveats

## Vendor-reported performance numbers

Numbers from NIO, Snap, Oracle, Microsoft, Criteo, and other company examples are workload-specific.

They demonstrate that the techniques can matter; they do **not** predict OpenProcessor's exact speedup.

## Community GitHub numbers

Numbers from independent repositories are specific to their hardware, models, versions, data, and benchmark methodology.

The main value is the experiment design and implementation pattern.

## Public/unofficial material

This research uses:

- public GitHub code
- public GitHub issues
- public NVIDIA forums/material
- public GTC sessions
- public company engineering material
- public NVIDIA customer case studies

It does not rely on stolen credentials, private source code, or inaccessible proprietary leaks.

## Model quality

Any optimization involving:

- FP16
- INT8
- changed resize
- changed interpolation
- changed normalization
- changed crop
- changed letterbox
- changed NMS

must be tested for acceptable output quality.

This is particularly important for embedding models.

---

# Final Architecture Summary

```text
                         OPENPROCESSOR

                    CONTROL / API PLANE
           +----------------------------------+
           | FastAPI                          |
           | auth / jobs / query / metadata  |
           | health / model controls          |
           +----------------+-----------------+
                            |
                            v

                    HIGH-THROUGHPUT INGEST
           +----------------------------------+
           | async workers                    |
           | bounded queues                   |
           | storage readers                  |
           | hash/dedupe                      |
           +----------------+-----------------+
                            |
                            | compressed media
                            v

                +----------------------------+
                |         TRITON             |
                |                            |
                |  DALI / decode once        |
                |           |                |
                |     +-----+------+         |
                |     |     |      |         |
                |     v     v      v         |
                |   YOLO   CLIP    PE        |
                |    TRT    TRT    TRT        |
                |                            |
                |          SCRFD TRT          |
                |              |             |
                |        GPU postprocess      |
                |              |             |
                |        face work queue      |
                |              |             |
                |         ArcFace TRT         |
                +-------------+--------------+
                              |
                              v

                    RESULT / ENRICHMENT
              +---------------+---------------+
              |                               |
              v                               v
      OpenSearch bulk queue               OCR queue
              |                               |
              v                               v
          indexes                       text enrichment
```

Core principles:

1. Keep compressed media compressed until the high-performance preprocessing layer where practical.
2. Decode once.
3. Reuse decoded data across models.
4. Run independent models concurrently when the GPU benefits.
5. Batch the real inference unit: images, faces, or crops.
6. Let Triton schedule GPU execution instead of manufacturing concurrency with huge Python worker counts.
7. Use bounded queues and backpressure.
8. Overlap storage, CPU, GPU, and database work.
9. Move expensive pre/postprocessing toward GPU execution when profiling proves it matters.
10. Tune dynamic batching from a minimal baseline.
11. Optimize all resident models together.
12. Keep FastAPI as the control plane, not the bulk-media dataplane.
13. Use `trtexec`, Perf Analyzer, Model Analyzer, Triton metrics, and Nsight rather than intuition.
14. Validate accuracy every time precision or preprocessing changes.
15. Treat published speedups as evidence to investigate, not promises.

---

# Short Action List

```text
1. Freeze benchmark corpus and current baseline.
2. Add Triton Perf Analyzer + per-stage metrics.
3. Replace ingest_batch's 8 sync threads with bounded async inference.
4. Run YOLO and MobileCLIP concurrently.
5. Pipeline directory read / infer / OpenSearch instead of batch barriers.
6. Prototype JPEG -> DALI -> YOLO.
7. Expand DALI to decode-once multi-model fan-out.
8. Globally batch ArcFace faces.
9. Move face postprocess/alignment GPU-side if still measurable.
10. Decouple OCR and OpenSearch from the GPU critical path.
11. Run Model Analyzer on the real multi-model workload.
12. Benchmark transport/shared memory.
13. Upgrade 26.06 -> 26.08 with engine rebuild/regression test.
14. Only then pursue lower-level TensorRT/CUDA micro-optimization.
```

The goal is to turn OpenProcessor from:

```text
a FastAPI application that calls fast Triton models
```

into:

```text
a high-throughput visual inference engine built around Triton.
```
