# OpenProcessor segmenter

A promptable segmentation service (Meta's SAM 3) that serves the
**segmenter leg** of the curation detection cascade. See
[`docs/CURATION.md`](../../docs/CURATION.md) for where that leg sits in
the pipeline.

## Role

The cascade's primary detector is a Triton-hosted box detector. When it
misses a region (unusual viewpoint, occlusion, an object class the
detector was never trained on) or when the VLM rejects its candidate,
the detection worker
([`scripts/curation/worker/client.py`](../../scripts/curation/worker/client.py))
routes that crop here. SAM 3 is a promptable segmentation model: pass
`text_prompt="<the thing you are curating>"` and it returns instances of
that thing directly — no segment-everything-then-filter step.

The worker routes here **selectively** (a minority of crops): each call
is on the order of a second or more on a mid-range GPU, too slow for an
every-crop fanout. The routing policy lives in the worker, not here.

**This service is domain-neutral.** `text_prompt` is a required
per-request field with no server-side default — what you segment for is
a deployment decision, not a property of the server. There are no class
names, dataset paths or thresholds baked in.

## API

### `GET /health`

Liveness + readiness. `loaded` stays `false` while the model pool is
still building (first boot also downloads weights), so the container
healthcheck probes the flag rather than the socket.

```jsonc
{"status": "healthy", "model": "sam3", "device": "cuda:0", "loaded": true, "instances": 2}
```

### `POST /segment`

```jsonc
{
  "crop_jpeg_b64": "<base64 jpeg>",  // data: prefix tolerated
  "text_prompt": "shipping label",    // REQUIRED — no default
  "max_candidates": 4                // optional, top-K by score
}
```

Returns:

```jsonc
{
  "candidates": [
    {
      "bbox_norm": [0.51, 0.62, 0.58, 0.67],   // crop-frame normalized
      "score": 0.88,
      "mask_iou": 0.92                          // rectangularity, null if masks disabled
    }
  ],
  "elapsed_ms": 2937.4,
  "crop_size": [256, 256],
  "prompt": "shipping label"
}
```

### `POST /segment/batch`

`{crops_jpeg_b64: [...], text_prompt, max_candidates}` → `results[]`
aligned 1:1 with the request order. N images run under one processor
lock and one HTTP round trip. The caller chunks; 4–16 per request is the
useful range — bigger batches hold a processor longer and cut overall
concurrency, smaller ones don't amortize the per-call overhead.

**Coordinate frame:** `bbox_norm` is normalized to the **submitted
image**. When that image is a crop of a larger frame, re-projecting to
the source frame is the caller's job — the worker does it via
`crop_norm_to_source_norm()` in
[`src/services/detection/cascade_detect.py`](../../src/services/detection/cascade_detect.py)
before persisting, the same contract the primary-detector path uses.

## Build and run

```bash
# Build (pins upstream SAM 3 to the Dockerfile's SAM3_SHA default).
docker compose --profile segmenter build segmenter

# Re-pin upstream SAM 3 to a specific commit.
docker compose --profile segmenter build --build-arg SAM3_SHA=<sha> segmenter

# Run it alongside the curation workers, and point them at it.
OP_SEGMENTER_URL=http://segmenter:8000 \
  docker compose --profile curation --profile segmenter up -d
```

The service ships under its own compose profile rather than `curation`
because it needs a GPU and a HuggingFace token — enabling it is an
explicit choice, and the cascade runs without it (`OP_SEGMENTER_URL` empty ⇒ the
segmenter leg is a clean no-op; see
`tests/curation/test_segmenter_optional.py`).

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | (required) | HuggingFace auth for the gated SAM 3 weights. `HUGGINGFACE_HUB_TOKEN` and `HUGGING_TOKEN` are also accepted. |
| `SEGMENTER_DEVICE` | `cuda:0` | Torch device to host the model on. |
| `SEGMENTER_LISTEN_PORT` | `8000` | HTTP port inside the container. Deliberately distinct from `SEGMENTER_PORT`, the host-side port compose maps (`env.template`) — sharing a name let `.env`'s host-port value leak into the container's own uvicorn bind and break it (F-75). |
| `SEGMENTER_LOG_LEVEL` | `info` | uvicorn / app log level. |
| `SEGMENTER_INSTANCES` | `2` | Processors in the pool = max concurrent forwards. |
| `SEGMENTER_SHARED_WEIGHTS` | `0` | `1` builds the model once and wraps it in N processors (see below). |
| `SEGMENTER_COMPILE` | `0` | `1` enables `torch.compile`. First call pays a 30–60 s warmup; steady state improves ~25–50%. |
| `SEGMENTER_ENABLE_MASKS` | `1` | `0` skips the mask head — faster, but `mask_iou` comes back `null`. |

Never put the token in the Dockerfile or commit it; compose loads it
from `.env`.

## Sizing and the weight-sharing tradeoff

A SAM 3 image model is roughly 4 GB in bf16, and the pool bounds
concurrency:

* **Independent weights** (`SEGMENTER_SHARED_WEIGHTS=0`, default) — each
  processor gets its own model copy. Simple and obviously safe; VRAM
  scales linearly with `SEGMENTER_INSTANCES`.
* **Shared weights** (`=1`) — one model, N processors over it.
  `Sam3Processor.set_image()` returns an explicit state dict rather than
  mutating the processor, and the model runs in eval mode with no
  parameter writes, so this is safe for inference. CUDA serializes
  kernel launches on a device anyway, so the parallelism you gain is
  overlapped Python pre/post-processing and HTTP overhead — what you
  really buy is the VRAM back. If shared init fails for any reason the
  service logs it and falls back to independent weights rather than
  failing to boot.

Weights stay fp32 and inference runs under bf16 autocast (matching the
upstream inference path); activation checkpointing is disabled at load,
since it is a training-time memory optimization that only costs time
during inference.

## Attribution

See [`NOTICE`](./NOTICE). SAM 3 is from
[facebookresearch/sam3](https://github.com/facebookresearch/sam3)
(Apache 2.0); the service shape draws on
[efwfe/Labely](https://github.com/efwfe/Labely)'s SAM 3 inference
pattern.
