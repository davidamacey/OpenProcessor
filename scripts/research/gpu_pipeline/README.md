# GPU pipeline prototypes (research)

Prototype code for `gpu:PR-0`, `PR-1` and `PR-2` of the research plan
([09g section 6](../../../docs/design/v070_rebuild/09g_research_gpu_pipeline.md),
[08 section 3](../../../docs/design/v070_rebuild/08_research_prototype_track.md)). Registration,
results and the reproduce commands are in
[docs/research/gpu_pipeline_prototypes.md](../../../docs/research/gpu_pipeline_prototypes.md).
Nothing here is imported by `src/` or `models/`. Every dataset path and GPU index is an argument.

## Environment

A Python 3.12 venv outside the repository, called by its binaries (never sourced):

```bash
python3.12 -m venv /path/to/venvs/op-gpu-research
/path/to/venvs/op-gpu-research/bin/pip install -r scripts/research/gpu_pipeline/requirements-gpu-research.txt
```

`requirements-gpu-research.lock.txt` is the resolved `pip freeze` of the run. The scripts import
the shipped `models/gpu_image_prep/1/` modules by path and read the detector and PE plans of a
model repository read-only.

## Scripts

| Script | What it does |
|---|---|
| `pr0_stage_split.py` | PR-0: the shipped algorithm out of Triton, per-stage wall time, GPU kernel ms per stage (`profile`), NVML samples, CPU-reference parity (`cpu-smoke` is tier 0) |
| `pr1_decoders.py` | PR-1: decoder matrix (shipped nvJPEG lanes against nvImageCodec and DALI), throughput, parity, GPU ms (`profile`), coverage on fixtures |
| `pr2_resize_ops.py` | PR-2: resize and crop arms (CuPy kernels, CV-CUDA, DALI), gates, P-box and PE parity, GPU ms (`profile`) |
| `run_tier1.sh` | the tier-1 command sequence of the three prototypes (environment variables documented in its header) |
| `set_p_sample.py` | seeded, hash-named, verified copy of a private photo sample to local NVMe |
| `exif_fixtures.py` | decoder-coverage fixtures from public images |
| `harness_common.py`, `cpu_reference.py`, `shipped_pipeline.py`, `decoder_arms.py`, `resize_arms.py`, `trt_runner.py`, `nsys_stats.py` | shared parts |

## Rules

- GPU commands run under the card lock: `CUDA_VISIBLE_DEVICES=N flock LOCK nice -n 10 ...`.
- Raw outputs go outside the repository; only small result files that back a published number are
  copied to `docs/benchmarks/` with `SHA256SUMS`.
- Private photo results are aggregates only.
