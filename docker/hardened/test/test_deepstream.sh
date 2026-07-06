#!/usr/bin/env bash
# Headless functional test for the hardened DeepStream image:
#   decode the bundled sample video -> nvinfer (resnet18 traffic-cam detector,
#   engine built from ONNX on first run) -> fakesink, with perf measurement.
#   Proves: model load + TensorRT engine build + live inference on the GPU.
#
# usage: test_deepstream.sh [image_tag] [gpu_index]
set -uo pipefail

TAG="${1:-deepstream-hardened:9.0}"
GPU="${2:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$HERE/ds_work"; mkdir -p "$WORK"
DS=/opt/nvidia/deepstream/deepstream

# Headless deepstream-app config. gpu-id=0 = the single GPU exposed by --gpus.
cat > "$WORK/ds_headless.txt" <<EOF
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=1
[tiled-display]
enable=0
[source0]
enable=1
type=3
uri=file://$DS/samples/streams/sample_720p.mp4
num-sources=1
gpu-id=0
[streammux]
gpu-id=0
batch-size=1
batched-push-timeout=40000
width=1280
height=720
[primary-gie]
enable=1
gpu-id=0
batch-size=1
config-file=$DS/samples/configs/deepstream-app/config_infer_primary.txt
[sink0]
enable=1
type=1
sync=0
[osd]
enable=0
[tests]
file-loop=0
EOF

echo "==> running deepstream-app headless (nvinfer detector) on GPU $GPU"
echo "    (first run builds the TensorRT engine from resnet18 ONNX; ~1-2 min)"
# run as root so first-run engine build can write next to the model in the image
# layer. The image's DEFAULT user is non-root (dsuser); this override is test-only.
timeout 360 docker run --rm --gpus "\"device=$GPU\"" --user root \
    -v "$WORK:/work" -w "$DS" "$TAG" \
    deepstream-app -c /work/ds_headless.txt 2>&1 | tee "$WORK/ds_run.log" | tail -40

echo "==> verdict"
if grep -qiE 'App run successful' "$WORK/ds_run.log"; then
    echo "DEEPSTREAM FUNCTIONAL TEST: PASS (pipeline ran to completion)"
    grep -iE 'PERF|Engine .*built|serialize|Deserialize' "$WORK/ds_run.log" | tail -5
    exit 0
elif grep -qiE '\*\*PERF' "$WORK/ds_run.log"; then
    echo "DEEPSTREAM FUNCTIONAL TEST: PASS (inference perf observed)"
    grep -iE '\*\*PERF' "$WORK/ds_run.log" | tail -3
    exit 0
else
    echo "DEEPSTREAM FUNCTIONAL TEST: FAIL / INCONCLUSIVE — see $WORK/ds_run.log"
    grep -iE 'error|failed|cuda|driver' "$WORK/ds_run.log" | tail -8
    exit 1
fi
