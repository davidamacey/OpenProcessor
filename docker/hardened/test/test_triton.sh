#!/usr/bin/env bash
# End-to-end functional test for the hardened Triton image:
#   1. generate a tiny ONNX model (repo venv)
#   2. build a TensorRT .plan from it with trtexec INSIDE the hardened image
#   3. serve both models (onnxruntime_onnx + tensorrt_plan) on the chosen GPU
#   4. run an HTTP inference against each and verify the output
#
# usage: test_triton.sh [image_tag] [gpu_index]
set -euo pipefail

TAG="${1:-triton-hardened:26.06}"
GPU="${2:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$HERE/model_repository"
VENV="$(cd "$HERE/../../.." && pwd)/.venv/bin/python"
NAME="triton-hardened-test"
PORT=8000

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> [1/4] generate ONNX test model"
"$VENV" "$HERE/make_test_model.py"

echo "==> [2/4] build TensorRT engine with trtexec inside $TAG (GPU $GPU)"
docker run --rm --gpus "\"device=$GPU\"" --user root \
    -v "$REPO:/models" --entrypoint trtexec "$TAG" \
    --onnx=/models/simple_onnx/1/model.onnx \
    --saveEngine=/models/simple_trt/1/model.plan \
    --minShapes=input:1x3x32x32 --optShapes=input:4x3x32x32 --maxShapes=input:8x3x32x32 \
    2>&1 | tail -6
test -f "$REPO/simple_trt/1/model.plan" && echo "    engine built: $(du -h "$REPO/simple_trt/1/model.plan" | cut -f1)"

echo "==> [3/4] start Triton serving both models (GPU $GPU)"
cleanup
docker run -d --name "$NAME" --gpus "\"device=$GPU\"" \
    -p "$PORT:8000" -v "$REPO:/models:ro" "$TAG" \
    --model-repository=/models --model-control-mode=none --strict-readiness=true >/dev/null

echo -n "    waiting for readiness"
for i in $(seq 1 60); do
    if curl -fsS "http://localhost:$PORT/v2/health/ready" >/dev/null 2>&1; then echo " ready"; break; fi
    echo -n "."; sleep 2
    if [ "$i" = 60 ]; then echo " TIMEOUT"; docker logs --tail 40 "$NAME"; exit 1; fi
done

echo "    loaded models:"; curl -fsS "http://localhost:$PORT/v2/models/simple_onnx/config" >/dev/null && echo "      simple_onnx OK"
curl -fsS "http://localhost:$PORT/v2/models/simple_trt/config" >/dev/null && echo "      simple_trt OK"

echo "==> [4/4] inference checks"
rc=0
"$VENV" "$HERE/infer_check.py" simple_onnx --url "http://localhost:$PORT" || rc=1
"$VENV" "$HERE/infer_check.py" simple_trt  --url "http://localhost:$PORT" || rc=1

echo "==> triton server version:"; docker exec "$NAME" tritonserver --version 2>/dev/null | head -1 || true
[ "$rc" = 0 ] && echo "==> TRITON FUNCTIONAL TEST: PASS" || echo "==> TRITON FUNCTIONAL TEST: FAIL"
exit $rc
