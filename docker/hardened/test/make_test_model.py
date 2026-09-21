#!/usr/bin/env python3
"""Generate a tiny CNN as ONNX + a Triton model repository for the hardened-image
functional test. Mirrors the openprocessor flow (ONNX -> TensorRT .plan -> serve).

Run with the repo venv:  .venv/bin/python docker/hardened/test/make_test_model.py
Writes: docker/hardened/test/model_repository/{simple_onnx,simple_trt}/...
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from torch import nn


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE / 'model_repository'
IN_C, IN_H, IN_W, N_CLS = 3, 32, 32, 10


class TinyNet(nn.Module):
    """conv -> relu -> global avg pool -> fc. Small enough for a fast TensorRT build."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(IN_C, 16, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, N_CLS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


CONFIG_ONNX = f"""\
name: "simple_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [{{ name: "input"  data_type: TYPE_FP32  dims: [ {IN_C}, {IN_H}, {IN_W} ] }}]
output [{{ name: "output" data_type: TYPE_FP32  dims: [ {N_CLS} ] }}]
instance_group [{{ kind: KIND_GPU count: 1 }}]
"""

CONFIG_TRT = f"""\
name: "simple_trt"
platform: "tensorrt_plan"
max_batch_size: 8
input [{{ name: "input"  data_type: TYPE_FP32  dims: [ {IN_C}, {IN_H}, {IN_W} ] }}]
output [{{ name: "output" data_type: TYPE_FP32  dims: [ {N_CLS} ] }}]
instance_group [{{ kind: KIND_GPU count: 1 }}]
"""


def main() -> None:
    torch.manual_seed(0)
    model = TinyNet().train(False)  # inference mode (no BN/dropout here, so effectively a no-op)

    onnx_dir = REPO / 'simple_onnx' / '1'
    trt_dir = REPO / 'simple_trt' / '1'
    onnx_dir.mkdir(parents=True, exist_ok=True)
    trt_dir.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, IN_C, IN_H, IN_W)
    onnx_path = onnx_dir / 'model.onnx'
    torch.onnx.export(
        model,
        dummy,
        onnx_path.as_posix(),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=18,
    )
    (REPO / 'simple_onnx' / 'config.pbtxt').write_text(CONFIG_ONNX)
    (REPO / 'simple_trt' / 'config.pbtxt').write_text(CONFIG_TRT)

    # fixed sample input + reference output so the served result can be checked
    sample = torch.randn(1, IN_C, IN_H, IN_W)
    with torch.no_grad():
        ref = model(sample).numpy()
    np.save(HERE / 'sample_input.npy', sample.numpy())
    np.save(HERE / 'reference_output.npy', ref)

    print(f'ONNX written: {onnx_path}  ({onnx_path.stat().st_size} bytes)')
    print(f'TRT config ready (engine built by trtexec at test time): {trt_dir}')
    print(f'sample_input {sample.shape} / reference_output {ref.shape} saved')


if __name__ == '__main__':
    main()
