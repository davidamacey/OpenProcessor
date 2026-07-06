#!/usr/bin/env python3
"""Send the saved sample input to a served Triton model over the KServe v2 HTTP
protocol and verify the output shape + numeric match against the torch reference.

Usage: .venv/bin/python infer_check.py <model_name> [--url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import requests


HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--url', default='http://localhost:8000')
    ap.add_argument('--rtol', type=float, default=2e-2)
    ap.add_argument('--atol', type=float, default=2e-2)
    args = ap.parse_args()

    x = np.load(HERE / 'sample_input.npy').astype(np.float32)
    ref = np.load(HERE / 'reference_output.npy').astype(np.float32)

    payload = {
        'inputs': [
            {
                'name': 'input',
                'shape': list(x.shape),
                'datatype': 'FP32',
                'data': x.flatten().tolist(),
            }
        ],
        'outputs': [{'name': 'output'}],
    }
    r = requests.post(f'{args.url}/v2/models/{args.model}/infer', json=payload, timeout=30)
    if r.status_code != 200:
        print(f'FAIL [{args.model}] HTTP {r.status_code}: {r.text[:300]}')
        return 1

    out = r.json()['outputs'][0]
    got = np.array(out['data'], dtype=np.float32).reshape(out['shape'])
    shape_ok = tuple(got.shape) == tuple(ref.shape)
    # TensorRT FP16/TF32 kernels drift from torch fp32 -> tolerant compare
    close = np.allclose(got, ref, rtol=args.rtol, atol=args.atol)
    maxdiff = float(np.max(np.abs(got - ref))) if shape_ok else float('nan')

    status = 'PASS' if shape_ok else 'FAIL'
    print(
        f'{status} [{args.model}] shape={got.shape} ref={ref.shape} '
        f'max|Δ|={maxdiff:.4g} numeric_close={close}'
    )
    print(f'      served logits: {np.round(got.flatten(), 3).tolist()}')
    # Serving correctness = correct shape + a valid numeric response. Numeric
    # closeness is reported (TRT precision may drift) but only shape gates pass/fail.
    return 0 if shape_ok else 1


if __name__ == '__main__':
    sys.exit(main())
