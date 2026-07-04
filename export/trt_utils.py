"""Shared TensorRT builder helpers for the export scripts.

Centralizes the version-sensitive bits of engine building so the per-model
export scripts don't each carry their own compatibility branches.
"""

from __future__ import annotations

import tensorrt as trt


def create_explicit_network(builder: trt.Builder) -> trt.INetworkDefinition:
    """Create an explicit-batch network across TensorRT versions.

    TRT >= 10 networks are always explicit-batch; the
    ``NetworkDefinitionCreationFlag.EXPLICIT_BATCH`` flag was deprecated in
    10.x and removed in newer majors. Pass it only where it still exists.
    """
    flag = getattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH', None)
    if flag is not None:
        return builder.create_network(1 << int(flag))
    return builder.create_network(0)
