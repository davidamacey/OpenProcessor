"""The frame engine: decode each photo once on the GPU and cut every model input from that frame.

A standalone process (its own image, ``docker/frame-engine/Dockerfile``) that calls the TensorRT
models in Triton as a client and answers the API with KServe v2 ``ModelInfer``, protocol version 1
(``src/domain/frame_engine_wire.py``). Design: docs/design/v070_rebuild/13_backend_speed_plan.md, D1.
"""
