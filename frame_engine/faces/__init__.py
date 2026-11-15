"""The ``faces`` and ``face_embed`` branches: SCRFD on passes cut from the resident frame, five-point
alignment sampled from the full-resolution frame on the GPU, ArcFace in bounded batches.

Reference implementation: InsightFace ``python-package`` at tag ``v0.7`` (commit
``3e6486942a1be2da0e5b475fac375ea73264bd21``), the release the ``buffalo_l`` models ship with:
``model_zoo/scrfd.py`` (decode, NMS), ``utils/face_align.py`` (``estimate_norm``, ``norm_crop``)
and ``model_zoo/arcface_onnx.py`` (the recogniser's input convention). Where this package departs
from it, the module that does says so and why.
"""
