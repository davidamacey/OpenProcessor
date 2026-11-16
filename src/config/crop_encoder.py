"""The crop encoder's supported input sizes and the catalog version each one's vectors belong to.

The same weights built at 224 (the default: about half the GPU time per crop) and at 336 (for
classes that differ in fine detail) make vectors that cannot be compared, so each size is its own
embedding-space version (``examples/embedding/catalog.tsv``). The deployment's ``PE_CROP_SIZE``
picks one; a project keeps the space it was created with.
"""

from __future__ import annotations


DEFAULT_CROP_SIZE = 224

#: Supported ``PE_CROP_SIZE`` -> the catalog ``(model_key, version)`` of its crop space.
CROP_ENCODERS: dict[int, tuple[str, int]] = {
    224: ('pe_core_l14_crop224', 1),
    336: ('pe_core_l14_crop336', 1),
}
