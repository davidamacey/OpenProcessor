"""The item space of a 0.6.0-layout index, for the harnesses that replay one.

Parity and write-path harnesses run against indexes copied from a 0.6.0 deployment, whose
vector field is the registry's legacy default (``legacy_names``); they build their kNN
queries against that space instead of resolving a bound project's.
"""

from __future__ import annotations

from src.domain.embedding_spaces import KnnMethodSpec, ResolvedSpace, Target, legacy_names


def legacy_item_space(dim: int = 1024) -> ResolvedSpace:
    names = legacy_names(Target.ITEM)
    return ResolvedSpace(
        space_key='pe_core_l14_336_v1',
        target=Target.ITEM,
        vector_field=names.vector_field,
        state_field=names.state_field,
        nested_path=None,
        dim=dim,
        index_role='items',
        model_key='pe_core_l14_336',
        model_version=1,
        triton_image_model=None,
        triton_text_model=None,
        knn_method=KnnMethodSpec(),
    )
