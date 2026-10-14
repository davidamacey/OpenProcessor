"""Wire models of the bake-off eval datasets."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvalDatasetClass(BaseModel):
    """One class with at least one object in the dataset's test split."""

    eval_class_id: int
    name: str
    registry_class_id: int | None = Field(
        description='Class-registry id (inverse of class_registry.json export_id_map); '
        'null when the export has no registry mapping for it'
    )
    registry_class_name: str | None = Field(
        description='Live registry name for registry_class_id, resolved by id lookup, '
        'never assumed equal to the eval dataset name; null when registry_class_id is '
        'null or no longer present in the registry (e.g. deleted, not merely deprecated)'
    )
    n_objects: int
    n_images: int


class EvalDataset(BaseModel):
    id: str = Field(description='export:<path under export_root> or external:<group>/<name>')
    source: Literal['export', 'external']
    group: str | None = Field(description='curated | public | sample for external; null for export')
    name: str
    path: str
    is_current: bool
    dataset_kind: Literal['multi_class', 'single_class', 'external']
    nc: int = Field(description='Size of the label space (data.yaml names)')
    classes: list[EvalDatasetClass]
    n_images: int
    n_objects: int
    n_background_images: int
    frozen_test_sha: str | None = Field(
        description='Test-split identity: which frames (filenames only), 16 hex'
    )
    test_label_sha: str | None = Field(
        description='Test-split content: which boxes (label file contents), 16 hex; '
        'the hash the evaluator verifies before scoring'
    )
    sha_source: Literal['manifest', 'computed'] = Field(
        description='Where test_label_sha came from: the export manifest / freeze lock, '
        'or computed from the label files'
    )
    dataset_sha: str | None
    exported_at: str | None
    unlabeled_items_on_exported_images: int | None
    frozen_ok: bool | None = Field(
        description='External only: the test split still matches its freeze lock'
    )


class EvalDatasetList(BaseModel):
    datasets: list[EvalDataset]
    count: int
