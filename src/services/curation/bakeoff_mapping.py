"""Bake-off class mapping: which model class scores against which dataset class.

A model x dataset pair gets a :class:`~scripts.curation.bakeoff.class_map.ClassMapping`
from a five-rule order (algorithm in ``scripts/curation/bakeoff/class_map.py``);
this module applies rules 1 and 5 (an explicit name map, else a placeholder the
evaluator resolves by name) and rules 2-4 (a training run's own class remap).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.curation.bakeoff import class_map as cmap
from src.schemas.bakeoff.models import TrainTestOverlap
from src.services.curation import eval_datasets as evd
from src.services.curation.bakeoff_resolve import BakeoffRequestError, RunModel


if TYPE_CHECKING:
    from scripts.curation.bakeoff.profile import BakeoffProfile
    from src.services.curation.eval_datasets import EvalDatasetRecord


NAME_MATCH_WARNING = 'resolved by name inside the evaluator'


# =============================================================================
# Class mapping
# =============================================================================


def scored_class_ids(dataset: EvalDatasetRecord, profile: BakeoffProfile) -> list[int]:
    """Classes present in the test split, narrowed by the profile's ``class_filter``."""
    present = dataset.present_class_ids
    if not profile.class_filter:
        return present
    wanted = {cmap.normalize_name(n) for n in profile.class_filter}
    return [c for c in present if cmap.normalize_name(dataset.class_names.get(c, '')) in wanted]


def _single_cls_mapping(
    run: RunModel, dataset: EvalDatasetRecord, scored: list[int]
) -> cmap.ClassMapping | None:
    """``None`` when the run covers exactly one class (then rules 2/3 apply)."""
    n = len(run.include_classes) if run.include_classes else len(run.class_names)
    if n == 1:
        return None
    if len(scored) == 1:
        return cmap.finalize(
            'single_class_fallback',
            {0: scored[0]},
            model_names=None,
            eval_names=dataset.class_names,
            scored_class_ids=scored,
        )
    raise BakeoffRequestError(
        422,
        f'single_cls run over {n} classes cannot be scored per class; '
        'compare it on a single-class export',
    )


def run_class_mapping(
    run: RunModel, dataset: EvalDatasetRecord, scored: list[int]
) -> cmap.ClassMapping:
    """Rules 2-4 for a training run."""
    if run.single_cls:
        fallback = _single_cls_mapping(run, dataset, scored)
        if fallback is not None:
            return fallback
    names = run.class_names
    # A single_cls model is named 'object'; do not cross-check that name.
    check_names = None if run.single_cls else names
    eval_map, eval_names = dataset.export_id_map, dataset.class_names
    warnings: list[str] = []
    if run.remap.mapping and eval_map:
        new_to_orig = {new: orig for orig, new in run.remap.mapping.items()}
        m2e, warnings = cmap.resolve_by_registry(
            new_to_orig, eval_map, model_names=check_names, eval_names=eval_names
        )
        method = 'run_class_remap'
    elif not run.remap.mapping and run.train_export_id_map and eval_map:
        m2e, warnings = cmap.resolve_by_registry(
            cmap.invert_export_id_map(run.train_export_id_map),
            eval_map,
            model_names=check_names,
            eval_names=eval_names,
        )
        method = 'registry_ids'
    else:
        m2e, method = cmap.resolve_by_names(names, eval_names), 'names'
    return cmap.finalize(
        method,
        m2e,
        model_names=names,
        eval_names=eval_names,
        scored_class_ids=scored,
        warnings=warnings,
    )


def explicit_class_mapping(
    class_map: dict[str, str] | None, dataset: EvalDatasetRecord, scored: list[int]
) -> cmap.ClassMapping:
    """Rules 1 and 5: an explicit name map, else a name-match placeholder for the evaluator."""
    if class_map is None:
        return cmap.ClassMapping('names', None, warnings=[NAME_MATCH_WARNING])
    m2e, warnings = cmap.resolve_explicit(class_map, dataset.class_names)
    return cmap.finalize(
        'explicit',
        m2e,
        model_names={int(k): v for k, v in class_map.items()},
        eval_names=dataset.class_names,
        scored_class_ids=scored,
        warnings=warnings,
    )


def _overlap(run: RunModel, dataset: EvalDatasetRecord) -> TrainTestOverlap | None:
    if run.train_export_dir is None:
        return None
    return TrainTestOverlap(**evd.train_test_overlap(run.train_export_dir, dataset))
