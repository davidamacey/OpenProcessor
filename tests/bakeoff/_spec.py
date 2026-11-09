"""A minimal resolved bake-off request: the job spec the API would build and its response."""

from __future__ import annotations

from src.schemas.bakeoff.job_spec import BakeoffJobSpec, JobDataset, JobModel
from src.schemas.bakeoff.run import AcceptedDataset, AcceptedModel, BakeoffRunAccepted


def spec_and_accepted(job_id: str, out_dir: str) -> tuple[BakeoffJobSpec, BakeoffRunAccepted]:
    model = JobModel(
        model='baseline:m1',
        display_name='m1',
        source='baseline',
        run_id=None,
        backend='ultralytics',
        weights='/w/m1.pt',
        imgsz=640,
        mode='full',
        backend_options={},
        triton_model=None,
        training_data=None,
        class_map_by_dataset={'export:e1': None},
        train_test_overlap_by_dataset={'export:e1': None},
    )
    spec = BakeoffJobSpec(
        job_id=job_id,
        profile=None,
        out_dir=out_dir,
        datasets=[
            JobDataset(
                id='export:e1',
                dir_name='export__e1',
                path='/exports/e1',
                test_label_sha='abc',
                frozen_test_sha='def',
                eval_class_ids=[1],
            )
        ],
        models=[model],
    )
    accepted = BakeoffRunAccepted(
        status='enqueued',
        job_id=job_id,
        profile='generic',
        datasets=[
            AcceptedDataset(
                id='export:e1',
                path='/exports/e1',
                frozen_test_sha='def',
                test_label_sha='abc',
                n_eval_classes=1,
            )
        ],
        models=[
            AcceptedModel(
                model='baseline:m1',
                display_name='m1',
                source='baseline',
                class_mapping={},
                train_test_overlap={},
            )
        ],
        warnings=[],
    )
    return spec, accepted
