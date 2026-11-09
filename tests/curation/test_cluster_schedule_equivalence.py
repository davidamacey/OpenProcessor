"""The job route and the synchronous route carry the same defaults for every parameter that
reaches the run (the retrain moved into a background job, it computes the same thing)."""

from __future__ import annotations

import inspect


RUN_PARAMS = ('class_id', 'cluster_id', 'run_vlm')


def test_job_route_and_sync_route_share_run_defaults() -> None:
    from src.routers.curation.pipeline_public import pipeline_auto_label
    from src.routers.curation.pipeline_start import pipeline_auto_label_start

    sync = inspect.signature(pipeline_auto_label).parameters
    job = inspect.signature(pipeline_auto_label_start).parameters
    for name in RUN_PARAMS:
        assert job[name].default == sync[name].default, name
