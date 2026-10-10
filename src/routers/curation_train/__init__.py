"""Training-pipeline router.

Mounts at ``{api_prefix}/train``. Each module owns one concern and exposes a
sub-router; this package assembles them in the published route order.
The routers stay thin: file-system access goes through
:mod:`src.services.training.jobs`, hyperparameter tables live in
:mod:`src.services.training.profiles`, preflight checks in
:mod:`src.services.training.preflight_checks`.

Endpoints:

    POST   {api_prefix}/train/preflight          -> PreflightReport
    POST   {api_prefix}/train/start              -> {job_id}        (writes job.json)
    POST   {api_prefix}/train/start_campaign     -> {campaign_id, job_ids}
    GET    {api_prefix}/train/status             -> most recent TrainJobStatus
    GET    {api_prefix}/train/status/{job_id}    -> TrainJobStatus
    GET    {api_prefix}/train/runs               -> paginated list
    GET    {api_prefix}/train/log/tail/{job_id}  -> tail N lines
    POST   {api_prefix}/train/cancel/{job_id}    -> drop cancel sentinel
    POST   {api_prefix}/train/cancel_campaign/{campaign_id}
    GET    {api_prefix}/train/profiles           -> profile table
    GET    {api_prefix}/train/presets            -> class-subset presets
    GET    {api_prefix}/train/augmentation_presets -> augmentation preset catalog
    GET    {api_prefix}/train/gpus               -> TrainGpuOptionsResponse
    GET    {api_prefix}/train/manifest/{job_id}   -> run lineage manifest
    GET    {api_prefix}/train/artifacts/{job_id}/{name}
        -> whitelisted run artifact (confusion_matrix.png, results.csv, ...)

Pre-flight contract: ``/start`` calls ``/preflight``
internally and refuses to write ``job.json`` if any check has severity
``block``. Pass ``?force=true`` to bypass; the report is still returned in
the 422 body so the UI can render it inline.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.config import get_curation_config
from src.routers.curation_train import artifacts, catalog, preflight, promote, runs, start


config = get_curation_config()

router = APIRouter(
    prefix='/train',
    tags=[f'{config.api_tag} - Train'],
)
for _sub in (
    preflight.router,
    start.router,
    runs.router,
    catalog.router,
    promote.router,
    artifacts.router,
):
    router.include_router(_sub)
