"""Tasks of the clustering family: the run and the projection sync."""

from __future__ import annotations

from src.services.clustering.job import cluster_projection_sync, cluster_run
from src.services.face_identity.person_partition import persons_cluster
from src.services.projects.guard import make_curation_opensearch
from src.tasks.runtime.register import register_job


register_job('cluster_run', cluster_run, client_factories={'opensearch': make_curation_opensearch})
register_job(
    'cluster_projection_sync',
    cluster_projection_sync,
    client_factories={'opensearch': make_curation_opensearch},
)
register_job('persons_cluster', persons_cluster)
