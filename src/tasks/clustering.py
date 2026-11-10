"""Tasks of the clustering family: the run and the projection sync."""

from __future__ import annotations

from src.services.clustering.job import cluster_projection_sync, cluster_run
from src.services.clustering.region_fp_job import region_fp_centroids
from src.services.curation.job_binding import opensearch_client
from src.services.face_identity.person_partition import persons_cluster
from src.tasks.runtime.register import register_job


register_job('cluster_run', cluster_run, client_factories={'opensearch': opensearch_client})
register_job(
    'cluster_projection_sync',
    cluster_projection_sync,
    client_factories={'opensearch': opensearch_client},
)
register_job(
    'region_fp_centroids',
    region_fp_centroids,
    client_factories={'opensearch': opensearch_client},
)
register_job('persons_cluster', persons_cluster)
