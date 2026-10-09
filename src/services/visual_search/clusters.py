"""FAISS cluster training, assignment, balance and album listing."""

import logging

from src.clients.opensearch.client import OpenSearchClient
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)


class ClusterMixin:
    opensearch: OpenSearchClient
    inference: InferenceService

    # =========================================================================
    # Clustering Operations (FAISS IVF - Industry Standard)
    # =========================================================================

    def _get_cluster_index(self, index_name: str):
        """Convert string index name to ClusterIndex enum."""
        from src.services.clustering import ClusterIndex

        name_map = {
            'global': ClusterIndex.GLOBAL,
            'vehicles': ClusterIndex.VEHICLES,
            'people': ClusterIndex.PEOPLE,
            'faces': ClusterIndex.FACES,
        }
        if index_name.lower() not in name_map:
            raise ValueError(
                f'Invalid index name: {index_name}. Must be one of: {list(name_map.keys())}'
            )
        return name_map[index_name.lower()]

    def _get_opensearch_index(self, index_name: str):
        """Convert string index name to OpenSearch IndexName enum."""
        from src.clients.opensearch.names import IndexName

        name_map = {
            'global': IndexName.GLOBAL,
            'vehicles': IndexName.VEHICLES,
            'people': IndexName.PEOPLE,
            'faces': IndexName.FACES,
        }
        if index_name.lower() not in name_map:
            raise ValueError(
                f'Invalid index name: {index_name}. Must be one of: {list(name_map.keys())}'
            )
        return name_map[index_name.lower()]

    async def train_clusters(
        self,
        index_name: str,
        n_clusters: int | None = None,
        max_samples: int | None = None,
    ) -> dict:
        """
        Train FAISS IVF clustering for an index.

        Extracts embeddings from OpenSearch and trains FAISS IVF index.
        This is typically run once initially, then periodically for rebalancing.

        Args:
            index_name: Which index to train (global, vehicles, people, faces)
            n_clusters: Number of clusters (uses default if None)
            max_samples: Maximum samples for training (None = all)

        Returns:
            Training stats including n_clusters, n_vectors, timing
        """
        import time

        from src.services.clustering import get_clustering_service

        cluster_index = self._get_cluster_index(index_name)
        opensearch_index = self._get_opensearch_index(index_name)

        logger.info(f'Training clusters for {index_name}...')
        start_time = time.time()

        # Extract embeddings from OpenSearch
        embeddings, doc_ids = await self.opensearch.get_all_embeddings(
            index_name=opensearch_index,
            max_docs=max_samples,
        )

        if len(embeddings) == 0:
            return {
                'status': 'error',
                'error': f'No embeddings found in {index_name} index',
            }

        # Train FAISS index
        clustering_service = get_clustering_service()
        stats = await clustering_service.train_index(
            index_name=cluster_index,
            embeddings=embeddings,
            n_clusters=n_clusters,
        )

        # Assign clusters to all documents
        assignments = clustering_service.assign_clusters_batch(cluster_index, embeddings)
        cluster_ids = [a.cluster_id for a in assignments]
        cluster_distances = [a.distance for a in assignments]

        # Update OpenSearch with cluster assignments
        updated = await self.opensearch.update_cluster_assignments(
            index_name=opensearch_index,
            doc_ids=doc_ids,
            cluster_ids=cluster_ids,
            cluster_distances=cluster_distances,
        )

        training_time = time.time() - start_time

        return {
            'status': 'success',
            'index_name': index_name,
            'n_vectors': stats.n_vectors,
            'n_clusters': stats.n_clusters,
            'avg_cluster_size': stats.avg_cluster_size,
            'empty_clusters': stats.empty_clusters,
            'documents_updated': updated,
            'training_time_s': round(training_time, 2),
        }

    async def assign_unclustered(self, index_name: str) -> dict:
        """
        Assign clusters to unclustered documents.

        Finds documents without cluster_id and assigns them to nearest centroid.

        Args:
            index_name: Which index to process

        Returns:
            Assignment stats
        """
        from src.services.clustering import get_clustering_service

        cluster_index = self._get_cluster_index(index_name)
        opensearch_index = self._get_opensearch_index(index_name)

        clustering_service = get_clustering_service()

        if not clustering_service.is_trained(cluster_index):
            return {
                'status': 'error',
                'error': f'Clusters not trained for {index_name}. Run train_clusters first.',
            }

        # Get unclustered embeddings
        embeddings, doc_ids = await self.opensearch.get_unclustered_embeddings(
            index_name=opensearch_index,
        )

        if len(embeddings) == 0:
            return {
                'status': 'success',
                'index_name': index_name,
                'documents_updated': 0,
                'message': 'No unclustered documents found',
            }

        # Assign to clusters
        assignments = clustering_service.assign_clusters_batch(cluster_index, embeddings)
        cluster_ids = [a.cluster_id for a in assignments]
        cluster_distances = [a.distance for a in assignments]

        # Update OpenSearch
        updated = await self.opensearch.update_cluster_assignments(
            index_name=opensearch_index,
            doc_ids=doc_ids,
            cluster_ids=cluster_ids,
            cluster_distances=cluster_distances,
        )

        return {
            'status': 'success',
            'index_name': index_name,
            'documents_found': len(embeddings),
            'documents_updated': updated,
        }

    async def get_cluster_stats(self, index_name: str) -> dict:
        """
        Get detailed cluster statistics.

        Returns FAISS stats and OpenSearch aggregations.

        Args:
            index_name: Which index

        Returns:
            Cluster statistics
        """
        from src.services.clustering import get_clustering_service

        cluster_index = self._get_cluster_index(index_name)
        opensearch_index = self._get_opensearch_index(index_name)

        clustering_service = get_clustering_service()

        # Get FAISS stats
        faiss_stats = clustering_service.get_stats(cluster_index)

        # Get OpenSearch cluster aggregations
        os_clusters = await self.opensearch.get_cluster_stats(opensearch_index)

        return {
            'status': 'success',
            'index_name': index_name,
            'faiss': {
                'is_trained': faiss_stats.is_trained,
                'n_clusters': faiss_stats.n_clusters,
                'n_vectors': faiss_stats.n_vectors,
                'avg_cluster_size': faiss_stats.avg_cluster_size,
                'min_cluster_size': faiss_stats.min_cluster_size,
                'max_cluster_size': faiss_stats.max_cluster_size,
                'empty_clusters': faiss_stats.empty_clusters,
                'trained_at': faiss_stats.trained_at,
            },
            'opensearch_clusters': os_clusters[:20],  # Top 20 clusters
            'total_clusters_in_opensearch': len(os_clusters),
        }

    async def check_cluster_balance(self, index_name: str) -> dict:
        """
        Check if clusters need rebalancing.

        Args:
            index_name: Which index to check

        Returns:
            Balance assessment with recommendation
        """
        from src.services.clustering import get_clustering_service

        cluster_index = self._get_cluster_index(index_name)

        clustering_service = get_clustering_service()

        if not clustering_service.is_trained(cluster_index):
            return {
                'status': 'error',
                'error': f'Clusters not trained for {index_name}',
            }

        balance = await clustering_service.check_balance(cluster_index)

        return {
            'status': 'success',
            'index_name': balance.index_name,
            'is_balanced': balance.is_balanced,
            'imbalance_ratio': round(balance.imbalance_ratio, 2),
            'empty_ratio': round(balance.empty_ratio, 4),
            'vectors_since_training': balance.vectors_since_training,
            'needs_rebalance': balance.needs_rebalance,
            'reason': balance.reason,
        }

    async def rebalance_clusters(self, index_name: str) -> dict:
        """
        Force rebalance clusters by re-training from current data.

        Args:
            index_name: Which index to rebalance

        Returns:
            Rebalancing stats
        """
        # Just call train_clusters which does a full retrain
        return await self.train_clusters(index_name=index_name)

    async def get_cluster_members(
        self,
        index_name: str,
        cluster_id: int,
        page: int = 0,
        size: int = 50,
    ) -> dict:
        """
        Get members of a specific cluster (album view).

        Args:
            index_name: Which index
            cluster_id: Cluster ID to retrieve
            page: Page number
            size: Page size

        Returns:
            Cluster members sorted by distance to centroid
        """
        opensearch_index = self._get_opensearch_index(index_name)

        members = await self.opensearch.get_cluster_members(
            index_name=opensearch_index,
            cluster_id=cluster_id,
            page=page,
            size=size,
        )

        return {
            'status': 'success',
            'index_name': index_name,
            'cluster_id': cluster_id,
            'page': page,
            'size': size,
            'count': len(members),
            'members': members,
        }

    async def list_albums(self, min_size: int = 5) -> dict:
        """
        List auto-generated albums (clusters) from global index.

        Args:
            min_size: Minimum cluster size to include

        Returns:
            List of albums with metadata
        """
        from src.clients.opensearch.names import IndexName

        # Get cluster stats from global index
        clusters = await self.opensearch.get_cluster_stats(IndexName.GLOBAL)

        # Filter by minimum size
        albums = [c for c in clusters if c['count'] >= min_size]

        # Sort by size descending
        albums.sort(key=lambda x: x['count'], reverse=True)

        return {
            'status': 'success',
            'total_albums': len(albums),
            'albums': albums[:100],  # Top 100 albums
        }
