"""Tests for :mod:`src.services.detection.fp_store`.

Proves the state-dir/prefix derivation
(``CurationConfig.state_dir`` + ``RegionFields.prefix``) actually drives
the on-disk location, rather than being hardcoded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.config import CurationConfig, RegionFields
from src.services.detection.fp_store import FalsePositiveCentroidStore, fp_store_dir


if TYPE_CHECKING:
    from pathlib import Path


class TestFpStoreDir:
    def test_default_uses_default_config_and_fields(self) -> None:
        expected = CurationConfig().state_dir / f'{RegionFields().prefix}_fp'
        assert fp_store_dir() == expected

    def test_derives_from_injected_state_dir(self, tmp_path: Path) -> None:
        cfg = CurationConfig(state_dir=tmp_path)
        result = fp_store_dir(cfg)
        assert result == tmp_path / 'region_fp'

    def test_derives_from_injected_prefix(self, tmp_path: Path) -> None:
        cfg = CurationConfig(state_dir=tmp_path)
        fields = RegionFields(prefix='legacy')
        result = fp_store_dir(cfg, fields)
        assert result == tmp_path / 'legacy_fp'

    def test_state_dir_and_prefix_together_are_not_hardcoded(self, tmp_path: Path) -> None:
        # Two distinct (state_dir, prefix) pairs must resolve to two
        # distinct, correctly-derived directories -- proof this isn't a
        # module-level constant computed once from the default singletons.
        cfg_a = CurationConfig(state_dir=tmp_path / 'deployment_a')
        cfg_b = CurationConfig(state_dir=tmp_path / 'deployment_b')
        fields_a = RegionFields(prefix='region')
        fields_b = RegionFields(prefix='widget')

        dir_a = fp_store_dir(cfg_a, fields_a)
        dir_b = fp_store_dir(cfg_b, fields_b)

        assert dir_a == tmp_path / 'deployment_a' / 'region_fp'
        assert dir_b == tmp_path / 'deployment_b' / 'widget_fp'
        assert dir_a != dir_b


class TestFalsePositiveCentroidStore:
    def test_directory_defaults_via_fp_store_dir(self, tmp_path: Path) -> None:
        cfg = CurationConfig(state_dir=tmp_path)
        fields = RegionFields(prefix='custom')
        store = FalsePositiveCentroidStore(config=cfg, fields=fields)
        assert store.directory == tmp_path / 'custom_fp'

    def test_explicit_directory_overrides_config_derivation(self, tmp_path: Path) -> None:
        explicit = tmp_path / 'explicit_dir'
        cfg = CurationConfig(state_dir=tmp_path / 'ignored')
        store = FalsePositiveCentroidStore(explicit, config=cfg)
        assert store.directory == explicit

    def test_exists_false_before_save(self, tmp_path: Path) -> None:
        cfg = CurationConfig(state_dir=tmp_path)
        store = FalsePositiveCentroidStore(config=cfg)
        assert store.exists() is False

    def test_load_returns_false_when_nothing_saved(self, tmp_path: Path) -> None:
        cfg = CurationConfig(state_dir=tmp_path)
        store = FalsePositiveCentroidStore(config=cfg)
        assert store.load() is False

    def test_save_then_load_round_trips_centroids_and_metadata(self, tmp_path: Path) -> None:
        cfg = CurationConfig(state_dir=tmp_path)
        fields = RegionFields(prefix='region')
        store = FalsePositiveCentroidStore(config=cfg, fields=fields)

        centroids = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        metadata = {'subtypes': ['glare', 'occluded']}
        store.save(centroids, metadata)

        assert store.exists() is True
        assert (tmp_path / 'region_fp' / 'centroids.faiss').is_file()
        assert (tmp_path / 'region_fp' / 'metadata.json').is_file()

        reloaded = FalsePositiveCentroidStore(config=cfg, fields=fields)
        assert reloaded.load() is True
        assert reloaded.metadata == metadata

    def test_search_returns_nearest_centroid_distance_and_index(self, tmp_path: Path) -> None:
        cfg = CurationConfig(state_dir=tmp_path)
        store = FalsePositiveCentroidStore(config=cfg)
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        store.save(centroids, {'subtypes': ['a', 'b']})

        dist, idx = store.search(np.array([[0.9, 0.1]], dtype=np.float32))
        assert idx[0] == 0
        # Plain L2, not faiss.IndexFlatL2's raw squared L2.
        # Squared distance is (0.1)^2 + (0.1)^2 = 0.02; plain L2 is its
        # square root.
        assert dist[0] == pytest.approx(np.sqrt(0.02), abs=1e-4)

    def test_search_returns_plain_l2_not_squared_l2(self, tmp_path: Path) -> None:
        """Orthogonal unit vectors -> sqrt(2); identical -> 0.

        Every caller (auto-assign FP threshold in orchestrator.py,
        suspected-FP threshold in regions_fp.py, both documented as "L2
        on unit-norm") assumes plain L2. faiss.IndexFlatL2 returns
        *squared* L2 by design -- orthogonal unit vectors give 2.0, not
        sqrt(2) -- so the store must take the square root before
        returning.
        """
        cfg = CurationConfig(state_dir=tmp_path)
        store = FalsePositiveCentroidStore(config=cfg)
        orthogonal = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        store.save(orthogonal, {'subtypes': ['a', 'b']})

        # Query identical to centroid 0 -> distance 0.
        dist_identical, idx_identical = store.search(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        assert idx_identical[0] == 0
        assert dist_identical[0] == pytest.approx(0.0, abs=1e-6)

        # Query orthogonal to both centroids (nearest is a tie broken by
        # index order) -> plain L2 between orthogonal unit vectors is
        # sqrt(2), not faiss's raw squared-L2 value of 2.0.
        dist_orthogonal, _idx = store.search(np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
        assert dist_orthogonal[0] == pytest.approx(np.sqrt(2.0), abs=1e-4)

    def test_two_deployments_with_different_prefixes_do_not_collide(self, tmp_path: Path) -> None:
        cfg = CurationConfig(state_dir=tmp_path)
        store_region = FalsePositiveCentroidStore(config=cfg, fields=RegionFields(prefix='region'))
        store_legacy = FalsePositiveCentroidStore(config=cfg, fields=RegionFields(prefix='legacy'))

        store_region.save(np.array([[1.0, 0.0]], dtype=np.float32), {'subtypes': ['region_a']})
        store_legacy.save(np.array([[0.0, 1.0]], dtype=np.float32), {'subtypes': ['legacy_a']})

        assert store_region.directory != store_legacy.directory
        reloaded_region = FalsePositiveCentroidStore(
            config=cfg, fields=RegionFields(prefix='region')
        )
        reloaded_legacy = FalsePositiveCentroidStore(
            config=cfg, fields=RegionFields(prefix='legacy')
        )
        assert reloaded_region.load() is True
        assert reloaded_legacy.load() is True
        assert reloaded_region.metadata == {'subtypes': ['region_a']}
        assert reloaded_legacy.metadata == {'subtypes': ['legacy_a']}
