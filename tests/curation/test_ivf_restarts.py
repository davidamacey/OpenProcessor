"""The IVF fit runs one k-means restart unless told otherwise.

Restarts keep the best of ``nredo`` fits and cost ``nredo`` times one; on real vectors one
restart matches five in purity, false-merge rate and compactness
(scripts/bench/ivf_nredo_agreement.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.services.curation.clustering.methods import ivf as ivf_mod
from src.services.curation.clustering.methods.ivf import IVFMethod
from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore


if TYPE_CHECKING:
    import pytest


def test_default_is_one_restart() -> None:
    assert ivf_mod.IVF_DEFAULT_NREDO == 1
    assert IVFMethod().nredo == 1


def test_the_fit_hands_the_restart_count_to_kmeans(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def fake_train(self: IVFCentroidStore, sample: np.ndarray, **kw: object):
        seen.append(int(kw['nredo']))  # type: ignore[call-overload]
        return np.eye(4, sample.shape[1], dtype=np.float32), 0.0

    monkeypatch.setattr(IVFCentroidStore, 'train', fake_train)
    monkeypatch.setattr(
        IVFCentroidStore,
        'assign_batch_with_distances',
        lambda _self, x: (np.zeros(len(x), dtype=np.int64), np.zeros(len(x), dtype=np.float32)),
    )
    x = np.random.default_rng(0).standard_normal((300, 8)).astype(np.float32)
    IVFMethod(n_clusters=4, persist=False)._fit(x, 4, False)
    assert seen == [1]
