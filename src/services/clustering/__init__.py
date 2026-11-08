"""Clustering on versioned partitions (research clustering plan 5.1 and 7).

``partitions`` writes one immutable partition; ``carry_over`` keeps cluster identities across
runs; ``run`` is the ``cluster_run`` job body; ``activation`` and ``projection`` serve a partition
to search; ``assign`` places a new item by the serving partition. Every function takes a session
the caller scoped (``project_tx``; ``core_tx`` for faces) and never commits.
"""
