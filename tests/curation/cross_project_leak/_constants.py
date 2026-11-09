"""Project slugs, the route prefix and per-project names the isolation tests share."""

from __future__ import annotations

import uuid


API = '/curation'
SCOPED = f'{API}/projects/{{project}}'
SLUGS = ('default', 'alpha', 'beta')
CLASS_NAMES = {'default': 'default_cardinal', 'alpha': 'alpha_zebra', 'beta': 'beta_heron'}
EMBED_DIM = 8


def cluster_uid(slug: str, n: int = 1) -> str:
    """The identity of seeded cluster ``n`` of project ``slug``."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f'{slug}-cluster-{n}'))
