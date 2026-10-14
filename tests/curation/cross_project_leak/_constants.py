"""Project slugs, the route prefix and per-project names the isolation tests share."""

from __future__ import annotations


API = '/curation'
SCOPED = f'{API}/projects/{{project}}'
SLUGS = ('default', 'alpha', 'beta')
CLASS_NAMES = {'default': 'default_cardinal', 'alpha': 'alpha_zebra', 'beta': 'beta_heron'}
EMBED_DIM = 8
