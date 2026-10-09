"""The per-project VLM scope policy: which crops the VLM class writers may label.

Distinct from the ingest policy (which governs what ingest stores and embeds):
this one limits the two automated VLM class writers, the continuous
``curation-vlm-worker`` and the ``auto_label`` VLM stage. An explicit request
(``POST /vlm/label_cluster/{id}``, a ``cluster_id``-scoped auto-label run, a
single-crop label) is never limited by it.

Scopes:

* ``all`` (default): every eligible crop, the behaviour before the policy existed.
* ``uncertain``: only crops whose detector confidence is below ``conf_max`` or
  was not recorded.
* ``representatives``: the ``per_cluster`` crops nearest each cluster centre, plus
  every unassigned crop.
* ``off``: no automated VLM class writes.

``max_crops_per_day`` (0 = unlimited) caps VLM class attempts per UTC day and
``sample_frac`` (0..1] keeps a stable hash-selected fraction of the crops.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


VlmScope = Literal['all', 'uncertain', 'representatives', 'off']


class VlmPolicyBody(BaseModel):
    model_config = ConfigDict(extra='forbid')

    scope: VlmScope = 'all'
    conf_max: float = Field(default=0.80, ge=0.0, le=1.0)
    per_cluster: int = Field(default=5, ge=1, le=100)
    max_crops_per_day: int = Field(default=0, ge=0)
    sample_frac: float = Field(default=1.0, gt=0.0, le=1.0)


class VlmPolicy(VlmPolicyBody):
    revision: int = 0
