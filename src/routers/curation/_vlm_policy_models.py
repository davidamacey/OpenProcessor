"""Wire models of the VLM-policy routes (the policy shapes live in
:mod:`src.services.curation.vlm_policy`)."""

from __future__ import annotations

from pydantic import Field

from src.services.curation.vlm_policy import VlmPolicyBody


class VlmPolicyUpdate(VlmPolicyBody):
    """``PUT`` body: the policy plus the revision the caller read."""

    expected_revision: int = Field(ge=0)
