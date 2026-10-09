"""
Generic vision-language-model (VLM) labeler service.

Talks to any OpenAI-compatible vision ``/chat/completions`` endpoint
(a deployment might run behind OpenWebUI, but nothing here names
that model) to:

- batch-classify item crops into one of a caller-supplied set of class
  names (closed- or open-vocabulary)
- verify whether a crop's sub-region-of-interest (e.g. a printed label
  on a product photo) is real, and read any text on it

This module composes :class:`VlmLabeler` from the operation families, one
module each: ``vlm_labeler_core.py`` (transport), ``vlm_labeler_classify.py``,
``vlm_labeler_verify.py``, ``vlm_labeler_visibility.py`` and
``vlm_labeler_combined.py``. Models and errors live in ``vlm_models.py``,
reply parsing in ``vlm_reply_parse.py``, class-name helpers in
``vlm_class_names.py``, transport in ``vlm_client.py`` and prompt data in
``vlm_prompts.py``.

Design notes
------------
- The upstream VLM is typically launched with a hard cap on images per
  prompt (e.g. ``--limit-mm-per-prompt '{"image":8}'``), so calls chunk
  crops into batches of at most ``max_images_per_call``.
- Uses tenacity (via ``vlm_client.post_chat_with_retry``) for
  retry-with-backoff on transient 5xx + connection errors.
- Robust JSON parsing — VLMs will sometimes wrap output in ```json
  fences despite the system prompt. We strip them and on parse failure
  return all-low-confidence fallbacks so the rest of the pipeline can
  still funnel the crop into the human-review queue rather than
  crashing the batch.
- The wire keys a prompt asks the VLM to return for the
  region-of-interest sub-annotation (``region_visible``,
  ``region_bbox_correct``, ``region_text``, ``region_confidence``) are
  read via a :class:`~src.config.RegionFields` instance rather than
  hardcoded literals, so a deployment with existing data under
  different field names (e.g. an overlay using ``roi_*``) is a config
  flip, not a code change. A :class:`~src.services.labeling
  .vlm_prompts.PromptPack`'s own templates ask the VLM for the matching
  key names — see that module's docstring.

Public surface
--------------
- :class:`VlmLabeler` — async client; one instance is intended to be
  shared across the FastAPI process.
"""

from __future__ import annotations

from src.services.labeling.vlm_labeler_classify import ClassifyOps
from src.services.labeling.vlm_labeler_combined import CombinedOps
from src.services.labeling.vlm_labeler_verify import VerifyOps
from src.services.labeling.vlm_labeler_visibility import VisibilityOps


class VlmLabeler(ClassifyOps, VerifyOps, VisibilityOps, CombinedOps):
    """Async client for an OpenAI-compatible vision-chat VLM endpoint."""


__all__ = ['VlmLabeler']
