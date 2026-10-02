---
sidebar_position: 2
title: Backend feature flags
---

# What each feature needs from the backend

Every feature below degrades to **absent, not broken**, when the backend
hasn't enabled it — the nav link or page section simply doesn't render, and
Cropwright never fires a request against a route the backend hasn't
mounted. None of this is a Cropwright build flag; it's all read from the
backend's own `/methods`, `/health`, and feature-specific availability
probes at runtime.

| Feature | Backend requirement |
| --- | --- |
| Region review tab, `/clusters` region gallery, sub-box edit | A region profile served on `GET {API_PREFIX}/health` (`region_profile`) |
| Score chips, mistakenness/uniqueness sort options | `OP_SCORES_ENABLED` |
| Diverse-selection overlay on `/clusters/[id]` | `OP_SELECT_DIVERSE_ENABLED` |
| Embedding-plot lasso tool on `/clusters` | `OP_VIZ_PROJECTION_ENABLED` |
| Semantic (embedding) search box | `OP_SEMANTIC_SEARCH_ENABLED` |
| `/train` (preflight, launch, promote) | The backend's trainer container/worker running |
| `/bakeoff` | The backend's on-demand evaluator container, and its `/bakeoff/*` router mounted |
| `/ingest` browser upload persisting bytes for later browsing | The backend's ingest-upload persistence (a pre-persistence backend shows a banner explaining uploads aren't browsable) |
| Projects (`/p/<project>`, `/projects`) | The backend's projects API; the project list and each project's served prefix come from it |
| Combine projects | The backend's combine router mounted (probed once; absent otherwise) |
| [Dataset import and Reprocess](../user-guide/dataset-import.md) | The backend's `datasets` routes (probed once per project via the formats list) |
| [Prompt-pack editor](../user-guide/prompt-packs.md) | The backend's prompt-pack routes (probed once per project) |
| [Region-profile editor](../user-guide/region-profiles.md) | The backend's region-profile routes (probed once per project) |
| [VLM registry](../user-guide/vlm-models.md) | The backend's VLM endpoint registry (probed once per deployment); per-run VLM pickers follow the `vlm` axis on `/methods` |
| Test on a crop | Prompt packs: a call the schema marks testable. Region profiles: the profile test route |
| Imported review tab | The tab served by the backend's review-tab vocabulary |
| [Keyboard shortcut editor](../user-guide/keyboard-shortcuts.md#customizing-shortcuts) | The backend's keymap routes |
| Model sharing | The backend's model status serving owner and sharing revision |
| Multi-box regions | A region profile; the per-write box limit comes from the served profile |
