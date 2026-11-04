# Feature tour

A route-by-route index of what Cropwright does. The task-oriented
description of each page is the user guide, in
`docs-site/docs/cropwright/user-guide/`; it is the one home for this
content, so this file only points to it. See the main [README](../README.md)
for setup and [CLAUDE.md](../CLAUDE.md) for the route and keyboard-shortcut
tables.

Everything that depends on an optional backend feature is **absent, not
broken**, when the backend does not serve it. What each feature needs is in
[Backend feature flags](../../docs-site/docs/cropwright/configuration/backend-feature-flags.md).
Every project page lives under `/p/<project>/`.

| Page | Path | User guide |
| --- | --- | --- |
| Projects, combine | `/projects`, `/projects/combine` | [Projects](../../docs-site/docs/cropwright/user-guide/projects.md), [Combine projects](../../docs-site/docs/cropwright/user-guide/combine-projects.md) |
| Dashboard | `dashboard` | [Dashboard](../../docs-site/docs/cropwright/user-guide/dashboard.md) |
| Ingest | `ingest` | [Ingest](../../docs-site/docs/cropwright/user-guide/ingest.md) |
| Dataset import and Reprocess | `datasets/import`, `datasets/imports` | [Dataset import](../../docs-site/docs/cropwright/user-guide/dataset-import.md) |
| Clusters, cluster detail, label operations | `clusters`, `clusters/[id]`, `label-ops` | [Clusters](../../docs-site/docs/cropwright/user-guide/clusters.md), [Assist](../../docs-site/docs/cropwright/user-guide/assist.md) |
| Review | `review` | [Review](../../docs-site/docs/cropwright/user-guide/review.md), [Multi-box regions](../../docs-site/docs/cropwright/user-guide/multi-box-regions.md) |
| Accuracy audit | `audit` | [Accuracy audit](../../docs-site/docs/cropwright/user-guide/audit.md) |
| Classes | `classes` | [Classes](../../docs-site/docs/cropwright/user-guide/classes.md) |
| Export | `export` | [Export](../../docs-site/docs/cropwright/user-guide/export.md) |
| Models | `models` | [Models](../../docs-site/docs/cropwright/user-guide/models.md) |
| Train | `train` | [Train](../../docs-site/docs/cropwright/user-guide/train.md) |
| Bake-off | `bakeoff` | [Model comparison](../../docs-site/docs/cropwright/user-guide/bakeoff.md) |
| Settings | `settings` | [Settings](../../docs-site/docs/cropwright/user-guide/settings.md) |
| Prompt packs, region profiles, open-vocabulary sets, VLM models, ingest policy | `settings/prompt-packs`, `settings/region-profiles`, `settings/open-vocab`, `settings/models`, `settings/ingest-policy` | [Prompt packs](../../docs-site/docs/cropwright/user-guide/prompt-packs.md), [Region profiles](../../docs-site/docs/cropwright/user-guide/region-profiles.md), [Open-vocabulary sets](../../docs-site/docs/cropwright/user-guide/open-vocabulary.md), [VLM models](../../docs-site/docs/cropwright/user-guide/vlm-models.md), [Ingest policy](../../docs-site/docs/cropwright/user-guide/ingest-policy.md) |
| Keyboard shortcuts overlay | press `` ` `` anywhere | [Keyboard shortcuts](../../docs-site/docs/cropwright/user-guide/keyboard-shortcuts.md) |

Screenshots are published with the documentation site, in
`docs-site/static/img/screenshots/cropwright/`. See
[Screenshots](../../docs-site/docs/cropwright/developer-guide/screenshots.md).
