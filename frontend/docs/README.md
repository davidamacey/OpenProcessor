# docs/ index

| Doc                                                                                    | What it is                                                                                                                      | Status                                                                               |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| [FEATURES.md](FEATURES.md)                                                             | Visual feature tour — every route, screenshot + explanation                                                                     | **current**, maintained                                                              |
| [screenshots/](screenshots/)                                                           | Source images for FEATURES.md + the README demo GIF                                                                             | **current**                                                                          |
| [system-audit-2026-09-11.md](system-audit-2026-09-11.md)                               | Independent full-stack audit (pipeline integrity, gaps, punch list)                                                             | historical snapshot — many findings since resolved, see root `CHANGELOG.md`          |
| [curation-strategy-plan-2026-09.md](curation-strategy-plan-2026-09.md)                 | Design proposal for the stackable curation/sorting strategy registries (StrategyBar, review sorts, overlays)                    | historical — the plan this shipped from; see `CHANGELOG.md` for what actually landed |
| [market-research-labeling-tools-2026-09.md](market-research-labeling-tools-2026-09.md) | Read-only market survey: this app vs. LightlyStudio and the broader commercial+OSS CV-labeling landscape                        | reference — still accurate as a market snapshot, not tied to a specific code version |
| [opensource-labeling-comparison-2026-09.md](opensource-labeling-comparison-2026-09.md) | Follow-up: head-to-head vs. CVAT / Label Studio / FiftyOne / LightlyStudio / Diffgram / Supervisely, plus a video-labeling note | reference — same caveat as above                                                     |

Root-level docs (`../README.md`, `../CLAUDE.md`, `../CHANGELOG.md`,
`../RUBRIC.md`) are the other half of the picture: README/CLAUDE.md are
kept current as the app changes; CHANGELOG.md is the authoritative
"what actually shipped" record; RUBRIC.md is the labeling-decision
reference for human labelers, independent of app version.

The two "historical" docs above aren't stale in the sense of being
wrong — they're accurate as of the date in the filename. Cross-check
against `CHANGELOG.md` and the current code before treating a specific
finding as still true.
