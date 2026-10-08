# Example annotation profiles

Cropwright ships with no domain built in. Region features (the region
review tab, the `/clusters` region gallery, sub-box editing, the
detections panel, the region drain on `/ingest`) appear only when the
OpenProcessor backend serves a region profile on
`GET {API_PREFIX}/health` (`region_profile`). The app then builds the
region slot from that profile: the profile's `display_name` labels the
tab and the gallery, and its `region_class_name` is the bound class.

The files in `annotation-profiles/` are tier-2 deployment profiles
(`annotation-profiles.json`). They are never bundled into the build. Use
one to replace the generic served slot with domain-specific copy, a
keymap, or hand-declared training cohorts.

| File                        | What it is                                                                                |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| `license-plate.json`        | Customizes the region slot of a backend running the `license_plate` region profile.       |
| `aircraft-tail-number.json` | Demo slot with a parent-frame sub-box and no false-positive state. Needs backend support. |
| `defect-code.json`          | Demo slot with no geometry at all (a closed text vocabulary). Needs backend support.      |

## Enabling an example

1. Configure the matching region profile on the backend (for example
   OpenProcessor's `OP_REGION_PROFILE_PATH=examples/region_profiles/license_plate.json`).
   The frontend alone cannot enable region features.
2. Bind-mount the example over the tier-2 path in the running container:
   `/usr/share/nginx/html/annotation-profiles.json`. Or copy it to
   `static/annotation-profiles.json` before a build.

A slot that uses the region routes (`/regions`, `/crops/{id}/region*`)
must use the served profile's `name` as its `key`. Otherwise it is
dropped with a warning, because the backend serves exactly one region
profile. With no region profile configured, every such slot is dropped.
