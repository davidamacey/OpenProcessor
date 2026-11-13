# P2a-0: Metadata (phase spec)

| Field | Value |
|---|---|
| Status | plan, 2026-11-13, added after the owner's decision that metadata is in scope. Starts when lane L-A0 has merged and the owner has answered D1 and Q6 ([90](90_owner_questions.md)); it may run while the engine effort (P1) works, because the files are disjoint ([00](00_photo_library_plan.md) section 2.4) |
| Master plan | [00_photo_library_plan.md](00_photo_library_plan.md): rules, data model (section 6), routes (section 7), exit bar (section 2) |
| Why a phase of its own | P2a-1's timeline, Places and smart albums are only as good as the capture times and positions under them. Proving the metadata first keeps P2a-1 provable and small |

Notation: `{L}` = `/curation/projects/{project}/library`; `media_id` is an image id, or `video:<id>` for a
scanned video.

---

## 1. Scope and non-goals

**In scope.** One metadata reader for photos **and** videos (the existing ExifTool reader, extended, not a
second one); correct GPS parsing (signed decimals, altitude, direction, speed, GPS UTC time, accuracy);
QuickTime/MP4 date handling; embedded GPS tracks from action cameras, drones, dashcams and phones; GPX files
(found in the source folders or uploaded) as tracks; geotagging photos without GPS from a track; import of
keywords, tags, titles and captions from embedded XMP/IPTC and from XMP sidecars; edits of capture time
(single, batch shift, time-zone set), position, title, caption, keywords and tags, stored in the database
and undoable; XMP sidecar export; a served metadata report; a minimal Cropwright report page.

**Non-goals.** Writing to any original file; videos as timeline items, proxies and playback (P3); the
library timeline, Places map and editing UI (P2a-1 builds them on these routes); star ratings (out, 00
section 2.2).

---

## 2. The one reader (lane L-M1)

### 2.1 What exists today (read on `int/b4-stateless`)

| Piece | Today |
|---|---|
| `src/clients/exiftool_client.py` | PyExifTool `ExifToolHelper` over one `exiftool -stay_open True -@ -` process per worker, `common_args=['-G', '-n']`, extra `-fast`, a watchdog timeout per read, typed `ExifToolError` for outages, log lines that never carry ExifTool output (it can hold a coordinate) |
| `src/domain/capture_metadata.py` | pure `parse_capture` over `EXIFTOOL_TAGS` (`EXIF:DateTimeOriginal`, `OffsetTimeOriginal`, `SubSec*`, `CreateDate`, make, model, lens, focal lengths, orientation, `GPSDateStamp`/`GPSTimeStamp`, `GPSStatus`, `GPSHPositioningError`, `GPSImgDirection`, `Composite:GPSLatitude/Longitude/Altitude`); offset only from `OffsetTimeOriginal` or GPS UTC; (0, 0) and void fixes are no location; `PARSER_VERSION = 'capture@1'` |
| `src/services/curation/capture_read.py`, `capture_fields.py` | ingest call and the `capture.*` mapping; `source_backfill` re-reads stale or unknown captures |

### 2.2 What changes (same modules; `PARSER_VERSION = 'capture@2'`)

| Change | Detail | Docs |
|---|---|---|
| Group family 1 | `common_args=['-G1', '-n']`: keys like `ExifIFD:DateTimeOriginal`, `XMP-dc:Subject`, `XMP-lr:HierarchicalSubject`, `Keys:CreationDate`, `UserData:GPSCoordinates`, so XMP namespaces and QuickTime atoms are told apart | [exiftool `-G`](https://exiftool.org/exiftool_pod.html) |
| Tags added (photos) | `GPS:GPSAltitudeRef` (through `Composite:GPSAltitude`, signed with `-n`), `GPS:GPSImgDirectionRef`, `GPS:GPSSpeed`/`GPSSpeedRef`, `Composite:GPSDateTime` (UTC), `GPS:GPSMapDatum`, `GPS:GPSMeasureMode`; serial numbers (`ExifIFD:SerialNumber`, `ExifIFD:BodySerialNumber`, maker-note serials) for per-camera clock fixes; `IFD0:Software` | ExifTool GPS and EXIF tag tables |
| Tags added (text) | `XMP-dc:Subject`, `XMP-lr:HierarchicalSubject`, `IPTC:Keywords`, `XMP-dc:Title`, `IPTC:ObjectName`, `XMP-dc:Description`, `IPTC:Caption-Abstract`, `IFD0:ImageDescription`, `XMP-photoshop:DateCreated`; ratings are not read into storage | ExifTool XMP and IPTC tag tables |
| Videos | no `-fast` for QuickTime files; tags `QuickTime:CreateDate`, `QuickTime:MediaCreateDate`, `Keys:CreationDate` (carries an offset on many phones), `UserData:DateTimeOriginal`, `Keys:GPSCoordinates`/`UserData:GPSCoordinates` (ISO 6709, parsed by ExifTool into signed decimals with `-n`), `QuickTime:Duration`, codec and size, `Keys:Make`/`Model` | [QuickTime tags](https://exiftool.org/TagNames/QuickTime.html) |
| QuickTime time zone | QuickTime integer dates "should be stored as UTC", but many cameras store local time, so ExifTool assumes no zone unless the `QuickTimeUTC` API option is set. Rule order: (1) a tag with an explicit offset (`Keys:CreationDate`); (2) the UTC time of the file's own GPS track, which also yields the offset; (3) the per-make rule table `src/domain/quicktime_time_rules.py` (data, one fixture per row) saying whether that make writes UTC, read with `-api QuickTimeUTC` when it does; (4) otherwise floating local time with `offset_source = 'none'` | same page, QuickTimeUTC note |
| Embedded tracks | a first pass detects a timed-metadata track (handler or sample description such as `gpmd` for GoPro GPMF, `camm`, subtitle `tx3g` streams of drones, dashcam GPS atoms); only then a second pass with `-ee3 -G3 -n` returns the per-sample `GPSDateTime`, latitude, longitude, altitude, speed and track, grouped by embedded document. ExifTool reads 128 kinds of timed GPS metadata | [`-ee`](https://exiftool.org/exiftool_pod.html), [geotagging and inverse geotagging](https://exiftool.org/geotag.html), QuickTime tags page |
| Thinning | points thinned to at most 1 Hz (first point of each second, plus the last), stored in `media_track_points`; a GeoJSON line simplified once for the map | |
| Location policy | `drop`: positions and tracks are not stored (only `has_location` and `has_track`); `store_coarse`: positions rounded as today and tracks thinned to one point per 30 s and rounded; `location_purge` deletes tracks | `src/domain/location_policy.py` |
| Bounds | per-file timeout scaled by size for `-ee3` (a long video is a long parse); an outage stays `exif_status = 'unknown'` and is retried by the backfill | existing watchdog |
| Sidecars read | for `photo.jpg`, ExifTool also reads `photo.xmp` and `photo.jpg.xmp` beside it (Lightroom and darktable conventions) and the matching file in the sidecar mirror tree; precedence per field: database edit > sidecar > embedded | |
| Apple edit files (`.AAE`) | an `IMG_1234.AAE` beside `IMG_1234.HEIC` is the iPhone's edit recipe, never ingested as media. It is an XML plist; `adjustmentData` is base64, then raw-deflate JSON (`com.apple.photo`, format versions 1.3 to 1.11 seen) with an `adjustments` list (Crop, SmartTone, SmartColor, WhiteBalance, DepthEffect, PortraitEffect, SourceSelect, Effect), or a binary plist for `com.apple.video.slomo`. Read with the standard library (`plistlib`, `base64`, `zlib`); store the format, version and adjustment identifiers per item, flag "edited in Apple Photos", and apply only Crop (and rotation) to the displayed derivative; other adjustments are recorded, not re-rendered. Check `osxphotos` (MIT) for its decoder before writing one | unknown format version: recorded, not applied |
| Re-read | bumping the parser version makes the existing capture backfill re-read every file with an older stamp; workspace projects use the same reader (one reader) | |

Libraries: ExifTool (free software under the same terms as Perl, per its docs) in the images that already
ship it; PyExifTool (already pinned, dual GPL-3.0-or-later or BSD, the BSD terms used); `gpxpy` for GPX
parsing (the lane verifies its licence and pins it); no other metadata reader is added.

### 2.3 Video scan without decoding

The source walk (bulk ingest walkers, `src/services/curation/bulk_ingest/`) routes video extensions (`.mp4
.mov .m4v .insv .lrv` as configured) to the job `library_metadata_scan`, which reads metadata only and writes
`library_videos` and `media_tracks`. Photos keep the ingest path. Nothing decodes a frame in P2a-0.

### 2.4 GPX tracks and geotagging

| Step | Rule |
|---|---|
| Sources | `.gpx` files found by the walk (read-only, like any original) and files uploaded to `POST {L}/tracks/gpx` (stored under the derivatives root) |
| Parse | `gpxpy`; every track segment becomes one `media_tracks` row |
| Geotag | for photos without a read or edited position whose effective capture time falls on a track (with an optional camera clock offset), position = linear interpolation between the two fixes around the time when they are at most `GeoMaxIntSecs` (1800 s) apart, else the nearest fix when within `GeoMaxExtSecs` (1800 s), else none. This is ExifTool's documented rule; ExifTool's own `-geotag` writes files, so the rule is implemented on the stored points and checked against `exiftool -geotag` run on temporary copies in the test |
| Result | a `location_edit`-kind library operation with `location_source = 'gpx'` or `'track'`, dry run first, undoable |

---

## 3. Edits stored in the database (lane L-M2)

| Edit | Storage | Effect |
|---|---|---|
| Capture time: set one, shift a selection by minutes, set a time-zone offset; filter by camera model and serial within a date range (the wrong-camera-clock case) | `capture_overrides` | effective `capture.taken_local`, `taken_utc`, `offset_minutes`; `time_source = 'user'` |
| Position | `capture_overrides` (`location_source = 'user'`) | effective `capture.location`; refused under `drop`, coarsened under `store_coarse` |
| Title, caption | `media_text` (`source = 'user'`) | `meta.title`, `meta.caption` |
| Keywords and tags | `keywords`, `media_keywords` (`source = 'user'`) | `meta.keyword_ids`, `meta.keyword_paths` |

Rules: the reader never writes an override table; the projection drain composes effective values (override
if present, else read) and is the only writer of those image-doc fields; each edit is a
`library_operations` row with per-media `before` values and undo (the per-item rule of
`docs/design/cq3_label_operations.md`: a later edit of the same field is kept); revert of a field deletes
the override and re-reads. Videos keep their effective values in `library_videos.capture` until P3 indexes
them.

---

## 4. XMP sidecar export (lane L-M3)

| Concern | Design |
|---|---|
| What | for each edited or imported media: `XMP-exif:DateTimeOriginal` and `XMP-exif:OffsetTimeOriginal`, `XMP-photoshop:DateCreated`, GPS (policy-filtered), `XMP-dc:Title`, `XMP-dc:Description`, `XMP-dc:Subject`, `XMP-lr:HierarchicalSubject`. Favourites have no standard tag and are not written |
| How | ExifTool writes a new XMP file with `-o <destination>.xmp` and the tag assignments, through the same `-stay_open` process; the original is only read |
| Where | per Q6: default a mirror tree `OP_DERIVATIVES_ROOT/sidecars/<project>/<path below the source root>.xmp`; option (b), if the owner allows it, next to the original only when the operator enabled it for the library and the source mount is writable. The writer refuses any destination that is an original file or not ending in `.xmp` (test) |
| Round trip | export → a fresh library over the same files with the sidecar tree configured as a sidecar root → identical effective values (bar 4) |

---

## 5. Tables and migrations

One revision `<next>_library_metadata.py` (lane L-M2's first commit): `capture_overrides`, `media_text`,
`keywords`, `media_keywords`, `library_videos`, `media_tracks`, `media_track_points`, `library_operations`,
`library_operation_members`, `library_projection_pending`, with the columns of 00 section 6.1, forced RLS.
OpenSearch: `capture_fields.py` gains the capture@2 fields; new `library_meta_fields.py` (`meta.*`).

---

## 6. Routes and wire models (`src/routers/curation/library_metadata.py`, models in `src/schemas/curation/library_metadata.py`)

| Route | Request | Response |
|---|---|---|
| `GET {L}` | | `LibrarySummary {photos_total, videos_total, tracks_total, photos_undated, size_class, coverage: dict[str, {done, pending, failed}]}` (P2a-1 adds fields) |
| `GET {L}/metadata/report` | | `MetadataReport {cameras: list[{make, model, serial_suffix, photos, videos, time_sources: dict[str, int], offsets: dict[str, int], first_taken, last_taken}], undated: int, unreadable: int, with_location: int, tracks: int, parser: str}` (serials shown as the last 4 characters only) |
| `GET {L}/media/{media_id}/metadata` | | `MediaMetadata {media_id, read: CaptureWire, effective: CaptureWire, overrides: list[{field, value, op_id}], title, caption, keywords: list[{keyword_id, path, source}], track_id \| None}` |
| `POST {L}/media/capture_time` | `{selection: MediaSelection, shift_minutes?: int, set_offset_minutes?: int, set_taken_local?: str, dry_run: bool}` | `MetadataOpResponse {dry_run, n_media, samples: list[{media_id, before, after}], op_id, job_id}` |
| `POST {L}/media/location` | `{selection, lat, lon, altitude_m?, dry_run}` | `MetadataOpResponse` |
| `POST {L}/media/text` | `{selection, title?, caption?, dry_run}` | `MetadataOpResponse` |
| `GET/POST {L}/keywords` | create `{name, parent_id?}` | `KeywordTree {keywords: list[{keyword_id, name, parent_id, path, count, source}]}` |
| `PATCH/DELETE {L}/keywords/{keyword_id}` | `{name?, parent_id?}` | `Keyword`; delete refuses with children (`409 keyword_has_children`) |
| `POST {L}/media/keywords` | `{selection, add: list[str], remove: list[str], dry_run}` (paths like `Travel/Portugal`) | `MetadataOpResponse` |
| `GET {L}/tracks` | `from`, `to`, `cursor` | `Tracks {tracks: list[TrackCard {track_id, source_kind, device, file_name, start_utc, end_utc, n_points, bbox}], next_cursor}` |
| `GET {L}/tracks/{track_id}` | `points_cursor` | `Track {...TrackCard, line: GeoJSON LineString, points: list[{t_utc, lat, lon, alt_m, speed_mps}], next_points_cursor, photos_along: list[{image_id, taken_utc, lat, lon}]}` |
| `POST {L}/tracks/gpx` | multipart `.gpx` files | `Tracks` |
| `POST {L}/tracks/{track_id}/geotag` | `{selection?, camera_offset_minutes: int = 0, dry_run}` | `MetadataOpResponse` |
| `GET {L}/videos` | `cursor` | `Videos {videos: list[{video_id, file_name, taken_local, duration_s, codec, has_track}], next_cursor}` |
| `POST {L}/sidecars/export` | `{selection?, destination: Literal['mirror','beside_original']}` | `JobRef` (`beside_original` refused unless enabled, `409 sidecars_beside_disabled`) |
| `GET {L}/operations`, `POST {L}/operations/{op_id}/undo` | | `LibraryOperations`, `MetadataOpResponse` |

`MediaSelection`: exactly one of `media_ids` (max 5,000) or a filter (`camera_model`, `serial_suffix`,
`taken_from`, `taken_to`, `has_location`, `folder_prefix`); never an exclusion list.

---

## 7. Jobs and ticks

| Job or tick | Queue | Singleton | Notes |
|---|---|---|---|
| `library_metadata_scan` | `library` | `exclusive_group='library_metadata'` | videos and GPX files; pausable, sliced |
| capture re-read | existing `source_backfill` capture step | existing | after the parser bump |
| projection drain (tick) | `periodic` | advisory `LIBRARY_PROJECTION_DRAIN`, obj = project lock key | composes effective capture, `meta.*` |
| `library_bulk_op` | `library` | none | edits above `OP_BATCH_SYNC_MAX` |
| `library_gpx_geotag` | `library` | `exclusive_group='library_metadata'` | |
| `library_sidecar_export` | `library` | `exclusive_group='library_sidecars'` | pausable |

New advisory class constants are added to `src/config/advisory_locks.py` with the next free numbers.

---

## 8. Cropwright (lane L-F0)

One page, `routes/p/[project]/library/metadata/+page.svelte`, rendering `{L}/metadata/report` (cameras,
time sources, offsets, undated and unreadable counts, tracks) with a "Export sidecars" action
(`lib/jobs/followJob.ts`). It uses `ui/PageShell.svelte`, `ui/DataTable.svelte`, `ui/EmptyState.svelte`.
P2a-1's library layout wraps this route; the page itself is not rewritten. API module
`lib/api_libraryMetadata.ts`, types `lib/types_libraryMetadata.ts`. The editing UI (info panel, batch time
shift dialog, keyword editor, track map) is P2a-1.

---

## 9. Lanes, order and verify commands

```
wave 0:  L-A0 storage roles (00 section 2.3)  ||  L-M4 metadata test set and parity harness
wave 1:  L-M2 schema + wire models first commit (contracts generated), then edits, drain, routes
         ||  L-M1 reader capture@2, video scan, tracks, GPX
wave 2:  L-M3 sidecar export and round trip  ||  L-F0 report page  ||  L-FIX
then:    the coordinator's sweep, the parity and round-trip runs, the owner reviews the report of the own library
```

| Lane | Model, effort | Owns | Verify (paste the output) |
|---|---|---|---|
| L-M4 | Sonnet, medium | `scripts/datasets/fetch_metadata_testset.py` (ExifTool per-make samples, GoPro `gpmf-parser` samples, Commons geotagged originals, any licensed drone/dashcam/phone files; manifest with licence per file; local only), `scripts/bench/library_metadata_parity.py` (runs the reference `exiftool -j -n -G1 -ee3` and `-p gpx.fmt -ee3`, compares with the served values, registered bars in its header) | `.venv/bin/python -m pytest tests/scripts/test_metadata_parity_harness.py -q --no-cov` |
| L-M1 | Sonnet, high (every ingest reads it) | `src/clients/exiftool_client.py`, `src/domain/capture_metadata.py`, new `src/domain/quicktime_time_rules.py`, new `src/domain/gps_track.py`, `src/services/curation/capture_read.py`, `src/clients/curation_opensearch/capture_fields.py`, the video and GPX routing in `src/services/curation/bulk_ingest/`, `src/services/library/metadata/scan.py`, `tracks.py` | `.venv/bin/python -m pytest tests/domain/test_capture_metadata.py tests/domain/test_quicktime_time_rules.py tests/domain/test_gps_track.py tests/clients/test_exiftool_client.py -q --no-cov` |
| L-M2 | Sonnet, high (write paths, undo, lock rule) | migration, `src/db/models/library_metadata.py`, repos, `src/schemas/curation/library_metadata.py`, `src/routers/curation/library_metadata.py`, `src/services/library/{scope,projection_drain}.py`, `src/services/library/metadata/{edits,keywords,text,geotag,report}.py`, `src/clients/curation_opensearch/library_meta_fields.py`, queue and job kinds, advisory class constants | `.venv/bin/python -m pytest tests/library/test_metadata_edits.py tests/library/test_projection_drain.py tests/library/test_geotag.py tests/library/test_scope.py -q --no-cov && .venv/bin/python scripts/codegen/generate_contracts.py --check` |
| L-M3 | Sonnet, high | `src/services/library/metadata/sidecars.py`, its route | `.venv/bin/python -m pytest tests/library/test_sidecar_export.py tests/test_library_no_source_writes.py -q --no-cov` |
| L-F0 | Sonnet, medium | the report page, API and types | `cd frontend && npm run check && npx vitest run src/lib/api_libraryMetadata.test.ts && npx playwright test e2e/stubbed/library/test_metadata_report.py` (through the stubbed pytest harness) |

---

## 10. Tests to write red first

| Test | Proves |
|---|---|
| `tests/domain/test_capture_metadata.py` (extend) | `-G1` keys; signed GPS and altitude below sea level; `GPSImgDirectionRef` magnetic vs true kept; GPS UTC datetime; (0, 0) rejected; text tags trimmed; ratings ignored |
| `tests/domain/test_quicktime_time_rules.py` | each rule row with a fixture; offset tag beats track UTC beats rule table; unknown make stays floating |
| `tests/domain/test_gps_track.py` | `-G3` documents to ordered points; thinning to 1 Hz; `store_coarse` thinning and rounding; `drop` stores nothing |
| `tests/library/test_geotag.py` | interpolation and extrapolation limits equal ExifTool's documented defaults; camera offset; photos with a read position untouched |
| `tests/library/test_metadata_edits.py` | batch shift by camera serial; undo keeps a later edit; the reader's re-read never overwrites an override; DST and a trip across time zones keep local and UTC consistent |
| `tests/library/test_sidecar_export.py` | round trip of every exported field; destination guard refuses originals and non-`.xmp` paths; `beside_original` refused unless enabled |
| `tests/test_library_one_metadata_reader.py` | only `src/clients/exiftool_client.py` starts ExifTool; only `src/domain/capture_metadata.py` maps its tags (AST and `rg` scan) |
| `tests/test_library_no_source_writes.py` | no write, rename or delete under a source root anywhere in `src/services/library/` (AST), except the sidecar writer's guarded `beside_original` path |

Pending wave runs: the parity harness on the full test set; a live scan of the GoPro samples; the checksum
manifest before and after edits and export.

---

## 11. Acceptance walk (owner)

1. Point a library at the owner's own folders (photos and videos); the scan finishes; open Library ›
   Metadata: cameras, time sources and offsets look right; the undated and unreadable counts are small and
   listed.
2. A camera whose clock was wrong: batch shift its photos by the right amount from the API or report page;
   the dry run shows before and after; apply; undo; apply again.
3. Videos from an action camera, a drone and a phone appear with tracks; the track count matches what the
   owner expects.
4. Upload a GPX of a walk; geotag the photos of a camera without GPS from it; spot-check 10.
5. Export sidecars to the mirror tree; open a few in another photo tool; the edits are there; the originals'
   checksums are unchanged.
6. The parity and round-trip reports against the bars of 00 section 2.
