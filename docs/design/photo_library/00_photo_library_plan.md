# Photo library: master plan (P1-P4)

| Field | Value |
|---|---|
| Status | plan, written 2026-11-13 on branch `plan/photo-library` (tree of `int/b4-stateless`), revised the same day after an independent review. Nothing here is built. Executes phase by phase on the owner's go |
| Read first | [VISION_AND_GOALS.md](../../VISION_AND_GOALS.md), [engineering_standards.md](../engineering_standards.md), the inputs [01_feature_research.md](01_feature_research.md) and [02_current_system_gap.md](02_current_system_gap.md) |
| Files in this folder | `00` (this file); [05_p2a0_metadata.md](05_p2a0_metadata.md) (P2a-0), [10_p2a1_organise.md](10_p2a1_organise.md) (P2a-1), [15_p2a2_people.md](15_p2a2_people.md) (P2a-2), [20_p2b_curate.md](20_p2b_curate.md), [30_p2c_create.md](30_p2c_create.md) (lane-ready phase specs); [40_p3_p4_video_outline.md](40_p3_p4_video_outline.md) (short outline); [90_owner_questions.md](90_owner_questions.md) (decisions for the owner) |
| Where this file and a phase file disagree | order and scope: this file wins; task detail: the phase file wins |
| Conventions | `{P}` = `/curation/projects/{project}`; `{L}` = `{P}/library`. Every route named here is **proposed**. Paths are relative to the repo root. Migrations are named `<next>_<topic>.py`: the coordinator assigns the next free number at merge time, so phases never collide |

Plan rules that bind every phase (from CLAUDE.md, the engineering standards and the owner):

1. **Thin frontend.** The backend serves data and decisions: buckets, counts, rule schemas, sort options,
   "hidden because" reasons, memory selections, empty-state copy. Cropwright never filters, counts or ranks.
   Missing data is a new route with a contract and a test, never a client workaround.
2. **Postgres for relational data, Celery for long jobs**, no job files, no file locks. Singletons use the
   mechanisms that exist on `int/b4-stateless`:
   - a **job** that must not run twice declares `exclusive_group` and `exclusive_scope` on its `JobKind`
     (`src/config/job_kinds/kind.py`); the partial unique index `uq_jobs_one_active` on `jobs`
     (`src/db/models/jobs.py`) allows one active job per (project, group, scope value). Per project: scope
     `''`; per calendar account: scope `account_id`; per exclusion rule: scope `rule_id`. A running job
     holds its row lease (`owner`, `lease_expires_at`);
   - a **periodic tick** runs through `src/services/periodic/runner.py` once per active, unpaused project
     with the project bound, and takes `pg_try_advisory_xact_lock(class, project lock_key)`; each library
     tick gets a new class constant in `src/config/advisory_locks.py` (next free after
     `IMAGE_COUNTS_DRAIN = 21`). A tick only claims bounded work or enqueues jobs; it never runs a long body.
3. **No fan-out in request handlers** (engineering standards principle 9).
4. **Wire contract.** Every route or field change runs `make contracts` and commits `contracts/` in the
   same commit. Wire models of a phase are published (Pydantic models + generated contracts) before any
   frontend lane of that phase starts.
5. **Class identity is the name.** Rules, exclusions and "Send to project" carry class names, never ids.
6. **Lock rule.** Person names, merges, confirmations, rejections, flags, album edits and exclusion rules
   are human decisions; no machine run overwrites them (`src/clients/occ_locks.py`, OCC helpers in
   `src/clients/occ.py`). The projection drain copies human values from Postgres; it never computes them.
7. **No duplicate logic.** One face-identity service serves the core `/persons` routes and library People;
   the superseded union-find grouping is deleted when the new method lands (section 5.3).
8. **Fail closed.** A workspace project refuses library routes; a photo whose face processing or projection
   is not current is not eligible for created outputs; an unmounted source root marks nothing missing.
9. **Originals stay in place.** The library writes only derivatives (thumbnails, renders) and databases.
   No route, job or fixture writes, moves or deletes a source file (checksum test every live run).
10. **Privacy.** Library data never leaves the host. Faces never reach a hosted model. Positions follow the
    project's location policy. Calendar tokens and events stay on the host. Plans and tests use generic
    wording and public data only; no private names.
11. **Docs and libraries first.** Every lane brief names the library or documented pattern it uses and cites
    the pages read; prior art (Immich, PhotoPrism, digiKam) is read, never copied.
12. **Measured phases.** A phase starts only after the previous one met its exit bar on the owner's walk.

---

## 0. Goals and principles

Every phase's exit bar (section 2) names the goals it proves.

| # | Goal |
|---|---|
| **G1** | **The product.** A self-hosted photo library that does what Apple Photos, Google Photos and Immich do, and more: deeper processing (larger models, object boxes, VLM captions, custom-trained models, GPS tracks), more information per photo and more control. Originals are never modified |
| **G2** | **No hardware excuse.** A phone does all of this on a small chip; a GPU server has no excuse. The hard parts are engineering: |
| G2a | fast first import **and** cheap incremental updates: new photos are processed without re-processing or re-grouping the library |
| G2b | everything a screen shows is precomputed (thumbnails, counts, people, places, events, memories); no screen does heavy work when opened; read paths have measured targets at 500k and 5M photos (section 10) |
| G2c | show only what is confident: uncertain faces go to an "Is this X?" review; human confirmations are permanent rules that survive every re-run and improve matching |
| G2d | larger models only when batched, at the right precision and without needless copies (the engine speed work, P1) |
| G2e | polish: an intuitive, professional UI held to the same speed bars |
| **G3** | **Create frames with the stored boxes**: faces, people and objects decide crops, pan and zoom targets and matched cuts, so no output cuts a head |
| **G4** | **Measured phases**: each phase ships and is proven on public data plus the owner's walk on the owner's own library before the next starts |

---

## 1. Product statement

**OpenProcessor is the engine; the library is the so-what.** Anyone can run a detector. What a person wants
from a photo collection of 50,000 to 5,000,000 images is to find, keep, leave out and relive the right
photos, without uploading anything and without moving a file.

OpenProcessor reads a whole photo library in place (folders under the source root, never copied) and
records, per photo: capture time and place, objects, faces and people, text in the image, quality and
aesthetics, near-duplicates and bursts. It writes only derivatives and databases. On that one processed
library it offers four jobs:

| Job | What the user gets | Phase |
|---|---|---|
| 1. Label and train | Today's Cropwright loop. A library selection becomes a training project in one step (Send to project) | exists; link in P2a-2 |
| 2. Organise | Correct metadata first: capture time, GPS and camera from every photo and video through one ExifTool reader, GPS tracks from action cameras, drones, dashcams, phones and GPX files, imported keywords, tags, titles and captions, edits (including a batch clock or time-zone shift) kept in the database and exportable as XMP sidecars (P2a-0). Then a timeline that scrolls millions of photos instantly, albums, rule-based smart albums, Places with tracks on the map, favourites (a heart and a Favourites album), pick/reject flags, keywords, hidden, archive, trash, duplicates, one search (P2a-1); People (P2a-2) | P2a-0, P2a-1, P2a-2 |
| 3. Curate | Leave any person, object, text, date or place out of the library views and every created output, at high recall, inherited by new photos, with a review queue and an audit. Burst stacks with an explainable top pick, assisted culling | P2b |
| 4. Create | Calendar-aware smart albums and memories (trips, events, people, holidays, birthdays, "on this day") with best-shot selection, chapters, local-VLM titles and captions, music, beat-synced cuts, pan and zoom. **The first "AI videos" are MP4 slideshows rendered from stills**; montages with video clips come in P4 | P2c (stills), P4 (clips) |

**Who it is for.** A power user or data scientist who owns a large collection and will not upload it. **What
makes it different:** in-place libraries with per-signal provenance; exclusion everywhere at high recall
(no product offers it, 01 section 5); memories beyond "on this day" with a real video (Immich's main gap,
01 section 1.3 item 8); the same data feeds custom model training.

**Not in scope (any phase):** pixel editing, sharing and public links, mobile apps, cloud sync, a separate
desktop app or service (seams in section 3.4; nothing is built), multi-user accounts (no authentication
exists, `SECURITY.md`; Q2).

---

## 2. Phase roadmap

Each phase ships, is walked and proven against its exit bar before the next starts. Bars use the public sets
of section 11.5, each chosen for what it can prove (Open Images and COCO carry no capture time or GPS, so they
prove only object signals). **The final gate of every phase is the owner's walk on the owner's own library**
(numbers only in committed text). Every rate bar states its minimum sample and a one-sided 95 %
Clopper-Pearson bound; a missed bar is recorded with its cause and never moved. Bars are registered in the
eval script header before the first score.

| Phase | Plain name | Entry bar (all must hold) | Exit bar (all must hold) |
|---|---|---|---|
| **P1** | The image engine (owned by the separate engine speed effort) | its own | N1 and N3-N10 delivered with contracts; the engine owner has confirmed the provisional N-table field names in writing (section 2.1); on the 500k real-EXIF set every photo has a placeholder and grid thumbnail, HEIC fixtures ingest, frame-target search answers. Goals: G2a, G2b, G2d |
| **P2a-0** | Metadata: one ExifTool reader for photos and videos, GPS tracks, imported keywords, tags, titles and captions, edits in the database, XMP sidecar export | L-A0 merged; owner answered D1 and Q6; may run while the engine effort works (disjoint files, section 2.4) | (1) parity with the ExifTool reference (`exiftool -j -n -G1 -ee3`) for 100 % of present capture fields on the metadata test set, n ≥ 2,000 files over ≥ 100 camera models (coordinates within 1e-7°, times and offsets exact); video capture times agree with the GPS UTC of their own track on every video fixture that has one; (2) embedded track points equal the ExifTool `-p gpx.fmt -ee3` reference (count, first and last fix) on every video fixture (GoPro samples plus licensed drone, dashcam and phone files); (3) GPX geotag positions match ExifTool's documented interpolation within 1e-6°; (4) capture-time shift, location, title, caption, keywords and tags round-trip through an exported XMP sidecar into a fresh library with identical values (test); (5) exactly one metadata reader in the tree (test); (6) originals byte-identical after edits and export (checksum); (7) the `drop` location policy stores no positions or tracks (test); (8) the owner reviews the served metadata report of the own library; (9) incremental: a re-read after the parser change touches only files with an older parser stamp, and adding 1,000 files reads only those (test). Goals: G1, G2a, G4 |
| **P2a-1** | Organise: timeline, albums, smart albums, Places | P1 exit; P2a-0 exit; owner answered Q2 | (1) budgets of section 10 met on the 500k real-EXIF set and on the 5M replica (bench header records hardware, OpenSearch heap, load); (2) scrub to any month shows thumbnails in ≤ 1 s and the no-scroll-jump test passes; (3) smart-album rules evaluate server-side and pick up new photos (test); (4) Places: served country agrees with an independent label on ≥ 0.98 of photos, n ≥ 1,000, lower bound ≥ 0.97 (section 11.5); (5) trash, empty trash and every write path leave the source tree byte-identical (checksum test); (6) the UI meets the client budgets of section 10 and passes the owner's walk on the owner's library at 1600 and 430 px. Goals: G1, G2b, G2e, G4 |
| **P2a-2** | People | P2a-1 exit; N1 and N7 delivered; owner answered Q3 (face model licence); faces on by default in library projects (owner decision 2026-11-13) | (1) accepted machine assignments precision ≥ 0.99, n ≥ 2,000 assignments, lower bound ≥ 0.985, on **held-out identities** (thresholds fitted on other identities); (2) photo recall of a person at accept + review band reported per face-side bin (P2b depends on it); (3) confirm, reject, merge, split survive a full regroup (test); (4) "forget this person" leaves zero face docs, prototypes, constraints and projections for that person (count test); (5) People budgets of section 10; (6) incremental: 1,000 new photos added to a 500k library are assigned within the 60 s budget with no full grouping run (test and bench); (7) owner walk. Goals: G2a, G2b, G2c, G4 |
| **P2b** | Curate | P2a-2 exit; owner answered Q4; recall bars registered | (1) exclusion recall ≥ 0.99 on photos where the identity's face side is ≥ 40 px (n ≥ 500 photos over ≥ 20 identities, lower bound ≥ 0.98) and ≥ 0.95 on all face-visible photos (n ≥ 1,000, lower bound ≥ 0.935); false exclusion ≤ 0.05 of other photos (n ≥ 2,000, upper bound ≤ 0.06); on the Commons face set interleaved into the 500k library (section 11.5); (2) a new photo of an excluded person is out of every view and output before it becomes visible (fail-closed test); (3) zero excluded-identity photos in created outputs across the eval (hard zero); (4) stack top-1 agreement ≥ 0.60, n ≥ 500 series, lower bound ≥ 0.56; (5) owner walk. Goals: G1, G2c, G4 |
| **P2c** | Create (stills) | P2b exit; owner answered Q5, Q7, Q8; N6/N7 present | (1) Google Calendar full and incremental sync, 410 recovery, `invalid_grant` → reconnect, disconnect-and-purge to zero, proven on a recorded API fake and a synthetic calendar fixture (public photos have no calendars), then on the owner's account; (2) memories: zero seeded excluded identities and zero utility-class photos in any memory (hard zero); (3) owner rates 20 memories from the public set and 20 from the own library with the rubric of 30 section 12: mean ≥ 4 of 5, none below 3 on "selection"; (4) a 60 s 1080p render ≤ 60 s with NVENC (≤ 180 s CPU); ≥ 90 % of cuts within 50 ms of a beat; (5) event boundary F1 ≥ 0.75 on 50 human-labelled real user streams; (6) subject-aware framing: in 40 rendered memories no face box is cut by the frame edge (hard zero), and matched cuts keep the subject centre within 0.05 of the frame width and its height ratio within 0.85-1.18 on ≥ 90 % of pairs (n ≥ 200, lower bound ≥ 0.86) (30 section 5.2); (7) memories, featured photos and events are built by the nightly and incremental jobs, and opening Memories reads only stored rows (budget test); (8) owner walk. Goals: G1, G2b, G2e, G3, G4 |
| **P3** | Video processing | P2c exit | outline only ([40](40_p3_p4_video_outline.md)) |
| **P4** | Video memories | P3 exit | outline only |

### 2.1 What the library needs from P1 (provisional until the engine owner confirms)

The engine effort owns these internals. **Every field name below is provisional**: the engine owner confirms
or renames them before P2a-1 starts, and this table is then updated once (no lane codes against an
unconfirmed name).

| # | Need | Minimum contract (provisional names) | Read by |
|---|---|---|---|
| N1 | Faces in curation ingest | `processing_plan.py` allows `faces`/`face_embed` for library projects; each face written to the project's `faces` role index (fields in section 6.2); `faces.state` on the image doc | P2a-2, P2b, P2c |
| N2 | Index role `faces` and size classes | **owned by this plan's lane L-A0, not by P1** (section 2.3), so P1 writes into a role that already exists | everything |
| N3 | Stored derivatives tier | ThumbHash placeholder (`derivatives.thumbhash`) and aspect on the image doc; grid thumbnail (long side 360 px, WebP) at ingest; preview (1600 px JPEG) lazily on first view; under `OP_DERIVATIVES_ROOT`, never in a source folder; content-addressed paths written to a temp file and renamed (atomic; concurrent writers write identical bytes, no lock); eviction of previews by a tick under an advisory lock with a size cap; disk full fails closed (`503 derivatives_full`, summary alert); the grid tier is rebuildable from originals by a backfill job and is not backed up | timeline, viewer, People covers, renders |
| N4 | HEIC/HEIF ingest; RAW via the embedded preview if the engine owner agrees | bulk ingest accepts `.heic .heif` and decodes once | timeline |
| N5 | Whole-photo OCR text | `text.ocr`, `text.ocr_at` on the image doc | search, text exclusion |
| N6 | Per-photo quality signals | `signals.sharpness`, `signals.exposure`, `signals.aesthetic` (0-1), `signals.utility_class` (`none/screenshot/document/receipt/whiteboard`), `signals.model`, `signals.version` | P2b culling, P2c |
| N7 | Per-face signals | `quality`, `eyes_open`, `yaw_deg`, `side_px` per face | P2a-2 prototypes, P2b, P2c |
| N8 | Library-scale ingest | a registered throughput for 500k photos; `embedding.mode=selected` default for library projects | budgets |
| N9 | Frame-target semantic search | `{P}/search/text` and `{P}/search/image` serve `target=frame`, `group_by=image` ([cluster plan](../cluster_search_makemodel_plan.md) section 4.1) | search |
| N10 | Moved and deleted files | `source_state` (`present/moved/missing`); moved bytes repointed by content key; **an unmounted root marks nothing missing** | trash, renders |

If an N item is late, the phase that reads it waits; no library lane builds a stand-in. Capture metadata
(time, offset, GPS, camera, keywords, captions) is **not** an engine need: P2a-0 owns the one ExifTool
reader (section 2.4).

### 2.2 Later, optional (outside every exit criterion), and what is out

Owner decisions of 2026-11-13 moved metadata, keywords, tags, captions, capture-time edits, GPS tracks,
GPX and XMP sidecars **into** P2a-0/P2a-1. What remains below is not part of P2a-P2c; each item is a later
phase only on the owner's request, and none has tables, fields or routes until then (no dead code).

| Item | Status | Why |
|---|---|---|
| Star ratings | **out** | the owner does not use them; favourites, pick/reject, explicit keep-top choices and the culling signals give memories and top-pick everything a rating would. No concrete benefit was found, so no question is asked. Ratings in imported XMP are not stored |
| Colour labels | later | pick/reject and keywords cover sorting |
| Folder view of the originals | later | search by path covers the need; a tree needs a new projection |
| Merge suggestions, unassigned-faces browser, "me", person birth date | later | P2b needs only grouping, review band, confirm/reject, merge, split, forget; birthday memories in P2c use calendar birthday events |
| OCR text overlay with copy in the viewer | later | needs per-line boxes from N5 |
| Video playback with a marker moving on the track | P3 | needs playable proxies (many action-camera and phone files are HEVC, which browsers do not all play) and videos as timeline items; P2a-1 shows the tracks on the map with a time slider and the photos along them, which needs no video decoding |

### 2.3 The `faces` index role and size classes (lane L-A0, owner of N2)

`PROJECT_INDEX_ROLES` feeds a database CHECK on `project_indexes.role`, so the role is a schema change.
L-A0 runs as soon as D1 is answered, in parallel with the engine effort, because P1's N1 writes into it.

| Change | Where |
|---|---|
| `PROJECT_INDEX_ROLES = ('images', 'items', 'faces')`; `projects.kind` (`workspace`/`library`, immutable) and `projects.size_class` (`s`/`m`/`l`, required for libraries, null otherwise) with CHECKs | `src/db/models/projects.py` |
| Migration `<next>_library_kind_faces_role.py`: add the two columns and CHECKs, drop and recreate the `project_indexes.role` CHECK with the new list | `src/db/migrations/versions/` |
| `IndexRole.FACES = 'faces'`; `INDEX_ROLE_ATTR[IndexRole.FACES] = 'faces_index'` | `src/config/curation_roles.py` |
| `CurationConfig.faces_index: str \| None` (None for workspace projects); the loops over `INDEX_ROLE_ATTR` skip an absent optional role | `src/config/curation.py` |
| Project records and resource names build the `faces` name only for `kind = 'library'` | `src/config/projects.py`, `src/services/projects/records.py`, `lifecycle.py` |
| Faces index body (512-d Lucene HNSW cosine) and shard counts per size class for all three roles: `s` 1/2/1, `m` 3/6/3, `l` 8/16/8 (images/items/faces), keeping shards in OpenSearch's 10-50 GB guidance; `_split` documented for growth | `src/clients/curation_opensearch/bodies_faces.py`, `base.py`, `lifecycle.py`, the schema stamp |
| Capacity counts faces shards; project delete removes the faces index; the face privacy purge targets the bound project's faces index | `src/services/projects/capacity.py`, delete path, `src/services/face_privacy.py` |
| Guard: ownership already comes from `project_indexes`; an allowlist change only if the faces service sends a new shape | `src/services/projects/guard_policy.py` |
| Guard tests (red first): a bound workspace project naming `{prefix}{slug}__faces` raises `CrossProjectAccess`; library A naming library B's faces index raises; library A's own faces index passes for the allowlisted shapes; unbound code naming any `__faces` index raises | `tests/curation/cross_project_leak/test_faces_role_isolation.py` |
| DB test: the role CHECK accepts `faces`; lifecycle refuses a `faces` row for a workspace project | `tests/db/test_project_kind_roles.py` (wave run) |

Verify (lane): `.venv/bin/python -m pytest tests/curation/cross_project_leak/test_faces_role_isolation.py
tests/projects -q --no-cov` and `.venv/bin/python scripts/codegen/generate_contracts.py --check`.
Adversarial review (isolation): an independent reviewer reads the L-A0 diff before merge.

### 2.4 Who owns the metadata files

P2a-0 replaces the current capture reader in place (no second extractor): it owns
`src/clients/exiftool_client.py`, `src/domain/capture_metadata.py`, `src/services/curation/capture_read.py`
and `src/clients/curation_opensearch/capture_fields.py` while it runs. The engine effort is told not to
edit them in that window; ingest calls the reader through its existing function.

---

## 3. Architecture overview

```
 Cropwright (nginx, :5184), same app
   /p/<project>/workspace/...   (labelling, existing)
   /p/<library>/library/...     (Library area; frontend/src/routes/p/[project]/library/**)
        │  only {L}/* and existing {P}/* routes
        ▼
 API (FastAPI :4603, N stateless replicas)
   src/routers/curation/library_*.py ─► src/services/library/** ─► src/domain/library_*.py (pure rules)
        │ OpenSearch (guarded, one bound project)        PostgreSQL (RLS on project_id)
        │   {prefix}{slug}__images  (photo-first)          projects.kind='library', persons, albums,
        │   {prefix}{slug}__items   (objects)              smart_albums, photo_annotations, operations,
        │   {prefix}{slug}__faces   (new role, L-A0)       exclusion_rules, events, calendar_*, memories
        ▼
 Celery + Valkey: queues `library` (CPU) and `media_render` (FFmpeg; NVENC when a GPU is assigned)
   periodic ticks (runner + advisory locks): projection drain, person assignment, trash expiry,
   calendar sync, memories nightly
 Triton (engine): detection, PE frame vectors, SCRFD + ArcFace, OCR, quality heads
 Derivatives: OP_DERIVATIVES_ROOT (thumbnails, previews, renders); sources mounted read-only
```

### 3.1 The photo-first projection and which writer wins

The library's unit is the photo. Every library predicate runs as **one query on the images index**, never
through the 10k image join (`src/services/curation/image_join.py`). Each image doc carries projected fields
(section 6.2) written by one recompute writer, the **library projection drain**
(`src/services/library/projection_drain.py`, modelled on `src/services/curation/image_counts_refresh.py`),
for every image id in `library_projection_pending`.

Human data (`lib.*`) has two writers, and the rule is fixed: **Postgres is the source of truth and the
highest `revision` wins.** A human write updates the `photo_annotations` row (revision + 1), then writes the
image doc's `lib.*` and `lib.revision` with OCC in the same request (below `OP_BATCH_SYNC_MAX`, else a job),
and enqueues the photo. The drain reads the row and writes only when the row's revision is greater than the
doc's `lib.revision`; on an OCC conflict it re-reads and re-checks. Neither writer can write a value that is
not in Postgres, and a stale drain cannot overwrite a newer write-through (test
`tests/library/test_annotation_writer_order.py` interleaves both orders).

Person merges do not wait for re-projection: queries expand a person id to the ids merged into it
(`persons.merged_into`), and the drain normalises `faces.person_ids` in the background.

### 3.2 Query-time predicates

Visibility (trash, archive, hidden, exclusion rules) is compiled per request from Postgres into `must_not`
clauses by one pure function (`src/domain/library_visibility.py`) used by the one library query builder.
A rule takes effect on the next request; new photos are covered once projected and are not eligible for
created outputs before that (section 5a).

### 3.3 Scale shape

- Size classes per section 2.3; growth past a class by `_split` in a maintenance job.
- Vectors: one PE frame vector per photo (about 8.4 KB stored, 42 GB at 5M est.); crop vectors only for
  selected classes (N8); 512-d face vectors.
- Thumbnails: grid tier at ingest (about 25 KB, about 125 GB at 5M est.); previews lazy and capped (N3).
- Timeline reads are aggregations and `search_after` pages on `capture.taken_local`; no deep `from`.

### 3.4 Seams for a later separate service or desktop app (not built)

| Seam | How it is kept |
|---|---|
| API | every library route under `{L}` in `src/routers/curation/library_*.py`, models in `src/schemas/curation/library_*.py` |
| Logic | `src/services/library/**` imports curation services only through named functions |
| Frontend | `frontend/src/routes/p/[project]/library/**`, `frontend/src/lib/library/**`, `frontend/src/lib/components/library/**`, `lib/api_library*.ts` |
| Jobs | queues `library` and `media_render`, compose profile `library` |

### 3.5 Videos and GPS tracks before P3

Video files are scanned for metadata only in P2a-0 (no decoding): each becomes a `library_videos` row
(capture time, duration, camera, GPS track) and is not yet a timeline item. GPS tracks, embedded in videos
or read from GPX files, are stored per source in `media_tracks` and `media_track_points` under the location
policy. P2a-1 draws them on the Places map with a time slider and the photos taken along them. P3 adds
videos to the images index as timeline items with playable proxies and the synced player; it adds to these
rows and rewrites nothing.

---

## 4. The library model decision (D1, the owner decides)

Options from [02](02_current_system_gap.md) section 4: **A** one library project; **B** a layer over many
projects; **C** a new library index family with its own guard binding; **D** library as the source, projects
derived.

**Recommendation: A for storage, D for the workflow ("library projects").** A library is a project with
`projects.kind = 'library'`, three index roles (`images`, `items`, `faces`), bound and guarded like any
project, its tables under the existing RLS. Training happens in workspace projects created or extended with
**Send to project** (combine copy: images by content key, classes by name). Several libraries may exist.

| Criterion | A + D workflow (recommended) | B | C | D as in 02 |
|---|---|---|---|---|
| Isolation guard | unchanged (one new role, section 2.3) | weakened or a new multi-project binding | new binding kind and name family | as C |
| Pipeline reuse | full | full per project, new fan-in | only if every pipeline is parameterised by index family (two-pipeline risk) | as C plus an ingest split |
| Postgres RLS | reused | reused | needs a `library_id` key everywhere | as C |
| Faces | project `faces` role | per project plus a cross-project person table | once, in the library | as C |
| Scale | size classes, photo-first projection | N queries plus merge, no global cursor | designed for size | as C |
| Cost | smallest | high | highest | highest |

Reasons: (1) the fail-closed isolation guarantee stays; (2) no duplicate pipeline; (3) RLS for free; (4)
training isolation preserved through derived projects (what D wanted); (5) P2a-1 can start right after P1.
**Cost:** a library project also has a labelling Workspace; the UI frames its labels as detector machine
labels. **What the owner loses with C or D:** a storage lane and an adversarial guard review before any P2a
feature, and about one more phase of time. This settles the [cluster plan](../cluster_search_makemodel_plan.md)
O1 as option b for library projects only; faces are on by default in library projects (owner decision 2026-11-13).

---

## 5. Backend services and jobs

### 5.1 Services (new package `src/services/library/`)

| Module (proposed) | Responsibility | Phase |
|---|---|---|
| `scope.py` | `require_library(project)`: `409 project_not_library` for workspace projects | P2a-0 |
| `metadata/` | the metadata scan of videos, GPX discovery and tracks, keywords and tags, titles and captions, capture and location overrides, batch time shift, GPX geotagging, XMP sidecar export, the metadata report (the reader itself stays `src/clients/exiftool_client.py` + `src/domain/capture_metadata.py`) | P2a-0 |
| `query.py` | the one library query builder (`LibraryFilter` + visibility) | P2a-1 |
| `timeline.py` | buckets, pages, locate, neighbours | P2a-1 |
| `projection_drain.py` | recompute photo projections (section 3.1) | P2a-0 |
| `annotations.py`, `trash.py` | favourite, pick/reject, archive, hidden, trash; operations and undo | P2a-1 |
| `albums.py`, `smart_albums.py` | collections; rules through `src/domain/library_rules.py` | P2a-1 |
| `places.py` | offline reverse geocoding (GeoNames) | P2a-1 |
| `search.py`, `facets.py` | combined search, suggest, facets | P2a-1 |
| `people/` | library People on the shared face-identity service (section 5.3): grouping, assignment, review band, merge, split, forget | P2a-2 |
| `send_to_project.py` | selection → workspace project through `src/services/projects/combine/` | P2a-2 |
| `exclusions.py`, `exclusion_sweep.py` | rules, preview, coverage, audit, recall sweep | P2b |
| `stacks.py`, `culling.py` | burst stacks (near-dup engine, new group kind), top pick, suggestions | P2b |
| `events.py`, `calendar/`, `memories/`, `render/` | events, calendar, memories, rendering | P2c |

### 5.2 Jobs, ticks and their singleton mechanism

Queues `TaskQueues.LIBRARY = 'library'` and `TaskQueues.MEDIA_RENDER = 'media_render'` (`src/config/queue.py`,
profile `library`); kinds in `src/config/job_kinds/library.py`, routes in `src/config/task_routes/library.py`.

| Job or tick | Kind | Queue | Singleton | Limits | Phase |
|---|---|---|---|---|---|
| Projection drain | tick 15 s; claims ≤ 2,000 pending rows per project per tick | `periodic` | advisory class `LIBRARY_PROJECTION_DRAIN`, obj = project lock key | tick budget 10 s | P2a-0 |
| Metadata scan of videos and GPX files; capture re-read after the reader change | job `library_metadata_scan` (the existing `source_backfill` capture step re-reads photos) | `library` | `exclusive_group='library_metadata'` | pausable, sliced | P2a-0 |
| Metadata edit or keyword op above `OP_BATCH_SYNC_MAX` | job `library_bulk_op` | `library` | none | soft 600 s | P2a-0 |
| GPX geotag apply | job `library_gpx_geotag` | `library` | `exclusive_group='library_metadata'` | | P2a-0 |
| XMP sidecar export | job `library_sidecar_export` | `library` | `exclusive_group='library_sidecars'` | pausable | P2a-0 |
| Bulk annotation or album op above `OP_BATCH_SYNC_MAX` | job `library_bulk_op` (P2a-0 kind) | `library` | none | soft 600 s | P2a-1 |
| Empty trash | job `library_trash_empty` | `library` | `exclusive_group='library_trash'` | soft 1 h, pausable | P2a-1 |
| Trash expiry | tick daily: enqueues `library_trash_empty` for expired photos | `periodic` | advisory `LIBRARY_TRASH_EXPIRE` | | P2a-1 |
| Places backfill, GeoNames load | jobs `library_places_backfill`, `library_geonames_load` | `library` | `exclusive_group='library_places'` | pausable | P2a-1 |
| Person assignment | tick 30 s; claims ≤ 1,000 new faces | `periodic` | advisory `LIBRARY_PERSON_ASSIGN` | tick budget 20 s | P2a-2 |
| Person grouping run | job `library_person_grouping` | `cluster` | `exclusive_group='library_people'` | pausable, sliced | P2a-2 |
| Forget person | job `library_person_forget` | `library` | `exclusive_group='library_people'` | | P2a-2 |
| Send to project | job `library_send_to_project` | `jobs` | `exclusive_group='combine_target'`, scope `target_slug` | | P2a-2 |
| Exclusion sweep | job `library_exclusion_sweep` | `library` | `exclusive_group='library_exclusion_sweep'`, scope `rule_id` | | P2b |
| Exclusion nightly | tick daily: enqueues sweeps | `periodic` | advisory `LIBRARY_EXCLUSION_NIGHTLY` | | P2b |
| Stack run | near-dup run with `group_kind='burst'` | near-dup queue | the near-dup kind's existing exclusivity | | P2b |
| Events build | job `library_events_build` | `library` | `exclusive_group='library_events'` | | P2c |
| Calendar sync | tick 15 min: enqueues one job per due account (±25 % jitter) | `periodic` → `library` | advisory `LIBRARY_CALENDAR_SYNC`; job `exclusive_group='library_calendar'`, scope `account_id` | soft 600 s | P2c |
| Memories nightly | tick daily 03:00: enqueues `library_memories_build` | `periodic` → `library` | advisory `LIBRARY_MEMORIES_NIGHTLY`; job `exclusive_group='library_memories'` | `pausable=True`, `slice_s` set, resumes per candidate kind, soft 2 h | P2c |
| Render | job `library_render` | `media_render` | none (worker concurrency) | soft 15 min | P2c |

### 5.3 One face-identity service (decision)

Core `/persons` (global `visual_search_faces`) and library People must not be two implementations.
**Decision: reuse one service, replace the old method.** `src/services/face_identity/` becomes the only
face-identity code, parameterised by the index and the bound scope. Lane L-P1 (P2a-2) moves grouping to the
mutual-kNN graph with average-linkage AHC (`src/services/curation/clustering/methods/ahc.py`), with the
quality gate, same-photo cannot-link, review band, stable ids and constraints of the cluster plan section 3.2,
and **deletes** the union-find grouping (`method='union_find'`) in the same change; the core `/persons`
routes call the same functions on the global index (the core API stands on its own, VISION_AND_GOALS). No
second grouping, assignment or merge code exists anywhere.

## 5a. Exclusion design (P2b; detail in [20](20_p2b_curate.md) section 2)

| Part | Design |
|---|---|
| Rules | `exclusion_rules` rows: subject kind (`person`, `class` by name, `text`, `date_range`, `annual_date`, `place`, `album`) and level |
| Levels | `show_less`: never a memory's subject, weight 0.2 in group shots; `block`: never in any created output, group shots included, left out of Send to project; `hide`: `block` plus removed from every browsing surface and the People list |
| Query time | `src/domain/library_visibility.py` compiles enabled rules into `must_not` clauses; every list goes through the one query builder (route-table test) |
| High recall | a person rule matches `faces.person_ids` or `faces.candidate_person_ids`; candidates from a looser threshold `t_exclude`, a recall sweep seeded by every confirmed face, a ±10 min context link, and a measured optional no-face body link |
| Inheritance | the assignment tick scores new faces against excluded persons at `t_exclude` in the same write; photos are not eligible for created outputs until processed and projected; under `hide`, unprocessed photos stay hidden (Q4) |
| Human override | confirm (must-link) or reject (cannot-link); photo-level force and release |
| Audit, reset | counts-only audit; coverage including "waiting for faces"; reset with a dry-run count |
| Exclusion vs forget | exclusion keeps the person's embeddings so new photos can be caught; "forget this person" (P2a-2) deletes them, after which new photos of that person are not recognised (the UI says so) |

## 5b. Memories engine design (P2c; detail in [30](30_p2c_create.md))

```
photos ──► events and trips (adaptive time gaps + GPS jumps, home cell) ──► candidates
calendar (Google, ICS, CalDAV) + holidays ──► matches (suggested, user-accepted) ──┘
candidates ──► eligibility (exclusions, utility class, quality, processed, stack tops)
          ──► memory score ──► selection (chapters, cluster-then-cap, MMR, people coverage, cover)
          ──► titles (calendar > holiday > birthday event > local VLM > template)
          ──► timeline (beats, holds, dissolves, pan and zoom) ──► browser preview ──► MP4 on the host
```

| Stage | Model or library | Licence (status) |
|---|---|---|
| Event segmentation | ours (PhotoTOC, Platt et al. 2003; Cooper et al. 2005) | free to implement |
| Holidays | `holidays` | MIT (01) |
| Google Calendar | `google-auth-oauthlib`, `google-auth`, `google-api-python-client` | Apache-2.0 (lane verifies pins) |
| ICS, recurrence | `icalendar`, `recurring-ical-events`, `python-dateutil` | BSD, LGPL-3.0 (library use), Apache-2.0/BSD |
| CalDAV | `caldav` | Apache-2.0 option (01) |
| Token encryption | `cryptography` Fernet / MultiFernet | Apache-2.0 or BSD |
| Quality input | P1 signals N6/N7 | per P1 |
| Diversity | cluster-then-cap, MMR (Carbonell and Goldstein 1998) | free |
| Titles | the project's local VLM activation | per catalogue model |
| Beats | Beat This! | code and weights MIT (README read 2026-11-13) |
| Framing | stored face, person and object boxes first; smart crop per output aspect (never a cut head), pan and zoom targets on the subject, matched cuts aligning subject position and size (prior art: smartcrop.js, Google AutoFlip, Netflix match cutting; 30 section 5.2) | ours over our boxes |
| Saliency (only where no boxes) | U-2-Net small (`u2netp`) on ONNX Runtime CPU | Apache-2.0 (01) |
| Compose and encode | MoviePy v2 + distribution FFmpeg, or a pure FFmpeg filtergraph (prototype decides) | MIT; FFmpeg build licence checked by an image test: no `--enable-nonfree` (30 section 5.3) |
| Music | user-supplied (Q7) | per track |
| Avoided | IQA-PyTorch (PolyForm NC), CR-FIQA (CC BY-NC), madmom models (CC BY-NC-SA), Remotion (company licence) | 01 section 3.8 |

---

## 6. Data model

### 6.1 PostgreSQL (every project table carries `project_id` under forced RLS, registered like the existing tables in `src/db/migrations/schema_0001/security.py`)

| Table | Columns | Phase |
|---|---|---|
| `projects`, `project_indexes` (change) | section 2.3 | L-A0 |
| `capture_overrides` | pk (`project_id`,`media_id text`), `taken_local timestamp null`, `offset_minutes smallint null`, `lat double null`, `lon double null`, `altitude_m real null`, `location_source text null` (`user/gpx/track`), `op_id uuid`, `revision bigint`, `updated_at`; `media_id` is an image id or `video:<id>` | P2a-0 |
| `media_text` | pk (`project_id`,`media_id`), `title text null`, `caption text null`, `title_source`, `caption_source` (`imported/user`), `revision`, `updated_at` | P2a-0 |
| `keywords` | `id uuid7`, `project_id`, `name`, `parent_id null`, `path text` (materialised `a/b/c`), `source` (`imported/user`), `created_at`; unique (`project_id`,`parent_id`,`name`). Flat tags are top-level keywords (`dc:subject`); hierarchical ones come from `lr:hierarchicalSubject` | P2a-0 |
| `media_keywords` | `project_id`, `media_id`, `keyword_id`, `source` (`imported/user`); pk the first three | P2a-0 |
| `library_videos` | `id uuid7`, `project_id`, `content_key`, `path`, `size_bytes`, `duration_s`, `codec`, `width`, `height`, `capture` (`jsonb`, the capture@2 fields), `has_track bool`, `source_state`, `scanned_at`; unique (`project_id`,`content_key`) | P2a-0 |
| `media_tracks` | `id uuid7`, `project_id`, `source_kind` (`video_embedded/gpx_file/gpx_upload`), `video_id null`, `file_name`, `device text null`, `start_utc`, `end_utc`, `n_points int`, `bbox` (four reals), `simplified jsonb` (GeoJSON line for the map, simplified once), `policy_applied` (`store/store_coarse`), `created_at` | P2a-0 |
| `media_track_points` | `track_id`, `seq int`, `t_utc timestamptz`, `lat double`, `lon double`, `alt_m real null`, `speed_mps real null`, `heading_deg real null`; pk (`track_id`,`seq`); thinned to at most 1 Hz | P2a-0 |
| `photo_annotations` | pk (`project_id`,`image_id text`), `favourite bool` (the heart; feeds the Favourites album, memories and key-photo choice), `pick text` (`none/pick/reject`), `archived bool`, `hidden bool`, `trashed_at timestamptz null`, `revision bigint`, `updated_at` | P2a-1 |
| `albums` | `id uuid7`, `project_id`, `title`, `description null`, `cover_image_id null`, `sort` (`position/taken_asc/taken_desc`), `member_count int`, `revision`, `created_at`, `updated_at` | P2a-1 |
| `album_members` | `album_id`, `image_id text`, `position numeric`, `added_at`, `library_op_id null`; pk (`album_id`,`image_id`) | P2a-1 |
| `smart_albums` | `id`, `project_id`, `title`, `kind` (`smart_album/saved_search`), `rule jsonb` (`library_rule/1`), `rule_version int`, `pinned bool`, `revision`, `created_at`, `updated_at` | P2a-1 |
| `library_operations` | `id`, `project_id`, `kind` (`capture_edit/location_edit/text_edit/keyword/gpx_geotag` in P2a-0; `annotate/album_add/album_remove/trash/restore` in P2a-1; later phases add kinds), `summary jsonb`, `n_photos int`, `undone_at null`, `created_at` | P2a-0 |
| `library_operation_members` | `op_id`, `media_id text`, `before jsonb`; pk both | P2a-0 |
| `library_projection_pending` | `project_id`, `image_id text`, `reason`, `enqueued_at`; pk (`project_id`,`image_id`) | P2a-0 |
| `geo_places` (deployment table, no RLS, written only by the loader job) | `geonames_id int pk`, `name`, `ascii_name`, `country_code char(2)`, `admin1_code`, `admin1_name`, `admin2_name`, `lat real`, `lon real`, `population int`, `dataset_version` | P2a-1 |
| `persons` | `id uuid7`, `project_id`, `name null`, `hidden bool`, `cover_face_id null`, `state` (`active/merged/forgotten`), `merged_into uuid null`, `face_count int`, `named_at null`, `revision`, `created_at`, `updated_at` | P2a-2 |
| `person_prototypes` | `person_id`, `face_id text`, `mode_tag null`, `weight real`; pk both; at most 8 per person | P2a-2 |
| `face_constraints` | `id`, `project_id`, `kind` (`must_link/cannot_link`), `face_id text`, `person_id`, `source` (`confirm/reject/split/merge`), `created_at`; unique (`face_id`,`person_id`,`kind`) | P2a-2 |
| `exclusion_rules`, `exclusion_audit`, `exclusion_photo_decisions`, `stack_picks`; `near_dup_runs.group_kind` | [20](20_p2b_curate.md) section 5 | P2b |
| `library_events`, `library_event_members`, `calendar_accounts`, `calendar_sources`, `calendar_events`, `calendar_matches`, `memories`, `memory_chapters`, `memory_members`, `music_tracks`, `renders` | [30](30_p2c_create.md) section 6 | P2c |

A forgotten person keeps its row as `state = 'forgotten'` with `name` set to null and no faces, so stale
references resolve to "forgotten" rather than erroring; nothing identifying remains. Settings that are not
relational go into `project_settings` as new keys (`library.trash_retention_days` default 30,
`library.faces_enabled`; later-phase keys in their files).

### 6.2 OpenSearch fields (additive mappings; one field module per group, like `count_fields.py`)

| Group | Fields | Writer | Phase |
|---|---|---|---|
| `derivatives` | `thumbhash` (not indexed), `aspect`, `grid_key`, `at` | P1 (N3) | P1 |
| `signals` | `sharpness`, `exposure`, `aesthetic`, `utility_class`, `model`, `version` | P1 (N6) | P1 |
| `text` | `ocr`, `ocr_at` | P1 (N5) | P1 |
| `source_state` | keyword | P1 (N10) | P1 |
| `faces` | `state`, `count` (P1); `person_ids`, `candidate_person_ids`, `projected_at` (drain) | P1 / drain | P1, P2a-2 |
| `capture` (capture@2, same group) | existing fields, plus `location_source` (`exif/user/gpx/track`), `subject_area`, `serial_number`, `software`; effective values = override if present, else read; `time_source = 'user'` for overridden times | the reader; overrides by the drain | P2a-0 |
| `meta` | `title`, `caption` (text), `keyword_ids`, `keyword_paths` (keyword[]), `revision` | drain from `media_text` and `media_keywords` | P2a-0 |
| `objects` | `class_names` (names of every class on the photo's items), `projected_at` | drain | P2a-1 |
| `place` | `country_code`, `country`, `admin1`, `admin2`, `locality`, `geonames_id` | drain; cleared by `location_purge` | P2a-1 |
| `lib` | `favourite`, `pick`, `archived`, `hidden`, `trashed_at`, `album_ids`, `revision` | write-through + drain (section 3.1) | P2a-1 |
| `capture` (extend) | `taken_month_day` (`MM-DD`), `taken_year` | drain | P2a-1 |
| `lib` (extend) | `excl_forced`, `excl_released` | P2b | P2b |
| `stack` | `group_id`, `is_top`, `size` | stack projection | P2b |
| `event` | `event_id`, `trip_id` | drain | P2c |

Faces index (L-A0 body; P1 writes detections, P2a-2 writes identity): `face_id` (content-derived from
`image_id` and the rounded box), `image_id`, `bbox_norm`, `landmarks`, `det_score`, `quality`, `yaw_deg`,
`eyes_open`, `side_px`, `embedding` (512-d), `capture_taken_local`, `person_id`, `person_state`
(`unassigned/suggested/assigned/confirmed`), `candidate_person_ids`, `assign_score`, `assign_source`
(`machine/human`), `model`. Machine runs never change a `confirmed` face or a face with a constraint.

---

## 7. Route list (all proposed; `{L}` = `/curation/projects/{project}/library`)

Style follows `contracts/openapi/curation.json`: snake_case, plural collections, `{x_id}` parameters,
actions as sub-paths, `dry_run` and `expected_revision` on writes. Wire models are in the phase files.

| Phase | Method | Path | Purpose |
|---|---|---|---|
| P2a-0 | GET | `{L}` | summary: counts, size class, coverage per signal |
| P2a-0 | GET | `{L}/metadata/report` | per-camera counts, time sources, offsets found, files without a capture time, videos and tracks found, unreadable files |
| P2a-0 | GET | `{L}/media/{media_id}/metadata` | effective and read values side by side, with sources |
| P2a-0 | POST | `{L}/media/capture_time` | single or batch: shift by minutes, set a time-zone offset, or set a time; filter by camera and serial; dry run with before/after samples; undoable |
| P2a-0 | POST | `{L}/media/location` | set a position for a selection (policy-filtered); undoable |
| P2a-0 | POST | `{L}/media/text` | set title and caption for a selection; undoable |
| P2a-0 | GET, POST | `{L}/keywords` | keyword tree with counts; create |
| P2a-0 | PATCH, DELETE | `{L}/keywords/{keyword_id}` | rename, move, delete |
| P2a-0 | POST | `{L}/media/keywords` | add or remove keywords and tags for a selection; undoable |
| P2a-0 | GET | `{L}/tracks` | tracks with time range, device and source |
| P2a-0 | GET | `{L}/tracks/{track_id}` | simplified line, points page, photos taken along it |
| P2a-0 | POST | `{L}/tracks/gpx` | upload GPX files as tracks |
| P2a-0 | POST | `{L}/tracks/{track_id}/geotag` | geotag photos without GPS from a track (dry run, job) |
| P2a-0 | GET | `{L}/videos` | scanned videos (metadata only until P3) |
| P2a-0 | POST | `{L}/sidecars/export` | write XMP sidecars of edited media (job) |
| P2a-0 | GET | `{L}/operations` | operations, newest first |
| P2a-0 | POST | `{L}/operations/{op_id}/undo` | undo one operation, keeping later edits |
| P2a-1 | GET | `{L}/timeline/buckets` | year/month/day buckets with counts, undated bucket |
| P2a-1 | GET | `{L}/timeline/photos` | tiles of one bucket (cursor) |
| P2a-1 | GET | `{L}/timeline/locate` | bucket and index of a date or photo |
| P2a-1 | GET | `{L}/photos/{image_id}` | photo detail with provenance |
| P2a-1 | GET | `{L}/photos/{image_id}/neighbours` | previous/next in a served context |
| P2a-1 | GET | `{L}/photos/{image_id}/preview` | preview tier |
| P2a-1 | POST | `{L}/photos/annotations` | favourite, pick/reject, archived, hidden for a selection; undoable |
| P2a-1 | POST | `{L}/photos/trash` | trash (index level only) |
| P2a-1 | POST | `{L}/photos/restore` | restore |
| P2a-1 | GET | `{L}/trash` | trash with days left |
| P2a-1 | POST | `{L}/trash/empty` | remove trashed photos' index docs and derivatives (job); never the source |
| P2a-1 | GET, POST | `{L}/albums` | list, create |
| P2a-1 | GET, PATCH, DELETE | `{L}/albums/{album_id}` | read, edit, delete |
| P2a-1 | GET | `{L}/albums/{album_id}/photos` | album photos |
| P2a-1 | POST | `{L}/albums/{album_id}/photos` | add a selection |
| P2a-1 | POST | `{L}/albums/{album_id}/photos/remove` | remove a selection |
| P2a-1 | POST | `{L}/albums/{album_id}/photos/reorder` | move photos |
| P2a-1 | GET | `{L}/rules/schema` | served rule fields and operators |
| P2a-1 | GET, POST | `{L}/smart_albums` | list, create |
| P2a-1 | GET, PUT, DELETE | `{L}/smart_albums/{smart_album_id}` | read, replace, delete |
| P2a-1 | GET | `{L}/smart_albums/{smart_album_id}/photos` | evaluated photos |
| P2a-1 | POST | `{L}/smart_albums/preview` | count and first page of an unsaved rule |
| P2a-1 | GET | `{L}/places` | places by level |
| P2a-1 | GET | `{L}/search` | text or example photo plus filter |
| P2a-1 | GET | `{L}/search/suggest` | type-ahead |
| P2a-1 | GET | `{L}/facets` | facet counts |
| P2a-1 | GET | `{L}/photos/by_content/{image_id}` | resolve a content id from a workspace project |
| P2a-2 | GET | `{L}/people` | people with counts and covers |
| P2a-2 | GET, PATCH | `{L}/people/{person_id}` | read; name, hide, cover |
| P2a-2 | GET | `{L}/people/{person_id}/photos` | photos of a person |
| P2a-2 | GET | `{L}/people/{person_id}/faces` | faces by state |
| P2a-2 | GET | `{L}/people/{person_id}/suggestions` | review-band faces |
| P2a-2 | POST | `{L}/people/{person_id}/faces/confirm` | must-link |
| P2a-2 | POST | `{L}/people/{person_id}/faces/reject` | cannot-link |
| P2a-2 | POST | `{L}/people/merge` | merge persons |
| P2a-2 | POST | `{L}/people/{person_id}/split` | split selected faces |
| P2a-2 | POST | `{L}/people/{person_id}/forget` | delete the person's faces, embeddings and derived data (dry run, confirm count) |
| P2a-2 | GET | `{L}/faces/{face_id}/thumbnail` | face crop |
| P2a-2 | POST | `{L}/people/grouping/runs` | start a grouping run |
| P2a-2 | GET | `{L}/people/grouping/status` | last run, coverage |
| P2a-2 | POST | `{L}/send_to_project` | copy a selection into a workspace project (dry run, job) |
| P2b | GET, POST | `{L}/exclusions` | list, create |
| P2b | GET, PATCH, DELETE | `{L}/exclusions/{rule_id}` | read, change, delete |
| P2b | POST | `{L}/exclusions/preview` | counts per surface before saving |
| P2b | GET | `{L}/exclusions/{rule_id}/coverage` | excluded, candidates, waiting, audit |
| P2b | POST | `{L}/exclusions/{rule_id}/sweep` | recall sweep (job) |
| P2b | GET | `{L}/exclusions/{rule_id}/candidates` | photo-level candidates with evidence |
| P2b | POST | `{L}/exclusions/{rule_id}/candidates/decide` | force or release photos |
| P2b | POST | `{L}/exclusions/reset` | delete every rule (dry-run count) |
| P2b | GET | `{L}/review/faces` | candidate queue |
| P2b | GET | `{L}/stacks` | stacks with top pick |
| P2b | GET | `{L}/stacks/{group_id}` | members with per-signal scores |
| P2b | POST | `{L}/stacks/{group_id}/top` | set the top |
| P2b | POST | `{L}/stacks/{group_id}/keep_top` | trash the rest (undoable) |
| P2b | POST | `{L}/stacks` | manual stack |
| P2b | DELETE | `{L}/stacks/{group_id}` | unstack |
| P2b | GET, PUT | `{L}/culling/profile` | thresholds and weights |
| P2b | GET | `{L}/culling/suggestions` | pick/reject suggestions with reasons |
| P2b | POST | `{L}/culling/apply` | apply as flags (undoable) |
| P2c | GET | `{L}/events` | events and trips |
| P2c | GET | `{L}/events/{event_id}` | one event |
| P2c | POST | `{L}/events/rebuild` | re-segment (job) |
| P2c | GET | `{L}/calendar/accounts` | connected calendars |
| P2c | POST | `{L}/calendar/accounts/google/start` | begin OAuth (PKCE) |
| P2c | POST | `{L}/calendar/accounts/google/complete` | finish OAuth with the redirected URL (auto or pasted) |
| P2c | POST | `{L}/calendar/accounts/{account_id}/reconnect` | re-consent after `needs_reauth` |
| P2c | POST | `{L}/calendar/accounts/ics` | ICS file or secret URL |
| P2c | POST | `{L}/calendar/accounts/caldav` | CalDAV account |
| P2c | DELETE | `{L}/calendar/accounts/{account_id}` | disconnect; `purge=true` revokes and deletes |
| P2c | GET, PATCH | `{L}/calendar/accounts/{account_id}/sources` | calendars to sync |
| P2c | POST | `{L}/calendar/accounts/{account_id}/sync` | sync now (job) |
| P2c | GET | `{L}/calendar/events` | events in a range |
| P2c | GET | `{L}/calendar/matches` | suggested matches with reasons |
| P2c | POST | `{L}/calendar/matches/{match_id}/accept` | accept |
| P2c | POST | `{L}/calendar/matches/{match_id}/reject` | reject |
| P2c | GET | `{L}/holidays` | holidays for a year |
| P2c | GET | `{L}/memories` | memories |
| P2c | GET, PATCH | `{L}/memories/{memory_id}` | read; title, save, dismiss |
| P2c | POST | `{L}/memories/{memory_id}/photos/remove` | drop photos |
| P2c | GET, PUT | `{L}/memories/{memory_id}/timeline` | slideshow timeline |
| P2c | POST | `{L}/memories/build` | build now (job) |
| P2c | POST | `{L}/memories/from_prompt` | prompt-built memory (job) |
| P2c | GET | `{L}/featured` | featured photos |
| P2c | GET, POST | `{L}/music` | tracks; upload with licence fields |
| P2c | DELETE | `{L}/music/{track_id}` | remove a track |
| P2c | POST | `{L}/memories/{memory_id}/renders` | render to MP4 (job) |
| P2c | GET | `{L}/renders/{render_id}` | status |
| P2c | GET | `{L}/renders/{render_id}/file` | the MP4 (range requests) |

Reused unchanged: `{P}/map`, `{P}/map/cells`, `{P}/near_duplicates/*`, `{P}/search/*` (N9),
`{P}/images/{image_id}/file`, `{P}/images/{image_id}/thumbnail`, `{P}/jobs/{job_id}`, `{P}/location_policy`.

---

## 8. Cropwright information architecture

### 8.1 Where the Library lives

For a library project, `frontend/src/lib/projectPaths.ts` `SECTION_TABLE` shows a `library` group first
(only when the served project `kind` is `library`):

```
Library | Workspace | Data v (Ingest, Classes) | Data quality | Model v | Settings
```

The Library page has a left sidebar (`ui/SidebarNav.svelte`, the 12f P11 pattern); entries appear with
their phase:

```
Library (P2a-1)     Collections (P2a-1)   People (P2a-2)   Curate (P2b)       Create (P2c)
  Photos              Albums                People            Exclusions         Memories
  Favourites          Smart albums                            Review people      Events and trips
  Places and tracks   Keywords                                Stacks, Culling    Calendars
  Search              Recently added
Utilities: Duplicates, Hidden, Archived, Trash; Metadata (report, time fixes, sidecar export: P2a-0)
```

Routes: `frontend/src/routes/p/[project]/library/...`; URL grammar `frontend/src/lib/library/libraryUrl.ts`
(pattern of `workspaceUrl.ts`).

### 8.2 Moving between the library and labelling or training

| From | Action | To |
|---|---|---|
| Library selection, album, smart album or search | **Send to project…** (P2a-2) | dialog: new or existing workspace project, served dry-run counts (photos, items, classes by name, photos left out by rules); a job; then the target Workspace |
| Library viewer | **Show objects** | the same project's Workspace Gallery, image scope |
| Workspace image details (workspace project) | **Show in library** (when a library holds the content id) | the library viewer via `{L}/photos/by_content/{image_id}` |
| Project switcher | libraries first, with a photo icon | |

### 8.3 Reused components

`ui/Lightbox.svelte`, `ui/SidebarNav.svelte`, `ui/Menu.svelte`, `ui/Popover.svelte`, `ui/Drawer.svelte`,
`ui/Tabs.svelte`, `ui/Segmented.svelte`, `ui/EmptyState.svelte`, `ui/Skeleton.svelte`, the contextual
toolbar pattern of `components/workspace/BulkBar.svelte`, `components/map/**` and
`lib/map/mapController.svelte.ts`, `lib/selection.svelte.ts` (extended), `lib/jobs/followJob.ts`, `lib/sse.ts`,
`components/review/**` patterns, `components/nearDup/NearDupGroupModal.svelte`. The 12f design system applies.

---

## 9. UX standards to meet

| Area | Standard | Source |
|---|---|---|
| Grid | justified layout by row height (`justified-layout`, MIT, geometry only); months as sections; section-level virtualisation with `virtua/svelte` (MIT, Svelte 5); at most about 600 tiles in the DOM | 01 section 2; <https://github.com/flickr/justified-layout>; <https://github.com/inokawa/virtua> |
| Scrubber | year/month scrubber from bucket counts; date bubble while dragging; jump loads only the target section | Immich, Google Photos (01) |
| No scroll jump | scroll anchoring when a section above the viewport changes height | Immich PR 16305 (01) |
| Placeholders | ThumbHash, then grid tier, then preview in the viewer | <https://github.com/evanw/thumbhash> |
| Keyboard | `?` overlay; arrows; Space or `x` select; Shift range; Ctrl/Cmd+A; Enter open; Esc; `F` favourite; `P`/`X`/`U` pick, reject, unflag; Del trash; `[`/`]` in the viewer; bound through `contracts/json/keymap_actions.json` | Lightroom, Immich (01) |
| Selection | ranges and queries, not loaded ids; date headers select a day or month server-side; counts from a dry run | 01 section 2; 12f P10 |
| Viewer | full screen, filmstrip, served info panel, zoom, prev/next in context, faces on request (P2a-2) | Apple, Immich |
| Mobile | 430 px: two columns, thin scrubber, bottom bulk bar | 12f section 6 |
| Accessibility | WCAG 2.2 AA; tiles are buttons with served alt text | 12f P9 |

---

## 10. Performance budgets

p95 server times, warm, one API replica; client times on the 1600x1000 profile. The bench header records
the hardware, OpenSearch heap and load average; three repetitions. 500k rows on the real-EXIF set; 5M rows on
the **scale replica** (section 11.6), a latency claim only, never a quality claim.

| View or call | 500k | 5M |
|---|---|---|
| `{L}/timeline/buckets` (month) | 150 ms | 400 ms |
| `{L}/timeline/photos` (one month, 200 tiles) | 120 ms | 250 ms |
| Scrub to any month → thumbnails painted (client) | 1.0 s | 1.5 s |
| First timeline paint (client) | 1.5 s | 2.0 s |
| Grid thumbnail fetch | 15 ms | 15 ms |
| Preview miss / hit | 400 / 20 ms | 400 / 20 ms |
| Scroll | 60 fps; heap ≤ 300 MB after 50k photos | same |
| `{L}/search` text + 3 filters, page 100 | 300 ms | 600 ms |
| `{L}/smart_albums/preview` | 300 ms | 800 ms |
| `{L}/facets` (6 facets) | 250 ms | 600 ms |
| `{P}/map/cells` with a library filter | 300 ms | 700 ms |
| Bulk annotation, 200 photos (sync path) | 500 ms | 500 ms |
| Metadata read with `-stay_open` (P2a-0), photos | registered by L-M4 before the run (target ≥ 50 files/s per worker) | same |
| `{L}/tracks/{track_id}` (simplified line + photos along it) | 200 ms | 300 ms |
| New photo visible after ingest | 30 s | 30 s |
| `{L}/people` (P2a-2) | 200 ms | 400 ms |
| `{L}/people/{person_id}/photos` page (P2a-2) | 150 ms | 300 ms |
| New face assigned after ingest (P2a-2) | 60 s | 60 s |
| Merge of a 20k-photo person visible in lists (P2a-2) | 5 s (id expansion) | 5 s |
| Drain normalisation after that merge | 60 s | 60 s |
| Exclusion predicate overhead (P2b) | ≤ 10 % | ≤ 10 % |
| Memories nightly build (P2c) | 10 min | 60 min, resumable |
| Render 60 s 1080p (P2c) | 60 s GPU / 180 s CPU | same |

---

## 11. Testing strategy

### 11.1 Levels

| Level | What | Where | When |
|---|---|---|---|
| Unit | rule grammar, visibility, bucket math, positions, events, scoring, MMR, timeline EDL, beat snapping, ICS, OAuth URL and PKCE, token crypto | `tests/domain/`, `tests/library/` | lane, red first |
| Service with fakes | against the existing OpenSearch and project fakes | `tests/library/` | lane |
| Guard and isolation | faces role isolation (section 2.3); library routes refuse workspace projects; no source writes (AST) | `tests/curation/cross_project_leak/`, `tests/library/`, `tests/test_library_no_source_writes.py` | lane |
| Database | migrations, RLS, CHECKs, repos | `tests/db/` | wave sweep |
| Contract | `make contracts-check` | `contracts/` | lane and sweep |
| Live | ingest → projection → views; regroup keeps corrections; forget to zero; trash empty with the source checksum manifest; calendar against a recorded fake | `tests/live/library/` | wave sweep |
| Stubbed e2e | grid DOM cap, scrubber, no scroll jump, keys, range selection, one primary per view | `frontend/e2e/stubbed/library/` | lane |
| Browser walk | screenshots at 1600 and 430 px, each viewed | coordinator | phase exit |
| Scale | budgets of section 10 | `scripts/bench/library_*.py` | phase exit |

### 11.2 Lane checks

Every lane runs, and pastes the output of, its verify command from its phase file (targeted pytest paths,
`.venv/bin/python scripts/codegen/generate_contracts.py --check`, `npm run check` and the named vitest or
stubbed e2e files for frontend lanes, pre-commit on touched files). Red first: each behaviour test is watched
failing before the change, and the hand-back names the failing test ids. Integration tests are listed as
"pending wave run".

### 11.3 Privacy and source safety

No face, position, calendar text or token in logs (log capture); purges return counts and a follow-up count
of 0; hosted VLM endpoints refused for library data unless Q5 allows text; a checksum manifest of the test
source tree before and after every live library test.

### 11.4 Statistics

Every rate bar reports n, the point estimate and the one-sided 95 % Clopper-Pearson bound
(`scipy.stats.binomtest(...).proportion_ci(method='exact')`; SciPy is already a dependency). A bar passes
only when both the point estimate and the bound meet it.

### 11.5 Eval sets and what each proves

Access and licence read from each source page on 2026-11-13. "Local only" sets are never committed; only
counts and rates are.

| Set | Capture time / GPS | Licence and access | Proves |
|---|---|---|---|
| **Metadata test set** (P2a-0) | yes: real files | ExifTool's per-make metadata samples (JPEG, 7,119 models from 109 makers; no licence stated, so local only; [source](https://exiftool.org/sample_images.html)); GoPro sample videos with GPMF GPS from `gpmf-parser` (Apache-2.0 or MIT; [source](https://github.com/gopro/gpmf-parser)); Commons geotagged originals; drone, dashcam and phone video files only where a source page states a licence, else the owner's own files (numbers only) | P2a-0 parity, track and round-trip bars |
| **MIRFLICKR-1M** (Huiskes and Lew) | EXIF time and date for most images, GPS rare; EXIF distributed with the set | images under Creative Commons Attribution, credits in metafiles; usable for benchmarking ([source](https://press.liacs.nl/mirflickr/)) | P2a-1 timeline budgets and buckets on 500k real capture times; the base of the 5M replica |
| **YFCC100M** metadata + Multimedia Commons pixels | date taken for all, longitude/latitude for about 48M, user id per photo | metadata under the Yahoo Webscope agreement (accepted by the coordinator before use; local only); pixels in the AWS `multimedia-commons` bucket (`--no-sign-request`); each photo under its uploader's CC licence ([AWS registry](https://registry.opendata.aws/multimedia-commons/), [paper](https://arxiv.org/abs/1503.01817)) | P2a-1 Places at scale (country agreement against the metadata's Places expansion labels), P2c events and trips on real per-user streams (50 streams labelled by a person) |
| **Wikimedia Commons geotagged originals** | real camera EXIF time and GPS (the existing fetcher keeps only files whose EXIF GPS matches the page) | per-file CC0, PD, CC BY, CC BY-SA; `scripts/datasets/fetch_geotagged_commons.py` | P2a-1 Places sample (fallback when YFCC terms are not accepted; 1,000 photos, human-checked country), map, single-author event series |
| **Commons face set** (CC-licensed identities with many photos, including non-frontal; being built locally; P2a-2 lane L-PE builds it if absent) | as Commons | per-file CC; identities from Commons person categories; local only | P2a-2 recall report, P2b recall bars, P2c seeded exclusions (photos interleaved into the 500k library by shifting them into real events) |
| **LFW** | none | no formal licence; local only (`scripts/datasets/fetch_face_proxy.py`) | P2a-2 precision on held-out identities; threshold fitting on the other identities |
| CALFW, CPLFW | none | used only if a licence is found on the source; the repo's fetcher already rejects CPLFW for having none | pose and age bins (optional) |
| Open Images V7 "Human face" boxes, COCO | **none** | CC BY 4.0 annotations, CC BY 2.0 images | detector recall and object classes only; never time, place or event bars |
| Photo triage benchmark (Chang et al. 2016) | none | terms checked before use; local only; if they forbid it, 500 series of the owner's own picks | P2b top pick |
| AVA | none | research only; evaluation only (90 section 4, E2) | aesthetic rank correlation |
| Owner's own library | real | owner's | the final gate of every phase; numbers only |

Where downloaded copies lack EXIF, the build step writes the source's own metadata values into the local
copies ("metadata-restored EXIF"): the values are real, and the step is recorded in the manifest.

### 11.6 The 5M scale replica (lane L-S1, P2a-1)

`scripts/bench/library_build_scale_replica.py` builds a separate bench library project from the 500k real
docs: ten copies with new content ids, times shifted by whole years per copy, the same vectors (and faces,
re-run after P2a-2), written through the normal bulk paths. The coordinator runs it only on the bench stack
under its own compose project name; every report labels the result "replica".

---

## 12. Risks

| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| R1 | **Exclusion misses a photo** (side view, back of head, small face, no face) | high / high | candidate band counted as excluded; fail closed on unprocessed photos; exemplar growth; measured body link; coverage and audit; recall bars with bounds; the UI states the limit |
| R2 | **Scale** at 5M | medium / high | size classes and `_split`; selected crop vectors; photo-first projection; merged-id expansion; budgets on the replica before exit |
| R3 | **Google Calendar must stay connected** | high / medium | owner-created Desktop client published "In production" (Q8); PKCE loopback plus paste-back; 15-min sync keeps the token in use; `invalid_grant` → reconnect prompt; ICS fallback (30 section 3) |
| R4 | P1 needs slip or are renamed | medium / high | provisional N-table confirmed by the engine owner before P2a-1; no stand-ins |
| R5 | Face model licence (SCRFD/ArcFace weights non-commercial research only) | certain / medium | `ATTRIBUTION.md`; Q3; `model` on every face so a swap re-embeds and regroups |
| R6 | Mixed concerns in a library project | medium / low | library labels framed as machine labels; custom schemes in derived projects |
| R7 | Frontend performance of the virtualised grid | medium / medium | proven libraries; DOM-cap test; a stubbed 5M scroll test before features |
| R8 | Biometric and calendar privacy; in some jurisdictions faces of third parties are special-category or biometric data (for example GDPR Article 9, Illinois BIPA) | low / high | faces on by default in library projects (owner decision), with a per-library switch, purge and "forget this person"; operator responsibility stated in the user docs; encrypted tokens; local VLM default; adversarial review of purge, forget and OAuth |
| R9 | Memories feel generic | medium / medium | published selection methods, rubric, editable titles, prompt-built memories |
| R10 | Render licences | medium / medium | image test fails on `--enable-nonfree` in `ffmpeg -buildconf` and records the build in `ATTRIBUTION.md`; user-supplied music with licence fields |
| R11 | Derivatives disk fills or is lost | medium / medium | N3: atomic writes, capped previews, `503 derivatives_full`, grid tier rebuildable by backfill (not backed up) |
| R13 | Video capture time ambiguity: QuickTime integer dates should be UTC but many cameras store local time | high / medium | prefer tags that carry an offset (for example `Keys:CreationDate`), then the GPS UTC of the file's own track, then a per-make rule table (`src/domain/quicktime_time_rules.py`) deciding whether to read with `-api QuickTimeUTC`; every rule has a fixture; unknown makes stay floating local time with `offset_source = 'none'` |
| R14 | Tracks are dense position data | medium / high | location policy applies (`drop`: no tracks; `store_coarse`: rounded and thinned); `location_purge` clears tracks; never in logs; never sent to a VLM |
| R12 | Token key loss or rotation | low / medium | Fernet key file in the library secrets directory, `MultiFernet` rotation with `secret_key_id`; database backups hold only ciphertext; restore without the key sets accounts to `needs_reauth` (fail closed) |

---

## 13. Traceability

### 13.1 Professional bar (01 section 1.3)

| # | Feature | Phase | Plan section |
|---|---|---|---|
| 1 | Instant timeline: justified grid, sections, scrubber, placeholders, no scroll jumps | P1 (N3), P2a-1 | 00 sections 9-10; 10 |
| 2 | People: precise clusters, bulk confirm/reject, merge, hide, name, incremental assignment, review queue | P2a-2 ("me" later) | 15 |
| 3 | Combined search (text, people, place, date, OCR, camera) with facets | P1 (N5, N9), P2a-1; person terms P2a-2 | 10 section 3; 15 |
| 4 | Nested all/any/none smart albums; people-driven live albums; live updates | P2a-1; person fields P2a-2 | 10 section 3.3; 15 |
| 5 | Duplicates and bursts in stacks with top pick, keep top | P2a-1 (Duplicates view), P2b | 10; 20 section 3 |
| 6 | Culling vocabulary and assisted culling | P2a-1 (pick/reject), P2b; stars out, colour labels later | 10; 20 section 4; 00 section 2.2 |
| 7 | Feature less/never; block vs show less; reset; utility filtering | P2b, P1 (N6), P2c | 20 section 2; 30 section 4.2 |
| 8 | Memories beyond "on this day" with music-synced video | P2c (stills), P4 (clips) | 30; 40 |
| 9 | Safety of the originals: trash, hidden, in-place | P2a-1 | 10; rule 9 |

### 13.2 Features the owner may have missed (01 section 1.4)

| Feature | Phase | Plan section or reason |
|---|---|---|
| Utility / screenshot classifier | P1 (N6) | 2.1; 30 section 4.2 |
| Stacks with top pick; RAW+JPEG and burst pairing | P2b | 20 section 3 |
| Pick/reject flags | P2a-1 | 10 |
| Stars | out (2.2) | favourites, flags and implicit signals cover it |
| Colour labels | later | 2.2 |
| Hierarchical keywords and tags; XMP import and sidecar export | P2a-0 (data), P2a-1 (UI, rules) | 05; Q6 decides where sidecars may be written; originals never modified |
| Nested smart albums, saved searches | P2a-1 | 10 section 3.3 |
| Date/time/GPS correction, time-zone shift, GPX | P2a-0 (data, batch shift), P2a-1 (UI) | 05; 10 |
| Trips; Recently added | P2c; P2a-1 | 30 section 2; 8.1 |
| Birthday memories | P2c from calendar birthday events; person birth date later | 30 section 4.1; 2.2 |
| GPS tracks from videos and GPX on the map; synced video playback | P2a-0 (data), P2a-1 (map, time slider), P3 (synced player) | 05; 10; 40 |
| Holiday-aware memories, feature a holiday less | P2c, P2b rule kind | 30; 20 section 2.2 |
| OCR overlay with copy | later, optional | 2.2 |
| Quality review queue | P2b (opt-in filter, off by default) | 20 section 4 |
| Folder view | later, optional | 2.2 |
| Live albums by people | P2a-2 (person rule fields) | 15 |
| Admin view of queues, re-run per signal, model choice | engine (P1) and existing Jobs routes | coverage per signal in `{L}` |
| Per-signal provenance | P1 fields, served in photo detail (P2a-1) | 6.2 |
| Query by example, more like these | P2a-1 (`example_image_id`) | embedding export not planned: the training export covers datasets |
| Fairness checks per subgroup | P1 and P2b eval where labels allow | no demographic inference is added |

### 13.3 Other 01 inventory rows

| Feature | Phase | Plan section or reason |
|---|---|---|
| Archive | P2a-1 | `lib.archived` |
| Locked folder | P2a-1 Hidden view | a password lock needs authentication (Q2) |
| Places map and names | P2a-1 | 10 |
| Pets | P2a-1 as object classes | named-pet identities not planned: no pet re-identification model |
| Never feature a person, hide dates | P2b | 20 |
| Prompt-built memory | P2c | 30 section 4.6 |
| Sharing, editing, automation workflows | not planned | out of scope (01); smart albums and rules cover rule-driven cases |
| Video (01 section 6) | P3, P4 | 40 |

### 13.4 Gap rows (02 section 3)

| Gap | Resolved by |
|---|---|
| Timeline buckets, locate, virtualised grid, persistent thumbnails | N3; 10 |
| Albums | 10 |
| Smart albums and the 10k join cap | photo-first projection (3.1); 10 |
| People in a library scope; People UI | 2.3 (faces role), N1; 15 |
| Merge/split, not this person | 15 |
| Hide a person everywhere | 20 (levels); P2a-2 `persons.hidden` for the People list |
| Places | 10 |
| Favourites, hidden, trash | 10 |
| Duplicates library framing | 10 (view), 20 (stacks) |
| Combined search | N9; 10; 15 |
| Memories, calendar | 30 |
| Exclusions inherited by new photos | 20 |
| Users | Q2 |
