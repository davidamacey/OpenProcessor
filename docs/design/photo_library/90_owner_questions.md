# Photo library: owner questions

Status: 2026-11-13, revised after the owner's decisions of the same day. Section 1 holds the decisions
(all answered). Section 2 records what the owner has decided. Section 3
lists what the plan settled from the docs (the owner may overrule any of it). Section 4 holds two questions
for the engine (P1) owner, not for this list.

## 1. Decisions (all 8 answered by the owner on 2026-11-13)

| # | Owner's answer |
|---|---|
| D1 | **A for storage with D's workflow**: a project with `kind = 'library'`; training in workspace projects via "Send to project" |
| Q2 | **(a)** one shared view per library |
| Q3 | **(a)** ship People on the current face models, stated in the docs; research a permissive replacement; swap by re-embedding |
| Q4 | **`block` by default, `hide` one click away**; under `hide`, unprocessed photos stay hidden until processed |
| Q5 | **(a) local by default, (b) text-only hosted titles as a per-library opt-in** |
| Q6 | **(a) mirror folder by default, (b) next to the originals as a per-library opt-in** |
| Q7 | **Own tracks added in the UI, plus a public-domain/CC0 starter set like Apple Photos' memory music**; a licensed music-service API (e.g. Epidemic Sound, Artlist) or AI music is a later advanced feature (30 section 5.4) |
| Q8 | **(a)** the operator's own Desktop-app client, published "In production"; ICS URL as the documented fallback |

The tables below keep each question's context and options.

### Library shape

| # | Question (one line of context) | Options | Recommendation | Consequence |
|---|---|---|---|---|
| **D1** | Where does the library live? A project of a new kind, or a new storage family ([00](00_photo_library_plan.md) section 4) | A one library project; B a layer over many projects; C a new library index family; D library as the source, projects derived | **A for storage with D's workflow**: a project with `kind = 'library'`; training in workspace projects filled by "Send to project" | Guard, row-level security and the pipeline are reused; P2a-0 can start at once. C or D add a storage lane and a guard review first |
| **Q2** | Are favourites, albums, hidden people and exclusions per user? There is no login today | (a) one shared view per library; (b) per user, after adding authentication | **(a)**; a person who wants a private view gets a separate library over the same folders | (b) needs an authentication project first |

### Faces and privacy

| # | Question | Options | Recommendation | Consequence |
|---|---|---|---|---|
| **Q3** | The face models' weights are licensed for non-commercial research only. Ship People on them? | (a) yes, stated in the docs, while a permissively licensed replacement is researched; (b) wait for a replacement | **(a)**; every face records its model, so a swap re-embeds and regroups without a schema change | (b) delays People and all of curation |
| **Q4** | Default level of a new exclusion, and photos whose faces are not processed yet | level `block` (never in memories or slideshows) or `hide` (also gone from the timeline and search); unprocessed photos under `hide`: hidden until processed, or shown | **`block` by default, `hide` one click away; under `hide`, unprocessed photos stay hidden until processed** | `hide` users may briefly miss new photos while faces run; the coverage view shows how many |
| **Q5** | May library text (dates, place names, people names, calendar titles) go to a hosted VLM for titles? Images never do | (a) never: local VLM or template titles; (b) text only, per-library opt-in | **(a) by default, (b) as an opt-in setting** | (a) keeps "library data never leaves the host"; without a local VLM, titles come from the calendar, holidays and templates |

### Metadata and outputs

| # | Question | Options | Recommendation | Consequence |
|---|---|---|---|---|
| **Q6** | Where may XMP sidecars be written? Originals are never modified either way | (a) only a mirror folder under the derivatives directory; (b) also next to the originals, per library, when the operator mounts the photo folders writable | **(a) by default, (b) as an opt-in** | (a) keeps the photo folders untouched; other tools find the sidecars only if pointed at the mirror. (b) is what Lightroom and darktable expect, but writes into the photo folders |
| **Q7** | Music for memory videos | (a) user-supplied tracks with licence fields; (b) also an optional fetch of a CC0/CC-BY starter set, credited on the end card | **(a) now, (b) later if wanted**; nothing can be bundled (Pixabay's licence forbids standalone redistribution, [summary](https://pixabay.com/service/license-summary/)) | the first video needs a track from the user; the UI says so |

### Calendar

| # | Question | Options | Recommendation | Consequence |
|---|---|---|---|---|
| **Q8** | Who owns the Google sign-in client? The connection must stay connected long term | (a) each operator creates a **Desktop app** client in their own Google Cloud project, publishes it **"In production"** and stays unverified (personal use, under 100 users, one "unverified app" screen at connect); (b) OpenProcessor publishes one verified client; (c) no Google sign-in, ICS secret URL only | **(a), with (c) as the documented fallback.** Read-only scopes; refresh token stored locally and encrypted; incremental sync tokens; a revoked or expired grant (`invalid_grant`) shows "reconnect" and a re-consent. Verified on 2026-11-13: 7-day refresh tokens apply only while the consent screen is in "Testing" ([OAuth 2.0](https://developers.google.com/identity/protocols/oauth2)); personal-use apps may stay unverified ([verification not needed](https://support.google.com/cloud/answer/13464323?hl=en)); Desktop clients use a loopback redirect with PKCE ([installed apps](https://developers.google.com/identity/protocols/oauth2/native-app)) | (a) costs each operator a 10-minute documented setup. (b) needs a privacy policy, a verified domain and ongoing Google review. (c) is a polled snapshot without birthdays as event types |

## 2. Decided by the owner (2026-11-13)

| Decision | Where it lands |
|---|---|
| Faces are **on by default** in library projects (off in workspace projects), with the per-library switch, purge and "forget this person" | 00 sections 4 and 6; 15 |
| Excluded (`block`, `hide`) photos are **left out of Send to project**; existing workspace projects are not changed | 20 section 2.1; 15 section 3 |
| **Emptying the trash never deletes an original**; it removes the library's records and derivatives | 10 section 3.2; 00 rule 9 |
| The first **"AI videos" are photo slideshows** rendered to MP4; montages with video clips are P4 | 00 section 1; 30; 40 |
| **P2a is split**: P2a-1 timeline, albums, smart albums, Places; P2a-2 People | 00 section 2; 10; 15 |
| **Metadata is in scope** with one ExifTool reader for photos and videos, GPS tracks from videos and GPX, capture-time and other edits kept in the database, XMP/IPTC import, XMP sidecar export; placed in its own phase **P2a-0** before P2a-1 | 00 section 2; 05 |
| **Favourites**: a heart and a Favourites album; favourites feed memories and key-photo choice | 10; 30 section 4.4 |
| **Star ratings are out**: no concrete benefit was found over favourites, pick/reject flags and the implicit signals (keep-top choices, culling signals), so no question is asked; ratings in imported XMP are not stored | 00 section 2.2 |
| Subject-aware framing of memories (smart crop per aspect, pan and zoom on the subject, matched cuts) | 30 section 5.2 |
| The other defaults in this plan are accepted | |

## 3. Settled from the docs (the owner may overrule)

| # | Question (source) | Answer | Basis |
|---|---|---|---|
| S1 | One library or many? | many allowed; one recommended per household | D1; `src/services/projects/capacity.py` |
| S2 | How is "home" known for trips? | suggested from the most frequent night-time place cell, confirmed once | 30 section 2 |
| S3 | Reverse geocoding data (02 section 6.4) | GeoNames `cities500` and admin codes (CC BY, credited), nearest lookup with SciPy; place names follow the location policy | [GeoNames export](https://www.geonames.org/export/); `reverse_geocoder` not chosen: LGPL-2.1, last release 2016 |
| S4 | Thumbnails at ingest or lazily? (02 section 6.7) | placeholder and grid thumbnail at ingest (about 125 GB at 5M est.), previews lazily into a capped store | 00 section 3.3, N3 |
| S5 | Licences to clear (01 section 7.6) | Beat This! MIT (README, 2026-11-13); SSCD and MagFace not used; Q-Align evaluation only; `justified-layout`, `virtua`, `thumbhash` MIT; smartcrop.js MIT (read, not used as code); FFmpeg build checked by an image test that fails on `--enable-nonfree`; ExifTool under the Perl terms; PyExifTool's BSD option | 00 sections 5b and 9; 30 section 5.3; 05 section 2 |
| S6 | Google push notifications? | not used (they need a public HTTPS endpoint); a jittered 15-minute sync-token poll | [Calendar sync guide](https://developers.google.com/workspace/calendar/api/guides/sync) |
| S7 | Calendar descriptions and attendees | descriptions only when a setting is on (default off); attendees never stored | data minimisation, 01 section 4.1 |
| S8 | Hide low-quality photos by default? | no; an opt-in filter | 20 section 4 |
| S9 | Core `/persons` and library People | one face-identity service; the old union-find grouping is deleted | no duplicate logic; 00 section 5.3 |
| S10 | Synced video playback with a moving map marker | P3 (needs playable proxies); P2a-1 shows tracks on the map with a time slider | 00 section 2.2; 40 |

## 4. For the engine (P1) owner, not this list

| # | Question | Plan's suggestion |
|---|---|---|
| E1 | HEIC/HEIF and RAW in the first library release | HEIC yes; RAW through the embedded JPEG preview; full RAW decode later |
| E2 | Training data for the aesthetic head | prompt-pair baseline on PE vectors, then the owner's own favourites and pick/reject flags; AVA for evaluation only (research-only terms) |
