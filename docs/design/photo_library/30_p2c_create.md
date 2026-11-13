# P2c: Create (stills): calendar, memories, slideshows (phase spec)

| Field | Value |
|---|---|
| Status | plan, 2026-11-13. Starts after the P2b exit bar and the owner's answers to Q5, Q7 and Q8 ([90](90_owner_questions.md)) |
| Master plan | [00_photo_library_plan.md](00_photo_library_plan.md); builds on [05_p2a0_metadata.md](05_p2a0_metadata.md), [10_p2a1_organise.md](10_p2a1_organise.md), [15_p2a2_people.md](15_p2a2_people.md) and [20_p2b_curate.md](20_p2b_curate.md) (eligibility uses `created_clauses`) |
| Research input | [01_feature_research.md](01_feature_research.md) sections 3 (memories science), 3.7 (music and polish), 4 (calendar) |

Notation as in 10: `{L}` = `/curation/projects/{project}/library`. Video clips are P4; everything here renders
from stills.

---

## 1. Scope and non-goals

**In scope.** Event and trip segmentation; Google Calendar (required) with a long-lived local connection,
ICS file and secret URL (secondary), CalDAV (secondary, last lane); holidays; event-to-photo matching as
suggestions; calendar-aware smart-album rule fields; the memories engine (candidates, eligibility, scoring,
selection, chapters, titles and captions with a local VLM, featured photos, prompt-built memories); a music
library with licence fields; beat tracking; pan and zoom from saliency and faces; a declarative slideshow
timeline previewed in the browser and rendered to MP4 on the host.

**Non-goals.** Video clips in memories (P4); push notifications from Google (they need a public HTTPS
endpoint); mapping calendar attendees to People (01 section 4.1 says never automatic; a manual mapping is a
later item); sharing renders; a licensed music-service connection (later, Q7).

**Needs from P1.** N3 previews (render input), N5 OCR (theme and utility checks), N6 signals (aesthetic,
sharpness, exposure, utility class), N7 face signals (eyes open, quality). P2c does not start without N6.

---

## 2. Events and trips (lane L-C1)

`src/domain/library_events.py` (pure) and `src/services/library/events.py`.

| Step | Rule | Source |
|---|---|---|
| Order | photos by `capture.taken_utc` when known, else `taken_local` with the library's home offset | 01 section 4.3 |
| Time boundaries | PhotoTOC adaptive gap: a boundary after photo N when `log(g_N) ≥ K + mean(log g_{N+i}, i = −d..d)`, `d = 10`; `K` calibrated on the labelled sample (start from the paper's value; the lane cites it from the paper) | Platt et al. 2003 (01 section 3.1) |
| Place boundaries | split where consecutive photos are more than 50 km apart with a gap above 30 min | 01 section 3.1 |
| Small events | fewer than 3 photos join the nearer neighbour if within 2 h, else stay as singles (not memory candidates) | |
| Home | the most frequent night-time (22:00-06:00 local) place cell, suggested once and confirmed by the user into `library.home_place` | Apple "frequent places" (01) |
| Trips | consecutive events whose centroids are more than 100 km from home, spanning at least 2 local days with no home-cell photo between; a multi-day all-day calendar event overlapping them merges them | 01 section 3.1 |
| Incremental | after ingest, only the affected time window (±7 days of the new photos) is re-segmented; event ids are kept by member overlap (Jaccard ≥ 0.5), like person ids | |
| Writes | `library_events`, `library_event_members`; `event.event_id`, `event.trip_id` projected on images by the projection drain | |
| Location policy | `drop`: time boundaries only, no trips, no centroids; `store_coarse`: centroids rounded like the positions | `src/domain/location_policy.py` |

Rule grammar additions (`library_rule/1`, additive fields served by `{L}/rules/schema`): `event` (`in`),
`trip` (`in`), `calendar_event` (`in`, `summary_contains`), `holiday` (`in`, holiday names for the
configured country).

---

## 3. Calendar (lane L-C2)

### 3.1 Google Calendar: the long-lived connection

The connection must stay connected the way Apple Calendar and open-source sync clients stay connected.
Verified on Google's pages on 2026-11-13:

| Fact | Consequence | Source |
|---|---|---|
| A project whose consent screen is External and in **Testing** gets refresh tokens that expire after 7 days | never ship in Testing; the owner publishes the consent screen | [OAuth 2.0 overview, "Refresh token expiration"](https://developers.google.com/identity/protocols/oauth2) |
| A refresh token also stops when the user revokes access, when it is unused for six months, when the account exceeds its live-token limit, or when time-based access ends; the error is `invalid_grant` | handle `invalid_grant` as "needs re-consent"; the 15-minute sync uses the token, so it never idles six months | same page |
| A personal-use app (fewer than 100 users) may stay unverified; users click through the "unverified app" screen | the owner's own Cloud project, published "In production", unverified, is allowed for self-hosted use | [When is verification not needed](https://support.google.com/cloud/answer/13464323?hl=en) |
| Installed apps use the "Desktop app" client type, are public clients (the secret is not confidential), must use PKCE (S256), and redirect to `http://127.0.0.1:<port>` | client type Desktop; PKCE always; loopback redirect | [OAuth for iOS and desktop apps](https://developers.google.com/identity/protocols/oauth2/native-app) |
| Web-server redirect URIs must be HTTPS, cannot be raw IP addresses (localhost exempt) and need a public-suffix domain | a LAN address such as a private IP cannot be a redirect target; this is why the loopback plus paste-back design below is used | [OAuth for web server apps, redirect URI validation](https://developers.google.com/identity/protocols/oauth2/web-server) |
| The device flow allows only OpenID, Drive file/appdata and YouTube scopes | no device-code flow for Calendar | [OAuth for TV and limited-input devices](https://developers.google.com/identity/protocols/oauth2/limited-input-device) |
| Revocation: `POST https://oauth2.googleapis.com/revoke` with the token; it removes every scope granted to the project for that user | disconnect revokes, then purges | native-app page above |
| Incremental sync: full `events.list`, keep `nextSyncToken`, send `syncToken` with the same parameters, deleted entries included, 410 Gone means wipe and full sync | the sync algorithm below | [Synchronize resources efficiently](https://developers.google.com/workspace/calendar/api/guides/sync) |
| Quotas: 600 requests/min per user per project; randomise sync times ±25 %; back off exponentially (32-64 s cap) on 403/429 `usageLimits` | jittered tick, Celery retry with backoff | [Manage quotas](https://developers.google.com/workspace/calendar/api/guides/quota) |

**Setup (owner's production path, documented in the user docs by this lane):**

1. In the owner's own Google Cloud project: enable the Calendar API; consent screen user type External,
   app name, the two scopes below; **Publish app** so the status is "In production" (no verification
   needed for personal use; the unverified-app screen stays).
2. Create an OAuth client of type **Desktop app**; download its JSON.
3. Place the JSON as the secret file `google_oauth_client.json` in the library secrets directory
   (`OP_LIBRARY_SECRETS_DIR`, default `/run/secrets/op_library`, a read-only mount like the VLM secrets of
   `src/services/labeling/vlm_endpoints.py`), next to `token_key` (Fernet keys, one per line, newest first, generated by the installer). Rotation uses
   `MultiFernet` (decrypt with any listed key, encrypt with the first) and a re-encrypt job; `secret_key_id`
   records which key wrote each row. Database backups hold only ciphertext; a restore without the key file
   sets every account to `needs_reauth` (fail closed), and the user reconnects.
4. In Cropwright: Library › Calendars › Connect Google Calendar.

**Scopes (read-only, narrowest pair):** `https://www.googleapis.com/auth/calendar.calendarlist.readonly`
and `https://www.googleapis.com/auth/calendar.events.readonly` (01 section 4.1).

**Consent flow** (libraries: `google-auth-oauthlib` `Flow` with PKCE, `google-auth` credentials, both
Apache-2.0; the lane cites their docs for the pinned versions):

1. `POST {L}/calendar/accounts/google/start` creates `state = <project slug>.<random 32 bytes>` and a PKCE
   verifier, keeps both in Valkey for 10 minutes (shared state, not in a process), and returns the
   authorisation URL with `access_type=offline`, `prompt=consent`, `code_challenge_method=S256` and
   `redirect_uri = http://127.0.0.1:<Cropwright port>/`.
2. **Browser on the host:** Google redirects to Cropwright's root with `code` and `state`; the root page
   (`routes/+page.svelte` gains a guarded handler) posts the full URL to
   `POST {L}/calendar/accounts/google/complete` and navigates to Library › Calendars.
3. **Browser elsewhere on the LAN:** the redirect to `127.0.0.1` fails to load on that machine; Cropwright's
   connect dialog says so in advance and offers a field: "paste the address from the browser's address bar".
   The same `complete` route takes it. PKCE and `state` make the pasted URL useless to anyone else.
4. `complete` exchanges the code, checks the granted scopes, encrypts the refresh token with Fernet
   (`cryptography`, Apache-2.0 or BSD) under `token_key`, stores it in `calendar_accounts.secret_ciphertext`
   with `secret_key_id`, lists calendars (`calendarList.list`) into `calendar_sources` (all disabled until the
   user picks), and never stores the access token.
5. Optional third path, documented only: an operator with a public-suffix domain over HTTPS may register a
   Web client instead; the same `complete` route works. Not required by any bar.

**Sync**: a periodic tick every 15 minutes (advisory class `LIBRARY_CALENDAR_SYNC`) enqueues the job
`library_calendar_sync` (queue `library`, `exclusive_group='library_calendar'`, `exclusive_scope='account_id'`,
so one active sync per account) for each account due, with ±25 % jitter per account; per enabled source: first a full `events.list` (`singleEvents=false`,
`showDeleted=true`, paged), then `syncToken` calls with the same parameters; recurring masters are expanded
locally over the photo date range with `dateutil.rrule.rrulestr` (python-dateutil), exceptions come as their
own instances with `recurringEventId`; cancelled entries delete local rows; `410` wipes the source's events and
re-runs the full sync; `403/429 usageLimits` retries with exponential backoff (Celery `autoretry_for`,
`retry_backoff`, cap 64 s).

**Revoked or expired grant.** A `RefreshError` carrying `invalid_grant` sets the account to
`needs_reauth`, stops its sync, writes an event on the events feed, and Cropwright shows a banner on the
Library and Calendars pages: "Google Calendar needs to be reconnected". `POST {L}/calendar/accounts/
{account_id}/reconnect` starts the same flow with `login_hint`; success replaces the token and keeps the
events and sync tokens.

**Disconnect.** `DELETE {L}/calendar/accounts/{account_id}?purge=true` revokes the token, deletes the
account's sources, events and matches, and answers with the counts; a follow-up count query returns 0
(test). `purge=false` keeps the events as a frozen snapshot and deletes only the secret.

**Stored fields only:** `summary`, `location` (text), `start`/`end` (date or dateTime with zone),
`recurrence`, `recurringEventId`, `originalStartTime`, `eventType`, `status`, `visibility`; `description`
only when the library setting `library.calendar_store_descriptions` is on (default off). Never attendees,
emails, conference data. Event text reaches a VLM only under Q5's answer and never for `private` events.

### 3.2 ICS and CalDAV (secondary)

| Path | Library | Notes |
|---|---|---|
| ICS file upload | `icalendar` (BSD) + `recurring-ical-events` (LGPL-3.0, used as a library) | a snapshot; `POST {L}/calendar/accounts/ics` with a file |
| ICS secret URL | same; polled on the sync tick with `ETag`/`If-Modified-Since` | the URL is a secret: encrypted like a token |
| CalDAV (iCloud with an app-specific password, Nextcloud, Fastmail) | `caldav` (its Apache-2.0 option) | last lane (L-C6); not in the exit bar |

### 3.3 Holidays and matching

| Concern | Design |
|---|---|
| Holidays | `holidays` (MIT, 01 section 3.8) for `library.holiday_country` and subdivision; `GET {L}/holidays?year=` |
| Birthdays | Google `eventType = birthday` (and ICS yearly events); a suggestion "Is this <person>'s birthday?" links the event to a person once (`calendar_events.person_id`, a human link; persons have no birth-date field in P2) |
| Matching | an event owns a library event when at least 60 % of the library event's photos fall inside its window (all-day: the local days in the event's zone); GPS agreement with the event's location text (offline lookup of place names in `geo_places`) raises the score; stored in `calendar_matches` as `suggested` with reasons (`time_overlap`, `place_agrees`); never applied silently; accepted matches give memory titles |
| Skipped event types | `focusTime`, `outOfOffice`, `workingLocation` (01 section 4.1); `fromGmail` kept as trip hints |

---

## 4. Memories engine (lane L-C3)

`src/services/library/memories/` with pure parts in `src/domain/memory_selection.py` and
`src/domain/memory_scoring.py`.

### 4.1 Candidates

| Kind | Generator | Minimum |
|---|---|---|
| `on_this_day` | photos from today ±3 days in earlier years (Immich's window) | 8 eligible photos |
| `trip` | each trip; resurfaced on anniversaries (1, 2, 5, 10 years) and within 30 days of its end | 15 |
| `event` | events with at least 30 photos and at least 1.5 × the library's median event size; every accepted calendar match | 15 |
| `person` | for important people (named, at least 200 photos): "<name> through the years", yearly slices; frequent pairs ("A and B") | 20 across 3 events |
| `holiday` | events overlapping a holiday of the configured country, across years | 10 across 2 years |
| `birthday` | photos of the linked person in the 3 days up to a linked birthday event, across years (Immich v3.3) | 8 |
| `theme` | prompt-pair themes over the PE frame vectors (beach, snow, sunsets, food, pets, concerts, mountains): radial k-NN with a calibrated `min_score` | 20 across 3 events |
| `prompt` | "make a memory of …": the local VLM turns the text into a `library_rule/1` document (JSON-schema constrained by the served rule schema) plus an optional semantic query; on failure the semantic query alone | 10 |

### 4.2 Eligibility (fail closed; one function, `eligible_clauses()`)

Built from P2b's `created_clauses(rules)` plus: not archived, hidden or trashed; `signals.utility_class =
none` (receipts, screenshots, documents never become memories); `faces.state = done`; not in
`library_projection_pending`; `source_state = present`; `signals.sharpness` above the culling profile's blur
threshold; stack tops only. A memory never contains a photo outside this set; the nightly build and every
render re-check it (a rule added after a memory was built removes the photo on the next read).

### 4.3 Scoring which memories to surface

`S = w1·log(n_eligible) + w2·importance(people) + w3·novelty + w4·occasion + w5·quality_top_k + w6·anniversary − w7·recently_shown − show_less_penalty`

| Term | Meaning |
|---|---|
| importance | sum over present persons of `log(1 + photo_count)` × 1.5 when named × 1.5 when often with "me" (Apple's "important groups", 01 section 3.5) |
| novelty | `away_from_home`, distinct place count, rarity of the place in the library |
| occasion | accepted calendar match, holiday, birthday |
| quality_top_k | mean `signals.aesthetic` of the best 10 eligible photos |
| anniversary | 1, 5, 10 years ago |
| recently_shown | decays with `seen_at` and dismissals of the same kind |
| show_less | person or place with a `show_less` rule as subject: the memory is not generated; as members: weight 0.2 |

Weights live in a config document kind `memories_profile` (revisioned, `src/services/config_store/`).
Surfacing: at most 3 new memories a day, at most one per kind, none with more than 50 % member overlap with a
saved or recently shown memory.

### 4.4 Selection inside a memory

1. Pool = eligible photos of the candidate; collapse stacks and duplicate groups to their tops.
2. Chapters = sub-events by the adaptive gap rule inside the memory (2-6 chapters; one for short pools).
3. Quota per chapter ∝ √(chapter size), total N from the length preset (short 30 s ≈ 12 photos, standard
   60 s ≈ 24, long 120 s ≈ 45) — cluster-then-cap (Google Clips, 01 section 3.4).
4. Within a chapter, greedy MMR (Carbonell and Goldstein 1998): `pick argmax q_i − λ·max_j sim(i, j)`,
   `λ = 0.5` default; `q_i = a·aesthetic + b·sharpness + c·prominence + d·favourite + e·pick`; favourites (the heart) are the
   strongest user signal: a favourite in the pool is always a key-photo candidate and the cover prefers one;
   `sim` = max of frame cosine, a time kernel (`exp(−Δt / 10 min)`) and the Jaccard of the people sets.
5. Prominence per photo (01 section 3.5): face area fraction, centrality, sharpness, frontal pose, eyes open,
   weighted by the person's importance; zero for blocked people (they are not eligible anyway).
6. Coverage: every important person in the pool appears at least once if they have an eligible photo
   (unless `show_less`).
7. Cover: the highest `q` with the subject's prominence, landscape preferred, faces inside the safe area.
8. Order: chronological; chapter order follows time.

The selection stores `reasons` per member (served), so the UI can say why a photo is in.

### 4.5 Titles and captions

Priority: user edit > accepted calendar summary (not `private`) > holiday name > birthday ("<name>'s
birthday") > local VLM > template ("<place>, <month year>").

| Concern | Design |
|---|---|
| VLM call | one per memory: input = structured facts (date range, place names, named people, chapter object classes, OCR snippets of signs) and up to 6 key photos at 768 px; output = JSON (`title`, `subtitle`, `chapters: [{title, caption}]`) validated by a schema; one retry, then the template |
| Endpoint | the project's active VLM activation (`src/services/config_store/`), only when its endpoint is local (`allow_external = false`); images never go to a hosted endpoint; text facts only if Q7 allows it |
| Prompt | a new prompt-pack kind `memory_titles` (editable, revisioned, like the labelling packs) |
| Provenance | `title_source`, the pack revision and the model in `memories.reasons` |

### 4.6 Featured photos and prompt-built memories

`GET {L}/featured`: 6 photos a day from earlier years, eligible, highest `q` × importance, MMR-diverse,
not featured in the last 90 days. `POST {L}/memories/from_prompt`: section 4.1 `prompt` kind as a job.

---

## 5. Slideshow timeline and rendering (lane L-C4)

### 5.1 Timeline (EDL) `memory_timeline/1` (`src/domain/memory_timeline.py`, pure)

```json
{"version": 1, "preset": "landscape_1080p", "fps": 30,
 "music": {"track_id": "…", "start_s": 0.0, "gain_db": -3, "fade_in_s": 1.0, "fade_out_s": 2.5},
 "title_card": {"title": "…", "subtitle": "…", "duration_s": 3.0},
 "items": [{"image_id": "…", "start_s": 3.0, "duration_s": 2.4, "transition": "dissolve",
            "transition_s": 0.5, "motion": {"from": [0.0, 0.05, 1.0, 0.95], "to": [0.08, 0.1, 0.92, 0.9]},
            "chapter_title": null, "caption": null}],
 "end_card": {"lines": ["Music: <title> by <artist>, <licence>"], "duration_s": 3.0}}
```

| Step | Rule | Tool |
|---|---|---|
| Beats | beats and downbeats per track, computed once at upload, stored as JSON (`beats_key`) | **Beat This!** (code and weights MIT per its README, verified 2026-11-13; CPU or GPU) <https://github.com/CPJKU/beat_this> |
| Cut placement | hold = 4 beats under 110 BPM, else 8 (about 2-2.5 s); chapter changes on downbeats; each cut snapped to the nearest beat, then to the frame grid (33 ms at 30 fps) | |
| Transitions | cut on beat; 0.5 s dissolve where onset strength is low (slow sections); no wipes | 01 section 3.7 |
| Pan and zoom | see section 5.2 (subject-aware framing) | |
| Colour | per-photo exposure and white-balance normalisation clamped to ±10 % toward the chapter median; an optional look (neutral default, warm, film) as `.cube` LUTs written by this project | |
| Titles and captions | rendered with Pillow to RGBA overlays in Inter (OFL, already bundled for Cropwright) | |
| Audio | fades from the timeline; ends on a downbeat; video length snaps to the music phrase | |

### 5.2 Subject-aware framing: smart crop, pan and zoom targets, matched cuts

The stored boxes (faces from P2a-2, person and object boxes from the detector) frame every shot; saliency is
added only where boxes are absent.

| Step | Rule | Prior art and tools |
|---|---|---|
| Subject box per photo | union of face boxes (weighted by the person's importance and prominence), else person boxes, else the largest object boxes, else a saliency map's main region; heads: a face box is grown 40 % upward and 20 % sideways to cover hair and chin | boxes are ours; saliency **U-2-Net** small (`u2netp`, Apache-2.0, 4.7 MB) on ONNX Runtime CPU, only for photos without boxes |
| Smart crop per output aspect | for each preset (16:9, 9:16, 1:1) choose the crop of that aspect that contains every grown face box with a 5 % margin and maximises the subject box coverage, then centres the remaining slack by the rule of thirds; when no crop of that aspect can hold all faces, the photo sits whole on a blurred fill (never cut a head) | content-aware cropping as in smartcrop.js (MIT; boost regions for faces; [source](https://github.com/jwagner/smartcrop.js)) and Google AutoFlip's per-shot crop paths (01 section 3.7); Twitter's study recommends against pure saliency crops for fairness reasons ([Yee et al., arXiv 2105.08667](https://arxiv.org/abs/2105.08667)), so detected boxes lead and saliency only fills gaps. The crop search is a small function over our boxes; smartcrop's own edge and skin heuristics are not used |
| Pan and zoom targets | start and end rectangles inside the smart crop: zoom at most 1.12 toward the subject box (or out from it on alternate shots); every grown face box stays inside both rectangles | Ken Burns convention (01 section 3.7) |
| Matched cuts | consecutive shots in a chapter are ordered (within the selection's time order constraints) so that the subject's centre and size line up; where two neighbours are close, the end rectangle of shot A and the start rectangle of shot B are solved together so the subject's normalised centre and height match across the cut | frame matching of Netflix's match-cutting work (shots aligned by the framing of a person; [blog](https://netflixtechblog.com/match-cutting-at-netflix-finding-cuts-with-smooth-visual-transitions-31c3fc14ae59), [Chen et al., WACV 2023, arXiv 2210.05766](https://arxiv.org/abs/2210.05766)) |
| Checks (exit bar) | (a) in 40 rendered memories, no grown face box with side ≥ 24 output pixels is cut by the frame edge in any frame (projected through the timeline's rectangles; hard zero); (b) across cuts marked matched, the subject centre moves ≤ 0.05 of the frame width and the subject height ratio stays within 0.85-1.18 on ≥ 90 % of pairs (n ≥ 200 pairs, lower bound ≥ 0.86) | `scripts/bench/library_eval_framing.py` (lane L-C5) |

### 5.3 Renderer

| Concern | Design |
|---|---|
| Composition and encode | **MoviePy v2** (MIT) composites from the timeline and encodes through the image's system FFmpeg (`h264_nvenc` when a GPU is assigned, else `libx264`), selected by setting MoviePy's FFmpeg binary to the system one |
| Prototype first | before committing, the lane times MoviePy and a pure FFmpeg filtergraph (`zoompan`, `xfade`, `overlay`) on the same timeline in `artifacts_local/` and keeps only the one that meets the budget of 00 section 10; the other is never committed |
| Worker | a `library-render` target in the root `Dockerfile` (multi-stage, slim base, non-root) with FFmpeg from the distribution packages; an image test runs `ffmpeg -buildconf` and **fails if `--enable-nonfree` is present** (a nonfree build may not be redistributed in the published image), records whether `--enable-gpl` is set and which libraries are linked, lists the NVENC encoders, and the lane adds the build's licence (LGPL or GPL, both compatible with this AGPL-3.0 repository) to `ATTRIBUTION.md`; the release checklist's Trivy scan covers the image; compose service `library-render` (profile `library`, queue `media_render`, GPU from `.env` `RENDER_GPU_ID`, empty for CPU) |
| Output | `OP_DERIVATIVES_ROOT/renders/<project>/<render_id>.mp4`, served by `{L}/renders/{render_id}/file` with range requests |
| Preview | Cropwright plays the served timeline in the browser (CSS transforms on preview images, Web Audio for music, served beat times); rendering only on "Export video" |

### 5.4 Music

Two sources (Q7, owner 2026-11-13):
- **The owner's own tracks:** upload in the UI (a Music page and the `MusicPicker` in the memory editor) with
  `title`, `artist`, `licence` (`own`, `public_domain`, `cc0`, `cc_by`, `other`), `licence_url`, `attribution`.
- **A starter set like Apple Photos' memory soundtracks:** a small, mood-tagged set of public-domain and CC0
  tracks (a few each for calm, warm, upbeat) that the library offers out of the box. Only tracks whose licence
  allows redistribution are eligible (public domain recordings, CC0); each track's source page and licence are
  recorded in a manifest and `ATTRIBUTION.md`. Lane L-C4 picks the tracks from documented sources (for example
  Musopen public-domain recordings or the Free Music Archive's CC0 tracks; verify each licence on its page) and
  ships them as an optional fetch on first use, so the image stays small. CC BY tracks are allowed only with
  attribution on the end card. Licences such as Pixabay's that forbid standalone redistribution are never bundled.
`cc_by` requires attribution and puts it on the end card. Files under `OP_DERIVATIVES_ROOT/music/<project>/`
(the starter set under a shared read-only `music/starter/`), never redistributed outside a render. Tempo from the
beat grid; energy from RMS on decoded PCM (numpy); a memory's default track matches its mood tag (the VLM title
call also returns `mood: calm|warm|upbeat`) and length.

Later, advanced (not P2c): a connection to a licensed music service with an API (for example Epidemic Sound or
Artlist, if their terms allow it for personal renders), or AI-generated music. Research the APIs and terms when
that work starts.

---

## 6. Tables and migrations

Revision `<next>_library_create.py` (lane L-C1 writes it in its first commit, with the wire models of section 7 and the generated contracts, so L-C2 and L-C4 code on a fixed
schema): `library_events`, `library_event_members`, `calendar_accounts`, `calendar_sources`,
`calendar_events`, `calendar_matches`, `memories`, `memory_chapters`, `memory_members`, `music_tracks`,
`renders`; RLS registration; config store
kinds `memories_profile` and prompt-pack kind `memory_titles`. OpenSearch: `event.event_id`,
`event.trip_id` in `library_event_fields.py`.

| Table | Columns |
|---|---|
| `library_events` | `id uuid7`, `project_id`, `kind` (`event/trip`), `parent_trip_id null`, `start_local`, `end_local`, `photo_count`, `centroid_lat null`, `centroid_lon null` (under the location policy), `place_label null`, `away_from_home bool`, `segmenter_version`, `created_at` |
| `library_event_members` | `event_id`, `image_id`; pk both |
| `calendar_accounts` | `id`, `project_id`, `provider` (`google/ics_file/ics_url/caldav`), `display_name`, `status` (`active/needs_reauth/disabled`), `secret_ciphertext bytea null`, `secret_key_id null`, `scopes text[]`, `last_sync_at`, `last_error null`, `created_at` |
| `calendar_sources` | `id`, `project_id`, `account_id`, `external_id`, `title`, `enabled`, `time_zone`, `sync_token null`, `full_synced_at` |
| `calendar_events` | `id`, `project_id`, `source_id`, `external_id`, `recurring_event_id null`, `original_start null`, `summary`, `description null`, `location_text null`, `start_at null`, `end_at null`, `start_date null`, `end_date null`, `time_zone`, `event_type`, `status`, `visibility`, `person_id null` (birthday link, human), `updated_at` |
| `calendar_matches` | `id`, `project_id`, `calendar_event_id`, `library_event_id`, `score`, `reasons jsonb`, `state` (`suggested/accepted/rejected`), `created_at` |
| `memories` | `id`, `project_id`, `kind` (`on_this_day/trip/event/person/holiday/birthday/calendar/theme/prompt`), `title`, `subtitle null`, `title_source` (`calendar/holiday/vlm/template/user`), `cover_image_id`, `period_start`, `period_end`, `score`, `state` (`suggested/saved/dismissed`), `reasons jsonb`, `generator_version`, `seen_at null`, `revision`, `created_at` |
| `memory_chapters` | `memory_id`, `idx`, `title`, `caption null`; pk both |
| `memory_members` | `memory_id`, `image_id`, `chapter_idx`, `position`, `role` (`cover/key/member`), `caption null`; pk (`memory_id`,`image_id`) |
| `music_tracks` | `id`, `project_id`, `title`, `artist`, `licence`, `licence_url`, `attribution null`, `storage_key`, `duration_s`, `bpm null`, `beats_key null`, `analysed_at null`, `created_at` |
| `renders` | `id`, `project_id`, `memory_id`, `timeline jsonb` (`memory_timeline/1`), `preset` (`landscape_1080p/portrait_1080p/square_1080p`), `music_track_id null`, `status`, `job_id`, `output_key null`, `duration_s`, `size_bytes`, `created_at` |

---

## 7. Routes and wire models

| Route | Request | Response |
|---|---|---|
| `GET {L}/events` | `kind`, `LibraryFilter`, `cursor` | `Events {events: list[EventCard {event_id, kind, title, start_local, end_local, photo_count, cover_url, place_label, calendar_match: {summary, state} \| None}], next_cursor}` |
| `GET {L}/events/{event_id}` | | `Event {...EventCard, chapters: list[{start_local, end_local, photo_count}]}` |
| `POST {L}/events/rebuild` | `{scope: Literal['incremental','full']}` | `JobRef` |
| `GET {L}/calendar/accounts` | | `CalendarAccounts {accounts: list[{account_id, provider, display_name, status, last_sync_at, last_error, sources_enabled}], google_client_configured: bool}` |
| `POST {L}/calendar/accounts/google/start` | `{account_id?: str}` (reconnect) | `OAuthStart {authorization_url, state, redirect_uri, expires_at, browser_hint: Literal['loopback','paste']}` |
| `POST {L}/calendar/accounts/google/complete` | `{redirected_url}` | `CalendarAccount`; `400 oauth_state_invalid`, `400 oauth_scope_missing` |
| `POST {L}/calendar/accounts/{account_id}/reconnect` | | `OAuthStart` |
| `POST {L}/calendar/accounts/ics` | multipart file, or `{url}` | `CalendarAccount` |
| `POST {L}/calendar/accounts/caldav` | `{server_url, username, password}` | `CalendarAccount` (L-C6) |
| `DELETE {L}/calendar/accounts/{account_id}` | `purge: bool` | `CalendarDisconnect {revoked, n_events_deleted, n_matches_deleted}` |
| `GET/PATCH {L}/calendar/accounts/{account_id}/sources` | patch `{sources: [{source_id, enabled}]}` | `CalendarSources {sources: list[{source_id, title, enabled, time_zone, last_synced_at}]}` |
| `POST {L}/calendar/accounts/{account_id}/sync` | | `JobRef` |
| `GET {L}/calendar/events` | `from`, `to` | `CalendarEvents {events: list[{calendar_event_id, summary, start, end, all_day, event_type, source_title}]}` (`summary` masked for `private`) |
| `GET {L}/calendar/matches` | `state` | `CalendarMatches {matches: list[{match_id, calendar_event, library_event: EventCard, score, reasons: list[{label, value}]}]}` |
| `POST {L}/calendar/matches/{match_id}/accept`, `/reject` | | `{state}` |
| `GET {L}/holidays` | `year` | `Holidays {country, subdivision, holidays: list[{date, name}]}` |
| `GET {L}/memories` | `state`, `kind`, `cursor` | `Memories {memories: list[MemoryCard {memory_id, kind, title, subtitle, cover_url, period_start, period_end, photo_count, state, is_new}], next_cursor}` |
| `GET/PATCH {L}/memories/{memory_id}` | patch `{title?, subtitle?, state?, expected_revision}` | `Memory {...MemoryCard, chapters: list[{idx, title, caption, photos: list[PhotoTile & {role, caption, reasons}]}], title_source, reasons}` |
| `POST {L}/memories/{memory_id}/photos/remove` | `{image_ids}` | `Memory` |
| `GET/PUT {L}/memories/{memory_id}/timeline` | put `{timeline, expected_revision}` (order, durations, captions, music, preset) | `MemoryTimeline {timeline, beats: list[float], preview_urls: dict[str, str], revision}` |
| `POST {L}/memories/build` | `{kinds?}` | `JobRef` |
| `POST {L}/memories/from_prompt` | `{prompt, length: Literal['short','standard','long']}` | `JobRef` |
| `GET {L}/featured` | | `Featured {photos: list[PhotoTile & {years_ago}]}` |
| `GET/POST {L}/music` | multipart upload with licence fields | `MusicTracks {tracks: list[{track_id, title, artist, licence, duration_s, bpm, analysed}]}` |
| `DELETE {L}/music/{track_id}` | | `204` |
| `POST {L}/memories/{memory_id}/renders` | `{preset, length?}` | `JobRef` with `render_id` |
| `GET {L}/renders/{render_id}` | | `Render {render_id, status, duration_s, size_bytes, file_url, created_at}` |
| `GET {L}/renders/{render_id}/file` | Range | `video/mp4` |

New settings keys in `project_settings`: `library.home_place`, `library.holiday_country`,
`library.memories_enabled` (default on), `library.vlm_titles` (`local`, `off`, or `external_text_only` if
Q5 allows), `library.calendar_store_descriptions` (default off). New `OP_` variables (added to
`env.template` and `src/config/`): `OP_LIBRARY_SECRETS_DIR`, `OP_DERIVATIVES_ROOT` (shared with P1's N3; if
the engine effort already named it, use that name).

---

## 8. Jobs and queues

| Job or tick | Queue | Singleton | Notes |
|---|---|---|---|
| `library_events_build` (after ingest; full on demand) | `library` | `exclusive_group='library_events'` | |
| calendar tick (15 min) → `library_calendar_sync` jobs | `periodic` → `library` | advisory `LIBRARY_CALENDAR_SYNC`; job `exclusive_group='library_calendar'`, `exclusive_scope='account_id'` | ±25 % jitter; backoff on 403/429; `needs_reauth` on `invalid_grant` |
| memories tick (daily 03:00 library-local) → `library_memories_build` | `periodic` → `library` | advisory `LIBRARY_MEMORIES_NIGHTLY`; job `exclusive_group='library_memories'` | pausable, sliced per candidate kind, resumable; also refreshes featured photos |
| `library_memory_from_prompt` | `library` | no | VLM call through the existing labelling client |
| `library_music_analyse` (on upload) | `media_render` | no | Beat This!, RMS |
| `library_render` | `media_render` | no | concurrency 1 per GPU, 2 on CPU |

---

## 9. Cropwright components

| Piece | Path (new) | Reuses |
|---|---|---|
| Memories home | `routes/p/[project]/library/memories/+page.svelte`, `lib/components/library/{MemoryCard,MemoryRail,FeaturedStrip}.svelte` | `ui/Card.svelte` |
| Memory page and player | `lib/components/library/{MemoryView,MemoryPlayer,MemoryEditor,MusicPicker}.svelte`, `lib/library/timelinePlayer.svelte.ts` (plays served timing; computes nothing) | `PhotoViewer.svelte`, `ui/Drawer.svelte` |
| Render export | `lib/components/library/RenderDialog.svelte` | `lib/jobs/followJob.ts` |
| Events and trips | `routes/p/[project]/library/events/**`, `lib/components/library/{EventList,TripCard}.svelte` | `PhotoGrid.svelte` |
| Calendars | `routes/p/[project]/library/calendars/+page.svelte`, `lib/components/library/{CalendarConnect,CalendarSources,CalendarMatches,ReconnectBanner}.svelte`; the root-page OAuth handler in `routes/+page.svelte` | `ui/DataTable.svelte`, `components/Modal.svelte` |
| Rule builder | new served fields appear without code changes (`RuleBuilder.svelte` renders the schema) | P2a |

---

## 10. Lanes, order and dependencies

```
wave 1:  L-C1 events, trips, the migration and wire models, rule fields || L-C2 Google Calendar, ICS, holidays, matching, token crypto
         || L-C4 music, beats, saliency, timeline domain, renderer, render worker image || L-C5 memories eval kit
wave 2:  L-C3 memories engine, titles, featured, from_prompt || L-CF1 calendars, events and trips UI, reconnect banner
wave 3:  L-CF2 memories UI, player, editor, render dialog || L-C6 CalDAV || L-FIX
then:    adversarial review of OAuth, token storage and purge (Opus, read-only); eval; owner walk
```

| Lane | Model, effort | Owns | Depends on |
|---|---|---|---|
| L-C1 | Sonnet, high | the migration and wire models (first commit), `src/db/models/library_create.py`, repos, `src/domain/library_events.py`, `src/services/library/events.py`, `library_event_fields.py`, event routes, rule-grammar fields in `src/domain/library_rules.py` | P2b |
| L-C2 | Sonnet, high (secrets, external network) | `src/services/library/calendar/**`, `src/routers/curation/library_calendar.py`, `src/config/library_secrets.py`, `env.template` keys, the user doc section on Google setup | L-C1 migration commit |
| L-C4 | Sonnet, high | `src/domain/memory_timeline.py`, `src/services/library/render/**`, `src/tasks/library_render.py`, `Dockerfile` target, compose service, `ATTRIBUTION.md` rows, music and render routes | L-C1 migration commit |
| L-C5 | Sonnet, medium | `scripts/bench/library_eval_memories.py` (YFCC user streams and Commons author series with the Commons face set seeded into real events as excluded identities; a synthetic calendar fixture, because public photos have no calendars; the owner-rating sheet with the rubric of section 12.1; numbers only), `scripts/bench/library_eval_framing.py`, `scripts/bench/library_bench_render.py` | P2b L-B4 |
| L-C3 | Sonnet, high | `src/services/library/memories/**`, `src/domain/memory_selection.py`, `memory_scoring.py`, memories and featured routes, prompt-pack kind `memory_titles`, config kind `memories_profile` | L-C1, L-C2, L-C4 (timeline builder) |
| L-CF1 | Sonnet, medium | calendars and events UI, root OAuth handler, reconnect banner | L-C1, L-C2 merged |
| L-CF2 | Sonnet, high | memories UI, player, editor, render dialog | L-C3, L-C4 merged |
| L-C6 | Sonnet, medium | CalDAV in `src/services/library/calendar/caldav.py` and its route | L-C2 |

Docs every brief cites: Google OAuth pages of section 3.1, Calendar API sync and quota guides,
google-auth-oauthlib and google-auth docs, `cryptography` Fernet, python-dateutil `rrulestr`, `icalendar`,
`recurring-ical-events`, `holidays`, Beat This! README, U-2-Net README, ONNX Runtime Python API, MoviePy v2
docs, FFmpeg encoder and filter docs, Celery retry options, Starlette `FileResponse` range support for the
pinned version, smartcrop.js README (read, not used as code), the match-cutting paper.

Verify commands (paste the output): L-C1 `.venv/bin/python -m pytest tests/domain/test_library_events.py
tests/library/test_events_incremental.py -q --no-cov && .venv/bin/python scripts/codegen/generate_contracts.py --check`;
L-C2 `... tests/library/test_calendar_*.py`; L-C3 `... tests/domain/test_memory_selection.py
tests/library/test_memory_eligibility.py tests/library/test_memory_titles.py`; L-C4 `... tests/domain/test_memory_timeline.py
tests/domain/test_framing.py tests/docker/test_render_image_licence.py`; L-C5 `... tests/scripts/test_memories_eval.py`;
frontend lanes `cd frontend && npm run check` plus their stubbed e2e files. Frontend lanes start after L-C1's
wire models merge. The memories build job is `pausable`, sliced per candidate kind and resumes after a
restart (test `tests/library/test_memories_resume.py`).

---

## 11. Tests to write red first

| Lane | Tests |
|---|---|
| L-C1 | `tests/domain/test_library_events.py`: adaptive gaps on synthetic time series with known boundaries, GPS splits, trips need two days away, a trip across time zones and a DST change keeps one event, `drop` policy gives no trips; `tests/library/test_events_incremental.py`: ids kept by overlap |
| L-C2 | `tests/library/test_calendar_oauth.py`: S256 challenge, `state` bound to the project, pasted URL with a wrong state refused, scope check; `test_calendar_tokens.py`: Fernet round trip, the refresh token never in logs (log capture), access token never stored; `test_calendar_sync.py` against a recorded fake of the Calendar API: full then incremental, deleted entries, 410 resets, 429 backoff, `invalid_grant` → `needs_reauth` and the sync stops; `test_calendar_purge.py`: counts then zero; `test_calendar_matching.py`: 60 % overlap rule, all-day local days, reasons |
| L-C4 | `tests/domain/test_memory_timeline.py`: cuts on beats within one frame, end on a downbeat, attribution on the end card for `cc_by`; `tests/domain/test_framing.py`: smart crop per aspect keeps every grown face box, blurred fill when impossible, matched-cut alignment within the thresholds of section 5.2, saliency used only without boxes; `tests/docker/test_render_image_licence.py`: fails on `--enable-nonfree`; `tests/library/test_render_job.py` with a 3-photo timeline (CPU encode, short, wave run) |
| L-C3 | `tests/domain/test_memory_selection.py`: MMR diversity, quota per chapter, people coverage, cover safe area; `tests/library/test_memory_eligibility.py`: a blocked person, a receipt, an unprocessed photo, a trashed photo and a pending projection are never selected (one fixture each), and a rule added after building removes the photo on read; `test_memory_titles.py`: priority order, private events never in the VLM request, hosted endpoint refused |
| L-CF1, L-CF2 | `e2e/stubbed/library/test_calendar_connect.py` (loopback and paste paths, reconnect banner on `needs_reauth`), `test_memories.py` (cards, dismiss, save), `test_memory_player.py` (plays served timing; scrub; edit order sends `expected_revision`) |

---

## 12. Acceptance walk (owner, live stack, 1600 and 430 px)

### 12.1 Rating rubric (owner is the rater; 1-5 each; the memory's score is the mean)

| Criterion | 1 | 5 |
|---|---|---|
| Selection | wrong or weak photos, duplicates | the photos the owner would have picked |
| Story and order | random | a clear arc with sensible chapters |
| Titles and captions | wrong or generic | right and specific |
| Polish (framing, cuts, music) | heads cut, jarring | professional |

### 12.2 Walk

1. Library › Calendars: connect Google with the owner's production client from the host browser; then
   disconnect without purge and reconnect from a LAN laptop with the paste path; pick two calendars; sync.
2. Revoke the app in the Google account settings; the next sync shows the reconnect banner; reconnect keeps
   the events.
3. Events and trips list matches what a person would call the trips; accept two calendar matches.
4. Memories home after the nightly build: on-this-day, a trip, a person, a holiday; none contains an excluded
   person or a receipt; dismiss one, save one, rename one.
5. Open a trip memory: chapters with titles from the calendar or the local VLM; play the preview with music;
   reorder two photos, change a caption.
6. Export video (landscape 1080p, standard length): renders within the budget; cuts land on beats; faces are
   never cropped; the end card credits the track.
7. "Make a memory of snowy mountains with person A": a sensible result or a clear "not enough photos".
8. Disconnect with purge: the counts, then zero events.
9. The eval report (owner ratings, budgets, exclusion zero-check) against the 00 section 2 bars.
