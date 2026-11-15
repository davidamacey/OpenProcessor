# Photo library: feature and science research

Status: research input for the photo-library planner. This is not a plan and
not current behaviour. Compiled 2026-11-13 from public sources only.

**What this is for.** OpenProcessor processes a library in place, without
moving the originals. It records faces, people, objects, places, time and
text. One processed library then serves four jobs:

1. labelling and training;
2. organising (timeline, albums, smart albums, People, Places);
3. curating (finding or excluding identities and objects);
4. creating (calendar-aware smart albums and "memories": best photos,
   slideshows and videos).

Cropwright is the frontend. It already has Gallery, Map, Clusters, Review,
Detections, Table, faces and semantic search. This document sets the bar
those jobs must reach (Apple Photos, Google Photos, Immich, Adobe Lightroom).
It also lists the published science and the open models that can get us
there.

**Confidence marks.**

| Mark | Meaning |
|---|---|
| **H** | Stated on the cited primary page (vendor docs, repo README, paper page), read on 2026-11-13. |
| **M** | Stated by a reputable secondary source (vendor community page, established tech press) or inferred from file names. Not confirmed on a primary page. |
| **U** | Unverified: from memory or a single weak source. Check before relying on it. |

All compute figures in our own words are **estimates**, marked "est.". No
number here was measured on our stack.

**Phase labels.** The owner wants a measured rollout. Every feature carries
the earliest phase it fits:

| Phase | Scope |
|---|---|
| **P1** | Image ML engine: signals computed per image (faces, embeddings, OCR, quality, aesthetics, EXIF/GPS, events). No user-facing library features beyond what Cropwright has. |
| **P2a** | Organise (images): timeline, albums, smart albums, People, Places, favourites/hidden/trash, duplicates. |
| **P2b** | Curate (images): exclude identities or objects, review queues, culling, stacks. |
| **P2c** | Create (images): memories, featured photos, calendar-aware smart albums, and slideshows rendered from stills. |
| **P3** | Video processing: keyframes, faces and people in video, video search, transcoding. |
| **P4** | Video memories: clip selection and mixed photo and clip montages. |

Open decision for the planner: a slideshow *rendered to an MP4* from stills
only needs image signals, so this doc puts it in P2c. Montages that include
video clips are P4.

---

## 1. Feature inventory

### 1.1 Comparison table

Key: Y = yes, P = partial, N = no, ? = not confirmed. Sources are listed in
§1.2. Confidence is H unless marked.

| Feature | Apple Photos | Google Photos | Immich | Lightroom / Classic | Ente | PhotoPrism | Synology Photos | digiKam | Phase |
|---|---|---|---|---|---|---|---|---|---|
| Timeline grid with a scrubbable scrollbar | Y (U on grid detail) | Y | Y | Y (grid) | Y | Y | Y | Y | P2a |
| Virtual scroll for very large libraries | Y (U) | Y | Y | ? | ? | ? | ? | ? | P2a |
| Year/month/day zoom levels | Y (M) | P (U) | P (month/day groups) | N | ? | P (calendar) | ? | Y (date views, U) | P2a |
| Favourites | Y | Y | Y | Flags, ratings, labels | Y | Y | Ratings | Pick, ratings, colour labels | P2a |
| Archive (hide from timeline, still searchable) | Hidden album (M) | Y | Y | N | Y (U) | Y | ? | N | P2a |
| Locked or private folder | Hidden, Face ID locked (M) | Locked Folder (M) | Locked folder | N | Hidden (U) | Private | ? | N | P2a |
| Trash with retention | 30 days (M) | 60 days if backed up (M) | Y | N | Y (U) | Y (U) | Y (U) | Y (U) | P2a |
| Exact duplicates | Y | Y (U) | Checksum | Pixel duplicate finder (M) | Y (U) | Y (U) | Y | Y | P2a |
| Near-duplicate stacks with a "top pick" | Duplicates album | Photo Stacks, AI top pick | CLIP-distance duplicates and manual stacks | Assisted Culling auto-stack (M) | ? | Stacks | Similar Items stacked within ±24 h | Similarity search | P2b |
| Manual albums | Y | Y | Y | Collections | Y | Y | Y | Y | P2a |
| Rule-based smart albums | Mac only: Photo, Title, Date, any/all | Live albums by people and pets | Planned ("Soon") | Smart Collections: rich, nested all/any/none | ? | Saved filters (U) | Conditional albums: people, objects, tags, place, camera, lens, exposure | Saved searches (U) | P2a |
| People clustering and naming | Y (face + upper body) | Y | Y (DBSCAN-like) | Y (Classic People view) | Y (on device) | Y | Y | Y | P1 → P2a |
| Pets | Y | Y | N (not on roadmap) | ? | ? | Pet labels | Subjects | ? | P1 → P2a |
| Merge people | Y | Y (U) | Y | Drag faces onto a stack | Y | Y (U) | Y (U) | Y (U) | P2a |
| "Not this person" or confirm suggestions | Review More Photos | Y (U) | Reassign faces (U) | Confirm or reject suggested names | Split groupings | Y (U) | ? | Y (U) | P2a |
| Hide a person from People | Y | Y (M) | Y | N | ? | ? | ? | ? | P2a |
| Never feature a person in memories | Y (less / never) | Y (show less / blocked) | N | N/A | ? | N/A | N/A | N/A | P2b |
| Hide dates or holidays from memories | Y (day, place, holiday) | Y (date ranges) | N | N/A | ? | N/A | N/A | N/A | P2c |
| Places map | Y | Y (U) | Global map | Map module (U) | Y (U) | Places | Places | Geolocation (U) | P2a |
| Trips | Y (Trips collection, M) | P (U) | N | N | ? | ? | ? | ? | P2c |
| Semantic or CLIP search | Y (on-device image-text embedding) | Y; Ask Photos (Gemini, opt-in) | Y (CLIP / SigLIP2, multilingual) | Sensei content search; semantic search in preview | Y (MobileCLIP) | Captions and labels (Ollama/OpenAI) | Subjects | N (U) | P1 |
| Search by text in the image (OCR) | Y (U) | Y (exact match with quotes) | Y (PaddleOCR, since v2.2.0) | ? | N (magic search only) | ? | ? | N (U) | P1 |
| Metadata facets (camera, lens, date, place, person) | P | P | Y | Y (very rich) | ? | Y (colour, chroma, quality) | Y | Y | P2a |
| Memories ("on this day") | Y | Y | Y (nightly job, birthday memories since v3.3) | N | ? | Moments | ? | N | P2c |
| Themed or smart memories (people, places, events) | Y (knowledge graph) | Y (themed memories) | Planned ("Soon") | N | ? | Moments (place/time) | ? | N | P2c |
| Memory video with music | Y (Memory Mixes and Looks) | Y (highlight video, beat-synced templates) | N | N | ? | N | ? | N | P2c (stills), P4 (clips) |
| Generative "make a memory from a prompt" | Y (Apple Intelligence) | Y (highlight video from a search) | N | N | N | N | N | N | P2c |
| Quality or best-shot culling | Utility filter and aesthetics (internal) | Top pick in stacks | Duplicates keep-rules (bytes, EXIF count) | Assisted Culling: focus, eyes, exposure | ? | Quality review | ? | Image Quality Scanner (classic + aesthetic DNN) | P1 → P2b |
| Receipts, documents and screenshots filter | Y (Utilities) | ? | N | Assisted Culling flags documents (M) | ? | Documents | ? | ? | P1 → P2a |
| Keywords and hierarchies | P | N | Tags (web) | Y (hierarchical keywords) | ? | Labels | Tags | Y (hierarchical tags) | P2a |
| Folder view of the originals | N | N | Y | Folders (Classic) | N | Y (U) | Y (U) | Y | P2a |
| Sharing (note only) | Shared albums | Shared albums, partner (U) | Albums, partner, public links | Shared albums | Y | Links (U) | Spaces (U) | N | out of scope |
| Editing (note only) | Y | Y | Crop, rotate, non-destructive mobile edits | Y (core product) | ? | N | ? | Y | out of scope |
| Automation workflows | N | N | Workflows (v3.0) | Presets (U) | N | N | ? | Batch Queue Manager | P2a+ |

### 1.2 Per-product notes with sources

**Apple Photos**

- **People and Pets.** A detector pairs faces with upper bodies. The face and
  upper-body embeddings feed a two-pass clustering:
  - pass one is conservative agglomerative clustering within a "moment" (a
    time and location group);
  - pass two is hierarchical agglomerative clustering with median linkage,
    face embeddings only, across moments.

  The distance rule is `D = min(F, αF + βT)`. Incremental assignment uses
  sparse coding over several exemplars per cluster, not a single centroid.
  Clustering runs overnight while the device charges. **H**.
  <https://machinelearning.apple.com/research/recognizing-people-photos>
- **Knowledge graph.** The same people data builds an on-device knowledge
  graph: important groups of people, frequent places, past trips, events.
  This graph powers Memories. **H** (same URL).
- **ANSA scene analyser.** It feeds visual search, Featured Photos, Memories,
  wallpaper suggestions, deduplication and scene classification. **H**.
  <https://machinelearning.apple.com/research/on-device-scene-analysis>
  - About 16M parameters, all heads under 9.7 ms on iPhone 13. **H**.
  - Uses "image aesthetics to pick the highest-quality images". **H**.
  - A *utility filter* (a linear layer on the embedding) keeps receipts and
    documents out of Memories, Sharing and Featured Photos. **H**.
  - The embedding is used for deduplication. **H**.
  - Memories uses the signals for cover ranking, titling and music
    alignment. **H**.
- **Iconic scenes.** Apple crowd-sources, under differential privacy (ε=1),
  how often each of about 4.5M location-category pairs is photographed. This
  ranks Memories key photos (since iOS 16) and Places. **H**.
  <https://machinelearning.apple.com/research/scenes-differential-privacy>
- **Feature less / never feature.** Applies to a person, a day, a place or a
  holiday. "Feature This Person Less" and "Never Feature This Person" affect
  Memories, Featured Photos and the Photos widget. Reset is under Featured
  Content. **H**.
  <https://support.apple.com/guide/personal-safety/feature-a-person-a-place-or-memory-less-often-ipsa4c9edc62/web>
  and <https://support.apple.com/guide/iphone/feature-someone-less-iph10a9dd2a1/ios>
- **People management on Mac.** Hide this Person, Merge N People, Review More
  Photos (confirm or reject). **H** for current steps, **M** for older
  "Confirm Additional Photos".
  <https://support.apple.com/guide/photos/find-and-name-people-and-pets-phtad9d981ab/mac>
- **Duplicates album.** Under Utilities since iOS 16. Detection runs while the
  device is locked and charging. **H**. <https://support.apple.com/en-us/102260>
- **Utilities and Trips.** The iOS 18 Utilities also covers receipts and
  handwriting; Trips groups by location and year. **M**.
  <https://www.macrumors.com/guide/ios-18-photos/>
- **Smart Albums (Mac).** Conditions are Photo kind, Title, Date and more.
  Operators are is / is not. Match any or all. **H**, but the page does not
  list every field.
  <https://support.apple.com/guide/photos/create-smart-albums-pht6d60ca71/mac>
- **Memory movies.**
  - Apple Intelligence "Create a memory movie": type a description, and it
    "finds the best photos and videos, crafts a storyline with unique
    chapters and a narrative arc, and sets it to music". **H**.
    <https://support.apple.com/guide/mac-help/use-apple-intelligence-in-photos-mchl35c53342/15.0/mac/15.0>
  - Memory Mixes suggest songs. **M**.
  - The 12 Memory Looks "analyze photos and video, and apply contrast and
    color adjustments to achieve a uniform aesthetic". **M**.
    <https://appleinsider.com/articles/21/06/07/memories-features-in-photos-app-gets-apple-music-integration-new-features-in-ios-15>
    and <https://www.macrumors.com/how-to/use-memory-looks-photos-app/>

**Google Photos**

- **Memories controls.** **H**.
  <https://support.google.com/photos/answer/9454489?hl=en&co=GENIE.Platform%3DAndroid>
  - Hide faces with two levels. "Show less: You won't get memories about
    them, but they may still appear in group photos". "Blocked: They won't
    appear in memories, including group photos with others".
  - Hide dates: you can add several, and a range is allowed (M).
  - Remove one photo, or a whole memory.
  - Turn off time-based or themed memory types, and creation types (Remix,
    Animations, Cinematic photos, Collages).
  - Stated limits: "Face hiding may not work perfectly in all photos", and
    the settings apply "only to the featured memories carousel".
- **Photo Stacks** (since Nov 2023). Groups "nearly identical photos of the
  same subject that were taken together within a short time frame". It
  suggests a changeable "Top pick" and offers "Keep this, delete rest" (M).
  **H**. <https://support.google.com/photos/answer/14169846?hl=en&co=GENIE.Platform%3DAndroid>
- **Live albums.** Choose people or pets, and matching photos are added
  automatically, new ones too. **M**.
  <https://blog.google/products-and-platforms/products/photos/keep-your-favorite-photos-date-live-albums/>
- **Search.** Classic search, plus Ask Photos (Gemini, opt-in, mobile only).
  Quotes force an exact match on text in photos. "me" is a labelled face.
  **H**. <https://support.google.com/photos/answer/15235862?hl=en&co=GENIE.Platform%3DAndroid>
  and <https://support.google.com/photos/answer/15318661?hl=en&co=GENIE.Platform%3DAndroid>
- **Memories curation intent.** "machine learning to curate ... so you don't
  have to parse through many duplicate shots ... where the photos have good
  quality and all the kids are smiling". **H** (product blog; no technical
  paper found).
  <https://blog.google/products-and-platforms/products/photos/relive-your-best-memories-new-features-google-photos/>
- **Highlight video.**
  - Oct 2023: "choose the best clips & photos ... add music, and sync the
    video to the music". **M**.
  - Dec 2025: templates with "automatic cuts synced to the music" and "Help
    me select". **M**/**H**.
    <https://support.google.com/photos/answer/6128862?hl=en&co=GENIE.Platform%3DAndroid>
    and <https://petapixel.com/2025/12/09/google-photos-gets-more-powerful-video-editing-tools-on-mobile/>
  - A "Beat Match" option for photo transitions was seen in testing. **M**.
    <https://www.androidauthority.com/google-photos-highlight-videos-beat-match-apk-teardown-3712456/>
- **Archive, Locked Folder, Trash (60 days if backed up).** **M**.
  <https://www.howtogeek.com/google-photos-archive-isnt-what-you-think-it-is/>

**Immich** (the open-source reference)

- **Feature table** (README): virtual scroll, scrubbable scrollbar, archive
  and favourites, global map, facial recognition, Memories ("x years ago"),
  stacked photos, tags, folder view, partner sharing. **H**.
  <https://github.com/immich-app/immich/blob/main/README.md>
- **Faces.** Clustering is "derived from DBSCAN" and incremental, with a
  nightly pass for unassigned faces. **H**.
  <https://docs.immich.app/features/facial-recognition>
  - Settings: minimum recognised faces (default 3, the core-point
    threshold); maximum recognition distance (keep between 0.3 and 0.7);
    minimum detection score (at least 0.5).
  - People can be merged, and hidden from Explore.
- **Search.** CLIP-family models through VectorChord on Postgres, including
  multilingual SigLIP2. Filters: people, location, camera and lens, date
  range, media type, archive / favourite / not-in-album, rating, file and
  path, tags, OCR. **H**. <https://docs.immich.app/features/searching>
- **OCR.** PaddleOCR since v2.2.0 (Oct 2025). v3.3.0 added PP-OCRv6 and
  TensorRT RTX. **H** for the roadmap entries.
  <https://immich.app/roadmap> and <https://github.com/immich-app/immich/discussions/23377>
- **Duplicates.** CLIP-embedding distance, `duplicateDetection.maxDistance`
  default 0.01, range 0.001 to 0.1. **H**/**M**.
  <https://docs.immich.app/administration/system-settings/>
  - The maintainers say Immich has no notion of duplicate *types*; it is
    semantic similarity only. **M**.
    <https://github.com/immich-app/immich/discussions/25831>
  - Keeper pre-selection: larger file, then more EXIF fields. **H**.
    <https://docs.immich.app/features/duplicates-utility>
- **Memories.** A nightly job in `server/src/services/memory.service.ts`
  builds "on this day" memories for today ±3 days. They are stored in
  `memory` / `memory_asset` tables; users can favourite one to keep it, or
  delete it. **M** (PR and issue discussion).
  <https://github.com/immich-app/immich/pull/15953>
  - Birthday memories (v3.3): the three days before a set birthday, at most
    25 assets. **M**. <https://github.com/immich-app/immich/pull/30831>
  - A content-aware "smart memories" PR was closed unmerged in June 2025.
    **H**. <https://github.com/immich-app/immich/pull/16764>
  - Smart memories and smart albums are still listed as planned. **H**.
    <https://immich.app/roadmap>
- **ML backends.** CUDA, ROCm, OpenVINO, ARM NN, RKNN. Tasks: smart search,
  faces, OCR. **H**. <https://docs.immich.app/features/ml-hardware-acceleration>

**Adobe Lightroom (cloud desktop / web) and Lightroom Classic**

- **Culling primitives.** Pick/reject flags, 0–5 star ratings and colour
  labels. These are the professional culling vocabulary, and the smart
  collection criteria below include Pick Flag, Rating, Label Color and Label
  Text. **M** (helpx blocks automated fetch; criteria list from
  <https://helpx.adobe.com/be_en/lightroom-classic/desktop/organize-photos-in-lightroom-classic/smart-collections-criteria-in-lightroom-classic.html>
  via search snippet and
  <https://photography.tutsplus.com/tutorials/how-to-use-smart-collections-in-adobe-photoshop-lightroom--cms-24130>).
- **Smart Collections (Classic).** **M**.
  - Fields: Rating, Pick Flag, Label Color/Text, Source (collection /
    folder), File Name/Type, Date, Camera Info, Location, Other Metadata
    (keywords, title, caption, job), Develop, Size, Any Searchable Text.
  - Operators: is, is not, greater/less than, in range, contains, starts
    with, ends with.
  - Match **all / any / none**. Option/Alt-click creates **nested rule
    groups**. This is the richest rule model in the inventory and the one to
    copy.
- **Keywords.** Hierarchical keyword lists. Named people become *People
  keywords*. **H** (people keyword: Adobe face-recognition help, via search
  snippet; 403 on direct fetch).
  <https://helpx.adobe.com/lightroom-classic/help/face-recognition.html>
- **People (Classic).** People view shows Named and Unnamed areas.
  Suggestions appear in a "Similar" area and are confirmed with a check mark.
  Bulk-confirm by multi-select. **M**. Same URL and
  <https://www.lightroomqueen.com/community/threads/all-pictures-with-a-certain-person-in-it.49826/>
- **AI search (cloud Lightroom).** Sensei auto-tags in the cloud for content
  search. Facet syntax such as `camera:`. "Improved Search" (semantic, e.g.
  "a happy dog in snow") is an English-only Technology Preview on the web.
  Classic only searches user keywords. **H**/**M**.
  <https://helpx.adobe.com/lightroom/web/create-albums-and-organize-photos/add-and-organize-photos/search-and-filter-photos.html>
  and <https://helpx.adobe.com/lv/lightroom-cc/web/create-albums-and-organize-photos/add-and-organize-photos/semantic-search.html>
- **Assisted Culling** (Lightroom and Classic). **M**: no primary helpx page
  was readable; community and press only.
  <https://community.adobe.com/questions-675/p-assisted-culling-feedback-thread-lrc-1627467>
  and <https://fstoppers.com/education/lightroom-classic-154-adds-duplicate-finder-and-smarter-group-culling-903364>
  - Select criteria: Subject Focus, Eye Focus, Eyes Open, each with a
    sensitivity slider. Reject criteria: Misfires, Exposure Issues.
  - Auto-stack by capture time plus visual similarity, per-face scores in
    groups, scores cached and recomputed after edits.
  - Multi-person support (Feb 2026). Classic 15.4 adds a pixel-based
    duplicate finder.
- **Stacks, versions, metadata panels.** Classic has stacks, virtual copies
  (versions) and a Metadata panel with EXIF/IPTC presets. **U**: well known,
  but no primary page was read in this pass.

**Others (short)**

- **Ente.** All ML on the device; MobileCLIP v1 for magic search; face
  merge and split; no dedicated OCR search. **H**.
  <https://ente.com/help/photos/features/search-and-discovery/machine-learning>
  and <https://ente.com/ml/>
- **PhotoPrism.** **H**.
  <https://docs.photoprism.app/user-guide/ai/> and
  <https://web.photoprism.app/features/>
  - Labels, captions via Ollama/OpenAI, NSFW detection.
  - People, Places, Moments, Calendar, Quality Review, Private, Stacks,
    Documents.
  - The auto quality score filters low-quality images from search (M).
- **Synology Photos.** **H**.
  <https://www.synology.com/en-global/knowledgebase/DSM/help/SynologyPhotos/managing_albums>
  and <https://www.synology.com/en-global/knowledgebase/DSM/tutorial/Quick_Start_Synology_Photos>
  - Automatic People, Subjects, Places, Tags and Recently Added albums.
  - Similar Items stacked within ±24 h of capture or upload.
  - Conditional albums on file name, description, people, objects, tags,
    location, camera, lens, focal length, exposure, aperture, ISO, folder,
    file type, date and rating.
- **digiKam.** Image Quality Scanner (renamed in 8.6.0). It sets Pick Labels
  from either blur / noise / exposure / compression or a deep-learning
  aesthetic model. **H**.
  <https://www.digikam.org/news/2025-03-15-8.6.0_release_announcement/>

### 1.3 The professional bar

These are the features whose absence makes a library feel amateur. All
products at the top of the market have them, or the power-user tools
(Lightroom, digiKam) have them.

1. **A timeline that scrolls a whole library instantly.**
   - Justified grid, month/day sections, a scrubber that jumps anywhere.
   - Blur placeholders, then thumbnails.
   - No scroll jumps while sections load. (§2)
2. **People you can trust.**
   - Clusters with high precision, bulk confirm/reject.
   - Merge, hide, name, and "me".
   - New photos assigned incrementally.
   - A review queue for low-confidence assignments.
3. **Search that combines signals.** Semantic text, person AND person,
   place, date range, OCR text, camera / lens, rating, "not in album", all in
   one query with facets (Immich, Lightroom, Synology).
4. **Rule-based smart albums** with nested all/any/none groups (Lightroom).
   People-driven live albums (Google). Rules that update as new photos are
   processed.
5. **Duplicates and bursts collapsed into stacks** with an explainable top
   pick and "keep this, trash the rest" (Google, Lightroom, Immich).
6. **Culling vocabulary.** Pick/reject flags, stars, colour labels, plus
   AI-assisted culling (focus, eyes open, exposure, misfires) whose
   thresholds the user can see and change.
7. **Featured content with negative controls.**
   - Feature a person, day, place or holiday less, or never.
   - Block versus show-less semantics.
   - Reset.
   - Utility and screenshot filtering, so receipts never become memories.
8. **Memories beyond "on this day".** Trips, events, people and themes, with
   a polished, music-synced video (Apple, Google). Immich lacks this, and it
   is the main gap OpenProcessor can fill.
9. **Safety of the originals.** Trash with retention, a hidden or locked
   area, and in-place libraries (Immich external libraries, Lightroom
   Classic, digiKam).

### 1.4 Features the owner may have missed

These are useful to a power user or data scientist, and none of them is in
the owner's list.

| Feature | Seen in | Why it matters for us | Phase |
|---|---|---|---|
| Utility / document / screenshot classifier that excludes images from memories and featured photos | Apple ANSA, PhotoPrism Documents, Lightroom Assisted Culling | Without it, receipts and screenshots end up in memories. It is a linear head on an existing embedding (Apple, H). | P1 |
| Stacks with a top pick; RAW+JPEG and burst pairing | Google, Immich, Lightroom, Synology | Collapses the grid, and feeds culling and memories. | P2b |
| Pick/reject flags, stars, colour labels | Lightroom, digiKam | The standard culling vocabulary; doubles as training labels ("keeper" vs "reject"). | P2a |
| Hierarchical keywords, and keywords written back to XMP sidecars | Lightroom, digiKam (U for write-back) | The originals stay in place, so sidecars carry our labels to other tools without touching the files. | P2a |
| Nested any/all/none smart-album rules, and saved searches | Lightroom | Power-user query building; the same rule engine can drive memories. | P2a |
| Date/time/GPS correction and time-zone shifting | Immich (metadata editing, H on roadmap) | Event segmentation and calendar matching break on wrong clocks; travel shifts time zones. | P2a |
| GPX track-log geotagging | Lightroom Map module (U) | Fills GPS for photos from cameras without GPS. | P2a |
| Trips and Recent Days collections | Apple (M) | Cheap and valuable output of event segmentation. | P2c |
| Birthday memories, and a birthday on each person | Immich v3.3 (M), Google Calendar birthday events (H) | Calendar-aware creation. | P2c |
| Holiday-aware memories, and "feature this holiday less" | Apple | Needs a holiday calendar (§4). | P2c |
| OCR text overlay with copy | Immich v2.3+ | Low cost with PP-OCRv5 already in our stack. | P2a |
| Quality review queue that hides low-quality images by default | PhotoPrism | Keeps blurred misfires out of the browsing views. | P2b |
| Folder view of the originals | Immich, digiKam, Synology | Matches the "originals in place" model. | P2a |
| Auto-adding albums by people (live albums) | Google | Shares one rule engine with exclusions. | P2a |
| Admin view of the job queues, re-run per signal, model choice per task | Immich admin | For data scientists: reproducibility and model swaps. | P1 |
| Per-signal provenance (model id and version, threshold, time) | none publish it | Lets a user trust and re-run a selection. Fits OpenProcessor's audit style. | P1 |
| Embedding export, query by example, "more like these" | Immich (CLIP); no full export | Feeds training and analysis, our job (1). | P1 |
| Fairness checks on the quality, eyes-open and face signals per subgroup | Apple (people), Google (Top Shot, Clips) | Both vendors publish subgroup tests. Best-shot ranking can be biased. | P1 |

---

## 2. UX patterns that make libraries feel fast

| Pattern | Evidence | Notes for Cropwright | Phase |
|---|---|---|---|
| **Sectioned virtual grid.** The server sends per-month counts first; the client estimates section heights, then loads a section's assets on demand. | Google Photos web: "subdivide a user's collection ... on initial load send the sections and counts", grouped by month; sections, then segments, then tiles, all absolutely positioned (Google Design, 2018; 403 on direct fetch, summary via search, **M**). <https://medium.com/google-design/google-photos-45b714dfbed1>. Immich does the same through `/timeline/buckets` (monthly summary) and `/timeline/bucket` (one month) (**M**, client PR). <https://github.com/hdmain/immich-desktop/pull/4> | Backend serves month buckets with counts and aspect ratios. The frontend positions tiles absolutely. | P2a |
| **Justified layout** by row-height adjustment, not spacing. | Google: layout of 100 photos about 2 ms, 1M photos about 1.5 s (**M**, same article). Immich uses `@immich/justified-layout-wasm` (Rust → WASM; claims 10M boxes in under 50 ms) (**H** for the claim on the README). <https://github.com/immich-app/justified-layout> | Reuse Immich's library (check its licence before adopting; U), or Flickr's `justified-layout`. Do not hand-roll it. | P2a |
| **No scroll jump on late section loads.** Anchor the scroll position when the height of a section above the viewport changes. | Immich PR #16305 fixed exactly this (**M**). <https://github.com/immich-app/immich/pull/16305> | Test case: scrub to 2015, wait, and the view must not move. | P2a |
| **Svelte timeline manager.** Immich web keeps a "timeline-manager" with month and day groups and a viewer-asset layer. | Directory listing: `timeline-manager.svelte.ts`, `timeline-month.svelte.ts`, `timeline-day.svelte.ts`, `group-insertion-cache.svelte.ts`, `internal/` (**H**). <https://github.com/immich-app/immich/tree/main/web/src/lib/managers/timeline-manager> | Same stack as Cropwright (Svelte). The closest prior art, but **read only**; the licence is AGPL-3.0 like ours (U). | P2a |
| **Progressive thumbnails.** A ThumbHash placeholder sent inline in JSON, then a 250 px WebP thumbnail for the grid, then a 1440 px JPEG preview for the viewer and ML. | Immich defaults (**M**, v1.117.0 release notes). <https://github.com/immich-app/immich/releases/tag/v1.117.0> | We already decode for ML. Generate a placeholder plus 2 sizes at ingest, next to the originals' index, never in the original's folder. | P1 |
| **Keyboard flows.** `?` overlay; arrows move focus; `x` toggles selection; Enter opens or toggles; Shift-drag selects a range; Ctrl-A; Del. | Immich (**M**). <https://github.com/immich-app/immich/pull/16713> and <https://github.com/immich-app/immich/discussions/3085> | Range selection must also work in search results; Immich had bugs there (#13692). | P2a |
| **Selection model.** Range selection across unloaded sections; date-header checkboxes select a whole day. | Immich issues (**M**). <https://github.com/immich-app/immich/issues/1243> | Selection must be held as ranges and queries, not only as loaded ids. | P2a |
| **People naming flow.** Name one cluster; the app suggests more ("Review More Photos", Lightroom "Similar"); bulk confirm with multi-select and one check; "No" means not this person. | Apple (**H**), Lightroom (**M**), Immich minimum-faces threshold (**H**). | Cropwright's Clusters and Review already do this for objects. Reuse them for faces. | P2a |
| **Map and timeline interplay.** Map clusters filter the grid; the date range filters the map; the map is a search facet. | Immich global map and location filters (**H**); Synology Places map (**H**). | Cropwright already has Map geo filters. Link them to the timeline range. | P2a |

---

## 3. The science behind memories and curation

Licence notes cover **code / weights**. Dataset terms (AVA, KonIQ and others
are research-only) may also bind the weights. Treat any weights trained on
research-only data as "check before shipping" (**U** for each specific
case). "Triton?" means whether it fits our Triton / TensorRT stack (est.).

### 3.1 Event segmentation (time and place gaps)

| Method | Idea | Licence / availability | Compute | Fit | Phase |
|---|---|---|---|---|---|
| PhotoTOC (Platt, Czerwinski, Field, 2003) | Time-gap clustering against an adaptive local average of gaps, then content clustering inside events. | Paper. <https://www.microsoft.com/en-us/research/publication/phototoc-automatic-clustering-for-browsing-personal-photographs/> **H** | Trivial, CPU | SQL / Python on timestamps | P1 |
| Temporal event clustering (Cooper, Foote, Girgensohn, Wilcox, ACM TOMM 2005) | Inter-photo similarity at several time scales, optionally with content; unsupervised. | Paper. <https://dl.acm.org/doi/10.1145/1083314.1083317> **H** | Light, CPU | Yes | P1 |
| "Time as essence" (Graham et al., JCDL 2002) | Time-based hierarchy (events within days within months) for browsing. | Paper. <https://dl.acm.org/doi/10.1145/544220.544301> **H** (existence) | Trivial | Yes | P2a |
| Apple "moments" | Time and location groups that bound upper-body matching. | <https://machinelearning.apple.com/research/recognizing-people-photos> **H** | n/a | Design reference | P1 |
| Density clustering on (time, lat, lon) with HDBSCAN | Practical modern baseline; variable density. | scikit-learn `HDBSCAN`, BSD-3 (**U**: check the docs page for the pinned version) | CPU, O(n log n) est. | Yes; run per user library | P1 |

Recommendation: a two-level segmenter.

1. Split events on time gaps using PhotoTOC-style adaptive thresholds, and
   split again on GPS jumps.
2. Group events into trips when the place is far from the home cluster, the
   most frequent location (Apple's "frequent places" and "trips", **H**).
3. Calendar events (§4) override or name segments.

### 3.2 Aesthetic and quality scoring (best shot)

| Model | Venue | What it scores | Licence | Compute (est.) | Triton? | Phase |
|---|---|---|---|---|---|---|
| LAION improved aesthetic predictor | community, 2022 | Aesthetics 1–10 from a CLIP embedding through an MLP head; trained on AVA plus SAC/LOGOS (checkpoint names) | Apache-2.0 code and weights (**H**). <https://github.com/christophschuhmann/improved-aesthetic-predictor> | Head is negligible, but it needs **CLIP ViT-L/14** embeddings (**M**, from file names), an extra encoder pass | Yes (ONNX → TRT) | P1 |
| NIMA | IEEE TIP 2018; arXiv 1709.05424 | Predicts the *distribution* of human ratings (AVA, TID2013) | Paper (**H**). <https://arxiv.org/abs/1709.05424>. Weights: third-party reimplementations only (U) | CNN-sized (MobileNet / Inception); light | Yes | P1 |
| MUSIQ | ICCV 2021; arXiv 2108.05997 | Multi-scale transformer, native resolution; checkpoints for AVA, KonIQ, SPAQ, PaQ-2-PiQ | google-research repo (Apache-2.0 at the repo root, **U**); dataset terms apply. <https://github.com/google-research/google-research/tree/master/musiq> | ViT-B-ish; moderate | Possible; variable-size input is awkward for TRT | P1 (optional) |
| CLIP-IQA | AAAI 2023; arXiv 2207.12396 | Zero-shot "look and feel" from paired CLIP prompts | Paper (**H**). <https://arxiv.org/abs/2207.12396>. The reference implementation sits in IQA-PyTorch, licensed **PolyForm Noncommercial 1.0.0** (**H**). <https://github.com/chaofengc/IQA-PyTorch> | Free if the CLIP embeddings exist | Yes (text-prompt cosine on our embeddings) | P1 |
| Q-Align | ICML 2024 | One LMM for IQA, IAA and VQA from discrete text levels; base appears to be mPLUG-Owl2 (**M**) | Licence files present, terms not read (**U**). <https://github.com/Q-Future/Q-Align> | Multi-billion-parameter LMM (est.); a GPU per batch | No (not a TRT fit); run as an offline re-ranker only on shortlisted candidates | P2c (re-rank) |
| Apple ANSA aesthetics | Apple, 2022 | Aesthetics head on a shared backbone | Not released (**H**) | n/a | n/a | design reference |

**IQA-PyTorch licence caution.** It is licensed PolyForm Noncommercial
1.0.0, so use it for evaluation only, never as a shipped dependency (**H**,
repo text).

Recommendation. Ship a **linear or MLP head on our own embedding**, the
Apple utility-filter pattern (**H**). Train it on AVA-style aesthetic labels
we may use, plus owner keep/reject flags (§1.4) as personal preference
signal. Use CLIP-IQA-style prompt pairs on the existing MobileCLIP2/PE
embeddings as a zero-cost baseline. Keep Q-Align as an offline judge for
evaluation and top-k re-ranking.

### 3.3 Burst and near-duplicate culling

| Method | Notes | Licence | Phase |
|---|---|---|---|
| Embedding-distance grouping inside a time window | What Immich does globally (CLIP distance ≤ 0.01), and Synology within ±24 h. Bursts often sit at 0.05–0.17, so a global threshold misses them (**M**, Immich discussion #32805). Add a capture-time window and a looser threshold inside it. | n/a | P1 |
| SSCD copy-detection descriptors (CVPR 2022) | Built for copy / near-copy fingerprints. More "forensic" than CLIP for exact-scene duplicates. | Code MIT; model licence not stated on the README (**U**). <https://github.com/facebookresearch/sscd-copy-detection> | P1 |
| Photo triage (Chang et al., SIGGRAPH 2016) | Learns human preference *within a series*. Dataset: 15,545 photos in 5,953 series, with pairwise preferences. | Paper and benchmark (**H**). <https://gfx.cs.princeton.edu/pubs/Chang_2016_ATF/index.php>; dataset terms U | P2b |
| Google Top Shot | Frame ranking from functional (lighting), objective (eyes open) and subjective (expression) attributes. Face-first, with a fallback frame-quality model. Tuned with Vizier; fairness tested per subgroup. | Not released (**H**). <https://research.google/blog/top-shot-on-pixel-3/> | design reference |
| Google Clips | Pairwise human preferences, turned into a continuous quality score, distilled into a MobileNet; moments clustered to drop near-duplicates, with a per-cluster cap. | Not released (**H**). <https://research.google/blog/automatic-photography-with-google-clips/> | design reference |
| Lightroom Assisted Culling | Subject/eye focus, eyes open, exposure, misfires; auto-stack. | Product (**M**) | design reference |

The burst ranker we can build from published parts: within each stack, score
these signals.

- sharpness: Laplacian variance on the face or subject crop;
- eyes open: from landmarks, or a small classifier;
- face quality: §3.5;
- aesthetic head: §3.2.

Then pick the top. Learn the weights from owner picks; Google tuned theirs
with a black-box optimiser (**H**).

### 3.4 Diversity selection

| Method | Use | Source |
|---|---|---|
| Maximal Marginal Relevance (Carbonell and Goldstein, SIGIR 1998) | Greedy: quality minus λ times maximum similarity to the items already picked. Simple and tunable. | <https://doi.org/10.1145/290941.291025> (**U**: DOI from memory) |
| Determinantal point processes (Kulesza and Taskar, 2012) | Quality times diversity, in one kernel; k-DPP for a fixed memory size. | <https://arxiv.org/abs/1207.6083> (**U**: id from memory) |
| Cluster-then-cap (Google Clips) | Cluster moments, cap each cluster. | **H** (Clips blog above) |

Recommendation: cluster-then-cap for the ordering of events, then MMR
within an event over [embedding, time, people-set]. Both are CPU-cheap.

### 3.5 Face quality and people prominence

| Model | Venue | Licence | Notes | Phase |
|---|---|---|---|---|
| MagFace | CVPR 2021 (oral) | **Apache-2.0** (**H**). <https://github.com/IrvingMeng/MagFace> | Quality = embedding magnitude, a by-product of a recognition model. Needs MagFace's own recognition weights; check their training-data terms (U). | P1 |
| CR-FIQA | CVPR 2023 | **CC BY-NC 4.0** (**H**). <https://github.com/fdbtrs/CR-FIQA> | Non-commercial; evaluation only. | eval |
| SCRFD detection score, face size, landmark pose | already in stack | InsightFace code MIT; **pretrained models non-commercial research only**, with contact addresses for commercial terms (**H**). <https://github.com/deepinsight/insightface> | Cheap proxies; already computed. ATTRIBUTION.md already tracks this. | P1 |

People prominence is our own feature, assembled from published cues. Per
photo and per person:

- face area fraction;
- centrality;
- sharpness;
- frontal pose;
- eyes open;
- the person's importance.

Importance comes from the library: frequency, co-occurrence with "me", and
named or not. Apple's knowledge graph uses "important groups of people"
(**H**), and Google cites "all the kids are smiling" (**H**). Exclusion
(§5) multiplies prominence by 0 (blocked) or applies a down-weight (show
less).

### 3.6 Highlight and clip selection (video memories; P3/P4)

| Tool / model | Purpose | Licence | Compute (est.) | Phase |
|---|---|---|---|---|
| TransNetV2 | Shot-boundary detection (F1 96.2 on BBC Planet Earth, per README) | **MIT** (**H**). <https://github.com/soCzech/TransNetV2> | Small CNN on low-res frames; GPU-friendly | P3 |
| PySceneDetect | Content/threshold scene cuts, CPU | BSD-3 (**U**) | CPU | P3 |
| Moment-DETR / QVHighlights | Text-query moment retrieval plus per-2-second saliency | Code **MIT**; annotations **CC BY-NC-SA 4.0** (**H**). <https://github.com/jayleicn/moment_detr> | Needs CLIP/SlowFast features | P4 |
| Google Clips moment score | Frame quality averaged into a segment score; pick the peaks | Design (**H**) | n/a | P4 |
| Per-keyframe image signals | Reuse §3.2–3.5 on keyframes (faces, aesthetics, people) | ours | as in images | P3 |

### 3.7 Music, pacing and polish (the "make it professional" brief)

**What the products do**

- **Apple.**
  - Music alignment and titling are driven by scene signals (**H**, ANSA).
  - Memory Mixes suggest songs; the 12 Memory Looks apply a uniform colour
    grade "designed to mimic what colorists at film studios do" (**M**).
  - Prompt-driven memory movies build "chapters and a narrative arc" (**H**).
  - Shares with Apple Music songs are blocked for copyright reasons (**M**,
    MacStories).
- **Google.**
  - Highlight videos "sync the video to the music". Templates with beat
    matching and automatic cuts arrived in Dec 2025, and photo-transition
    Beat Match was in testing (**M**).
  - Memories can be built on "shared shape or color" (**M**, product blog).
- **Published research.**
  - *audeosynth* (Liao, Yu, Liu, Cheng; ACM TOG 34(4), SIGGRAPH 2015)
    extracts music and video temporal features. It solves cut placement and
    sync as an optimisation by MCMC (**H**).
    <https://history.siggraph.org/learning/audeosynth-music-driven-video-montage/>
  - *AutoFlip* (Google, 2020) does saliency-aware reframing: shot detection,
    then face/object/text signals, then per-scene crop paths (MediaPipe, now
    "legacy") (**H**).
    <https://research.google/blog/autoflip-an-open-source-framework-for-intelligent-video-reframing/>

**Pipeline we can build, with open parts**

| Step | Approach | Open tools / models | Licence | Compute (est.) | Phase |
|---|---|---|---|---|---|
| Story and order | Chronological by default. A VLM proposes chapters ("arrival", "the beach", "dinner"), a title and one-line captions from date, place, people names and image descriptions. The user edits. | Our VLM service (vLLM catalogue) | per catalogue model | One VLM call per memory plus one per chapter | P2c |
| Shot list | Pick N stills per chapter: quality (§3.2) × prominence (§3.5) with MMR diversity (§3.4); exclusions applied first (§5). | ours | – | CPU | P2c |
| Music | User-supplied or licence-safe library; tag by mood and tempo. | Pixabay Content License: free, no attribution, no standalone redistribution (**H** summary). <https://pixabay.com/service/license-summary/>; Free Music Archive / Kevin MacLeod CC-BY tracks (**U**, licence per track) | – | – | P2c |
| Beat and downbeat tracking | Place cuts on beats or downbeats; longer holds on slow sections. | **Beat This!** (ISMIR 2024): code and weights **MIT** per README (**H**; one third-party claims CC BY-NC-SA, so verify the LICENSE file). <https://github.com/CPJKU/beat_this>. `librosa` beat tracker (ISC, **U**). **madmom**: code BSD, but **models CC BY-NC-SA 4.0** (**H**), so avoid it. <https://github.com/CPJKU/madmom> | Beat This runs on GPU or CPU; seconds per song (est.) | P2c |
| Ken Burns / pan-zoom | Pan from low to high saliency; never crop faces; keep subjects within safe margins. | **U-2-Net** salient-object detection, **Apache-2.0**, 176 MB full or 4.7 MB small model (**H**). <https://github.com/xuebinqin/U-2-Net>. Faces and boxes we already have | Small CNN, Triton-friendly | P2c |
| Colour consistency | One grade per memory: match the white balance and exposure of each still to a chapter reference, or apply a LUT "look". | FFmpeg `lut3d`/`colorbalance` filters (**U** filter names); Apple's Looks as the product reference (**M**) | CPU/GPU | P2c |
| Titles and captions | Title card (event name from calendar or VLM, date, place); lower-third captions per chapter. | FFmpeg `drawtext` (**U**) or an HTML/SVG renderer | – | P2c |
| Transitions | Cut on beat; cross-dissolve on slow sections; no gimmick wipes (pro convention, **U**). | FFmpeg `xfade` (**U**) | – | P2c |
| Render | Declarative timeline (JSON) rendered server-side with GPU encode. | **FFmpeg** (LGPL/GPL by build, **U**) with NVENC; **MoviePy v2** (MIT, uses FFmpeg; v1 unmaintained) (**H**). <https://github.com/Zulko/moviepy>. **Remotion** is *not* open for companies over 3 people: company licence required (**H**). <https://www.remotion.dev/docs/license/pricing.md> | GPU encode, about real-time or faster (est.) | P2c |
| Clips (video memories) | Shot detection, then clip score, then trim to beats. | TransNetV2 (MIT), Moment-DETR code (MIT; NC data) | GPU | P4 |
| Reframing | Saliency-aware crop paths for portrait or landscape output. | AutoFlip (MediaPipe, legacy; Apache-2.0 **U**) | CPU | P4 |

Where a VLM adds real value:

- chapter titles and captions grounded in date, place and names;
- choosing the narrative order across chapters;
- rejecting off-topic stills ("is this a receipt?");
- natural-language memory requests (Apple's "Create a memory movie", Google
  highlight video from a search).

Keep the VLM out of per-image hot paths. Run it per memory.

### 3.8 Models and methods worth adopting

| Item | Use | Licence | Phase |
|---|---|---|---|
| Adaptive time-gap + GPS event segmentation (PhotoTOC / Cooper style) | Events, trips, memory candidates | papers (free to implement) | P1 |
| Utility / screenshot / document head on our embedding | Exclude from memories and featured photos | ours (Apple pattern) | P1 |
| Aesthetic head on our embedding (LAION-style MLP), with CLIP-IQA prompt pairs as a baseline | Best shot, covers | LAION code Apache-2.0; method only for CLIP-IQA | P1 |
| MagFace-style or detector-score face quality, plus landmark eyes-open | People prominence, burst pick | MagFace Apache-2.0; check weights' data (U) | P1 |
| Near-duplicate stacks: time window plus embedding distance, optionally SSCD | Stacks, top pick | SSCD code MIT, weights U | P1 → P2b |
| MMR / cluster-then-cap diversity | Memory and slideshow selection | free | P2c |
| `holidays` (Python) | Holiday memories and exclusions | MIT, 250 country codes (**H**). <https://github.com/vacanza/holidays> | P2c |
| Beat This! | Beat-synced cuts | MIT (**H**, verify) | P2c |
| U-2-Net (small) | Ken Burns saliency | Apache-2.0 (**H**) | P2c |
| FFmpeg + MoviePy v2 | Render | LGPL/GPL (U) + MIT | P2c |
| TransNetV2 | Video shots | MIT | P3 |
| Moment-DETR (code only, our own data) | Text-query clip selection | MIT code | P4 |
| Q-Align | Offline judge and re-ranker | U | P2c (eval) |
| **Avoid shipping**: IQA-PyTorch (PolyForm NC), CR-FIQA (CC BY-NC), madmom models (CC BY-NC-SA), Remotion (company licence) | – | – | – |

---

## 4. Calendar integration

### 4.1 Google Calendar API (required)

**Scopes** (read-only choices). **H**.
<https://developers.google.com/workspace/calendar/api/auth>

| Scope | Grants | Use |
|---|---|---|
| `calendar.calendarlist.readonly` | List subscribed calendars | Calendar picker |
| `calendar.events.readonly` | "View events on all your calendars" | **Default choice**: titles, times, places, attendees |
| `calendar.events.owned.readonly` | Events on calendars the user owns | Tighter alternative |
| `calendar.readonly` | All calendars and settings, read-only | Broader than needed |

Request the narrowest set: `calendar.calendarlist.readonly` +
`calendar.events.readonly`. The scope page does not say which scopes are
"sensitive" or "restricted". It says public apps using certain scopes need
verification (**H**).

**OAuth flow for a self-hosted, local-first app.** **H**.
<https://developers.google.com/identity/protocols/oauth2/native-app>

- Use the installed-app flow: a **loopback IP redirect** on a random port,
  **PKCE S256**, and `state` for CSRF.
- The client secret is not confidential in installed apps. Custom URI
  schemes and the OOB copy-paste flow are no longer supported.
- Store the refresh token securely; it is "always returned for installed
  applications".
- **Pitfall.** A Cloud project with an External user type in "Testing"
  status gets refresh tokens that **expire after 7 days**. The fix is to
  publish the app, which may need verification, or use an Internal Workspace
  app. There is also a cap of 100 refresh tokens per account per client id.
  **H** (Google OAuth docs, via search).
  <https://developers.google.com/identity/protocols/oauth2>
- This shapes the setup UX: either each operator brings their own Cloud
  project in "In production" status, or OpenProcessor publishes a verified
  client. That is an owner decision.
- Our server is not a desktop app. The browser does the consent screen, so
  the redirect must land on the OpenProcessor host. A loopback redirect only
  works when the browser runs on the same host. A LAN deployment needs
  another documented redirect option (**U**; the planner must check Google's
  rules for web-server clients on private hosts).

**Incremental sync.** **H**.
<https://developers.google.com/workspace/calendar/api/guides/sync>

1. Do a full `events.list` and store `nextSyncToken`, which appears on the
   last page only.
2. Then send `syncToken` with the same query parameters every time;
   disallowed parameters return 400.
3. Results always include deleted entries.
4. Page with `pageToken` until a new sync token arrives.
5. On **410 Gone**, wipe local events and run a full sync.

Push notifications need a public HTTPS endpoint, which does not fit a
loopback-only deployment. Poll with sync tokens on a jittered schedule
instead; Google recommends randomising sync times by ±25% (**H**, quota
page).

**Quotas.** 10,000 requests/min per project, 600/min per user per project.
Rate-limit responses are 403 or 429 with `usageLimits`. Use truncated
exponential backoff (max 32–64 s). **H**.
<https://developers.google.com/workspace/calendar/api/guides/quota>. One
library's sync costs a handful of requests, far inside the limits (est.).

**Event fields useful for memories.** **H**.
<https://developers.google.com/workspace/calendar/api/v3/reference/events>

| Field | Use |
|---|---|
| `summary`, `description` | Memory title and VLM context |
| `location` | Free text; geocode it and check it against photo GPS |
| `start` / `end` | `date` for all-day, `dateTime` + `timeZone` for timed; the photo-matching window |
| `attendees[]` (`email`, `displayName`, `self`, `responseStatus`) | Link to named People when the user maps them; never automatic |
| `recurrence`, `recurringEventId`, `originalStartTime` | Annual events (anniversaries); occurrence identity |
| `eventType` (birthday, focusTime, outOfOffice, workingLocation, fromGmail), `birthdayProperties` | Birthday memories; skip focus and working-location noise |
| `status` (cancelled), `transparency`, `visibility` | Drop cancelled events; respect private events |

**Local-first privacy design.**

- Tokens are stored on the host only, encrypted at rest (key from `.env`).
  Never log them.
- Event data is stored in the project's own store and never sent anywhere
  else. The only outbound traffic is to Google's API.
- The user chooses which calendars to sync and can **disconnect and purge**
  (revoke the token and delete the events).
- Matching runs locally. Event titles reach the VLM only if the VLM is local
  or the user agrees.
- Store only the fields we use. Drop `conferenceData` and attendee emails
  unless the user maps them to People.

### 4.2 Secondary paths

| Path | How | Library | Licence | Notes |
|---|---|---|---|---|
| ICS file import | Apple Calendar: File > Export > Export (.ics) (**H**). <https://support.apple.com/guide/calendar/import-or-export-calendars-icl1023/mac>. Google: Import & export, which gives a .zip of .ics files, or the "Secret address in iCal format" (**H**). <https://support.google.com/calendar/answer/37111?hl=en> | `icalendar` (BSD, **U**) + `recurring-ical-events` for RRULE/EXDATE expansion | LGPL-3.0 (**H**) | A snapshot: it does not update itself |
| ICS URL subscription | Poll the secret iCal URL | same | – | The URL is a secret: store it like a token |
| CalDAV (iCloud, Nextcloud, Fastmail) | iCloud needs 2FA and an **app-specific password** (**H**). <https://support.apple.com/en-us/102654> | `caldav` | GPL-3.0-or-later **OR** Apache-2.0 (**H**). <https://pypi.org/project/caldav/> | Pick the Apache-2.0 side |

### 4.3 Matching events to photos

- Normalise everything to UTC with the photo's capture time-zone offset
  (EXIF `OffsetTimeOriginal` when present, **U** tag name). Camera clocks
  drift, so allow a per-camera offset correction (§1.4).
- All-day events: use the local day in the event's time zone. For multi-day
  trips, overlap the event range with event segments (§3.1). An event "owns"
  a segment when most of the segment's photos fall inside it.
- Raise confidence when the GPS matches the geocoded `location`. Holidays
  come from `holidays` by country and region of the home location.
  Birthdays come from the `birthday` event type or the person record.
- Store matches as suggestions with a reason (time overlap, GPS distance).
  Never silently rename a user's album.

---

## 5. Excluding people

**What vendors offer**

| Product | Control | Scope | Source |
|---|---|---|---|
| Apple | Feature This Person Less / Never Feature; per day, place, holiday | Memories, Featured Photos, widget; reset available | **H** (personal-safety guide) |
| Apple | Hide this Person | People album only; the photos stay | **H** (Mac People guide) |
| Google | Hide faces: Show less vs Blocked | Featured memories carousel only; "may not work perfectly" | **H** |
| Immich | Hide person | Explore and the detail view only; no memory-level block | **H** |

None of them offers "remove from every library view". All three keep the
photos in the timeline and search. Google states its limits. A third-party
guide notes Apple misses photos whose face is not recognised (**M**,
Pocket-lint).

**What a robust "exclude this identity everywhere" needs**

The scope the owner asked for: memories, featured photos, smart albums,
search results and, optionally, the timeline.

1. **High-recall identity capture.**
   - Seed the identity from the named cluster, then search the face index
     with a *looser* threshold than for naming.
   - Use the face-plus-upper-body within-moment trick from Apple (**H**) to
     catch profile shots, backs of heads and small faces.
   - Recall matters more than precision here: a false hide is cheap and
     reversible, a missed one is the harm.
2. **Review queue for the uncertain band.** Matches between the naming
   threshold and the loose threshold go to a Review queue (Cropwright
   already has Review). Each one is confirmed or rejected, and the answers
   feed the identity's exemplars. Apple uses multiple exemplars per cluster
   (**H**).
3. **Exclusion as a first-class predicate, applied at query time.** Every
   library query (timeline, search, albums, memories, slideshow builder)
   filters through the exclusion set, so it holds for photos added later.
   Do not delete or move anything; the originals stay in place.
4. **Propagation to new photos.** Incremental face assignment runs at
   ingest. A new face near an excluded identity is excluded at once, then
   queued for review if it is in the uncertain band.
5. **Levels.**
   - *Block*: never in any created item, including group shots (Google).
   - *Show less*: no memories about them; group shots allowed.
   - *Hide from views*: excluded from the timeline and search as well.
   - Each level is reversible, and there is a full reset (Apple).
6. **Fail closed.** If an identity's assignment is stale or face processing
   has not finished on a photo, that photo is not eligible for *created*
   outputs (memories, slideshows) until processed. This matches the
   project's fail-closed rule.
7. **Beyond faces.**
   - The same predicate takes object classes (our detector) and text (OCR).
   - Dates and places are exclusions too (Google hide dates, Apple feature
     a day or place less).
8. **Video (P3).** Faces sampled from keyframes and tracked through shots.
   A clip containing a blocked identity is not eligible for P4 memories.
9. **Audit.** Keep a per-identity log of what was excluded and why
   (threshold, review decision), so the user can verify coverage.

Phase: the identity and the review queue in P2b; the exclusion predicate
everywhere in P2b; video in P3.

---

## 6. Video (later phases)

OpenProcessor processes images only today.

| Need | Detail | Tools | Phase |
|---|---|---|---|
| Decode and transcode | GPU decode (NVDEC) for analysis; H.264/HEVC proxies for web playback; HLS for long clips. Immich added web real-time transcoding and HLS in v3.0 (**H**, roadmap). | FFmpeg with NVDEC/NVENC (**U** build licence); the project's GPU-decode plan | P3 |
| Keyframes | Shot boundaries (TransNetV2) plus uniform sampling inside shots; cap per minute. | TransNetV2 (MIT) | P3 |
| Faces and people in video | Detect on keyframes, track inside shots, one identity per track (the face-quality-best frame becomes the exemplar). | existing SCRFD/ArcFace + a tracker (U) | P3 |
| Video search | Embed keyframes; OCR on keyframes; per-clip time ranges in results. | existing MobileCLIP2/PE, PP-OCRv5 | P3 |
| Clip scoring | Keyframe quality × people prominence × motion; Clips-style segment average and peak picking (**H** design). | ours | P4 |
| Live / motion photos | Pair the still with its clip; the clip is a free short memory segment. | Immich supports LivePhoto/MotionPhoto (**H**) | P3 |
| Memory video | Mixed stills and clips cut to beats (§3.7); clip trims snap to beats. | §3.7 stack | P4 |

---

## 7. Open questions for the planner

1. **P2c scope.** Is a stills-only rendered slideshow (MP4) part of P2c, as
   proposed here, or does all rendering wait for P4?
2. **Google OAuth client.** Each operator brings a Cloud project in
   production status, or OpenProcessor publishes a verified client. LAN
   deployments need a documented redirect path (§4.1).
3. **Aesthetic head training data.** Which aesthetic labels may we use
   commercially (AVA terms are U)? Or should the head train only on owner
   picks?
4. **Exclusion default level.** Should "Hide from views" (timeline and
   search) be the default, or only the memory-level block?
5. **XMP sidecar write-back.** Do we write sidecars at all, given the
   "originals untouched" rule? Sidecars are separate files next to the
   originals, which may still count as touching the library folder.
6. **Licences to clear before adoption.** Verify each before adopting:
   - Beat This! weights;
   - SSCD weights;
   - MagFace weights' training data;
   - Q-Align;
   - Immich's `justified-layout` crate;
   - FFmpeg build flags.
