# 00d: the research owner questions, de-duplicated (one list)

| Field | Value |
|---|---|
| Status | open, written 2026-11-04 by the research integration pass; none answered yet |
| Source | the section 10 owner questions of the seven research plans (112 raw: GPU pipeline J1-J12, clustering J1-J17, embeddings J1-J14, near-duplicates J1-J15, spatial J1-J15, plate text J1-J13, auto-label J1-J26). Merged where two plans ask the same thing; the "From" column keeps every source, so no raw question is lost |
| How to answer | one word per row is enough ("yes", "no", or the other option). Rows marked **answer before coding** block wave W2.0 or W2.1 (they change a W2.1 lane's scope or the first prototype window); every other row is needed only by the wave named. The plan already follows each recommendation and marks the place "pending RD-n", so a "no" is a small, named removal |
| Where answers go | the decision register of [00_index.md](00_index.md) section 6 (one row per answered RD-n), then the research plan's own J-row is marked answered |

## A. Placement in the program

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|
| RD-1 | **Answer before coding.** Build every research seam into the v0.7.0 rebuild now (the tables of [02d](02d_research_seam_schema.md), the job-control changes, the new lanes 1F, ES-A, ES-B, ES-C, 4F, ND-A, ND-B, ND-C and the W4 capture lane), even where the research winner lands later? | yes; each seam has a 0.7.0 writer and reader. Per-topic fallbacks if you say no to one: clustering records runs only (no memberships or activation); the re-index job moves to v0.8.0 while the registry and readers stay | W2.1 | gpu J4; clu J1; emb J1; nd J1 (seams part); al J2; sp J1 (seams part); pt section 5 (creation-time seams) |
| RD-2 | **Answer before coding.** Run the prototypes that feed a v0.7.0 decision on idle GPU time during W2-W4 (windows PW-1..PW-3 of [00e](00e_research_track.md), about 31 GPU hours: GPU PR-0..PR-6, plate P0-P6, near-duplicates P0-P2, spatial P-0/P-1 on CPU), amending MQ-9; everything else after the tag (RG-1..RG-6, about 62 GPU hours)? | yes; they touch no product file and run only when the program uses no GPU | W2.1 (PW-1 starts there) | gpu J1; nd J1 (timing); pt J4; al J1; sp J9; clu J15; emb section 6.5 |
| RD-3 | Land registration-plate text inside v0.7.0 as its own wave W5e, after W5d and before W6? | yes (you named plate text a must; it needs W5c's OCR output first) | before W5c | pt J3 |
| RD-4 | **Answer before coding.** The plate tables in their own revision in W5e, or in the W2.1 revision now? | their own revision in W5e (keeps W2.1 smaller; Alembic adds them cleanly) | W2.1 | pt J7 |
| RD-5 | Capture metadata at ingest (time with offset and source, camera, lens, focal length, a `has_location` flag) in W4, with a capture-date filter as its reader? | yes; it is the only signal that waiting can lose | W4 | sp J1 |
| RD-6 | One GitHub milestone per post-tag research wave (RW-1..RW-6), with #466 and #398 staying on v0.8.0? | yes | after the tag | gpu J11; al J16 |

## B. Research rules: code, data, people

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|
| RD-7 | **Answer before coding.** Track the prototype code in `scripts/research/<topic>/` with exact pins and a committed lock file (not a gitignored folder)? | yes; every published number needs a reproduce command from a clean checkout | PW-1 | gpu J2; clu protocol section 2; emb 6.1; nd 6.1; al 6.1; sp section 8 R-1; pt section 8 |
| RD-8 | **Answer before coding.** Set P: a seeded sample copied to local NVMe per topic; public text says only "met / not met on set P"; an aggregate number only with your yes per note; never an image, name, path, location value or plate string? | yes | PW-1 | gpu J3; clu J7; nd J12; sp J11; al J11 |
| RD-9 | Set P, faces and plates never go to a hosted model unless you name a sample; faces on set P are judged by the local VLM and you only? | yes | PW-1 | clu J8, J9 |
| RD-10 | **Answer before coding.** May the research download and use, locally only and never committed, publishing aggregates with citations: LFW, CALFW, CPLFW; CO3D and Objectron; the COCO and Open Images 1M crop set; California-ND, INRIA Holidays and Copydays, a DISC21 sample, SSCD weights; MSLS, the YFCC100M metadata and Exif pack, Pascal3D+, PACO-LVIS (after its annotation licence is read), MegaScenes; UFPR-ALPR and LPLC (RodoSol only if needed)? | yes, local only, aggregates only | PW-1 (Copydays, DISC21, UFPR-ALPR, LPLC); the rest after the tag | clu J10, J11; emb J6; nd J11; sp J10; pt J5 |
| RD-11 | Models with unclear training-data or weight terms (SALAD, AnyLoc ViT-G, Orient Anything, the colour-name table, GeoCalib, SSCD, a dedicated plate reader): measurement only, and shipping only as an opt-in installer group with the gap stated in ATTRIBUTION? | yes | PW-2 | sp J7; nd J9 (SSCD); pt J6 |
| RD-12 | About 15 hours of your (or a delegate's) time over the research: the clustering blind re-judge (8-10 h, a second person for a 30-sheet human ceiling if one exists who has not seen the disagreement lists), the auto-label review study (about 1 h, Label Studio in the research venv), plate transcription, two near-duplicate and spatial checks; a CSV with aliased images as the re-judge tool until Cropwright has review screens? | yes | after the tag (plate transcription in PW-1) | clu J12, J14; al J8 |
| RD-13 | The hosted visual judge only after it passes planted-sheet agreement on public data; the Message Batches client only if you provide a key? | yes | after the tag | clu J13; al J26 |

## C. GPU and serving

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|
| RD-14 | If CPU Huffman frees enough GPU time, let a deployment choose CPU or GPU Huffman in its serving profile (never adaptive at run time)? | yes, one measured default | W5b | gpu J5 |
| RD-15 | A C++ Triton backend in the repository if the through-Triton gap reaches the bar? | only on the bar, with its own plan file | W5b | gpu J6 |
| RD-16 | One Triton per card as the two-GPU default? | decide after `gpu:PR-7`; either answer is a configuration change | RW-1 | gpu J7 |
| RD-17 | Keep the custom ctypes nvJPEG lanes until a library reaches GPU Huffman at real batch sizes with equal pixels? | yes; re-test at every Triton image move | W5b | gpu J8 |
| RD-18 | DCGM profiling fields in the bench and monitoring stacks (the exporter may need extra capabilities)? | yes; off in the minimal install | W2.4 | gpu J9 |
| RD-19 | One INT8 second round (3 GPU hours) after the tag, closing the line for good if it fails? | yes | RW-1 | gpu J10 |
| RD-20 | Hardware with a JPEG decoder (A100 or H100 class) in scope? | no; the `hw_jpeg_decoder` column keeps the door open | - | gpu J12 |
| RD-21 | May the PE text tower move to a GPU TensorRT plan if it passes, with the CPU plan kept selectable per deployment? | yes | RW-2 | emb J8 |
| RD-22 | A VRAM budget for an optional embedder (2 GiB text, 4 GiB vision); PE-Core-G not offered unless you ask? | yes | RW-2 | emb J9 |

## D. Clustering

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|
| RD-23 | **Answer before coding.** Membership snapshots in PostgreSQL (about 150 MB per 1M items) rather than a Parquet file per partition? | PostgreSQL: diff, carry-over, rollback and review are queries | W2.1 (1F's tables) | clu J2 |
| RD-24 | Activate a new partition automatically when it moves no anchored or validated item and keeps id stability at 0.95 or more, else wait for a person? | yes | W2.3 | clu J3 |
| RD-25 | A dedicated `cluster` queue and `cluster-worker`, CPU in v0.7.0; RAPIDS only after the GPU agreement run passes? | yes | W2.1 | clu J4 |
| RD-26 | igraph and leidenalg (GPL) as product dependencies if Leiden wins? | decide only on a Leiden PASS; fallback scikit-learn average linkage | RW-3 | clu J5 |
| RD-27 | Average-linkage clustering above 150k items: block-wise with scikit-learn, no C++ build in this repository? | yes | RW-3 | clu J6 |
| RD-28 | Verification tables and a local `cluster_verify` job in the product with the Cropwright review screens (v0.8.0)? | yes, later | v0.8.0 | clu J16 |
| RD-29 | Move the cluster-refresh and IVF-retrain environment values into the `cluster_profile` and retire the variables? | yes, with the partition change in W2.3 | W2.3 | clu J17 |

## E. Embeddings and search

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|
| RD-30 | Keep the 0.6.0 vector field names as registry data instead of renaming them in the upgrade? | keep (a rename rewrites every vector for no behaviour change) | W2.2 | emb J2 |
| RD-31 | Delete the in-process torch and onnx text encoders in W4 (every service requires Triton under D-69)? | yes | W4 | emb J3 |
| RD-32 | Bump OpenSearch 3.6 to 3.9 only after the 1M index runs, as its own change with a parity gate? | yes, not inside v0.7.0 | RW-2 | emb J4 |
| RD-33 | Combine across different embedding spaces: copy matching spaces and mark the rest deferred (refuse only when the target lacks a target type)? | yes | W2.3 | emb J7 |
| RD-34 | The core `visual_search_*` indexes stay outside the registry in v0.7.0? | yes | W2.2 | emb J10 |
| RD-35 | The in-product evaluation is a hard gate for a space flip, with `force=true` as your waiver? | yes | W2.2 | emb J11 |
| RD-36 | New box spaces as separate nested fields (write isolation) rather than a second leaf? | yes | W2.2 | emb J12 |
| RD-37 | Adopt the #347 plan's calls as a block (staged A-D with its stop rule, generic names in tracked files, publish aggregates, the catalog's default VLM for captions, multilingual skipped), with the embeddings plan owning the encoder and frame experiments and clustering reading their verdicts? | yes | RW-2 | emb J5, J13 |
| RD-38 | Close #435 if PE frame search misses its bar, else build a frame route over the images index (no third `target`)? | yes | RW-2 | emb J14 |

## F. Near-duplicates

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|
| RD-39 | At a holdout freeze, move the held-out item's item-level group only (not every object on near-duplicate frames)? | yes | W2.3 | nd J2 |
| RD-40 | A separate looser "leak" cut for the holdout, equal to the dedup cut until the leakage prototype? | yes | W2.3 | nd J3 |
| RD-41 | Freeze and training export refuse on stale near-duplicate state (409, with a rescore button) instead of starting a rescore themselves? | yes | W2.3 | nd J4 |
| RD-42 | After a freeze, move new near-duplicates of held-out items into the holdout automatically (audited), rather than only excluding them from training? | yes | W2.3 | nd J5 |
| RD-43 | SHA-256 of the bytes replaces `imohash` as the exact duplicate key, enforced by a unique constraint? | yes; the cost is measured | W2.3 | nd J6 |
| RD-44 | Core `/ingest` groups asynchronously (the response lists candidates; the group appears after the core job)? | yes (a wire change for core clients) | W4 | nd J7 |
| RD-45 | Export: `collapse_near_duplicates` reading the stored groups replaces the per-request `dedup_threshold`? | yes | W2.3 | nd J8 |
| RD-46 | A perceptual hash tier: PDQ only if it passes, else none; a copy-detection model only past its stop rule? | yes | RW-4 | nd J9 |
| RD-47 | Per-project thresholds through a revisioned `near_dup_profile`, seeded from today's values? | yes | W2.3 | nd J10 |
| RD-48 | Region boxes inherit their item's group (no region-level grouping)? | yes | W2.3 | nd J13 |
| RD-49 | Retire `OP_CROP_DUP_THRESHOLD` instead of validating it? | yes | W2.3 | nd J14 |
| RD-50 | Rename the core index fields to `near_dup_*`, keeping the `/query/duplicates` answers? | yes | W4 | nd J15 |

## G. Spatial attributes and capture metadata

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|
| RD-51 | **Answer before coding.** Store no GPS coordinates in v0.7.0 (only `has_location`); coordinates later behind a per-project policy whose default is "drop"? | yes (a "store now" answer adds one column to the W2.1 schema) | W2.1 | sp J2 |
| RD-52 | Never store a camera body serial raw (a keyed hash only if bursts need it)? | yes | W4 | sp J3 |
| RD-53 | Back-fill capture metadata on upgraded 0.6.0 projects (header-only read; a missing file counted)? | yes | W4 | sp J4 |
| RD-54 | Strip embedded GPS losslessly from exported copies (ExifTool in the jobs image) unless the export asks to keep it, rather than only documenting it? | strip | W4 | sp J5 |
| RD-55 | Colour and view tags after the tag, each only after its prototype bar? | yes | RW-5 | sp J6 |
| RD-56 | pycolmap only for matching and reconstruction (Kornia only if throughput is too low)? | yes | RW-5 | sp J8 |
| RD-57 | Dense reconstruction, splats and a viewer only after a GO (COLMAP dense plus gsplat first; OpenMVS only with its AGPL terms reviewed)? | yes | RW-5 | sp J12 |
| RD-58 | A geo-photo timeline in v0.8.0 under the location policy; reverse geocoding only if it needs place names? | yes | v0.8.0 | sp J13 |
| RD-59 | A scene product lives in curation, project-scoped, not in the core API? | yes | RW-5 | sp J14 |
| RD-60 | The former #400 calls: run the attribute experiments on the research harness now, and file "one vote per near-duplicate group" as its own loop item? | yes to both | RW-5 | sp J15 |

## H. Registration-plate text

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|
| RD-61 | Default privacy mode and deployment ceiling `raw` (today's behaviour), operators lowering the ceiling? | yes | W4 | pt J1 |
| RD-62 | Approve the privacy model (modes off, keyed, raw; the ceiling; master secret plus per-project salt; a CLI-only purge; the sink test)? | yes | W4 | pt J2 |
| RD-63 | Defer keyed-mode near-match keys until the misread rate is measured? | yes | W5e | pt J8 |
| RD-64 | No issuing-jurisdiction field in v0.7.0? | yes | W5e | pt J9 |
| RD-65 | Raw-mode retention optional (none keeps text until a purge)? | yes | W5e | pt J10 |
| RD-66 | Consensus writes the group's reading into `text` and keeps each frame's own reading in `text_frame`? | yes | W5e | pt J11 |
| RD-67 | A detector trained on your private plate-box set may be used in the plate-box prototype and in your deployments, never shipped publicly? | yes | PW-3 | pt J12 |
| RD-68 | Burst grouping for plates reads the W4 capture time (no separate `captured_at` field), if the linking prototype shows time windows beat near-duplicate groups? | yes | W5e | pt J13; sp section 5.1 |

## I. The auto-label assistant

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|
| RD-69 | Make the `autolabel` job pausable at stage and chunk boundaries too (assist runs stay cancel-only)? | yes | W2.3 | al J3 |
| RD-70 | Cancel is final for every job kind; "stop for now" is pause (today a cancelled loop can resume)? | yes | W2.2 | al J4 |
| RD-71 | When a budget limit is reached, pause the run so the budget can be raised, rather than completing it? | pause | W2.3 | al J5 |
| RD-72 | Propagated labels stay out of the training export until a person accepts them (revisit after the teacher-student prototype)? | yes | W2.3 | al J6 |
| RD-73 | Fold the `autolabel` job's VLM sweep into the loop as `strategy: sweep` in v0.8.0? | yes | v0.8.0 | al J7 |
| RD-74 | Read the daily VLM cap from `vlm_usage_hourly` and allow an optional price per million tokens on an endpoint, both in v0.8.0? | yes | v0.8.0 | al J9 |
| RD-75 | Retention: loop proposals and selections deleted 90 days after the run, `vlm_usage_hourly` kept 400 days? | yes | W2.1 | al J10 |
| RD-76 | Box proposals stay in the region stage, not in the loop (revisit after #84)? | yes | - | al J12 |
| RD-77 | Loop slice length 600 s with a 30 / 35 minute slice limit, tuned after the live test? | yes | W2.3 | al J13 |
| RD-78 | The first product precision claim needs a Wilson lower bound of at least 0.93 and a point estimate of at least 0.95 against a blind human audit, on a public 10,000-image project? | yes | RW-6 | al J14 |
| RD-79 | The loop starts a training run only on a click (never by itself, never a promotion) in the first release? | yes | v0.8.0 | al J15 |
| RD-80 | Adopt the detector-class-roots (Z5) protocol calls as a block: about 35,200 train images with ids and hashes only; bar A; `within_key` unbarred; the `k<hash>` root key with `root_name` on the wire; `detector_class` with hold-back as the default after adoption; the IVF pool including `none`; no root cap but an audit above 20,000 members; a held-back review set; keys by normalized name; the hosted judge only after its agreement check? | yes | RW-6 | al J17-J26 |

## Count

112 raw questions; 80 after merging (6 placement, 7 rules, 9 GPU, 7 clustering, 9 embeddings,
12 near-duplicates, 10 spatial, 8 plate text, 12 auto-label). **Answer before coding: RD-1, RD-2,
RD-4, RD-7, RD-8, RD-10, RD-23, RD-51** (eight).
